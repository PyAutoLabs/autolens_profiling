"""vmap-batch memory probe utilities.

Given a JAX-traceable likelihood function and a parameter pytree, measure
the compiled program's VRAM footprint at two batch sizes and extrapolate
the maximum batch_size that fits a target device memory budget.

The probe avoids running the steady-state timing loop — it only does the
compile + first-call sequence required to read ``compiled.memory_analysis()``.
Two batch sizes (default 2 and 4) are needed so we can decompose the
total program memory into a per-replica linear coefficient + a constant
overhead term:

    memory_at_batch(B) ≈ constant_overhead + B * per_replica_memory

Then the largest batch fitting in ``vram_budget`` is::

    floor((vram_budget - constant_overhead) / per_replica_memory)

In practice JAX rematerialisation can make the relationship sub-linear at
large batch, so we add a safety floor (``vram_budget`` defaults to ~70 GB
on an 80 GB A100) and cap at ``max_batch`` (default 64) to keep compile
times tractable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProbeSample:
    """One probe sample: the compiled vmap program's memory cost at a given batch.

    ``peak_bytes`` is the field XLA reports as ``peak_memory_in_bytes`` on the
    compiled program — the maximum simultaneous device allocation across the
    full computation, including rematerialised activations. This is what we
    extrapolate against for sizing the production batch.

    ``output_bytes`` + ``temp_bytes`` are retained for backward compat with older
    JAX versions that don't expose ``peak_memory_in_bytes`` (we fall back to
    the sum, which over-counts but is conservative).
    """

    batch_size: int
    peak_bytes: int
    output_bytes: int = 0
    temp_bytes: int = 0

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1024**2

    @property
    def total_mb(self) -> float:
        """Conservative upper bound (output+temp sum). Use ``peak_mb`` if present."""
        return (self.output_bytes + self.temp_bytes) / 1024**2

    @property
    def effective_mb(self) -> float:
        """The memory number to extrapolate against — prefer peak, fall back to sum."""
        if self.peak_bytes > 0:
            return self.peak_mb
        return self.total_mb


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a vmap memory probe across multiple batch sizes."""

    dataset: str
    model: str
    instrument: str
    samples: list[ProbeSample] = field(default_factory=list)

    @property
    def per_replica_mb(self) -> float:
        """Linear coefficient of (peak) program memory vs batch_size.

        Computed from the two extreme samples. If only one sample exists,
        we fall back to ``effective_mb / batch_size`` (assumes zero constant
        overhead — conservative for the upper bound).
        """
        if len(self.samples) < 2:
            s = self.samples[0]
            return s.effective_mb / s.batch_size
        s_lo, s_hi = self.samples[0], self.samples[-1]
        return (s_hi.effective_mb - s_lo.effective_mb) / (s_hi.batch_size - s_lo.batch_size)

    @property
    def constant_overhead_mb(self) -> float:
        """Intercept of the program-memory-vs-batch linear fit."""
        if len(self.samples) < 2:
            return 0.0
        s_lo = self.samples[0]
        return s_lo.effective_mb - s_lo.batch_size * self.per_replica_mb


def probe_vmap_memory(
    func,
    args_pytree,
    batch_sizes: Sequence[int] = (1, 4),
    *,
    dataset: str = "",
    model: str = "",
    instrument: str = "",
) -> ProbeResult:
    """JIT-vmap ``func`` at each batch size, read ``compiled.memory_analysis()``.

    ``func`` must accept a pytree of parameters whose leaves are JAX arrays.
    ``args_pytree`` is the single-replica pytree (NOT pre-batched); we
    broadcast it to each batch size internally.

    Reads ``peak_memory_in_bytes`` when available (post-JAX-0.4.X) — this is
    the maximum simultaneous device allocation including rematerialised
    activations, and is the correct number to extrapolate against. Falls back
    to ``output_size_in_bytes + temp_size_in_bytes`` for older JAX versions
    (an over-estimate, but conservative).

    For cells with expensive compile (delaunay: 10-30 min/batch), call with
    ``batch_sizes=(1,)`` and accept single-point extrapolation. For cheap-compile
    cells (mge / pixelization: ~10 s/batch), use ``(1, 4, 16)`` for a multi-point
    fit that catches XLA rematerialisation non-linearity.
    """
    import jax
    import jax.numpy as jnp

    samples: list[ProbeSample] = []
    for B in batch_sizes:
        parameters = jax.tree_util.tree_map(
            lambda leaf, B=B: jnp.broadcast_to(leaf, (B, *leaf.shape)),
            args_pytree,
        )
        vmapped = jax.jit(jax.vmap(func))
        lowered = vmapped.lower(parameters)
        compiled = lowered.compile()
        mem = compiled.memory_analysis()
        peak_bytes = int(getattr(mem, "peak_memory_in_bytes", 0))
        samples.append(
            ProbeSample(
                batch_size=int(B),
                peak_bytes=peak_bytes,
                output_bytes=int(mem.output_size_in_bytes),
                temp_bytes=int(mem.temp_size_in_bytes),
            )
        )
    return ProbeResult(dataset=dataset, model=model, instrument=instrument, samples=samples)


def recommend_batch_size(
    probe: ProbeResult,
    *,
    vram_budget_gb: float = 65.0,
    safety_factor: float = 1.15,
    max_batch: int = 64,
) -> int:
    """Recommend the largest batch_size that fits ``vram_budget_gb`` on device.

    Computed by linear extrapolation of (peak) memory vs batch_size from the
    probe samples; capped at ``max_batch`` to keep XLA compile time tractable.

    Defaults targeted at A100 80 GB:
    - ``vram_budget_gb=65`` — leaves ~15 GB headroom for JAX runtime, CUDA driver
      allocations, allocator fragmentation, and per-call activation slack that
      static analysis doesn't account for.
    - ``safety_factor=1.15`` — multiplier on the static peak estimate, per the
      industry rule-of-thumb that XLA static analysis under-counts the real
      runtime peak by ~10-15% on complex graphs.
    - ``max_batch=64`` — compile time scales superlinearly with batch on some
      cells (notably delaunay), and 64 is roughly where diminishing returns kick
      in for production samplers.
    """
    budget_mb = vram_budget_gb * 1024
    per_replica = probe.per_replica_mb * safety_factor
    overhead = probe.constant_overhead_mb * safety_factor
    if per_replica <= 0:
        return max_batch
    raw = int((budget_mb - overhead) / per_replica)
    return max(1, min(raw, max_batch))


def write_probe_json(
    probe: ProbeResult,
    recommended_batch_size: int,
    output_path: Path,
    *,
    vram_budget_gb: float = 65.0,
    safety_factor: float = 1.15,
    max_batch_cap: int = 64,
    extra: dict | None = None,
) -> None:
    """Serialise probe samples + recommendation to JSON."""
    import jax

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": jax.default_backend(),
        "dataset": probe.dataset,
        "model": probe.model,
        "instrument": probe.instrument,
        "samples": [asdict(s) for s in probe.samples],
        "per_replica_mb": round(probe.per_replica_mb, 3),
        "constant_overhead_mb": round(probe.constant_overhead_mb, 3),
        "vram_budget_gb": vram_budget_gb,
        "safety_factor": safety_factor,
        "max_batch_cap": max_batch_cap,
        "recommended_batch_size": recommended_batch_size,
    }
    if extra is not None:
        payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2))
