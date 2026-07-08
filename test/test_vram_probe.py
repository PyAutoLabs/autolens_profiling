"""Unit tests for ``vram.probe`` extrapolation math.

These tests construct synthetic ``ProbeResult`` objects (no JAX dependency) and
verify the linear-fit + budget-extrapolation logic. Lets us iterate on the math
without paying for HPC probe-job cycles.

Run::

    cd autolens_profiling
    python -m pytest test/test_vram_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `vram` importable without installing the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vram.probe import ProbeResult, ProbeSample, recommend_batch_size  # noqa: E402


def _peak_sample(batch: int, peak_mb: float) -> ProbeSample:
    """Helper: construct a ProbeSample with only peak_bytes set."""
    return ProbeSample(batch_size=batch, peak_bytes=int(peak_mb * 1024**2))


def _legacy_sample(batch: int, output_mb: float, temp_mb: float) -> ProbeSample:
    """Helper: ProbeSample with no peak (legacy JAX fallback to output+temp)."""
    return ProbeSample(
        batch_size=batch,
        peak_bytes=0,
        output_bytes=int(output_mb * 1024**2),
        temp_bytes=int(temp_mb * 1024**2),
    )


def test_per_replica_two_point_fit():
    """Linear coefficient from two samples."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[_peak_sample(1, 100.0), _peak_sample(4, 400.0)],
    )
    assert probe.per_replica_mb == 100.0
    assert probe.constant_overhead_mb == 0.0


def test_per_replica_with_overhead():
    """Linear fit with non-zero intercept."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[_peak_sample(1, 200.0), _peak_sample(4, 500.0)],
    )
    # peak(B) = 200 + (B-1)*100 = 100 + 100*B → per_replica=100, overhead=100
    assert probe.per_replica_mb == 100.0
    assert probe.constant_overhead_mb == 100.0


def test_per_replica_single_sample_assumes_zero_overhead():
    """Single sample falls back to peak / batch as the per-replica estimate."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[_peak_sample(4, 800.0)],
    )
    assert probe.per_replica_mb == 200.0
    assert probe.constant_overhead_mb == 0.0


def test_recommend_capped_at_max_batch():
    """Tiny per-replica → recommendation is capped, not unbounded."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[_peak_sample(1, 1.0), _peak_sample(4, 4.0)],
    )
    assert recommend_batch_size(probe, max_batch=64) == 64


def test_recommend_respects_budget():
    """Large per-replica → recommendation drops below the cap."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[_peak_sample(1, 5_000.0), _peak_sample(4, 20_000.0)],
    )
    # per_replica = 5 GB. budget = 65 GB, safety = 1.15 → 5.75 GB / replica → ~11.3
    rec = recommend_batch_size(probe, vram_budget_gb=65.0, safety_factor=1.15)
    assert 10 <= rec <= 12, f"expected ~11, got {rec}"


def test_recommend_floor_at_1():
    """Per-replica > budget → still returns at least 1."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[_peak_sample(1, 200_000.0)],  # 200 GB per replica — absurd
    )
    rec = recommend_batch_size(probe, vram_budget_gb=65.0, safety_factor=1.15)
    assert rec == 1


def test_legacy_fallback_uses_output_plus_temp():
    """If peak_memory_in_bytes is absent (older JAX), fall back to output+temp."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[
            _legacy_sample(1, output_mb=10.0, temp_mb=90.0),  # 100 MB total
            _legacy_sample(4, output_mb=10.0, temp_mb=390.0),  # 400 MB total
        ],
    )
    assert probe.per_replica_mb == 100.0
    assert probe.constant_overhead_mb == 0.0


def test_safety_factor_tightens_recommendation():
    """Larger safety_factor → smaller recommended batch."""
    probe = ProbeResult(
        dataset="t",
        model="t",
        instrument="t",
        samples=[_peak_sample(1, 1_000.0), _peak_sample(4, 4_000.0)],
    )
    no_safety = recommend_batch_size(probe, vram_budget_gb=65.0, safety_factor=1.0)
    with_safety = recommend_batch_size(probe, vram_budget_gb=65.0, safety_factor=1.5)
    assert with_safety < no_safety
