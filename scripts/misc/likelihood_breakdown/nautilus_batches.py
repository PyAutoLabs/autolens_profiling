"""Captured Nautilus proposal batches — the recorder, the file format and the lane rules.

Certified-solver phase C1 (autolens_profiling#304). Two cells share this module:

- ``scripts/imaging/likelihood_breakdown/nautilus_batch_capture.py`` runs a real
  ``af.Nautilus`` fit and records every batched parameter array that reaches
  ``Fitness.call_wrap`` (:class:`BatchRecorder`, :func:`recording`, :func:`save`);
- ``scripts/imaging/likelihood_breakdown/fixed_light_trace.py --lanes captured``
  replays those lanes (:func:`load`, :func:`sample_calls`, :func:`timed_window`,
  :func:`chunked`, :func:`rate_summary`).

Everything here except :func:`recording` is pure numpy, so the selection and
summary rules are unit-tested without JAX (``scripts/misc/test/test_nautilus_batches.py``).

The recorder
------------

:func:`recording` patches two methods for the duration of a ``with`` block and
restores both on exit — nothing in any library is edited:

1. ``nautilus.Sampler.evaluate_likelihood`` — to read the sampler's state at the
   moment it asks for a batch: how many bounds it holds, whether it has finished
   exploring, and ``n_like`` so far. That is the Nautilus *phase* of the batch
   (:func:`phase_of`). The wrapper forwards the call unchanged.
2. ``autofit.non_linear.fitness.Fitness.call_wrap`` — the host-side batched entry
   Nautilus is handed as its ``likelihood`` (``Nautilus.fit_x1_cpu``,
   ``likelihood=fitness.call_wrap, vectorized=fitness.use_jax_vmap``). The
   wrapper copies the ``(n, n_dim)`` physical parameter array it receives, calls
   the original, and copies the figure of merit it returns. Call index = the
   order ``call_wrap`` was entered.

The phase rule (stated once, used by both cells)
------------------------------------------------

``phase_of(n_bounds, explored)``:

- ``prior`` — the sampler holds only its first bound, the unit cube, so the
  batch is a set of prior draws (Nautilus fills its first ``n_live`` points this
  way);
- ``exploration`` — more than one bound, exploration not finished;
- ``sampling`` — ``Sampler.explored`` is True (the shells are being refined).

``early`` / ``late`` is the call-index split: a call is ``early`` if its call
index is below the median call index of the calls replayed (``n // 2``), else
``late``. Both splits are reported; neither is a Nautilus internal quantity that
a later Nautilus version could redefine silently, except ``explored``, which the
recorder reads straight off the sampler.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Bump when the npz key set changes; ``load`` refuses a file it does not know.
SCHEMA_VERSION = 1

#: Phase labels, in run order.
PHASES = ("prior", "exploration", "sampling")


def phase_of(n_bounds: int, explored: bool) -> str:
    """The Nautilus phase of a batch, from the sampler state when it was requested."""
    if explored:
        return "sampling"
    if int(n_bounds) <= 1:
        return "prior"
    return "exploration"


class CaptureBudgetReached(Exception):
    """Raised from inside ``call_wrap`` once the capture's wall-clock budget is spent.

    The batch that crosses the budget IS recorded first; the exception then
    unwinds Nautilus, and the capture cell catches it and writes the file.
    """


@dataclass
class BatchRecorder:
    """Everything the two patched methods saw, in call order."""

    deadline: float | None = None
    parameters: list = field(default_factory=list)
    figure_of_merit: list = field(default_factory=list)
    call_wall_s: list = field(default_factory=list)
    call_phase: list = field(default_factory=list)
    call_n_bounds: list = field(default_factory=list)
    call_explored: list = field(default_factory=list)
    call_n_like_before: list = field(default_factory=list)
    #: The sampler state set by the evaluate_likelihood wrapper, consumed by the
    #: next call_wrap. ``None`` if call_wrap was reached some other way.
    _pending_state: dict | None = None

    @property
    def n_calls(self) -> int:
        return len(self.parameters)

    @property
    def n_lanes(self) -> int:
        return int(sum(p.shape[0] for p in self.parameters))

    def note_sampler_state(self, sampler) -> None:
        n_bounds = len(getattr(sampler, "bounds", []) or [])
        explored = bool(getattr(sampler, "explored", False))
        self._pending_state = {
            "n_bounds": n_bounds,
            "explored": explored,
            "n_like_before": int(getattr(sampler, "n_like", -1)),
            "phase": phase_of(n_bounds, explored),
        }

    def record(self, parameters, figure_of_merit, wall_s: float) -> None:
        params = np.array(parameters, dtype=float, copy=True)
        if params.ndim == 1:
            params = params[None, :]
        fom = np.array(figure_of_merit, dtype=float, copy=True).reshape(-1)
        if fom.shape[0] != params.shape[0]:
            raise AssertionError(
                f"call_wrap returned {fom.shape[0]} figures of merit for {params.shape[0]} "
                f"parameter vectors — the batch axis is not the lane axis"
            )
        state = self._pending_state or {
            "n_bounds": -1,
            "explored": False,
            "n_like_before": -1,
            "phase": "unknown",
        }
        self._pending_state = None
        self.parameters.append(params)
        self.figure_of_merit.append(fom)
        self.call_wall_s.append(float(wall_s))
        self.call_phase.append(state["phase"])
        self.call_n_bounds.append(int(state["n_bounds"]))
        self.call_explored.append(bool(state["explored"]))
        self.call_n_like_before.append(int(state["n_like_before"]))


@contextlib.contextmanager
def recording(recorder: BatchRecorder):
    """Patch ``Sampler.evaluate_likelihood`` and ``Fitness.call_wrap``; restore both on exit."""
    import nautilus
    from autofit.non_linear.fitness import Fitness

    original_call_wrap = Fitness.call_wrap
    original_evaluate = nautilus.Sampler.evaluate_likelihood

    def evaluate_likelihood(self, points):
        recorder.note_sampler_state(self)
        return original_evaluate(self, points)

    def call_wrap(self, parameters):
        start = time.perf_counter()
        figure_of_merit = original_call_wrap(self, parameters)
        recorder.record(parameters, figure_of_merit, time.perf_counter() - start)
        if recorder.deadline is not None and time.monotonic() >= recorder.deadline:
            raise CaptureBudgetReached(
                f"capture wall-clock budget reached after {recorder.n_calls} calls"
            )
        return figure_of_merit

    call_wrap.__doc__ = original_call_wrap.__doc__
    Fitness.call_wrap = call_wrap
    nautilus.Sampler.evaluate_likelihood = evaluate_likelihood
    try:
        yield recorder
    finally:
        Fitness.call_wrap = original_call_wrap
        nautilus.Sampler.evaluate_likelihood = original_evaluate


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def save(path: Path, recorder: BatchRecorder, meta: dict) -> dict:
    """Write the npz; returns the per-file summary the provenance JSON records."""
    if recorder.n_calls == 0:
        raise ValueError("no batches were recorded — nothing reached Fitness.call_wrap")
    sizes = np.array([p.shape[0] for p in recorder.parameters], dtype=int)
    call_index = np.repeat(np.arange(recorder.n_calls, dtype=int), sizes)
    lane_in_call = np.concatenate([np.arange(s, dtype=int) for s in sizes])
    arrays = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=int),
        "parameters": np.concatenate(recorder.parameters, axis=0),
        "figure_of_merit": np.concatenate(recorder.figure_of_merit, axis=0),
        "call_index": call_index,
        "lane_in_call": lane_in_call,
        "call_size": sizes,
        "call_wall_s": np.array(recorder.call_wall_s, dtype=float),
        "call_phase": np.array(recorder.call_phase, dtype="U16"),
        "call_n_bounds": np.array(recorder.call_n_bounds, dtype=int),
        "call_explored": np.array(recorder.call_explored, dtype=bool),
        "call_n_like_before": np.array(recorder.call_n_like_before, dtype=int),
        "meta_json": np.array(json.dumps(meta, sort_keys=True, default=str)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "n_calls": int(recorder.n_calls),
        "n_lanes": int(sizes.sum()),
        "call_sizes": {str(int(s)): int((sizes == s).sum()) for s in np.unique(sizes)},
        "calls_by_phase": {
            p: int(sum(1 for q in recorder.call_phase if q == p))
            for p in (*PHASES, "unknown")
            if any(q == p for q in recorder.call_phase)
        },
    }


def load(path: Path) -> dict:
    """Read a captured-batches npz back, refusing an unknown schema."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    version = int(out.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {version}, this checkout reads {SCHEMA_VERSION}")
    out["meta"] = json.loads(str(out.pop("meta_json")))
    return out


# ---------------------------------------------------------------------------
# The lane rules
# ---------------------------------------------------------------------------


def sample_calls(n_calls: int, k: int | None) -> np.ndarray:
    """``k`` call indices evenly spaced over the run (all calls when ``k`` is None).

    Deterministic, first and last call always included, so a sampled rate still
    spans every phase of the run.
    """
    if n_calls < 1:
        raise ValueError("no calls to sample")
    if k is None or k >= n_calls:
        return np.arange(n_calls, dtype=int)
    if k < 1:
        raise ValueError(f"--batch-sample must be >= 1 (got {k})")
    return np.unique(np.round(np.linspace(0, n_calls - 1, int(k))).astype(int))


def lanes_of_calls(call_index: np.ndarray, calls: np.ndarray) -> np.ndarray:
    """Lane (row) indices of every lane belonging to *calls*, in call order."""
    mask = np.isin(call_index, calls)
    return np.flatnonzero(mask)


def timed_window(call_index: np.ndarray, call_phase: np.ndarray, batch: int) -> np.ndarray:
    """The ``batch`` lanes the timed pass replays.

    RULE: consecutive captured calls, in call order, starting at the FIRST call
    that is not a ``prior`` batch (the first batch proposed from a learned
    bound — what the bulk of a run evaluates); lanes are concatenated across
    calls and truncated to ``batch``. At ``batch == n_batch`` this is exactly one
    real Nautilus batch. If the post-prior lanes run out, the window starts
    earlier (the latest ``batch`` lanes of the capture), and if the capture holds
    fewer than ``batch`` lanes in total it is an error, never a silent repeat.
    """
    n_lanes = int(call_index.shape[0])
    if batch > n_lanes:
        raise ValueError(
            f"the capture holds {n_lanes} lanes; a timed batch of {batch} would have to "
            f"repeat lanes, which is not a real batch"
        )
    not_prior = np.flatnonzero(call_phase != "prior")
    start_call = int(not_prior[0]) if not_prior.size else 0
    start = int(np.flatnonzero(call_index >= start_call)[0])
    start = min(start, n_lanes - batch)
    return np.arange(start, start + batch, dtype=int)


def chunked(indices: np.ndarray, size: int) -> list[tuple[np.ndarray, int]]:
    """Split *indices* into chunks of exactly ``size`` (one compiled batch shape).

    The final chunk is padded by repeating its last index; returns
    ``[(chunk, n_real), ...]`` so the caller drops the padded tail.
    """
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    out = []
    for start in range(0, len(indices), size):
        chunk = np.asarray(indices[start : start + size], dtype=int)
        n_real = int(chunk.shape[0])
        if n_real < size:
            chunk = np.concatenate([chunk, np.full(size - n_real, chunk[-1], dtype=int)])
        out.append((chunk, n_real))
    return out


def rate_summary(certified, passes, budget: int) -> dict:
    """Uncertified-lane rate and the pass distribution against the budget."""
    certified = np.asarray(certified, dtype=bool)
    passes = np.asarray(passes, dtype=int)
    n = int(certified.shape[0])
    n_uncertified = int((~certified).sum())
    values, counts = np.unique(passes, return_counts=True) if n else ([], [])
    return {
        "n_lanes": n,
        "n_uncertified": n_uncertified,
        "uncertified_rate": (n_uncertified / n) if n else None,
        "max_passes": int(passes.max()) if n else None,
        "mean_passes": float(passes.mean()) if n else None,
        "n_at_budget": int((passes >= int(budget)).sum()) if n else 0,
        "budget": int(budget),
        "passes_histogram": {str(int(v)): int(c) for v, c in zip(values, counts)},
    }


def grouped_rate_summaries(certified, passes, budget: int, labels) -> dict:
    """:func:`rate_summary` per distinct label (phase, or early / late)."""
    labels = np.asarray(labels)
    certified = np.asarray(certified, dtype=bool)
    passes = np.asarray(passes, dtype=int)
    return {
        str(label): rate_summary(certified[labels == label], passes[labels == label], budget)
        for label in dict.fromkeys(labels.tolist())
    }


def early_late(call_index_of_lane: np.ndarray, calls_replayed: np.ndarray) -> np.ndarray:
    """``early`` / ``late`` per lane: below / at-or-above the median replayed call."""
    ordered = np.sort(np.asarray(calls_replayed, dtype=int))
    split = int(ordered[len(ordered) // 2]) if ordered.size else 0
    return np.where(np.asarray(call_index_of_lane) < split, "early", "late")
