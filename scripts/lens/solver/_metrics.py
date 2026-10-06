"""Accuracy metrics of a positive-only solve against the corpus reference ``x_ref``.

All metrics compare a candidate's raw-coordinate reconstruction ``x`` with the fnnls reference
``system.x_ref`` of the same ``(Q, q)``:

``amp_rel_l2``
    ``||x - x_ref||_2 / ||x_ref||_2`` over all columns.
``amp_rel_max``
    ``max_j |x_j - x_ref_j| / x_ref_j`` over the columns the reference keeps active
    (``x_ref_j > 0``) — the worst relative error on any amplitude that carries light. Note it
    is dominated by any barely-active reference column (an ``x_ref_j`` of 1e-9 next to a
    maximum of 1e2 turns a 1e-2 absolute slip into 1e7), so it is paired with
``amp_rel_max_sig``
    The same restricted to *significant* columns, ``x_ref_j > SIG_FRACTION * max(x_ref)``
    (``SIG_FRACTION = 1e-6``).
``flux_rel_all``
    Signed ``(sum x - sum x_ref) / |sum x_ref|`` — the summed-amplitude (intensity-sum) error.
    Amplitudes are linear-profile intensities, so this is a flux *proxy*, not a photometric
    flux.
``flux_rel_source``
    The same over ``meta["source_column_index_list"]``; ``None`` when the system's source
    columns are unknown (or the reference source sum is zero).
``flux_inactive_rel``
    ``sum(x_j over reference-inactive columns) / sum(x_ref)``, where a column is
    reference-inactive iff ``x_ref_j <= SIG_FRACTION * max(x_ref)`` — the spurious mass a solver
    puts on columns the reference leaves (numerically) at zero. ``amp_rel_max`` and
    ``amp_rel_max_sig`` score only reference-*active* columns, so they cannot see it; this is the
    metric the released raw stop is blind to (added 2026-09-30, post-hoc to the phase-1 rule).
``active_set_mismatch``
    The number of columns whose activity differs between ``x`` and ``x_ref``, activity being
    ``v_j > SIG_FRACTION * max(v)`` for each vector on its own scale.
``objective_gap``
    ``(f(x) - f(x_ref)) / |f(x_ref)|`` with ``f(x) = 0.5 x^T Q x - q^T x``; positive means worse
    than the reference.
``kkt_residual_scaled``
    ``max(primal_violation, dual_violation, complementarity)`` of
    ``hazards._likelihood.nnls_optimality_metrics`` — the scale-normalised KKT residual the
    ``scripts/imaging/rectangular/hazards.py`` solver-diagnostic leg reports (its three
    components are kept alongside).

A non-finite ``x`` scores every metric ``None`` and ``finite: False``.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = str(_profiling_root() / "scripts" / "misc")
if _misc not in sys.path:
    sys.path.insert(0, _misc)

from hazards._likelihood import nnls_optimality_metrics  # noqa: E402

SIG_FRACTION = 1.0e-6

METRIC_KEYS = (
    "amp_rel_l2",
    "amp_rel_max",
    "amp_rel_max_sig",
    "flux_rel_all",
    "flux_rel_source",
    "flux_inactive_rel",
    "active_set_mismatch",
    "objective_gap",
    "kkt_residual_scaled",
    "primal_violation",
    "dual_violation",
    "complementarity",
)


def _objective(Q, q, x) -> float:
    return float(0.5 * x @ Q @ x - q @ x)


def _rel(a: float, b: float) -> float | None:
    return None if b == 0.0 else float((a - b) / abs(b))


def _inactive_flux_rel(x, x_ref) -> float | None:
    """Spurious mass on reference-inactive columns, relative to the reference total."""
    total = float(np.sum(x_ref))
    if x_ref.size == 0 or total == 0.0:
        return None
    inactive = x_ref <= SIG_FRACTION * float(np.max(x_ref))
    return float(np.sum(x[inactive]) / total)


def _active_set_mismatch(x, x_ref) -> int | None:
    """Columns whose activity (``v > SIG_FRACTION * max v``) differs between ``x`` and ``x_ref``."""
    if x_ref.size == 0:
        return None
    act = x > SIG_FRACTION * float(np.max(x))
    act_ref = x_ref > SIG_FRACTION * float(np.max(x_ref))
    return int(np.sum(act != act_ref))


def metrics(x, system) -> dict:
    """The accuracy metrics of ``x`` against ``system.x_ref`` (see the module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    x_ref = np.asarray(system.x_ref, dtype=np.float64)
    Q = np.asarray(system.Q, dtype=np.float64)
    q = np.asarray(system.q, dtype=np.float64)
    if x.shape != x_ref.shape:
        raise ValueError(f"x shape {x.shape} != x_ref shape {x_ref.shape}")

    if not np.all(np.isfinite(x)):
        return {"finite": False, **dict.fromkeys(METRIC_KEYS)}

    active = x_ref > 0
    significant = x_ref > SIG_FRACTION * float(np.max(x_ref)) if x_ref.size else active
    ref_norm = float(np.linalg.norm(x_ref))
    out = {
        "finite": True,
        "amp_rel_l2": float(np.linalg.norm(x - x_ref) / ref_norm) if ref_norm > 0 else None,
        "amp_rel_max": (
            float(np.max(np.abs(x[active] - x_ref[active]) / x_ref[active]))
            if np.any(active)
            else None
        ),
        "amp_rel_max_sig": (
            float(np.max(np.abs(x[significant] - x_ref[significant]) / x_ref[significant]))
            if np.any(significant)
            else None
        ),
        "flux_rel_all": _rel(float(np.sum(x)), float(np.sum(x_ref))),
        "flux_rel_source": None,
        "flux_inactive_rel": _inactive_flux_rel(x, x_ref),
        "active_set_mismatch": _active_set_mismatch(x, x_ref),
        "objective_gap": _rel(_objective(Q, q, x), _objective(Q, q, x_ref)),
    }
    cols = (system.meta or {}).get("source_column_index_list")
    if cols:
        idx = np.asarray(cols, dtype=int)
        out["flux_rel_source"] = _rel(float(np.sum(x[idx])), float(np.sum(x_ref[idx])))

    opt = nnls_optimality_metrics(Q, q, x)
    out["primal_violation"] = opt["primal_violation"]
    out["dual_violation"] = opt["dual_violation"]
    out["complementarity"] = opt["complementarity"]
    out["kkt_residual_scaled"] = max(
        opt["primal_violation"], opt["dual_violation"], opt["complementarity"]
    )
    return out


def timing(fn, Q, q, repeats: int = 5, **kwargs) -> float:
    """Median warm wall-clock (seconds) of ``fn(Q, q, **kwargs)`` over ``repeats`` calls.

    One untimed call first absorbs compilation / first-touch cost. ``fn`` must block until its
    result is ready (``Candidate.kernel`` does).
    """
    fn(Q, q, **kwargs)
    walls = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        fn(Q, q, **kwargs)
        walls.append(time.perf_counter() - t0)
    return float(statistics.median(walls))


def worst_abs(values) -> float | None:
    """``max |v|`` over the non-None, finite values; ``None`` if there are none."""
    vals = [abs(v) for v in values if v is not None and math.isfinite(v)]
    return max(vals) if vals else None


def median_or_none(values) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(statistics.median(vals)) if vals else None
