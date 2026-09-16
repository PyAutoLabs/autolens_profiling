"""JAX-free pieces shared by every CPU fixed-lens-light kernel module.

Why this module exists
----------------------

``fixed_light_cpu_kernels`` holds the CPU kernel rows, but it imports
``active_set_steps`` at module level and that imports JAX. A numba cell whose
whole premise is ``"jax" not in sys.modules`` therefore cannot reach any of it —
not even the four helpers that have nothing to do with JAX (the timer, the
thread block, the solver's view of the system, the numpy evidence scorer).

Those four live here instead, and ``fixed_light_cpu_kernels`` imports them from
this module rather than defining its own copies, so a row measured through
``fixed_light_numpy_solvers`` and a row measured through
``fixed_light_cpu_kernels`` are timed by **the same function** rather than by two
that agree today.

Nothing in this module may import JAX, ``active_set_steps``,
``library_solver_injection`` or ``fixed_light_cpu_kernels`` — at module level or
inside a function. The numba cells' ``"jax" not in sys.modules`` assert is the
test of that statement.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "BLAS_THREAD_VARS",
    "JAX_THREAD_VARS",
    "log_evidence_terms_np",
    "scored_np",
    "solver_view_of",
    "thread_block",
    "time_median",
]

#: The knobs OpenBLAS / MKL / OpenMP read at import. Recorded, never set here —
#: setting them after numpy is imported does nothing, which is the whole reason
#: a thread-scaling cell forks one subprocess per thread setting.
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: The knob XLA reads for its CPU intra-op thread pool. Disjoint from the above:
#: it does not touch BLAS, and the BLAS knobs do not touch it.
JAX_THREAD_VARS = ("NPROC",)


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def thread_block() -> dict:
    """Both thread-knob families, as this process was started with.

    Recorded on **every** CPU timing — the JAX rows and the numpy rows alike —
    because the two families are disjoint and a reader cannot tell from a
    millisecond which one was throttling it.
    """
    return {
        "blas": {name: os.environ.get(name) for name in BLAS_THREAD_VARS},
        "jax_pool": {name: os.environ.get(name) for name in JAX_THREAD_VARS},
        "n_threads_blas": _int_or_none(os.environ.get("OMP_NUM_THREADS")),
        "n_threads_jax_pool": _int_or_none(os.environ.get("NPROC")),
        "numba_num_threads": _int_or_none(os.environ.get("NUMBA_NUM_THREADS")),
        "cpu_count_os": os.cpu_count(),
        "note": (
            "NPROC sizes XLA's CPU intra-op pool (JAX rows). The BLAS knobs pin "
            "OpenBLAS/MKL/OpenMP (these rows) and are read once at import, so "
            "they can only be set before the process starts — hence one "
            "subprocess per thread setting. NUMBA_NUM_THREADS is numba's own, "
            "read once when its threading layer initialises."
        ),
    }


def time_median(fn: Callable[[], Any], *, n_repeats: int = 10, n_warmup: int = 2) -> dict:
    """Median wall time of ``fn`` over ``n_repeats`` calls, in milliseconds.

    Median, not mean: a CPU row on a laptop picks up scheduler noise, and one
    descheduled repeat should move the reported number by nothing. ``min`` is
    reported beside it as the cleanest observation and ``max`` as the noise
    scale, so a reader can see when the machine was not quiet.

    The warm-up calls are discarded — the first call through a numpy/scipy path
    pays for lazily-loaded BLAS kernels that no later call pays again.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1 (got {n_repeats})")

    for _ in range(max(0, n_warmup)):
        fn()

    samples = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)

    return {
        "ms": float(statistics.median(samples)),
        "ms_min": float(min(samples)),
        "ms_max": float(max(samples)),
        "n_repeats": int(n_repeats),
        "n_warmup": int(max(0, n_warmup)),
        "timer": "time.perf_counter, median of n_repeats",
    }


def solver_view_of(system) -> dict:
    """The QP the library's positive-only solver is actually handed.

    ``(curvature_reg_matrix, data_vector)`` restricted to ``ids_to_keep`` when
    the library is edge-zeroing, plus a ``scatter_back`` that puts a subset
    solution into the full parameter vector exactly as the library does.
    """
    crm_full = np.asarray(system.curvature_reg_matrix, dtype=float)
    dv_full = np.asarray(system.data_vector, dtype=float)
    ids = system.ids_to_keep
    n_full = int(system.n_params)

    if ids is None:

        def scatter_back(x_sub):
            return np.asarray(x_sub, dtype=float)

        return {
            "curvature_reg_matrix": crm_full,
            "data_vector": dv_full,
            "ids_to_keep": None,
            "n_full": n_full,
            "n_seen_by_solver": n_full,
            "scatter_back": scatter_back,
        }

    ids = np.asarray(ids, dtype=int)

    def scatter_back(x_sub):
        full = np.zeros(n_full, dtype=float)
        full[ids] = np.asarray(x_sub, dtype=float)
        return full

    return {
        "curvature_reg_matrix": crm_full[ids][:, ids],
        "data_vector": dv_full[ids],
        "ids_to_keep": ids,
        "n_full": n_full,
        "n_seen_by_solver": int(ids.shape[0]),
        "scatter_back": scatter_back,
    }


def _log_det_cholesky_np(matrix) -> float:
    """``2 * sum(log(diag(cholesky(M))))`` — the library's log-det formula, in numpy."""
    return float(2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(np.asarray(matrix, dtype=float))))))


def log_evidence_terms_np(system, x) -> dict:
    """Every evidence term of an iterate, in **numpy**.

    Term for term the arithmetic of
    ``likelihood_breakdown.reconstruction_steps.log_evidence_terms``, which is
    where the existing pins were calibrated — but that function is ``jnp`` and a
    numba CPU cell may not import JAX to score a row it measured without one. The
    two agree to fp64 round-off, and ``test_fixed_light_numpy_solvers.py`` pins
    that agreement against the JAX original rather than leaving it asserted in
    prose.

        ``-2 ln e = chi^2 + s^T H s + ln det(F + lambda H) - ln det(H) + noise_norm``

    The regularised terms use the rank-stripped (mapper-only) blocks: the full
    ``H`` is rank-deficient by construction (non-regularised linear components
    such as an MGE lens light), so its log-det is ``-inf``.
    """
    x = np.asarray(x, dtype=float)

    data = np.asarray(system.dataset.data.array, dtype=float)
    noise_map = np.asarray(system.dataset.noise_map.array, dtype=float)
    model_data = np.asarray(system.model_data_from(x), dtype=float)

    residual = data - model_data
    chi_squared = float(np.sum((residual / noise_map) ** 2))

    reduced_indices = np.asarray(system.mapper_indices)
    reg_reduced = np.asarray(system.reg_reduced, dtype=float)
    curv_reg_reduced = np.asarray(system.curv_reg_reduced, dtype=float)

    s_reduced = x[reduced_indices]
    regularization_term = float(s_reduced @ (reg_reduced @ s_reduced))

    log_det_curvature_reg = _log_det_cholesky_np(curv_reg_reduced)
    log_det_regularization = _log_det_cholesky_np(reg_reduced)

    noise_normalization = float(np.sum(np.log(2 * np.pi * noise_map**2)))

    log_evidence = -0.5 * (
        chi_squared
        + regularization_term
        + log_det_curvature_reg
        - log_det_regularization
        + noise_normalization
    )

    return {
        "chi_squared": chi_squared,
        "regularization_term": regularization_term,
        "log_det_curvature_reg": log_det_curvature_reg,
        "log_det_regularization": log_det_regularization,
        "noise_normalization": noise_normalization,
        "log_evidence": float(log_evidence),
    }


def scored_np(system, x_full, reference_log_evidence) -> dict:
    """Evidence of a full-length reconstruction, and its Δ against the reference.

    The numpy twin of ``fixed_light_cpu_kernels._scored``. A row is never left as
    a bare millisecond: a kernel that is fast because it solved a different
    problem says so, in nats, beside its timing.
    """
    terms = log_evidence_terms_np(system, x_full)
    log_evidence = terms["log_evidence"]
    return {
        "log_evidence": log_evidence,
        "d_log_evidence_vs_library": log_evidence - float(reference_log_evidence),
        "n_negative_entries": int(np.sum(np.asarray(x_full, dtype=float) < 0.0)),
    }
