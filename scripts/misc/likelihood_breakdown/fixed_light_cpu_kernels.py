"""CPU-appropriate positivity kernels for the fixed-lens-light (S3) system.

Why this module exists
----------------------

Phases 0 and 1 of the ``fixed-lens-light-profiling`` epic measured the
fixed-lens-light likelihood **only on an A100**, and measured it in JAX. Neither
is what a CPU user runs. On the CPU the library's own positivity solver is not
``jax_nnls``'s PDIP at all: ``reconstruction_positive_only_from`` branches on
``xp`` and takes the **numpy** path, an ``fnnls`` Cholesky-updating NNLS
(Bro & de Jong 1997) seeded from the sign of the dense unconstrained solve
(``inversion_util.py``). And the certified active set costs something quite
different there too: the JAX twin factorises the **full** ``(n, n)`` matrix every
pass because a jit needs a static shape, while a numpy implementation factorises
only the free block, which shrinks as the active set fills.

So a CPU row set is not the A100 row set with a different device string. This
module is the CPU row set (autolens_profiling#253, phase 2):

1. **library numpy NNLS** — the library's own CPU production kernel, called
   exactly as ``Inversion.reconstruction`` calls it, with its ``stats`` dict
   read back for the iteration counts;
2. **scipy unconstrained** — ``cho_factor`` / ``cho_solve``, the floor: what the
   solve costs with positivity dropped entirely;
3. **numpy certified active set** — ``active_set_steps.active_set_certified``,
   restricted Cholesky per pass, at the certifying pass budget. **The CPU
   production candidate.**

Every row is scored, never left as a bare millisecond: each returns the
log-evidence of its own reconstruction and its difference from the library's,
so a row that is fast because it solved a different problem says so.

What the solver actually sees
-----------------------------

Under edge zeroing the library subsets ``F + λH`` and ``D`` to
``solve_ids_to_keep`` *before* calling its positive-only solver and scatters
exact zeros back afterwards (``abstract.py:607-618``). Rows 1 and 3 are
therefore timed on that **subset**, which is the QP the library's own CPU kernel
receives and the one phase 0's pass budgets were certified on. Row 2 is reported
on the subset *and* on the full system, because dropping positivity in the
library also silently drops edge zeroing (``abstract.py:540``) — the full-system
number is the one comparable with the library's positive-negative route and with
phase 0's A2 Cholesky; the subset number is the one comparable with rows 1 and 3.

Threads
-------

A CPU millisecond is meaningless without its thread count, and there are **two
disjoint knob families**: ``NPROC`` sizes XLA's CPU intra-op pool and governs
JAX rows; ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS``
pin OpenBLAS / MKL and govern *these* rows, without touching JAX. Both are
recorded by :func:`thread_block`. The BLAS knobs are read **once, at library
import**, so a thread setting cannot be changed inside a running process: the
cell runs one subprocess per thread setting, and :func:`thread_block` records
what that subprocess was actually started with.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.linalg

from likelihood_breakdown import active_set_steps, fixed_light_system

__all__ = [
    "BLAS_THREAD_VARS",
    "cpu_kernel_rows",
    "library_numpy_nnls_row",
    "numpy_certified_row",
    "scipy_unconstrained_row",
    "solver_view_of",
    "thread_block",
    "time_median",
]

#: The knobs OpenBLAS / MKL / OpenMP read at import. Recorded, never set here —
#: setting them after numpy is imported does nothing, which is the whole reason
#: the cell forks one subprocess per thread setting.
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
        "cpu_count_os": os.cpu_count(),
        "note": (
            "NPROC sizes XLA's CPU intra-op pool (JAX rows). The BLAS knobs pin "
            "OpenBLAS/MKL/OpenMP (these rows) and are read once at import, so "
            "they can only be set before the process starts — hence one "
            "subprocess per thread setting."
        ),
    }


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


#: The numpy Jacobi scaling, **promoted** to
#: ``likelihood_breakdown.fixed_light_system.jacobi_scaled_np`` so the numba CPU
#: cell can reach it without importing this module (which pulls in
#: ``active_set_steps``, and with it JAX). This name is an alias of that
#: function — the same object, not a second copy — so every row measured through
#: it is measured through the promoted implementation.
_jacobi_scaled_np = fixed_light_system.jacobi_scaled_np


def _scored(system, x_full, reference_log_evidence):
    """Evidence of a full-length reconstruction, and its Δ against the library."""
    log_evidence = float(system.log_evidence(x_full))
    return {
        "log_evidence": log_evidence,
        "d_log_evidence_vs_library": log_evidence - reference_log_evidence,
        "n_negative_entries": int(np.sum(np.asarray(x_full, dtype=float) < 0.0)),
    }


def library_numpy_nnls_row(system, view, reference_log_evidence, *, n_repeats: int = 10) -> dict:
    """The library's own CPU positivity solver — ``fnnls_cholesky``, ``xp=np``.

    Called exactly as ``Inversion.reconstruction`` calls it on the NumPy path:
    the same entry point, the same already-subset matrices, the same
    dense-sign warm start the library builds internally. The ``stats`` dict the
    library writes is read **after** the call (it is populated in place), giving
    the outer/inner iteration counts and the passive-set size beside the timing.
    """
    from autoarray.inversion.inversion import inversion_util

    crm = view["curvature_reg_matrix"]
    dv = view["data_vector"]

    def _call():
        return inversion_util.reconstruction_positive_only_from(
            data_vector=dv,
            curvature_reg_matrix=crm,
            settings=None,
            xp=np,
            fingerprint=None,
        )

    x_sub = _call()
    row = {
        "label": "library numpy NNLS (fnnls_cholesky, xp=np)",
        "entry_point": (
            "autoarray.inversion.inversion.inversion_util."
            "reconstruction_positive_only_from (NumPy branch)"
        ),
        "problem": "the edge-zeroed subset, as the library hands it",
        "n": int(view["n_seen_by_solver"]),
    }
    row.update(time_median(_call, n_repeats=n_repeats))

    # The iteration counts come from the solver's own stats dict, which
    # ``reconstruction_positive_only_from`` does not return. Re-run the inner
    # call once with a stats dict of our own so the counts are the library's,
    # not a re-implementation's.
    try:
        from autoarray.util.fnnls import fnnls_cholesky

        stats: dict = {}
        fnnls_cholesky(
            crm,
            dv.T,
            P_initial=np.linalg.solve(crm, dv) > 0,
            stats=stats,
        )
        row["outer_iterations"] = int(stats.get("outer_iterations", -1))
        row["inner_iterations"] = int(stats.get("inner_iterations", -1))
        row["n_passive"] = int(stats.get("n_passive", -1))
    except Exception as exc:  # noqa: BLE001 — diagnostics must not lose the timing
        row["iteration_counts_error"] = repr(exc)

    row.update(_scored(system, view["scatter_back"](x_sub), reference_log_evidence))
    return row


def scipy_unconstrained_row(system, view, reference_log_evidence, *, n_repeats: int = 10) -> dict:
    """``cho_factor`` / ``cho_solve`` with positivity dropped — the floor.

    Reported twice. On the **subset** it is the honest floor for rows 1 and 3:
    the same QP with the constraint removed. On the **full** system it is what
    the library's positive-negative route solves, because turning positivity off
    turns edge zeroing off with it — and it is the row phase 0 called A2.
    """

    def _row_for(crm, dv, scatter_back, n, label, problem):
        def _call():
            c, low = scipy.linalg.cho_factor(crm, lower=True, check_finite=False)
            return scipy.linalg.cho_solve((c, low), dv, check_finite=False)

        x_sub = _call()
        out = {
            "label": label,
            "entry_point": "scipy.linalg.cho_factor / cho_solve",
            "problem": problem,
            "n": int(n),
        }
        out.update(time_median(_call, n_repeats=n_repeats))
        out.update(_scored(system, scatter_back(x_sub), reference_log_evidence))
        return out

    subset = _row_for(
        view["curvature_reg_matrix"],
        view["data_vector"],
        view["scatter_back"],
        view["n_seen_by_solver"],
        "scipy unconstrained (subset)",
        "the edge-zeroed subset — comparable with the two positivity rows",
    )

    crm_full = np.asarray(system.curvature_reg_matrix, dtype=float)
    dv_full = np.asarray(system.data_vector, dtype=float)
    full = _row_for(
        crm_full,
        dv_full,
        lambda x: np.asarray(x, dtype=float),
        system.n_params,
        "scipy unconstrained (full system)",
        (
            "the FULL system — what the library's positive-negative route solves, "
            "because Inversion.solve_ids_to_keep returns None the moment "
            "positivity is off (abstract.py:540), and what phase 0 called A2"
        ),
    )

    return {"subset": subset, "full_system": full}


def numpy_certified_row(
    system,
    view,
    reference_log_evidence,
    *,
    pass_budget: int,
    n_repeats: int = 10,
    tau_rel: float = active_set_steps.TAU_REL_DEFAULT,
) -> dict:
    """The numpy certified active set — restricted Cholesky per pass.

    The CPU production candidate, and **not** the JAX twin's cost: the jit-able
    version factorises the full ``(n, n)`` matrix every pass because its shape
    must be static, while this one factorises only the free block, which is
    smaller every pass once the active set fills. The row therefore records
    ``n_factorisations`` and the free-block size per pass as well as the
    millisecond, so the difference between the two implementations is visible
    rather than inferred.

    The timing includes the pass-0 unconstrained solve, because a production
    call pays for it: the scheme is seeded from the sign of that solve.
    """
    crm = view["curvature_reg_matrix"]
    dv = view["data_vector"]

    Q, q, d_scale = _jacobi_scaled_np(crm, dv)

    def _call():
        x_unc = active_set_steps.unconstrained_solve(Q, q)
        return active_set_steps.active_set_certified(
            Q,
            q,
            x_unc,
            policy="free_all",
            max_passes=int(pass_budget),
            tau_rel=tau_rel,
        )

    result = _call()
    row = {
        "label": "numpy certified active set (restricted Cholesky per pass)",
        "entry_point": "likelihood_breakdown.active_set_steps.active_set_certified",
        "problem": "the edge-zeroed subset, as the library hands it",
        "n": int(view["n_seen_by_solver"]),
        "pass_budget": int(pass_budget),
        "tau_rel": float(tau_rel),
        "certified": bool(result.certified),
        "passes_to_certification": result.passes_to_certification,
        "passes_run": int(result.passes_run),
        "n_factorisations": int(result.n_factorisations),
        "n_free_per_pass": [int(h["n_free"]) for h in result.history],
        "n_primal_violations_per_pass": [int(h["n_primal_violations"]) for h in result.history],
        "n_dual_violations_per_pass": [int(h["n_dual_violations"]) for h in result.history],
        "includes_pass_zero_unconstrained_solve": True,
        "implementation_note": (
            "restricted Cholesky on the free block, which shrinks as the active "
            "set fills — NOT the full-size masked solve the jit-able JAX twin "
            "must use for a static shape. The two are the same algorithm at "
            "different cost."
        ),
    }
    row.update(time_median(_call, n_repeats=n_repeats))
    row.update(_scored(system, view["scatter_back"](result.x * d_scale), reference_log_evidence))
    return row


def cpu_kernel_rows(
    system,
    *,
    pass_budget: int,
    n_repeats: int = 10,
    tau_rel: float = active_set_steps.TAU_REL_DEFAULT,
) -> dict:
    """Every CPU kernel row for one :class:`FixedLightSystem`, plus its gates.

    Returns a dict with ``threads``, ``rows`` and ``equalities`` — the last
    being the checks that make the milliseconds worth reading: the two
    positivity rows must reach the *same* positive solution as each other and as
    the library's own, to 1e-8 relative in log-evidence.
    """
    view = solver_view_of(system)
    x_library = np.asarray(system.inversion.reconstruction, dtype=float)
    reference_log_evidence = float(system.log_evidence(x_library))

    rows = {
        "library_numpy_nnls": library_numpy_nnls_row(
            system, view, reference_log_evidence, n_repeats=n_repeats
        ),
        "scipy_unconstrained": scipy_unconstrained_row(
            system, view, reference_log_evidence, n_repeats=n_repeats
        ),
        "numpy_certified_active_set": numpy_certified_row(
            system,
            view,
            reference_log_evidence,
            pass_budget=pass_budget,
            n_repeats=n_repeats,
            tau_rel=tau_rel,
        ),
    }

    nnls_ev = rows["library_numpy_nnls"]["log_evidence"]
    cert_ev = rows["numpy_certified_active_set"]["log_evidence"]
    scale = max(abs(reference_log_evidence), 1e-300)

    equalities = [
        {
            "check": "library numpy NNLS == the library's own reconstruction",
            "rel_diff": abs(nnls_ev - reference_log_evidence) / scale,
            "rtol": 1e-8,
        },
        {
            "check": "numpy certified active set == library numpy NNLS",
            "rel_diff": abs(cert_ev - nnls_ev) / scale,
            "rtol": 1e-8,
        },
    ]
    for entry in equalities:
        entry["status"] = "PASS" if entry["rel_diff"] <= entry["rtol"] else "FAIL"

    return {
        "threads": thread_block(),
        "reference_log_evidence": reference_log_evidence,
        "reference": "the library's own reconstruction of the S3 system",
        "n_full_system": int(view["n_full"]),
        "n_seen_by_solver": int(view["n_seen_by_solver"]),
        "n_edge_zeroed": int(view["n_full"] - view["n_seen_by_solver"]),
        "rows": rows,
        "equalities": equalities,
    }
