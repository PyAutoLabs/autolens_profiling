"""NumPy-path positivity solvers for the fixed-lens-light (S3) system — jax-free.

Why this module exists
----------------------

``library_solver_injection`` injects a solver into the library's positive-only
entry point on the **JAX** path; ``fixed_light_cpu_kernels`` holds the CPU kernel
rows. Both import ``active_set_steps`` at module level, and that imports JAX. A
numba cell whose whole premise is ``"jax" not in sys.modules`` can import
neither. This module is the numpy-path half of both, with the JAX half removed
rather than guarded:

- :func:`nnls_factor_reuse` — the candidate kernel (one Cholesky factorisation,
  reused);
- :func:`library_nnls_solve` / :func:`library_reconstruction_positive_only` — the
  library's own cold and memo paths, called as production calls them, so every
  row is timed through the same shape;
- :func:`numpy_solver_injected` — the injection seam, with the ``xp`` branch
  inverted relative to ``certified_solver_injected``: a JAX call is delegated to
  the original untouched, a numpy call is dispatched to the candidate;
- the row builders, which score every kernel against the library's own answer
  rather than leaving it as a bare millisecond.

**Nothing here may import jax, ``active_set_steps``, ``library_solver_injection``
or ``fixed_light_cpu_kernels``** — at module level or inside a function.

The cost model these kernels are aimed at
-----------------------------------------

On a source-only (S3) Delaunay ``N = 1500`` HST system the library's
``fnnls_cholesky`` converges in **zero** outer iterations: roughly 1495 of 1500
entries are already free and only a handful are negative. Its cost is therefore
not the active-set scheme at all — it is **two back-to-back full
factorisations**. ``reconstruction_positive_only_from`` builds its warm start as
``np.linalg.solve(curvature_reg_matrix, data_vector) > 0``
(``inversion_util.py:435``), an LU factorisation of the whole ``(n, n)`` matrix
whose factor is then **thrown away**; ``fnnls_cholesky`` immediately factorises
the passive block again with ``slg.cholesky`` (``fnnls.py:105``), and at
``k ≈ n`` that block is essentially the same matrix. A ``k x k`` fancy-index copy
and the ``O(n^2)`` gradient rebuilds sit on top.

The lever is the **factorisation count**, not the active-set scheme — which is
why the candidate here is the textbook Bro & de Jong (1997) iteration started
from a factorisation that is *kept*, rather than a different iteration.
"""

from __future__ import annotations

import contextlib
import inspect

import numpy as np
import scipy.linalg
from autoarray.util.cholesky_funcs import (
    _cho_solve_buffer,
    choldeleteindexes_inplace,
    cholinsertlast_inplace,
)
from autoarray.util.fnnls import fix_constraint_cholesky, fnnls_cholesky

from likelihood_breakdown.fixed_light_cpu_common import (
    scored_np,
    solver_view_of,
    time_median,
)

__all__ = [
    "LIBRARY_POSITIVE_ONLY_DOTTED",
    "factor_reuse_row",
    "library_cold_row",
    "library_nnls_solve",
    "library_reconstruction_positive_only",
    "memo_warm_row",
    "nnls_factor_reuse",
    "numpy_solver_injected",
    "unconstrained_row",
]

#: The dotted name :func:`numpy_solver_injected` rebinds. Recorded verbatim into
#: the results JSON so a reader of the numbers knows exactly what was patched.
LIBRARY_POSITIVE_ONLY_DOTTED = (
    "autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from"
)

#: ``fnnls.py:65`` — the library's own active-set tolerance, ``eps * n``.
_EPSILON = 2.2204e-16

#: The relative tolerance every equivalence pin in this module is judged at.
EQUIVALENCE_RTOL = 1.0e-9


# ===================================================================
# (a) The candidate kernel
# ===================================================================


def nnls_factor_reuse(ZTZ, ZTx, *, stats: dict | None = None) -> np.ndarray:
    """Bro & de Jong (1997) NNLS started from **one** Cholesky factorisation.

    What it is
        The same active-set iteration the library runs
        (``autoarray/util/fnnls.py``), on the same in-place Cholesky kernels
        (``autoarray/util/cholesky_funcs.py``), differing in exactly one place:
        the factorisation that produces the warm start is the factorisation the
        iteration then uses.

    Why one factorisation suffices
        The library spends two. It seeds the passive set from the *sign* of the
        unconstrained solution, computed with ``np.linalg.solve`` — an LU
        factorisation of the full ``(n, n)`` matrix whose factor is discarded at
        the end of the expression (``inversion_util.py:435``) — and
        ``fnnls_cholesky`` then factorises the passive block from scratch
        (``fnnls.py:105``). Here a single ``cho_factor`` gives *both*: the
        unconstrained solve (hence the sign seed) via ``cho_solve``, and the
        ``k = n`` factor the iteration starts from. Nothing about the algorithm
        changes — the seed, the tolerance, the outer loop and the constraint
        fixer are the library's.

    What the downdates cost
        The seed's non-positive entries have to leave the passive set, and each
        departure is a Cholesky **downdate** rather than a refactorisation:
        ``choldeleteindexes_inplace`` shifts the surviving rows and re-triangulates
        the trailing block with Givens rotations, ``O((k - i)^2)`` for the column
        at position ``i``. With a handful of negatives out of ``n ≈ 1500`` that is
        a rounding error against the ``O(n^3 / 3)`` factorisation it replaces;
        with a passive set that is mostly *empty* it is not, and this kernel is
        the wrong one for that regime. The measured regime is recorded in the row
        (``n_seed_negatives``), so the reader can see which one they are in.

    Harness prototype, not library code
        This lives in ``autolens_profiling`` and is measured against the
        library's own answer. Nothing in PyAutoArray is changed by this module;
        whether the saving earns a library change is the campaign's verdict, not
        this function's.

    Parameters
    ----------
    ZTZ
        ``F + lambda H`` — the curvature-plus-regularization matrix, ``(n, n)``.
    ZTx
        ``D`` — the data vector, ``(n,)``.
    stats
        Filled on return, when given, with ``n_factorisations``, ``n_downdates``,
        ``outer_iterations``, ``inner_iterations``, ``n_passive``,
        ``passive_set``, ``seed_source`` and ``n_seed_negatives``.
    """
    ZTZ = np.asarray(ZTZ, dtype=float)
    ZTx = np.asarray(ZTx, dtype=float)

    n = ZTZ.shape[0]
    tolerance = _EPSILON * n
    max_repetitions = 3
    no_update = 0
    outer_iterations = 0
    inner_iterations = 0
    n_downdates = 0

    # 1. The ONE factorisation. `lower=False` gives the upper factor `U` with
    #    `ZTZ = U.T @ U`, which is the convention `U_buffer` holds throughout
    #    `fnnls_cholesky` (`slg.cholesky` defaults to the same).
    c, _lower = scipy.linalg.cho_factor(ZTZ, lower=False, check_finite=False)
    x_unc = scipy.linalg.cho_solve((c, False), ZTx, check_finite=False)
    n_factorisations = 1

    # 2. The seed: the sign of the unconstrained solution — the library's own
    #    rule (`inversion_util.py:435`), read off the solve we already have.
    P = x_unc > 0
    n_seed_negatives = int(np.count_nonzero(~P))

    if P.all():
        # Every entry is already free: the unconstrained solution IS the NNLS
        # solution, and the library's outer loop would confirm that in zero
        # iterations after paying for a second factorisation.
        if stats is not None:
            stats.update(
                {
                    "n_factorisations": n_factorisations,
                    "n_downdates": 0,
                    "outer_iterations": 0,
                    "inner_iterations": 0,
                    "n_passive": int(n),
                    "passive_set": np.arange(n, dtype=int),
                    "seed_source": "factor_reuse_unconstrained",
                    "n_seed_negatives": 0,
                }
            )
        return np.asarray(x_unc, dtype=float)

    # 3. Keep the factor. `cho_factor` leaves the *input's* lower triangle in
    #    place below the diagonal (it is documented as containing "random data"),
    #    and the buffer kernels below index the whole `(n, n)` array, so the
    #    lower triangle is zeroed rather than inherited. The buffer is a fresh,
    #    writeable, C-contiguous float64 array: the numba kernels reject a
    #    read-only or non-contiguous one at compile time.
    U_buffer = np.zeros((n, n), dtype=float)
    U_buffer[:, :] = np.triu(c)
    k_active = n
    P_inorder = np.arange(n, dtype=int)
    P = np.ones(n, dtype=bool)

    s_chol = np.zeros(n, dtype=float)
    s_chol[:] = x_unc

    # 4. Drop the seed's non-positive entries and re-solve, repeating while any
    #    survivor is non-positive. This is `fnnls.py:134-151` verbatim in intent:
    #    the alpha -> 0 limit of `fix_constraint_cholesky`, which a warm start
    #    cannot use because it has no previous feasible iterate to interpolate
    #    from (every violator would give 0/0).
    while P_inorder.size and np.min(s_chol[P_inorder]) <= tolerance:
        id_delete = np.where(s_chol[P_inorder] <= tolerance)[0]

        k_active = choldeleteindexes_inplace(U_buffer, k_active, id_delete)
        n_downdates += int(id_delete.size)

        P[P_inorder[id_delete]] = False
        P_inorder = np.delete(P_inorder, id_delete)

        s_chol[~P] = 0.0

        if P_inorder.size:
            s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTx[P_inorder])

        inner_iterations += 1
        if inner_iterations > 10000:
            raise RuntimeError("nnls_factor_reuse: warm-start repair did not terminate")

    d = s_chol.copy()
    w = ZTx - ZTZ @ d

    # 5. The outer loop. Copied from `autoarray/util/fnnls.py:164-216` — the
    #    library's own Bro & de Jong iteration, calling the library's own
    #    `fix_constraint_cholesky` and its in-place Cholesky kernels. It is
    #    reproduced rather than called because `fnnls_cholesky` owns its factor
    #    from the first line and offers no way in with one already built; every
    #    line below that differs from the library's is a bug in this kernel.
    while (not np.all(P)) and np.max(w[~P]) > tolerance:
        current_P = P.copy()
        idmax = int(np.argmax(w * ~P))
        P_inorder = np.append(P_inorder, idmax)

        if k_active == 0:
            # Cold start (or a passive set emptied by the constraint fixer):
            # there is no factor to extend, so build the 1 x 1 one.
            U = scipy.linalg.cholesky(ZTZ[P_inorder][:, P_inorder])
            k_active = U.shape[0]
            U_buffer[:k_active, :k_active] = U
            n_factorisations += 1
        else:
            k_active = cholinsertlast_inplace(U_buffer, k_active, ZTZ[idmax][P_inorder])

        s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTx[P_inorder])

        P[idmax] = True
        while np.any(P) and np.min(s_chol[P]) <= tolerance:
            s_chol, d, P, P_inorder, k_active = fix_constraint_cholesky(
                ZTx=ZTx,
                s_chol=s_chol,
                d=d,
                P=P,
                P_inorder=P_inorder,
                U_buffer=U_buffer,
                k_active=k_active,
                tolerance=tolerance,
            )

            inner_iterations += 1
            if inner_iterations > 10000:
                raise RuntimeError("nnls_factor_reuse: constraint fixer did not terminate")

        d = s_chol.copy()
        w = ZTx - ZTZ @ d
        outer_iterations += 1

        if outer_iterations > 10000:
            raise RuntimeError("nnls_factor_reuse: outer loop did not terminate")

        if np.all(current_P == P):
            no_update += 1
        else:
            no_update = 0

        if no_update >= max_repetitions:
            break

    if not np.all(np.isfinite(d)):
        raise np.linalg.LinAlgError(
            "nnls_factor_reuse produced a non-finite solution "
            f"({np.count_nonzero(~np.isfinite(d))} of {d.size} entries)."
        )

    if stats is not None:
        stats.update(
            {
                "n_factorisations": int(n_factorisations),
                "n_downdates": int(n_downdates),
                "outer_iterations": int(outer_iterations),
                "inner_iterations": int(inner_iterations),
                "n_passive": int(P_inorder.size),
                "passive_set": P_inorder.copy(),
                "seed_source": "factor_reuse_dense_sign",
                "n_seed_negatives": n_seed_negatives,
            }
        )

    return d


# ===================================================================
# (b) The library's own paths, called as production calls them
# ===================================================================


def library_nnls_solve(ZTZ, ZTx, *, stats: dict | None = None) -> np.ndarray:
    """``fnnls_cholesky`` with the production cold warm start.

    Reproduces ``reconstruction_positive_only_from``'s un-memoized branch
    (``inversion_util.py:432-437``) — including the ``np.linalg.solve`` whose LU
    factor is discarded — so the reference row and the candidate row are timed
    through the same shape and the difference between them is the kernel and not
    the call wrapper.
    """
    ZTZ = np.asarray(ZTZ, dtype=float)
    ZTx = np.asarray(ZTx, dtype=float)
    solution = fnnls_cholesky(
        ZTZ,
        ZTx.T,
        P_initial=np.linalg.solve(ZTZ, ZTx) > 0,
        stats=stats,
    )
    if stats is not None:
        # `fnnls_cholesky` knows nothing of these two; the library adds them
        # after it returns (`inversion_util.py:412-413, 437`).
        stats.setdefault("seed_source", "dense")
        stats.setdefault("warm_start_fallback", False)
        # One LU (the discarded sign seed) plus one Cholesky of the passive block.
        stats.setdefault("n_factorisations", 2)
        stats.setdefault("n_downdates", None)
    return solution


def library_reconstruction_positive_only(
    data_vector,
    curvature_reg_matrix,
    *,
    settings=None,
    fingerprint=None,
):
    """The library's positive-only entry point, on the NumPy branch.

    The whole production call — memo lookup, seeded or dense solve, fallback
    guard, memo write-back — rather than ``fnnls_cholesky`` alone, so a memo row
    measures the path production takes and not a re-implementation of it.
    """
    from autoarray.inversion.inversion import inversion_util

    return inversion_util.reconstruction_positive_only_from(
        data_vector=np.asarray(data_vector, dtype=float),
        curvature_reg_matrix=np.asarray(curvature_reg_matrix, dtype=float),
        settings=settings,
        xp=np,
        fingerprint=fingerprint,
    )


# ===================================================================
# (c) The injection seam
# ===================================================================


@contextlib.contextmanager
def numpy_solver_injected(solver, *, label: str):
    """Rebind the library's positive-only solve to ``solver`` on the NumPy path.

    The mirror image of ``library_solver_injection.certified_solver_injected``,
    with the ``xp`` branch inverted: a **JAX** call is delegated to the original
    untouched (and counted), a **numpy** call is dispatched to ``solver``. A
    numba cell only ever produces the second kind, but the delegation is written
    and tested anyway — a patch that silently mis-routed a JAX call would be
    invisible in a process that has no JAX in it.

    ``solver`` is called as ``solver(curvature_reg_matrix, data_vector,
    stats=...)`` — the ``(ZTZ, ZTx)`` order of :func:`nnls_factor_reuse` and of
    ``fnnls_cholesky``, not the library entry point's ``(D, F + lambda H)``.

    Yields a mutable dict, ``{"numpy": n, "jax": n, "label": label,
    "last_stats": dict | None}``. The counters exist to be **asserted**: a patch
    that never fires would report the unpatched route's timing wearing the
    candidate's label.

    Restoration is by identity. On exit the attribute must still be the wrapper
    this manager installed; if something else has rebound it in the meantime the
    original is *not* restored over the top — that is a bug in the caller's
    nesting, and silently winning the race would hide it.

    Install it **outside** ``call_accounting.install``
        ``call_accounting.function_site`` rebinds the same dotted name, and both
        rebind by module attribute. Installed outside, this manager is in place
        first and the accounting wrapper closes over ``patched``, so the
        decomposition's solver site attributes the injected solve rather than the
        library's. Installed inside, ``uninstall()`` would restore the library
        function over the injection and the row would measure the wrong kernel.
        ``__wrapped__``, ``__name__`` and ``__doc__`` are carried over so a site
        that introspects the callable still resolves it.
    """
    from autoarray.inversion.inversion import inversion_util

    original = inversion_util.reconstruction_positive_only_from
    counts: dict = {"numpy": 0, "jax": 0, "label": label, "last_stats": None}

    # `factor` is the Cholesky-factor out-dict lever 3 of #267 added to the
    # library entry point (PyAutoArray `b4322c3e`): `AbstractInversion.reconstruction`
    # now passes one down so `log_det_curvature_reg_matrix_term` can read
    # `log det(F + lambda*H)` off the solve's factor instead of factorizing the same
    # matrix again. Read off the signature rather than assumed, so this manager
    # works against a pre-lever-3 library too — where the parameter does not exist
    # and forwarding it would be a `TypeError`.
    original_takes_factor = "factor" in inspect.signature(original).parameters

    def patched(
        data_vector,
        curvature_reg_matrix,
        settings=None,
        xp=np,
        fingerprint=None,
        factor=None,
    ):
        if xp.__name__.startswith("jax"):
            counts["jax"] += 1
            kwargs = {}
            if original_takes_factor:
                kwargs["factor"] = factor
            return original(
                data_vector=data_vector,
                curvature_reg_matrix=curvature_reg_matrix,
                settings=settings,
                xp=xp,
                fingerprint=fingerprint,
                **kwargs,
            )
        counts["numpy"] += 1
        if factor is not None:
            # Mirror the library's own contract: `reconstruction_positive_only_from`
            # clears this out-dict on entry so that a caller reading it after a solve
            # that did not publish a factor sees "no factor" rather than one belonging
            # to some other matrix. The injected solvers here keep their factorisation
            # for their own reuse and publish nothing, so leaving the dict empty is
            # what makes lever 3's `log_det_curvature_reg_matrix_term` decline the
            # fast path and take the unchanged dense route on the injected routes —
            # which is correct, and is why the `d_np` rows stay comparable with the
            # library rows they are measured against.
            factor.clear()
        stats: dict = {}
        counts["last_stats"] = stats
        return solver(
            np.asarray(curvature_reg_matrix, dtype=float),
            np.asarray(data_vector, dtype=float),
            stats=stats,
        )

    patched.__wrapped__ = original
    patched.__name__ = getattr(original, "__name__", "reconstruction_positive_only_from")
    patched.__doc__ = getattr(original, "__doc__", None)

    inversion_util.reconstruction_positive_only_from = patched
    try:
        yield counts
    finally:
        current = inversion_util.reconstruction_positive_only_from
        if current is not patched:
            raise RuntimeError(
                f"numpy_solver_injected({label!r}): "
                f"{LIBRARY_POSITIVE_ONLY_DOTTED} was rebound by something else while "
                f"the injection was open (found {current!r}, expected the injected "
                f"wrapper). Restoring the original here would clobber whatever holds "
                f"it now; fix the nesting instead."
            )
        inversion_util.reconstruction_positive_only_from = original


# ===================================================================
# (d) The row builders
# ===================================================================


def _iteration_fields(stats: dict) -> dict:
    """The counts every solver row carries, with ``None`` where a kernel has none."""
    return {
        "n_factorisations": stats.get("n_factorisations"),
        "n_downdates": stats.get("n_downdates"),
        "outer_iterations": stats.get("outer_iterations"),
        "inner_iterations": stats.get("inner_iterations"),
        "n_passive": stats.get("n_passive"),
        "n_seed_negatives": stats.get("n_seed_negatives"),
        "seed_source": stats.get("seed_source"),
        "warm_start_errors": stats.get("warm_start_errors"),
        "warm_start_fallback": stats.get("warm_start_fallback"),
    }


def _equivalence(x_full, reference_x, *, rtol: float = EQUIVALENCE_RTOL) -> dict:
    """Max absolute reconstruction difference against the library's own answer."""
    got = np.asarray(x_full, dtype=float)
    ref = np.asarray(reference_x, dtype=float)
    max_abs = float(np.max(np.abs(got - ref))) if ref.size else 0.0
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    return {
        "max_abs_diff_reconstruction_vs_library": max_abs,
        "equivalence_pin": {"rtol": rtol, "passed": bool(max_abs / scale <= rtol)},
    }


def library_cold_row(
    system,
    view,
    reference_log_evidence,
    reference_x,
    *,
    n_repeats: int = 10,
) -> dict:
    """``K0`` — the library's cold production solve. The reference row."""
    crm = view["curvature_reg_matrix"]
    dv = view["data_vector"]

    def _call():
        return library_nnls_solve(crm, dv)

    stats: dict = {}
    x_sub = library_nnls_solve(crm, dv, stats=stats)
    x_full = view["scatter_back"](x_sub)

    row = {
        "label": "library fnnls_cholesky, cold (dense-sign start, memo off)",
        "entry_point": "autoarray.util.fnnls.fnnls_cholesky",
        "problem": "the edge-zeroed subset, as the library hands it",
        "n": int(view["n_seen_by_solver"]),
        **_iteration_fields(stats),
    }
    row.update(time_median(_call, n_repeats=n_repeats))
    row.update(scored_np(system, x_full, reference_log_evidence))
    row.update(_equivalence(x_full, reference_x))
    return row


def factor_reuse_row(
    system,
    view,
    reference_log_evidence,
    reference_x,
    *,
    n_repeats: int = 10,
) -> dict:
    """``K3`` — :func:`nnls_factor_reuse`: one factorisation, downdates, same loop."""
    crm = view["curvature_reg_matrix"]
    dv = view["data_vector"]

    def _call():
        return nnls_factor_reuse(crm, dv)

    stats: dict = {}
    x_sub = nnls_factor_reuse(crm, dv, stats=stats)
    x_full = view["scatter_back"](x_sub)

    row = {
        "label": "factor-reuse NNLS (one Cholesky, downdates, library outer loop)",
        "entry_point": "likelihood_breakdown.fixed_light_numpy_solvers.nnls_factor_reuse",
        "problem": "the edge-zeroed subset, as the library hands it",
        "n": int(view["n_seen_by_solver"]),
        **_iteration_fields(stats),
    }
    row.update(time_median(_call, n_repeats=n_repeats))
    row.update(scored_np(system, x_full, reference_log_evidence))
    row.update(_equivalence(x_full, reference_x))
    return row


def unconstrained_row(
    system,
    view,
    reference_log_evidence,
    reference_x,
    *,
    n_repeats: int = 10,
) -> dict:
    """``K4`` — ``cho_factor``/``cho_solve``, positivity dropped. The floor.

    RECORDED, never a headline: it solves a **different problem**. Its
    ``equivalence_pin`` is the string ``"recorded"`` rather than a boolean,
    because there is no sense in which an infeasible minimiser passes or fails an
    equivalence to the constrained one — its Δ log-evidence and its negative
    count are the numbers to read.
    """
    crm = view["curvature_reg_matrix"]
    dv = view["data_vector"]

    def _call():
        c, low = scipy.linalg.cho_factor(crm, lower=True, check_finite=False)
        return scipy.linalg.cho_solve((c, low), dv, check_finite=False)

    x_sub = _call()
    x_full = view["scatter_back"](x_sub)
    ref = np.asarray(reference_x, dtype=float)

    row = {
        "label": "scipy unconstrained (cho_factor/cho_solve) — the floor, RECORDED",
        "entry_point": "scipy.linalg.cho_factor / cho_solve",
        "problem": (
            "the edge-zeroed subset with positivity dropped — a DIFFERENT, infeasible "
            "minimiser, reported to bound the solve from below and never quoted as a "
            "candidate"
        ),
        "n": int(view["n_seen_by_solver"]),
        "n_factorisations": 1,
        "n_downdates": 0,
        "outer_iterations": 0,
        "inner_iterations": 0,
        "n_passive": int(np.count_nonzero(np.asarray(x_sub, dtype=float) > 0.0)),
        "n_seed_negatives": None,
        "seed_source": "unconstrained",
        "warm_start_errors": None,
        "warm_start_fallback": None,
    }
    row.update(time_median(_call, n_repeats=n_repeats))
    row.update(scored_np(system, x_full, reference_log_evidence))
    row["max_abs_diff_reconstruction_vs_library"] = (
        float(np.max(np.abs(np.asarray(x_full, dtype=float) - ref))) if ref.size else 0.0
    )
    row["equivalence_pin"] = "recorded"
    return row


def memo_warm_row(
    systems,
    reference_log_evidence,
    reference_x,
    *,
    label: str,
    fingerprint: str,
    settings,
) -> dict:
    """``K1`` — the library's memo path, over a **stream** of nearby systems.

    The cross-evaluation warm-start memo
    (``autoarray/inversion/inversion/nnls_memo.py``) seeds each solve from the
    previous solve's final passive set. Its benefit only exists **between
    different evaluations**, so this row is timed one call per system over a
    stream rather than ``n_repeats`` calls on one system: re-solving the same
    matrix would warm-start from its own answer and report zero iterations,
    which is not a number production ever sees.

    The first call is the cold one (the memo is empty) and is recorded
    separately; the quoted ``ms`` is the **median of the rest**.

    The equivalence pin is taken on a final, genuinely warm re-solve of
    ``systems[0]`` — the reference problem, seeded from a *neighbour's* passive
    set. Pinning the stream's own first call would pin a cold solve, which is the
    library cold row by construction and proves nothing about the memo.

    ``AUTOARRAY_NNLS_WARM_START`` is set to ``"1"`` for the duration and restored
    afterwards, and the memo dict is cleared before and after the row: a cell
    that runs with the memo off by default must not leave this row reading an env
    it cannot see, nor leave a seed behind for the next row.
    """
    import os
    import time

    from autoarray.inversion.inversion import nnls_memo

    systems = list(systems)
    if len(systems) < 2:
        raise ValueError(
            f"memo_warm_row needs a stream of at least 2 systems (got {len(systems)}); "
            f"the memo's benefit only exists between different evaluations."
        )

    env_before = os.environ.get("AUTOARRAY_NNLS_WARM_START")
    os.environ["AUTOARRAY_NNLS_WARM_START"] = "1"
    nnls_memo._nnls_passive_set_memo.clear()

    calls: list[dict] = []
    view_0 = None
    x_full_warm = None

    def _timed_solve(view):
        crm = view["curvature_reg_matrix"]
        dv = view["data_vector"]
        started = time.perf_counter()
        x_sub = library_reconstruction_positive_only(
            dv, crm, settings=settings, fingerprint=fingerprint
        )
        return (time.perf_counter() - started) * 1e3, x_sub

    try:
        for index, one_system in enumerate(systems):
            view = solver_view_of(one_system)
            if index == 0:
                view_0 = view

            ms, _x_sub = _timed_solve(view)

            entry = nnls_memo._nnls_passive_set_memo.get(
                nnls_memo.memo_key(n=int(view["n_seen_by_solver"]), fingerprint=fingerprint)
            )
            calls.append(
                {
                    "index": index,
                    "n": int(view["n_seen_by_solver"]),
                    "ms": ms,
                    "cold": index == 0,
                    "memo_entry_present_after": entry is not None,
                    "memo_n_passive_after": (
                        None if entry is None else int(entry.passive_set.size)
                    ),
                    "memo_dense_error_fraction": (
                        None if entry is None else float(entry.dense_error_fraction)
                    ),
                }
            )

        # The pin: the reference problem re-solved with the memo holding the LAST
        # stream member's passive set — a warm solve of a matrix the seed did not
        # come from, which is exactly what production does.
        warm_ms, x_sub_warm = _timed_solve(view_0)
        x_full_warm = view_0["scatter_back"](x_sub_warm)
    finally:
        if env_before is None:
            os.environ.pop("AUTOARRAY_NNLS_WARM_START", None)
        else:
            os.environ["AUTOARRAY_NNLS_WARM_START"] = env_before
        nnls_memo._nnls_passive_set_memo.clear()

    warm = [call["ms"] for call in calls[1:]]

    row = {
        "label": label,
        "entry_point": LIBRARY_POSITIVE_ONLY_DOTTED,
        "problem": "the edge-zeroed subset, as the library hands it",
        "n": calls[0]["n"],
        "n_factorisations": None,
        "n_downdates": None,
        "outer_iterations": None,
        "inner_iterations": None,
        "n_passive": calls[-1]["memo_n_passive_after"],
        "n_seed_negatives": None,
        "seed_source": "memo",
        "warm_start_errors": None,
        "warm_start_fallback": None,
        "ms": float(np.median(warm)),
        "ms_min": float(np.min(warm)),
        "ms_max": float(np.max(warm)),
        "n_repeats": len(warm),
        "n_warmup": 0,
        "timer": "time.perf_counter, median over the stream after the cold call",
        "cold_call_ms": calls[0]["ms"],
        "pin_resolve_ms": warm_ms,
        "n_stream": len(calls),
        "calls": calls,
        "nnls_warm_start_env_during_row": "1",
        "note": (
            "One call per system, not n_repeats calls on one system: the memo seeds "
            "from the PREVIOUS evaluation's passive set, so repeating one solve would "
            "warm-start it from its own answer and report a solve production never "
            "gets. The equivalence pin is a final warm re-solve of systems[0]."
        ),
    }
    row.update(scored_np(systems[0], x_full_warm, reference_log_evidence))
    row.update(_equivalence(x_full_warm, reference_x))
    return row
