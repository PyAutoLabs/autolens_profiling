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
- :func:`fnnls_kernel_injected` — the seam **one level below** that one, which
  rebinds ``fnnls_cholesky`` itself rather than the entry point around it, so an
  injected kernel still receives the library's memo seed and still publishes the
  ``factor`` lever 3's log det reads (lever 4b, #276);
- :func:`active_set_jaccard` / :func:`kkt_residuals` / :func:`evidence_delta` /
  :func:`log_det_factor_check` — the metrics the lever 4b witness gates on, here
  rather than in the cell so a test can import them without running a cell;
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
    "LIBRARY_FNNLS_KERNEL_DOTTED",
    "LIBRARY_POSITIVE_ONLY_DOTTED",
    "WITNESS_EVIDENCE_RTOL",
    "WITNESS_LOG_DET_ATOL",
    "active_set_jaccard",
    "evidence_delta",
    "factor_reuse_row",
    "fnnls_cholesky_permuted",
    "fnnls_kernel_injected",
    "iteration_row",
    "kkt_residuals",
    "library_cold_row",
    "library_nnls_solve",
    "library_reconstruction_positive_only",
    "log_det_factor_check",
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

#: The dotted name :func:`fnnls_kernel_injected` rebinds — the NNLS kernel
#: itself, one level below :data:`LIBRARY_POSITIVE_ONLY_DOTTED`. It is the
#: ``fnnls`` module's attribute, because ``reconstruction_positive_only_from``
#: imports the name inside its own body (``inversion_util.py:397``) and so
#: resolves it from that module on every call.
LIBRARY_FNNLS_KERNEL_DOTTED = "autoarray.util.fnnls.fnnls_cholesky"

#: The lever 4b witness's W3 gate: Δ log evidence, relative. The phase's Witness
#: pin, and the same number every other equivalence gate in this campaign uses.
WITNESS_EVIDENCE_RTOL = 1.0e-9

#: The lever 4b witness's W6 gate, in nats. ``log_det_from_passive_cholesky_from``
#: agrees with a dense ``slogdet`` of the same matrix to <= 2e-12 nats — the
#: original number from the issue. The full-size identity run exceeded it at
#: floating-point roundoff; the approved gate also permits 32 scalar spacings
#: and independently verifies the passive factor reconstruction (1e-12 relative).
WITNESS_LOG_DET_ATOL = 2.0e-12
WITNESS_LOG_DET_ULPS = 32
WITNESS_FACTOR_RTOL = 1.0e-12

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


def fnnls_cholesky_permuted(
    ZTZ,
    ZTx,
    P_initial=np.zeros(0, dtype=int),
    stats: dict | None = None,
    factor: dict | None = None,
):
    """Run the library FNNLS iteration after one passive-first permutation.

    The symmetric gather is paid once.  Thereafter the initial passive block is
    contiguous and every Cholesky insertion reads a row from the gathered
    matrix.  Active-set choices retain NumPy's original-coordinate ``argmax``
    tie rule, so the permutation does not deliberately change the algorithm.
    """
    ZTZ = np.asarray(ZTZ)
    ZTx = np.asarray(ZTx)
    P_initial = np.asarray(P_initial)

    n = ZTZ.shape[0]
    tolerance = _EPSILON * n
    max_repetitions = 3

    initial_mask = np.zeros(n, dtype=bool)
    if P_initial.dtype == bool:
        initial_mask[:] = P_initial
        passive_original = np.where(P_initial)[0].astype(int)
    else:
        passive_original = P_initial.astype(int)
        initial_mask[passive_original] = True

    active_original = np.flatnonzero(~initial_mask)
    perm = np.concatenate((passive_original, active_original))

    # The sole full symmetric gather.  np.ix_ expresses the intended single
    # n-by-n result directly; all later matrix access stays in this coordinate
    # system.
    ZTZp = ZTZ[np.ix_(perm, perm)]
    ZTxp = ZTx[perm]

    k_seed = passive_original.size
    P = np.zeros(n, dtype=bool)
    P[:k_seed] = True
    P_inorder = np.arange(k_seed, dtype=int)

    d = np.zeros(n)
    w = ZTxp.copy()
    s_chol = np.zeros(n)
    U_buffer = np.zeros((n, n))
    k_active = 0
    loop_count = 0
    loop_count2 = 0
    no_update = 0

    if k_seed:
        U = scipy.linalg.cholesky(ZTZp[:k_seed, :k_seed], check_finite=False)
        k_active = k_seed
        U_buffer[:k_active, :k_active] = U
        s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTxp[P_inorder])

        # A seed is not a feasible previous iterate.  Delete every non-positive
        # member and re-solve until the reduced seed is feasible, exactly as the
        # library kernel does before entering its outer loop.
        while P_inorder.size and np.min(s_chol[P_inorder]) <= tolerance:
            id_delete = np.where(s_chol[P_inorder] <= tolerance)[0]
            k_active = choldeleteindexes_inplace(U_buffer, k_active, id_delete)
            P[P_inorder[id_delete]] = False
            P_inorder = np.delete(P_inorder, id_delete)
            s_chol[~P] = 0.0
            if P_inorder.size:
                s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTxp[P_inorder])
            loop_count2 += 1
            if loop_count2 > 10000:
                raise RuntimeError

        d = s_chol.copy()
        w = ZTxp - ZTZp @ d

    while (not np.all(P)) and np.max(w[~P]) > tolerance:
        current_P = P.copy()

        # np.argmax(w_original * ~P_original) selects the smallest original
        # index on a tie.  A plain argmax in permuted coordinates would silently
        # change that rule whenever ``perm`` is non-identity.
        active_p = np.flatnonzero(~P)
        max_w = np.max(w[active_p])
        tied_p = active_p[w[active_p] == max_w]
        idmax = int(tied_p[np.argmin(perm[tied_p])])
        P_inorder = np.append(P_inorder, idmax)

        if k_active == 0:
            U = scipy.linalg.cholesky(
                ZTZp[idmax : idmax + 1, idmax : idmax + 1],
                check_finite=False,
            )
            k_active = 1
            U_buffer[0, 0] = U[0, 0]
        else:
            k_active = cholinsertlast_inplace(U_buffer, k_active, ZTZp[idmax, P_inorder])

        s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTxp[P_inorder])
        P[idmax] = True

        while np.any(P) and np.min(s_chol[P]) <= tolerance:
            s_chol, d, P, P_inorder, k_active = fix_constraint_cholesky(
                ZTx=ZTxp,
                s_chol=s_chol,
                d=d,
                P=P,
                P_inorder=P_inorder,
                U_buffer=U_buffer,
                k_active=k_active,
                tolerance=tolerance,
            )
            loop_count2 += 1
            if loop_count2 > 10000:
                raise RuntimeError

        d = s_chol.copy()
        w = ZTxp - ZTZp @ d
        loop_count += 1
        if loop_count > 10000:
            raise RuntimeError

        if np.all(current_P == P):
            no_update += 1
        else:
            no_update = 0
        if no_update >= max_repetitions:
            break

    if not np.all(np.isfinite(d)):
        raise np.linalg.LinAlgError(
            "fnnls_cholesky_permuted produced a non-finite solution "
            f"({np.count_nonzero(~np.isfinite(d))} of {d.size} entries)."
        )

    passive_final_original = perm[P_inorder]
    result = np.empty_like(d)
    result[perm] = d

    if stats is not None:
        final_mask_original = np.zeros(n, dtype=bool)
        final_mask_original[passive_final_original] = True
        stats["outer_iterations"] = loop_count
        stats["inner_iterations"] = loop_count2
        stats["passive_set"] = passive_final_original.copy()
        stats["n_passive"] = int(P_inorder.size)
        stats["warm_start_errors"] = int(np.count_nonzero(initial_mask != final_mask_original))
        stats["n_perm_gathers"] = 1

    if factor is not None:
        factor["U_buffer"] = U_buffer
        factor["k_active"] = int(k_active)
        factor["passive_set"] = passive_final_original.copy()
        factor["matrix_shape"] = tuple(ZTZ.shape)

    return result


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
    # Likewise PyAutoArray #566's `solver` / `stats`: declared by the wrapper (so the
    # signature still covers the library's) and forwarded on the delegated JAX path
    # only to a library that has them. This is `library_solver_injection._library_kwargs`
    # inlined: that module imports JAX, which this module must never do (see
    # `test_the_numpy_modules_import_no_jax_at_any_level`). On the numpy path the
    # injected kernel *is* the solve, so the library's `solver` choice does not reach
    # it. The wrapper's keyword must be spelled `solver` (the library passes it by
    # name), so the injected kernel is held as `injected`.
    original_forwarded = frozenset(
        name for name in ("solver", "stats") if name in inspect.signature(original).parameters
    )
    injected = solver

    def patched(
        data_vector,
        curvature_reg_matrix,
        settings=None,
        xp=np,
        fingerprint=None,
        factor=None,
        solver="pdip",
        stats=None,
    ):
        if xp.__name__.startswith("jax"):
            counts["jax"] += 1
            kwargs = {}
            if original_takes_factor:
                kwargs["factor"] = factor
            for name, value in (("solver", solver), ("stats", stats)):
                if name in original_forwarded:
                    kwargs[name] = value
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
        solve_stats: dict = {}
        counts["last_stats"] = solve_stats
        return injected(
            np.asarray(curvature_reg_matrix, dtype=float),
            np.asarray(data_vector, dtype=float),
            stats=solve_stats,
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


@contextlib.contextmanager
def fnnls_kernel_injected(kernel, *, label: str):
    """Rebind the NNLS **kernel** ``fnnls_cholesky`` itself to ``kernel``.

    One level below :func:`numpy_solver_injected`, and for a reason that is the
    whole design of lever 4b.

    Why not the entry-point seam
        ``numpy_solver_injected`` replaces
        ``reconstruction_positive_only_from`` wholesale and calls its solver as
        ``solver(ZTZ, ZTx, stats=...)``. Two things the library does are
        therefore **not** done for the injected solver: the warm-start memo's
        seed is never passed (production runs ``fnnls_cholesky(P_initial=entry
        .passive_set)`` at ~2 outer iterations, so a kernel behind that seam
        re-seeds from the dense sign and a ``b`` vs candidate delta measures the
        seed rather than the kernel), and the ``factor`` out-dict is
        ``clear()``-ed, so lever 3's ``log_det_from_passive_cholesky_from`` fast
        path is declined and the route pays the dense log det instead. A
        candidate whose win *is* the memo-seeded passive block cannot be
        measured through it. It has to be a drop-in for ``fnnls_cholesky``.

    What is rebound, and why that name
        ``autoarray.util.fnnls.fnnls_cholesky`` — the module attribute, not a
        name on ``inversion_util``. ``reconstruction_positive_only_from`` does
        its ``from autoarray.util.fnnls import fnnls_cholesky`` **inside the
        function body** (``inversion_util.py:397``), so the name is resolved
        from the ``fnnls`` module on every call and rebinding it there is what
        both of its call sites (``:424`` the memo-seeded attempt, ``:443`` the
        dense-sign solve) actually see. Rebinding ``inversion_util
        .fnnls_cholesky`` would create an attribute nothing reads. Note the
        module-level ``fnnls_cholesky`` imported at the top of *this* module is
        bound once at import and is deliberately unaffected: it stays the
        genuine library kernel, which is what the witness's control pass and
        :func:`library_nnls_solve` need.

    ``kernel`` is called exactly as the library calls it —
    ``kernel(ZTZ, ZTx, P_initial=..., stats=..., factor=...)`` — with the
    library's own defaults, so a kernel that only forwards is transparent and
    the memo seed and the factor out-dict reach it untouched.

    Yields a mutable dict, ``{"calls": n, "label": label, "last_stats": dict |
    None, "last_factor_keys": list | None}``. The counter exists to be
    **asserted**: an injection that never fired would report the library
    kernel's answer wearing the candidate's label. ``last_stats`` is the dict
    the *library* handed down and the kernel filled, read after the call (the
    library adds ``seed_source`` / ``warm_start_fallback`` to it afterwards, so
    a caller must read it after the evaluation, not at the point the kernel
    returns); ``last_factor_keys`` is what the kernel published into ``factor``,
    the contract ``log_det_from_passive_cholesky_from`` reads.

    Restoration is by identity, exactly as :func:`numpy_solver_injected` does
    it. On exit the attribute must still be the wrapper this manager installed;
    if something else has rebound it in the meantime the original is *not*
    restored over the top — that is a bug in the caller's nesting, and silently
    winning the race would hide it.

    Install it **outside** ``call_accounting.install``
        ``call_accounting.function_site("fnnls", fnnls_module, "fnnls_cholesky")``
        rebinds this same attribute (``fixed_light_numba.py:1259``). Installed
        outside, this manager is in place first and the accounting wrapper
        closes over ``patched``, so the decomposition's ``solver.fnnls_cholesky``
        site attributes the injected kernel; installed inside, ``uninstall()``
        would restore the library kernel over the injection and the row would
        measure the wrong thing. ``__wrapped__``, ``__name__`` and ``__doc__``
        are carried over so a site that introspects the callable still resolves
        it.
    """
    from autoarray.util import fnnls as fnnls_module

    original = fnnls_module.fnnls_cholesky
    counts: dict = {
        "calls": 0,
        "label": label,
        "last_stats": None,
        "last_factor_keys": None,
    }

    def patched(
        ZTZ,
        ZTx,
        P_initial=np.zeros(0, dtype=int),
        stats: dict | None = None,
        factor: dict | None = None,
    ):
        counts["calls"] += 1
        result = kernel(ZTZ, ZTx, P_initial=P_initial, stats=stats, factor=factor)
        # After the call, never before: `fnnls_cholesky` publishes both dicts on
        # a successful return only, and a call that raises must leave the
        # previous call's record rather than a half-filled one.
        counts["last_stats"] = stats
        counts["last_factor_keys"] = None if factor is None else sorted(factor)
        return result

    patched.__wrapped__ = original
    patched.__name__ = getattr(original, "__name__", "fnnls_cholesky")
    patched.__doc__ = getattr(original, "__doc__", None)

    fnnls_module.fnnls_cholesky = patched
    try:
        yield counts
    finally:
        current = fnnls_module.fnnls_cholesky
        if current is not patched:
            raise RuntimeError(
                f"fnnls_kernel_injected({label!r}): "
                f"{LIBRARY_FNNLS_KERNEL_DOTTED} was rebound by something else while "
                f"the injection was open (found {current!r}, expected the injected "
                f"wrapper). Restoring the original here would clobber whatever holds "
                f"it now; fix the nesting instead."
            )
        fnnls_module.fnnls_cholesky = original


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


# ===================================================================
# (e) The lever 4b witness metrics
# ===================================================================
#
# These four functions ARE the witness's gates. They live here rather than in
# `scripts/imaging/likelihood_breakdown/fixed_light_numba_s4b_witness.py`
# because that cell — like every cell in this repo — is a top-to-bottom script
# with no `__main__` guard: importing it to test one metric would build an HST
# Delaunay N=1500 system and run the whole witness. A metric that cannot be
# tested on hand-built inputs is a metric nobody has checked, and the red
# control in `scripts/misc/test/test_fixed_light_s4b.py` (a deliberately wrong
# kernel that must FAIL W2 and W3) depends on calling them directly.


def active_set_jaccard(passive_a, passive_b) -> dict:
    """W2's metric: the Jaccard index of two solves' final passive sets.

    ``|A ∩ B| / |A ∪ B|`` over the passive *indices*, order-insensitive — the
    active-set solver appends and deletes, so ``stats["passive_set"]`` is in
    insertion order and two solves that agree on the set can disagree on the
    order. The order matters to the Cholesky factor (W6 guards that); it does
    not matter to "did the two kernels land on the same active set", which is
    what this measures.

    Two empty sets are ``1.0`` (identical, vacuously), not ``0/0``.
    """
    set_a = {int(index) for index in np.asarray(passive_a, dtype=int).ravel()}
    set_b = {int(index) for index in np.asarray(passive_b, dtype=int).ravel()}
    union = set_a | set_b
    intersection = set_a & set_b
    return {
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "n_passive_a": len(set_a),
        "n_passive_b": len(set_b),
        "n_intersection": len(intersection),
        "n_union": len(union),
        "only_in_a": sorted(set_a - set_b),
        "only_in_b": sorted(set_b - set_a),
        "identical": set_a == set_b,
    }


def kkt_residuals(ZTZ, ZTx, x, passive_set) -> dict:
    """The two numbers that say whether an NNLS answer is optimal.

    With ``w = ZTx - ZTZ @ x`` and ``P`` the passive set, the KKT conditions of
    ``min |Zs - x|^2  s.t.  s >= 0`` are ``w[~P] <= 0`` and ``x[P] > 0`` (the
    library's own loop runs until ``max(w[~P]) <= eps * n``; ``fnnls.py:65``).
    So ``max(w[~P])`` and ``min(x[P])`` are reported together: they are what
    turns "the two kernels chose different active sets" into either "one of them
    is wrong" or "both are optimal to the solver's tolerance and the difference
    is a knife-edge index whose value is ~0".

    That second reading is exactly the question W2 hands to the human, which is
    why this is computed for **both** passes of every draw that disagrees rather
    than only for the candidate.

    ``max_w_active`` is ``None`` when the passive set is everything (no active
    index to violate), ``min_x_passive`` is ``None`` when it is empty.
    """
    ZTZ = np.asarray(ZTZ, dtype=float)
    ZTx = np.asarray(ZTx, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    n = x.size
    mask = np.zeros(n, dtype=bool)
    indices = np.asarray(passive_set, dtype=int).ravel()
    if indices.size:
        mask[indices] = True

    w = ZTx - ZTZ @ x

    return {
        "max_w_active": None if bool(mask.all()) else float(np.max(w[~mask])),
        "min_x_passive": None if not bool(mask.any()) else float(np.min(x[mask])),
        "max_abs_w_passive": None if not bool(mask.any()) else float(np.max(np.abs(w[mask]))),
        "solver_tolerance": float(_EPSILON * n),
        "n": int(n),
    }


def evidence_delta(value, reference, *, rtol: float = WITNESS_EVIDENCE_RTOL) -> dict:
    """W3's metric: Δ log evidence in nats, and relative, against the library.

    Both are reported. Nats are the physical quantity a reader of this campaign
    compares against a sampler's evidence differences; the relative number is
    what the gate is written in, because the phase's Witness pin is relative.
    """
    value = float(value)
    reference = float(reference)
    delta = value - reference
    relative = abs(delta) / max(abs(reference), 1e-300)
    return {
        "log_evidence": value,
        "log_evidence_reference": reference,
        "delta_nats": delta,
        "relative_difference": relative,
        "bit_identical": value == reference,
        "rtol": rtol,
        "within_rtol": bool(relative <= rtol),
    }


def log_det_factor_check(matrix, factor, *, atol: float = WITNESS_LOG_DET_ATOL) -> dict:
    """W6's metric: the published ``factor`` really is a factor of ``matrix``.

    ``log_det_from_passive_cholesky_from(matrix, **factor)`` against
    ``np.linalg.slogdet(matrix)``. This is the contract lever 3 depends on: a
    candidate kernel that returned the right reconstruction but published a
    factor whose ``passive_set`` order did not match its ``U_buffer`` would
    silently corrupt ``log det(F + λH)`` — the evidence term — while every
    active-set and reconstruction check passed.

    The four keys are checked for presence first, and reported, so a kernel that
    publishes nothing fails with "no factor" rather than a ``TypeError``.
    """
    from autoarray.inversion.inversion import inversion_util

    matrix = np.asarray(matrix, dtype=float)
    expected_keys = ("U_buffer", "k_active", "passive_set", "matrix_shape")
    keys = [] if factor is None else sorted(factor)
    missing = [key for key in expected_keys if factor is None or key not in factor]

    row = {
        "factor_keys": keys,
        "missing_keys": missing,
        "expected_keys": list(expected_keys),
        "atol_nats": atol,
        "matrix_shape": list(matrix.shape),
    }

    if missing:
        row.update(
            {
                "log_det_from_factor": None,
                "log_det_slogdet": None,
                "delta_nats": None,
                "within_atol": False,
                "error": f"factor is missing {missing}",
            }
        )
        return row

    row["k_active"] = int(factor["k_active"])
    row["factor_matrix_shape"] = list(factor["matrix_shape"])
    row["shape_agrees"] = tuple(factor["matrix_shape"]) == tuple(matrix.shape)
    passive_set = np.asarray(factor["passive_set"], dtype=int).ravel()
    U_buffer = np.asarray(factor["U_buffer"])
    k_active = int(factor["k_active"])
    structure_agrees = (
        0 <= k_active <= matrix.shape[0]
        and passive_set.size == k_active
        and np.unique(passive_set).size == k_active
        and np.all((0 <= passive_set) & (passive_set < matrix.shape[0]))
        and U_buffer.ndim == 2
        and U_buffer.shape[0] >= k_active
        and U_buffer.shape[1] >= k_active
    )
    row["structure_agrees"] = bool(structure_agrees)
    if not structure_agrees or not row["shape_agrees"]:
        row.update(within_atol=False, error="invalid factor structure or matrix shape")
        return row

    # Approved 2026-09-18: the library identity itself differs from slogdet by
    # 1.27e-11 nats at N=1500. Keep the original failure visible, allow only 32
    # spacings of the scalar determinant, and independently verify the factor's
    # matrix reconstruction so a determinant-preserving permutation cannot pass.
    upper = np.triu(U_buffer[:k_active, :k_active])
    passive_matrix = matrix[np.ix_(passive_set, passive_set)]
    residual = (
        float(np.max(np.abs(upper.T @ upper - passive_matrix)))
        / max(float(np.max(np.abs(passive_matrix))), np.finfo(float).tiny)
        if k_active
        else 0.0
    )
    row["factor_relative_residual"] = residual
    row["factor_rtol"] = WITNESS_FACTOR_RTOL

    try:
        sign, slogdet = np.linalg.slogdet(matrix)
        from_factor = float(
            inversion_util.log_det_from_passive_cholesky_from(
                matrix=matrix,
                U_buffer=U_buffer,
                k_active=k_active,
                passive_set=passive_set,
            )
        )
    except (IndexError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        row.update(
            {
                "log_det_from_factor": None,
                "log_det_slogdet": None,
                "delta_nats": None,
                "within_atol": False,
                "error": f"invalid factor: {type(error).__name__}: {error}",
            }
        )
        return row
    delta = from_factor - float(slogdet)
    effective_atol = max(atol, WITNESS_LOG_DET_ULPS * float(np.spacing(abs(slogdet))))

    row.update(
        {
            "log_det_from_factor": from_factor,
            "log_det_slogdet": float(slogdet),
            "slogdet_sign": float(sign),
            "delta_nats": delta,
            "original_within_atol": bool(abs(delta) <= atol),
            "effective_atol_nats": effective_atol,
            "allowed_spacings": WITNESS_LOG_DET_ULPS,
            "within_atol": bool(
                np.isfinite(from_factor)
                and np.isfinite(slogdet)
                and abs(delta) <= effective_atol
                and np.isfinite(residual)
                and residual <= WITNESS_FACTOR_RTOL
                and row["shape_agrees"]
                and structure_agrees
                and sign > 0
            ),
        }
    )
    return row


def iteration_row(stats: dict | None) -> dict:
    """W5's RECORDED row: what one solve's ``stats`` says it cost.

    The library's five keys plus the two ``reconstruction_positive_only_from``
    adds afterwards. ``None`` where a key is absent rather than a ``KeyError``:
    W5 is a record, and a kernel that publishes fewer keys is a fact about that
    kernel to be read in the table, not a crash.
    """
    stats = stats or {}
    return {
        "outer_iterations": stats.get("outer_iterations"),
        "inner_iterations": stats.get("inner_iterations"),
        "n_passive": stats.get("n_passive"),
        "warm_start_errors": stats.get("warm_start_errors"),
        "seed_source": stats.get("seed_source"),
        "warm_start_fallback": stats.get("warm_start_fallback"),
    }
