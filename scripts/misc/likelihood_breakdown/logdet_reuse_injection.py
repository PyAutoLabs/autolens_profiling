"""Harness-level reuse of the certified solve's Cholesky factor for ``log det(F + lambda*H)``.

Why this module exists
----------------------

Phase 1 of the ``hst-gpu-non-solver-residue`` epic (autolens_profiling#268)
found that on the A100 the evidence term ``log_det_curvature_reg_matrix_term``
runs its own cuSOLVER Cholesky of ``M = F + lambda*H`` (0.89 ms, 2.8 % of the
HST Delaunay N=1500 fp64 call), right after the positive-only solve has
factorised the same system. The NumPy path already avoids this through the
block-determinant identity (``inversion_util.log_det_from_passive_cholesky_from``,
<= 2e-12 nats); the JAX path does not, because the library's certified solver
(``autoarray.util.jax_active_set.solve_certified``, #566) throws its factor away.

Phase 4 (autolens_profiling#303) asks whether keeping that factor is worth
>= 0.5 ms of the fused whole call **without changing the answer** (1e-9
relative). Nothing in PyAutoArray moves: exactly as ``library_solver_injection``
and ``psf_cube_injection`` do, this module rebinds two library entry points for
the duration of a scoped context, and a candidate that clears the lever becomes
a separate PyAutoArray prompt via /intake.

The identity
------------

For ANY split of the indices into F and Z and any symmetric positive-definite M,

    log det M = log det M_FF + log det(M_ZZ - M_ZF M_FF^-1 M_FZ).

It needs nothing from the solve except a Cholesky factor of M_FF, so it is exact
whichever set the search stopped on — including a lane whose certificate failed
and whose ``x`` came from the PDIP fallback: that lane's factor is still a
factor of a principal block of the same M.

The Jacobi scaling. With ``general.inversion.nnls_jacobi_preconditioning`` on
(the packaged default) ``reconstruction_positive_only_from``
(``inversion_util.py:433-436`` at PyAutoArray 681938ae) hands the solver

    D = 1 / sqrt(diag M),   Q_pc = (M * D[:, None]) * D[None, :]   ( = D M D ),

so the stashed factor is of ``Q_pc``'s free block, and

    log det M = log det Q_pc - 2 sum log D = log det Q_pc + sum log diag M.

With preconditioning off, D = 1 and the same formulas hold with no correction.

The two paired rebinds (active only while tracing)
--------------------------------------------------

(a) ``autoarray.util.jax_active_set.solve_certified`` ->
    :func:`solve_certified_stashing`. The library's
    ``solve_certified_with_fallback`` calls ``solve_certified`` as a MODULE
    GLOBAL of ``jax_active_set`` at call time (``jax_active_set.py:340``), and
    ``_certified_positive_only_from`` imports ``solve_certified_with_fallback``
    inside the function (``inversion_util.py:260``), so rebinding the module
    attribute is picked up by every trace entered inside the context. The
    wrapper runs the library's own ``active_set_search`` and then the SAME ops
    ``masked_solve`` runs (``jax_active_set.py:123-154``: identity-masked
    ``Q``, zeroed ``q``, ``jnp.linalg.cholesky``, ``cho_solve``) inline, so the
    returned ``x`` is the library's, bit for bit (pinned by the unit tests),
    and it additionally stashes ``(Q_pc, L, fixed)`` in the context.

(b) ``AbstractInversion.log_det_curvature_reg_matrix_term`` (a property) ->
    :func:`patched_log_det`. It evaluates ``self.reconstruction`` first (as the
    library's NumPy fast path does), reads the stash, and returns
    ``2 sum log diag L + log det S + sum log diag M`` where S is the Schur
    complement over Z (:func:`schur_log_det_from`):

    - Z is gathered into a STATIC ``k_slots`` slot array with
      ``jnp.nonzero(size=k_slots)``, unused slots padded with the identity;
    - ``Y = L^-1 Q[solved, Z]`` (one triangular solve, the fixed rows zeroed so
      ``Y^T Y = Q_ZF Q_FF^-1 Q_FZ`` exactly), ``S = Q_ZZ - Y^T Y``, and a
      ``k_slots x k_slots`` Cholesky;
    - ``|Z| > k_slots`` takes the library's dense route
      (``self._log_det_symmetric_from(M)``) through ``lax.cond`` — a real branch
      under scalar ``jit``, which is phase B's adopted composition.

    The columns ``Q[:, Z]`` are re-derived from M with the library's own scaling
    formula rather than sliced out of the stashed ``Q_pc``, so the
    full-coverage (Delaunay) and edge-zeroed (rectangular) cases are ONE code
    path; ``Q_pc`` is kept in the stash for the shape check.

The edge-zeroed subset. When the solve covers only ``ids = solve_ids_to_keep``
(rectangular: the mesh's edge pixels are held at zero and never enter the
solve), the log det of the FULL ``curvature_reg_matrix_reduced`` is still
wanted. The complement of ``ids`` joins Z, and its static size ``n_edge``
(``n_total - len(ids)``, a Python int even under ``jit`` because
``zeroed_ids_to_keep`` uses ``nonzero(size=n_keep)``) is added to ``k_max``:
``k_slots = k_max + n_edge``. Whether that stays small is recorded
(:func:`candidate_provenance`); the note decides whether the rectangular row is
a lever.

What is delegated to the library unchanged
------------------------------------------

Every other call of the log det goes to the ORIGINAL property, untouched, and
is counted under ``counts["delegated"]`` with the reason in
``counts["delegated_reasons"]``:

- the NumPy backend (the eager ``FitImaging`` the cell builds; it has its own
  fnnls-factor fast path);
- ``Settings.log_det_method == "slogdet"`` (a different value by construction);
- no regularization, no positive-only solver, or an inversion with
  unregularized linear objects (``curvature_reg_matrix_reduced`` would then be
  a ``mapper_indices`` subset the solve was not handed);
- no stash (the solve was not the certified one, e.g. PDIP or an MGE system),
  a stash already claimed by another inversion in the same trace, or a stashed
  factor whose size does not match the solved system.

A second access of the property by the SAME inversion in the same trace returns
the value computed on the first (``counts["jax_cached"]``), so the candidate
never falls back to a second dense factorisation.

The candidates
--------------

``control``
    No patch at all — the library's dense ``_log_det_symmetric_from`` as
    shipped. The context still yields a counts dict so the cell's code path is
    identical for every row.
``schur_k32`` / ``schur_k64`` / ``schur_k256``
    The Schur-complement log det with ``k_max`` = 32 / 64 / 256 slots (plus
    ``n_edge`` on an edge-zeroed system). ``schur_k256`` is a pre-registered
    AMENDMENT added after the RTX screen found |Z| = 4 on the fiducial but
    51-250 on the draws (k32 / k64 overflow to the dense branch on 8 / 7 of 9
    rows), before any A100 data.

JAX is imported inside functions, never at module level, like every
``likelihood_breakdown`` module that the CPU-only unit tests import.
"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CANDIDATES",
    "LIBRARY_LOG_DET_DOTTED",
    "LIBRARY_SOLVE_CERTIFIED_DOTTED",
    "LOGDET_GATE_NOTE",
    "LOGDET_LEVER_MS",
    "Candidate",
    "candidate_provenance",
    "logdet_gate_rows",
    "logdet_reuse_injected",
    "schur_log_det_from",
    "solve_certified_stashing",
]

#: The dotted names this module rebinds, recorded verbatim into the results JSON.
LIBRARY_SOLVE_CERTIFIED_DOTTED = "autoarray.util.jax_active_set.solve_certified"
LIBRARY_LOG_DET_DOTTED = (
    "autoarray.inversion.inversion.abstract.AbstractInversion.log_det_curvature_reg_matrix_term"
)

#: Pre-registered lever threshold (#303): the whole-call saving at the pin, measured
#: in the SAME process as the unmodified-library control, that makes a candidate a
#: lever worth a PyAutoArray prompt.
LOGDET_LEVER_MS = 0.5


@dataclass(frozen=True)
class Candidate:
    """One row of the phase-4 grid."""

    name: str
    label: str
    #: ``control`` | ``lever``.
    kind: str
    #: Schur slot budget before ``n_edge`` is added; ``None`` for the control.
    k_max: int | None
    library_change_if_wins: str


_LIBRARY_CHANGE = (
    "jax_active_set.solve_certified: return (or publish through an out-dict) the final "
    "masked Cholesky factor L and the fixed set; AbstractInversion.reconstruction: store "
    "them on the inversion (as _nnls_factor already is on the NumPy path); "
    "log_det_curvature_reg_matrix_term: a JAX fast path beside the NumPy one at "
    "abstract.py:1012 — 2 sum log diag L + a static-size Schur complement over the fixed "
    "(and edge) set, lax.cond to the dense route when it overflows"
)

#: Insertion order is the submit array's order (task 0 = control).
CANDIDATES: dict[str, Candidate] = {
    "control": Candidate(
        name="control",
        label="library dense Cholesky of F + lambda*H as shipped (no patch)",
        kind="control",
        k_max=None,
        library_change_if_wins="none — this is the library",
    ),
    "schur_k32": Candidate(
        name="schur_k32",
        label="certified-solve factor + Schur complement, k_max 32 (+ n_edge)",
        kind="lever",
        k_max=32,
        library_change_if_wins=_LIBRARY_CHANGE,
    ),
    "schur_k64": Candidate(
        name="schur_k64",
        label="certified-solve factor + Schur complement, k_max 64 (+ n_edge)",
        kind="lever",
        k_max=64,
        library_change_if_wins=_LIBRARY_CHANGE,
    ),
    # AMENDMENT (pre-registered before any A100 data): the RTX screen measured |Z| = 4
    # on the fiducial but 51-250 on the 8 draws, so k32 / k64 take the dense branch on
    # 8 / 7 of 9 rows. k_max 256 covers every screened row; its extra cost is a
    # 256-RHS triangular solve and a 256x256 Cholesky.
    "schur_k256": Candidate(
        name="schur_k256",
        label="certified-solve factor + Schur complement, k_max 256 (+ n_edge)",
        kind="lever",
        k_max=256,
        library_change_if_wins=_LIBRARY_CHANGE,
    ),
}


# ---------------------------------------------------------------------------
# The trace-scoped state
# ---------------------------------------------------------------------------


@dataclass
class _State:
    candidate: str
    k_max: int
    counts: dict
    #: The most recent certified solve seen in the trace: ``{"seq", "Q_pc", "L",
    #: "fixed", "claimed_by"}``. Only the latest is kept — an inversion's solve is a
    #: ``cached_property`` and the log det evaluates it first, so the latest solve
    #: when the log det runs is that inversion's own.
    latest: dict | None = None
    seq: int = 0
    #: ``id(inversion) -> (seq, value)``: a second access returns the first value.
    cache: dict = field(default_factory=dict)
    #: Per-log-det traced observables, for an UNTIMED reporting program to return.
    observed: list = field(default_factory=list)


#: The active context's state, or ``None`` outside one. Module-level so the two
#: rebinds (a plain function and a property getter) can share it.
_ACTIVE: _State | None = None


def _is_jax(xp) -> bool:
    return getattr(xp, "__name__", "").startswith("jax")


def _delegate(state: _State, reason: str) -> None:
    state.counts["delegated"] += 1
    state.counts["delegated_reasons"][reason] = state.counts["delegated_reasons"].get(reason, 0) + 1


# ---------------------------------------------------------------------------
# (a) the solver: the library's solve, keeping the factor
# ---------------------------------------------------------------------------


def solve_certified_stashing(Q, q, permanent=None, pass_budget=16, tau_rel=1.0e-9):
    """``jax_active_set.solve_certified``, bit for bit, keeping its final factor.

    Runs the library's ``active_set_search`` and then the SAME ops as
    ``jax_active_set.masked_solve`` inline (so the Cholesky factor ``L`` is
    reachable), and stashes ``(Q_pc, L, fixed)`` for :func:`patched_log_det`.
    Every op below is written inside THIS function on purpose: the stage map
    attributes it to ``certified_active_set_solve`` by function name.
    """
    import jax.numpy as jnp
    from autoarray.util import jax_active_set
    from jax.scipy.linalg import cho_solve

    Q, q = jax_active_set._as_float64(Q, q)

    fixed, certified, passes = jax_active_set.active_set_search(
        Q, q, permanent=permanent, pass_budget=pass_budget, tau_rel=tau_rel
    )

    # == jax_active_set.masked_solve(Q, q, fixed), op for op.
    keep = ~fixed
    Q_masked = jnp.where(keep[:, None] & keep[None, :], Q, 0.0)
    Q_masked = Q_masked + jnp.diag(jnp.where(fixed, 1.0, 0.0).astype(Q.dtype))
    q_masked = jnp.where(fixed, 0.0, q)
    L = jnp.linalg.cholesky(Q_masked)
    x = cho_solve((L, True), q_masked)

    state = _ACTIVE
    if state is not None:
        state.seq += 1
        state.latest = {"seq": state.seq, "Q_pc": Q, "L": L, "fixed": fixed, "claimed_by": None}
        state.counts["solves_stashed"] += 1

    return x, certified, passes


# ---------------------------------------------------------------------------
# (b) the log det: the block-determinant identity on the stashed factor
# ---------------------------------------------------------------------------


def z_mask_from(fixed, ids, n_total: int):
    """The full-space Z: the solve's fixed set, plus every index the solve never saw."""
    import jax.numpy as jnp

    if ids is None:
        return fixed
    return jnp.ones((n_total,), dtype=bool).at[ids].set(fixed)


def schur_log_det_from(M, L, fixed, ids, k_slots: int, jacobi: bool):
    """``log det M`` from the solve's factor ``L`` plus a Schur complement over Z.

    Parameters
    ----------
    M
        The ``(n_total, n_total)`` SPD matrix whose log det is wanted
        (``curvature_reg_matrix_reduced``).
    L
        Lower Cholesky factor of the solve's identity-masked matrix, over the
        ``n_s`` solved indices: ``Q_pc`` with the fixed rows / columns replaced
        by the identity, ``Q_pc = D M[ids][:, ids] D`` (``D = 1`` without Jacobi).
    fixed
        Boolean ``(n_s,)`` fixed set the factor was built on.
    ids
        The ``(n_s,)`` global indices the solve covered, or ``None`` for all.
    k_slots
        Static slot count for Z; the caller guarantees ``|Z| <= k_slots``
        (otherwise it takes the dense route).
    jacobi
        Whether the solver saw the Jacobi-scaled ``D M D``.

    Returns
    -------
    ``log det M`` (fp64 scalar).
    """
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    n_total = M.shape[0]
    z = z_mask_from(fixed, ids, n_total)
    count = jnp.sum(z)

    diag_M = jnp.diag(M)
    D = 1.0 / jnp.sqrt(diag_M) if jacobi else jnp.ones_like(diag_M)

    zidx = jnp.nonzero(z, size=k_slots, fill_value=0)[0]
    valid = jnp.arange(k_slots) < count

    # Q[solved, Z] in the scaled space, with the library's own formula
    # (M * D[:, None]) * D[None, :] applied to the k gathered columns only.
    cols = M[:, zidx]
    if ids is not None:
        cols = cols[ids]
        D_rows = D[ids]
    else:
        D_rows = D
    C = (cols * D_rows[:, None]) * D[zidx][None, :]
    # Zeroing the FIXED rows makes Y = L^-1 C exactly [L_FF^-1 Q_FZ ; 0]: L is
    # block-structured (identity on the fixed rows), so Y^T Y = Q_ZF Q_FF^-1 Q_FZ.
    C = jnp.where((~fixed)[:, None] & valid[None, :], C, 0.0)
    Y = solve_triangular(L, C, lower=True)

    S = (M[zidx][:, zidx] * D[zidx][:, None]) * D[zidx][None, :] - Y.T @ Y
    S = jnp.where(valid[:, None] & valid[None, :], S, 0.0) + jnp.diag(
        jnp.where(valid, 0.0, 1.0).astype(M.dtype)
    )

    log_det_FF = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    log_det_S = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(S))))
    # log det M = log det(D M D) - 2 sum log D = log det(D M D) + sum log diag M.
    jacobi_term = jnp.sum(jnp.log(diag_M)) if jacobi else 0.0
    return log_det_FF + log_det_S + jacobi_term


def _use_jacobi() -> bool:
    """The same config read ``reconstruction_positive_only_from`` makes (with its default)."""
    from autonerves import conf

    try:
        return bool(conf.instance["general"]["inversion"]["nnls_jacobi_preconditioning"])
    except KeyError:
        return True


def patched_log_det(self):
    """The rebound ``log_det_curvature_reg_matrix_term`` getter (see the module docstring)."""
    import jax
    import jax.numpy as jnp
    from autoarray.inversion.regularization.abstract import AbstractRegularization

    state = _ACTIVE
    original = _ORIGINAL_PROPERTY["fget"]
    if state is None:  # pragma: no cover — the property is only rebound inside a context
        return original(self)

    if not _is_jax(self._xp):
        _delegate(state, "numpy")
        return original(self)
    if not self.has(cls=AbstractRegularization):
        _delegate(state, "no_regularization")
        return original(self)
    if self.settings.log_det_method != "cholesky":
        _delegate(state, "log_det_method")
        return original(self)
    if not self.settings.use_positive_only_solver or not self.all_linear_obj_have_regularization:
        _delegate(state, "not_covered")
        return original(self)

    # The factor comes from the solve (a cached_property); evaluate it first, as the
    # library's NumPy fast path does.
    _ = self.reconstruction

    cached = state.cache.get(id(self))
    if cached is not None and state.latest is not None and cached[0] == state.latest["seq"]:
        state.counts["jax_cached"] += 1
        return cached[1]

    entry = state.latest
    if entry is None:
        _delegate(state, "no_stash")
        return original(self)
    if entry["claimed_by"] is not None and entry["claimed_by"] != id(self):
        _delegate(state, "stash_claimed_by_another_inversion")
        return original(self)

    M = self.curvature_reg_matrix_reduced
    ids = self.solve_ids_to_keep
    n_total = int(M.shape[0])
    n_s = n_total if ids is None else int(ids.shape[0])
    if tuple(entry["L"].shape) != (n_s, n_s):
        _delegate(state, "shape_mismatch")
        return original(self)

    entry["claimed_by"] = id(self)
    state.counts["jax"] += 1

    n_edge = n_total - n_s
    k_slots = int(state.k_max) + n_edge
    jacobi = _use_jacobi()
    L, fixed = entry["L"], entry["fixed"]

    count = jnp.sum(z_mask_from(fixed, ids, n_total))
    fits = count <= k_slots

    value = jax.lax.cond(
        fits,
        lambda: schur_log_det_from(M, L, fixed, ids, k_slots, jacobi),
        lambda: jnp.asarray(self._log_det_symmetric_from(M), dtype=M.dtype),
    )

    state.observed.append(
        {
            "z_count": count,
            "overflow": ~fits,
            "k_slots": k_slots,
            "n_edge": n_edge,
            "n_total": n_total,
            "n_solved": n_s,
            "jacobi": jacobi,
        }
    )
    state.cache[id(self)] = (entry["seq"], value)
    return value


#: Filled while a context is active: the library's original getter.
_ORIGINAL_PROPERTY: dict = {}


def _assert_signature_covers(original, patched, dotted: str) -> None:
    """Fail at injection time if the library's function has grown a parameter."""
    _kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    library_params = {
        name for name, p in inspect.signature(original).parameters.items() if p.kind in _kinds
    }
    wrapper_params = set(inspect.signature(patched).parameters)
    missing = sorted(library_params - wrapper_params)
    if missing:
        raise TypeError(
            f"{dotted} takes {missing}, which this harness wrapper does not accept. The library "
            f"has changed under the injection: add the parameter to the wrapper and forward it, "
            f"or the injected route is a different call from the library's own. (Do NOT change "
            f"the library — this module exists so nothing in PyAutoArray has to move.)"
        )


def _new_counts() -> dict:
    return {
        "jax": 0,
        "jax_cached": 0,
        "delegated": 0,
        "delegated_reasons": {},
        "solves_stashed": 0,
    }


@contextlib.contextmanager
def logdet_reuse_injected(candidate: str):
    """Rebind the certified solve and the curvature log det to *candidate*.

    Scoped: both originals are restored on exit, including on an exception.
    ``control`` installs no patch at all. Yields a mutable counts dict
    (``jax``, ``jax_cached``, ``delegated``, ``delegated_reasons``,
    ``solves_stashed``, and ``observed`` — the per-log-det traced observables
    ``z_count`` / ``overflow`` an UNTIMED reporting program may return as
    outputs) so the caller can assert the candidate was actually taken.

    The patch acts at TRACE time: enter the context around a FRESH ``jax.jit``
    of a NEW function object (one lowering per context) — jax caches traces by
    function identity, so re-jitting a function traced outside the context
    hands back the unpatched program.
    """
    global _ACTIVE

    if candidate not in CANDIDATES:
        raise ValueError(
            f"unknown log-det candidate {candidate!r}; expected one of {list(CANDIDATES)}"
        )

    counts = _new_counts()
    if candidate == "control":
        counts["observed"] = []
        yield counts
        return

    if _ACTIVE is not None:
        raise RuntimeError("logdet_reuse_injected does not nest")

    from autoarray.inversion.inversion.abstract import AbstractInversion
    from autoarray.util import jax_active_set

    original_solve = jax_active_set.solve_certified
    original_property = AbstractInversion.__dict__["log_det_curvature_reg_matrix_term"]
    if not isinstance(original_property, property):
        raise TypeError(
            f"{LIBRARY_LOG_DET_DOTTED} is no longer a plain property "
            f"({type(original_property).__name__}); the rebind would not take"
        )
    _assert_signature_covers(
        original_solve, solve_certified_stashing, LIBRARY_SOLVE_CERTIFIED_DOTTED
    )

    state = _State(candidate=candidate, k_max=int(CANDIDATES[candidate].k_max), counts=counts)
    counts["observed"] = state.observed
    _ORIGINAL_PROPERTY["fget"] = original_property.fget
    _ACTIVE = state
    jax_active_set.solve_certified = solve_certified_stashing
    AbstractInversion.log_det_curvature_reg_matrix_term = property(patched_log_det)
    try:
        yield counts
    finally:
        jax_active_set.solve_certified = original_solve
        AbstractInversion.log_det_curvature_reg_matrix_term = original_property
        _ACTIVE = None
        _ORIGINAL_PROPERTY.clear()


def candidate_provenance(candidate: str, n_total: int | None = None, n_solved: int | None = None):
    """What *candidate* computes, as JSON-ready data (the slot budget it actually used)."""
    spec = CANDIDATES[candidate]
    n_edge = None if n_total is None or n_solved is None else int(n_total) - int(n_solved)
    k_slots = None if spec.k_max is None or n_edge is None else int(spec.k_max) + n_edge
    return {
        "name": spec.name,
        "label": spec.label,
        "kind": spec.kind,
        "k_max": spec.k_max,
        "n_total": n_total,
        "n_solved": n_solved,
        "n_edge": n_edge,
        "k_slots": k_slots,
        "library_change_if_wins": spec.library_change_if_wins,
        "patched": (
            None
            if candidate == "control"
            else [LIBRARY_SOLVE_CERTIFIED_DOTTED, LIBRARY_LOG_DET_DOTTED]
        ),
        "identity": (
            "log det M = 2 sum log diag L + log det S + sum log diag M, S the Schur complement "
            "of the solved free block over Z = fixed set U unsolved (edge) indices, in the "
            "Jacobi-scaled space D M D with D = 1/sqrt(diag M) (inversion_util.py:433-436); "
            "exact for ANY split, so a PDIP-fallback lane is covered"
            if candidate != "control"
            else "the library's dense 2 sum log diag cholesky(M)"
        ),
    }


#: The pre-registered gate, verbatim into the results JSON (``gate.note``).
LOGDET_GATE_NOTE = (
    "Pre-registered before any A100 data (#303): per task, the candidate route (library "
    "certified solver via al.Settings, PDIP fallback, packaged budget, scalar jit) is pinned "
    "against the UNMODIFIED library route (the same Settings and composition, compiled "
    "WITHOUT the log-det rebind, in the same process) at 1e-9 relative on the fiducial and "
    "on 8 seeded draws (flds.random_draws(draw_seed, 8)); every row is GATED, control "
    "included (control's reference is a second compilation of its own program). Route b "
    "(library PDIP) is recorded, not gated: phase 2/3 placed a 2.0-2.6e-9 certified-vs-PDIP "
    "residual in the SOLVER path on the A100, which this phase does not touch. Also gated: "
    "|reconciliation_pct| <= 5, unjoined_ms == 0, and a non-control candidate must be taken "
    "on the JAX path (calls >= 1) in every compiled program. The JSON is written before the "
    "gate raises. Lever threshold (never a gate): >= 0.5 ms whole-call saving vs the "
    "unmodified library route measured in the SAME task (phase B saw ~3 % node-to-node "
    "differences, so rows are compared within a task only)."
)


def logdet_gate_rows(
    fiducial_pin: dict, draw_rows: list[dict], rtol: float
) -> tuple[list[dict], list[str]]:
    """Apply the pre-registered phase-4 pin gate; return ``(gated_rows, failed)``.

    Every row (the fiducial pin and each draw) is gated on ``rel_diff`` — the
    candidate route against the unmodified library route in the same
    composition — and annotated IN PLACE with ``gated``, ``gated_on`` and
    ``status``. ``rel_diff_vs_b_pdip`` (when present) is recorded as
    ``within_rtol_vs_b_pdip`` and never gated.
    """
    gated: list[dict] = []
    failed: list[str] = []
    rows = [(fiducial_pin, fiducial_pin.get("pin") or "fiducial")] + [
        (row, f"draw {row['draw']} ({row.get('draw_name', '?')})") for row in draw_rows
    ]
    for row, label in rows:
        ok = bool(row["rel_diff"] <= rtol)
        row.update(gated=True, gated_on="rel_diff", status="PASS" if ok else "FAIL")
        if "rel_diff_vs_b_pdip" in row:
            row["within_rtol_vs_b_pdip"] = bool(row["rel_diff_vs_b_pdip"] <= rtol)
        gated.append(row)
        if not ok:
            failed.append(label)
    return gated, failed


def dense_log_det_np(matrix) -> float:
    """NumPy reference: ``2 sum log diag cholesky(matrix)`` (for tests and the report)."""
    return float(2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(np.asarray(matrix))))))
