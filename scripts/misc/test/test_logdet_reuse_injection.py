"""Unit tests for the phase-4 log-det reuse candidates (``logdet_reuse_injection.py``, #303).

**CPU JAX in fp64, no imaging dataset, no GPU.** The Schur-complement log det
must be the dense log det of the SAME matrix — ``2 sum log diag cholesky(M)`` in
NumPy is the ground truth — on random SPD matrices with random fixed sets,
including an empty Z, a Z that exactly fills the slots, the Jacobi-scaled and
unscaled systems, and an edge-zeroed subset whose complement joins Z.

What a wrong candidate would look like, and what catches it: a Schur complement
built without zeroing the fixed rows of ``Q[:, Z]`` double-counts the identity
block — the random-Z test; a missing Jacobi term is off by ``sum log diag M``
nats — the scaled/unscaled pair; a solver wrapper that is not the library's
solve changes ``x`` — the bitwise test; a rebind that never takes, or never
comes off — the rebind-site and restore tests; an overflow that silently uses a
truncated Z — the forced-overflow test.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

aa = pytest.importorskip("autoarray")


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import logdet_reuse_injection as lri  # noqa: E402

LEVERS = [n for n, c in lri.CANDIDATES.items() if c.kind == "lever"]


def _spd(n: int, seed: int, scale_spread: float = 3.0) -> np.ndarray:
    """A random SPD matrix with a spread of diagonal scales (so Jacobi matters)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    M = A @ A.T + n * np.eye(n)
    s = np.exp(rng.uniform(-scale_spread, scale_spread, size=n))
    return (M * s[:, None]) * s[None, :]


def _factor_of(M: np.ndarray, fixed: np.ndarray, ids, jacobi: bool):
    """The factor the solver would keep: chol of the identity-masked (scaled) solved block."""
    Ms = M if ids is None else M[np.ix_(ids, ids)]
    D = 1.0 / np.sqrt(np.diag(Ms)) if jacobi else np.ones(Ms.shape[0])
    Q = (Ms * D[:, None]) * D[None, :]
    keep = ~fixed
    Qm = np.where(keep[:, None] & keep[None, :], Q, 0.0) + np.diag(np.where(fixed, 1.0, 0.0))
    return np.linalg.cholesky(Qm)


def _schur(M, fixed, ids, k_slots, jacobi):
    L = _factor_of(M, fixed, ids, jacobi)
    return float(
        jax.jit(
            lambda M_, L_, f_: lri.schur_log_det_from(
                M_, L_, f_, None if ids is None else jnp.asarray(ids), k_slots, jacobi
            )
        )(jnp.asarray(M), jnp.asarray(L), jnp.asarray(fixed))
    )


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("jacobi", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test__schur_log_det_is_the_dense_log_det_for_a_random_fixed_set(seed, jacobi):
    n = 48
    M = _spd(n, seed)
    rng = np.random.default_rng(100 + seed)
    fixed = rng.random(n) < 0.2
    got = _schur(M, fixed, None, k_slots=32, jacobi=jacobi)
    ref = lri.dense_log_det_np(M)
    assert abs(got - ref) <= 1e-10 * abs(ref)


@pytest.mark.parametrize("jacobi", [True, False])
def test__an_empty_z_is_the_factor_alone(jacobi):
    M = _spd(30, 7)
    fixed = np.zeros(30, dtype=bool)
    got = _schur(M, fixed, None, k_slots=8, jacobi=jacobi)
    assert abs(got - lri.dense_log_det_np(M)) <= 1e-10 * abs(lri.dense_log_det_np(M))


def test__a_z_that_exactly_fills_the_slots_is_exact():
    n, k = 40, 12
    M = _spd(n, 11)
    fixed = np.zeros(n, dtype=bool)
    fixed[np.random.default_rng(3).choice(n, size=k, replace=False)] = True
    got = _schur(M, fixed, None, k_slots=k, jacobi=True)
    ref = lri.dense_log_det_np(M)
    assert abs(got - ref) <= 1e-10 * abs(ref)


def test__everything_fixed_is_the_schur_complement_alone():
    n = 16
    M = _spd(n, 5)
    fixed = np.ones(n, dtype=bool)
    got = _schur(M, fixed, None, k_slots=n, jacobi=True)
    assert abs(got - lri.dense_log_det_np(M)) <= 1e-10 * abs(lri.dense_log_det_np(M))


@pytest.mark.parametrize("jacobi", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test__an_edge_zeroed_subset_joins_the_complement_to_z(seed, jacobi):
    """Rectangular: the solve covers ``ids``; the unsolved edge indices are in Z too."""
    n = 50
    M = _spd(n, 20 + seed)
    rng = np.random.default_rng(40 + seed)
    edge = np.sort(rng.choice(n, size=9, replace=False))
    ids = np.setdiff1d(np.arange(n), edge)
    fixed = rng.random(ids.size) < 0.15
    got = _schur(M, fixed, ids, k_slots=int(fixed.sum()) + edge.size + 3, jacobi=jacobi)
    ref = lri.dense_log_det_np(M)
    assert abs(got - ref) <= 1e-10 * abs(ref)


def test__the_z_mask_marks_fixed_and_unsolved_indices():
    fixed = jnp.asarray([False, True, False])
    ids = jnp.asarray([0, 2, 4])
    z = np.asarray(lri.z_mask_from(fixed, ids, 5))
    assert z.tolist() == [False, True, True, True, False]


# ---------------------------------------------------------------------------
# The solver wrapper is the library's solve
# ---------------------------------------------------------------------------


def _qp(n: int, seed: int):
    M = _spd(n, seed, scale_spread=1.0)
    D = 1.0 / np.sqrt(np.diag(M))
    Q = (M * D[:, None]) * D[None, :]
    q = np.random.default_rng(seed + 1).normal(size=n)  # ~half negative: a real active set
    return jnp.asarray(Q), jnp.asarray(q)


@pytest.mark.parametrize("seed", range(4))
def test__the_stashing_solve_returns_the_librarys_x_bit_for_bit(seed):
    from autoarray.util import jax_active_set

    Q, q = _qp(60, seed)
    x_lib, c_lib, p_lib = jax.jit(jax_active_set.solve_certified)(Q, q)
    x_st, c_st, p_st = jax.jit(lri.solve_certified_stashing)(Q, q)
    assert np.array_equal(np.asarray(x_lib), np.asarray(x_st))
    assert bool(c_lib) == bool(c_st) and int(p_lib) == int(p_st)
    assert int(np.sum(np.asarray(x_st) == 0.0)) > 0, "fixture has no active set"


def test__the_wrapper_covers_the_library_signature():
    import inspect

    from autoarray.util import jax_active_set

    lib = inspect.signature(jax_active_set.solve_certified).parameters
    ours = inspect.signature(lri.solve_certified_stashing).parameters
    assert list(lib) == list(ours)
    for name in lib:
        assert lib[name].default == ours[name].default


# ---------------------------------------------------------------------------
# The rebind: where it takes, and that it comes off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LEVERS)
def test__the_library_dispatch_reaches_the_stashing_solve(name):
    """``reconstruction_positive_only_from(solver="certified", xp=jnp)`` goes through
    ``solve_certified_with_fallback``, which looks ``solve_certified`` up as a module
    global — so the module-attribute rebind is the one that takes."""
    from autoarray.inversion.inversion import inversion_util

    M = _spd(40, 9, scale_spread=1.0)
    q = np.random.default_rng(10).normal(size=40)

    def make_solve():
        # A NEW function object per jit: jax caches traces by function identity, so a
        # second jax.jit of the SAME function would hand back the pre-context trace.
        def solve(M_, q_):
            return inversion_util.reconstruction_positive_only_from(
                data_vector=q_, curvature_reg_matrix=M_, xp=jnp, solver="certified"
            )

        return solve

    x_ref = jax.jit(make_solve())(jnp.asarray(M), jnp.asarray(q))
    with lri.logdet_reuse_injected(name) as counts:
        x_inj = jax.jit(make_solve())(jnp.asarray(M), jnp.asarray(q))
    assert counts["solves_stashed"] == 1
    assert np.array_equal(np.asarray(x_ref), np.asarray(x_inj))


@pytest.mark.parametrize("name", list(lri.CANDIDATES))
def test__the_library_entry_points_are_restored_after_the_context(name):
    from autoarray.inversion.inversion.abstract import AbstractInversion
    from autoarray.util import jax_active_set

    solve = jax_active_set.solve_certified
    prop = AbstractInversion.__dict__["log_det_curvature_reg_matrix_term"]
    with pytest.raises(RuntimeError, match="boom"):
        with lri.logdet_reuse_injected(name):
            if name != "control":
                assert jax_active_set.solve_certified is lri.solve_certified_stashing
                assert (
                    AbstractInversion.__dict__["log_det_curvature_reg_matrix_term"].fget
                    is lri.patched_log_det
                )
            raise RuntimeError("boom")
    assert jax_active_set.solve_certified is solve
    assert AbstractInversion.__dict__["log_det_curvature_reg_matrix_term"] is prop
    assert lri._ACTIVE is None


def test__an_unknown_candidate_is_refused():
    with pytest.raises(ValueError, match="unknown log-det candidate"):
        with lri.logdet_reuse_injected("schur_k8"):
            pass


# ---------------------------------------------------------------------------
# The rebound property on a minimal inversion stand-in
# ---------------------------------------------------------------------------


class _Settings:
    def __init__(self, log_det_method="cholesky", use_positive_only_solver=True):
        self.log_det_method = log_det_method
        self.use_positive_only_solver = use_positive_only_solver


class _FakeInversion:
    """Exactly the attributes ``patched_log_det`` reads, with the library's semantics:
    ``reconstruction`` runs the (rebound) certified solve on the Jacobi-scaled
    solved block, once; ``_log_det_symmetric_from`` is the dense route."""

    def __init__(self, M, q, ids=None, xp=jnp, settings=None, regularized=True):
        self._M = M
        self._q = q
        self._ids = ids
        self._xp = xp
        self.settings = settings or _Settings()
        self._regularized = regularized
        self.all_linear_obj_have_regularization = True
        self.dense_calls = 0
        self._reconstruction = None

    def has(self, cls):
        return self._regularized

    @property
    def curvature_reg_matrix_reduced(self):
        return self._M

    @property
    def solve_ids_to_keep(self):
        return self._ids

    @property
    def reconstruction(self):
        if self._reconstruction is None:
            from autoarray.util import jax_active_set

            Ms = self._M if self._ids is None else self._M[self._ids][:, self._ids]
            qs = self._q if self._ids is None else self._q[self._ids]
            D = 1.0 / jnp.sqrt(jnp.diag(Ms))
            x, _, _ = jax_active_set.solve_certified((Ms * D[:, None]) * D[None, :], qs * D)
            self._reconstruction = x * D
        return self._reconstruction

    def _log_det_symmetric_from(self, matrix, sparse=False):
        self.dense_calls += 1
        return 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(matrix))))


def _run_log_det(name, make, k_max=None):
    """Trace one log det inside a fresh jit; return (value, observed, counts, inversion)."""
    holder = {}
    with lri.logdet_reuse_injected(name) as counts:
        if k_max is not None:
            lri._ACTIVE.k_max = k_max

        def fn(M, q):
            inv = make(M, q)
            holder["inv"] = inv
            value = lri.patched_log_det(inv)
            obs = counts["observed"][-1] if counts["observed"] else None
            return value, (obs["z_count"], obs["overflow"]) if obs else (0, False)

        M = jnp.asarray(_spd(48, 31, scale_spread=1.0))
        q = jnp.asarray(np.random.default_rng(32).normal(size=48))
        value, (z_count, overflow) = jax.jit(fn)(M, q)
    return float(value), int(z_count), bool(overflow), counts, holder["inv"], np.asarray(M)


@pytest.mark.parametrize("name", LEVERS)
def test__the_rebound_property_is_the_dense_log_det(name):
    value, z_count, overflow, counts, inv, M = _run_log_det(name, _FakeInversion)
    assert counts["jax"] == 1 and counts["delegated"] == 0
    assert counts["solves_stashed"] == 1
    assert 0 < z_count <= lri.CANDIDATES[name].k_max and not overflow
    # lax.cond traces BOTH branches once; the run-time branch is the overflow flag
    assert inv.dense_calls == 1
    ref = lri.dense_log_det_np(M)
    assert abs(value - ref) <= 1e-10 * abs(ref)


@pytest.mark.parametrize("name", LEVERS)
def test__a_forced_overflow_takes_the_dense_route_and_is_still_exact(name):
    value, z_count, overflow, counts, inv, M = _run_log_det(name, _FakeInversion, k_max=1)
    assert z_count > 1 and overflow
    # the overflow flag selected the dense (library) branch of the cond
    assert inv.dense_calls == 1
    ref = lri.dense_log_det_np(M)
    assert abs(value - ref) <= 1e-10 * abs(ref)


@pytest.mark.parametrize("name", LEVERS)
def test__an_edge_zeroed_inversion_counts_the_edges_into_the_slots(name):
    edge = np.array([0, 5, 17, 47])
    ids = jnp.asarray(np.setdiff1d(np.arange(48), edge))

    def make(M, q):
        return _FakeInversion(M, q, ids=ids)

    value, z_count, overflow, counts, inv, M = _run_log_det(name, make)
    obs = counts["observed"][-1]
    assert obs["n_edge"] == 4 and obs["k_slots"] == lri.CANDIDATES[name].k_max + 4
    assert z_count >= 4 and not overflow
    ref = lri.dense_log_det_np(M)
    assert abs(value - ref) <= 1e-10 * abs(ref)


@pytest.mark.parametrize("name", LEVERS)
def test__delegation_is_counted_with_its_reason(name):
    M = _spd(20, 2, scale_spread=1.0)
    q = np.random.default_rng(0).normal(size=20)
    Mj, qj = jnp.asarray(M), jnp.asarray(q)
    cases = (
        (_FakeInversion(M, q, xp=np), "numpy"),
        (_FakeInversion(Mj, qj, regularized=False), "no_regularization"),
        (_FakeInversion(Mj, qj, settings=_Settings(log_det_method="slogdet")), "log_det_method"),
        (_FakeInversion(Mj, qj, settings=_Settings(use_positive_only_solver=False)), "not_covered"),
    )
    with lri.logdet_reuse_injected(name) as counts:
        original = lri._ORIGINAL_PROPERTY["fget"]
        lri._ORIGINAL_PROPERTY["fget"] = lambda self: "delegated"
        try:
            for fake, reason in cases:
                assert lri.patched_log_det(fake) == "delegated"
                assert counts["delegated_reasons"][reason] == 1
            # no stash: a reconstruction exists but no certified solve ran in the context
            fake = _FakeInversion(Mj, qj)
            fake._reconstruction = jnp.zeros(20)
            assert lri.patched_log_det(fake) == "delegated"
            assert counts["delegated_reasons"]["no_stash"] == 1
        finally:
            lri._ORIGINAL_PROPERTY["fget"] = original
    assert counts["jax"] == 0 and counts["delegated"] == 5


@pytest.mark.parametrize("name", LEVERS)
def test__a_second_access_is_cached_and_another_inversion_is_delegated(name):
    M = jnp.asarray(_spd(30, 4, scale_spread=1.0))
    q = jnp.asarray(np.random.default_rng(4).normal(size=30))
    with lri.logdet_reuse_injected(name) as counts:
        inv = _FakeInversion(M, q)
        first = lri.patched_log_det(inv)
        second = lri.patched_log_det(inv)
        assert first is second
        assert counts["jax"] == 1 and counts["jax_cached"] == 1
        other = _FakeInversion(M, q)
        other._reconstruction = jnp.zeros(30)  # "solved" without a new stash
        original = lri._ORIGINAL_PROPERTY["fget"]
        lri._ORIGINAL_PROPERTY["fget"] = lambda self: "delegated"
        assert lri.patched_log_det(other) == "delegated"
        lri._ORIGINAL_PROPERTY["fget"] = original
        assert counts["delegated_reasons"]["stash_claimed_by_another_inversion"] == 1


def test__control_patches_nothing():
    from autoarray.util import jax_active_set

    solve = jax_active_set.solve_certified
    with lri.logdet_reuse_injected("control") as counts:
        assert jax_active_set.solve_certified is solve
        assert lri._ACTIVE is None
    assert counts["jax"] == 0 and counts["observed"] == []


# ---------------------------------------------------------------------------
# Registry, provenance and the pre-registered gate
# ---------------------------------------------------------------------------


def test__the_registry_is_control_then_the_two_slot_budgets():
    assert list(lri.CANDIDATES) == ["control", "schur_k32", "schur_k64"]
    assert lri.CANDIDATES["control"].k_max is None
    assert [lri.CANDIDATES[n].k_max for n in LEVERS] == [32, 64]
    for spec in lri.CANDIDATES.values():
        assert spec.library_change_if_wins


def test__the_cell_offers_exactly_the_registry():
    import ast

    tree = ast.parse(
        (ROOT / "scripts/imaging/likelihood_breakdown/fixed_light_trace.py").read_text()
    )
    choices = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--logdet-candidate"
        ):
            choices = next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "choices")
    assert choices is not None, "fixed_light_trace.py does not declare --logdet-candidate"
    assert list(choices) == list(lri.CANDIDATES)


def test__provenance_records_the_slot_budget_actually_used():
    rec = lri.candidate_provenance("schur_k32", n_total=1521, n_solved=1369)
    assert rec["n_edge"] == 152 and rec["k_slots"] == 184
    assert rec["patched"] == [lri.LIBRARY_SOLVE_CERTIFIED_DOTTED, lri.LIBRARY_LOG_DET_DOTTED]
    assert lri.candidate_provenance("control", 1500, 1500)["patched"] is None


def _pin(rel, **extra):
    return {"pin": "fiducial", "rel_diff": rel, **extra}


def test__gate__every_row_is_gated_on_the_unmodified_library_route():
    fid = _pin(1e-12, rel_diff_vs_b_pdip=3e-9)
    draws = [{"draw": k, "draw_name": f"d{k}", "rel_diff": 1e-11} for k in range(8)]
    gated, failed = lri.logdet_gate_rows(fid, draws, 1e-9)
    assert len(gated) == 9 and not failed
    assert all(r["gated"] and r["gated_on"] == "rel_diff" and r["status"] == "PASS" for r in gated)
    assert fid["within_rtol_vs_b_pdip"] is False  # recorded, never gated


def test__gate__a_moved_answer_fails_by_name():
    fid = _pin(1e-12)
    draws = [{"draw": 0, "draw_name": "a", "rel_diff": 2e-9}]
    _, failed = lri.logdet_gate_rows(fid, draws, 1e-9)
    assert failed == ["draw 0 (a)"]


def test__gate__the_note_records_the_pre_registration_and_the_lever():
    assert "Pre-registered" in lri.LOGDET_GATE_NOTE
    assert "1e-9" in lri.LOGDET_GATE_NOTE and "0.5 ms" in lri.LOGDET_GATE_NOTE
    assert lri.LOGDET_LEVER_MS == 0.5
