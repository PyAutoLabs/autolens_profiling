"""Numerical and red-control tests for the CPU permutation experiment (#276)."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT / "scripts" / "misc"), str(Path(__file__).parent)]
from likelihood_breakdown import fixed_light_numpy_solvers as solvers
from likelihood_breakdown.fixed_light_s4b_checks import compare, evaluate_fit
from test_fixed_light_numba import tiny as _tiny
from test_fixed_light_numba import tiny_s3_pair as _tiny_pair

tiny, tiny_s3_pair = _tiny, _tiny_pair


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("start", ("empty", "all", "boolean", "unordered"))
def test_permuted_matches_library_and_factor(seed, start):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(27, 19))
    matrix = a.T @ a + np.eye(19)
    rhs = rng.normal(size=19)
    initial = {
        "empty": np.array([], dtype=int),
        "all": np.ones(19, dtype=bool),
        "boolean": np.linalg.solve(matrix, rhs) > 0,
        "unordered": np.array([14, 3, 17, 2, 0, 8]),
    }[start]
    before = (matrix.copy(), rhs.copy(), initial.copy())
    reference_stats, stats, factor = {}, {}, {}
    expected = solvers.fnnls_cholesky(matrix, rhs, P_initial=initial, stats=reference_stats)
    actual = solvers.fnnls_cholesky_permuted(
        matrix, rhs, P_initial=initial, stats=stats, factor=factor
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert solvers.active_set_jaccard(stats["passive_set"], reference_stats["passive_set"])[
        "identical"
    ]
    assert solvers.log_det_factor_check(matrix, factor)["within_atol"]
    k = factor["k_active"]
    u = factor["U_buffer"][:k, :k]
    np.testing.assert_allclose(
        u.T @ u, matrix[np.ix_(factor["passive_set"], factor["passive_set"])], atol=1e-12
    )
    for original, current in zip(before, (matrix, rhs, initial), strict=True):
        np.testing.assert_array_equal(original, current)


def test_seam_restores_after_error_and_refuses_foreign_rebinding():
    from autoarray.util import fnnls

    original = fnnls.fnnls_cholesky
    with pytest.raises(ValueError, match="deliberate"):
        with solvers.fnnls_kernel_injected(original, label="error"):
            raise ValueError("deliberate")
    assert fnnls.fnnls_cholesky is original

    def foreign(*a, **kw):
        return None

    try:
        with pytest.raises(RuntimeError, match="rebound"):
            with solvers.fnnls_kernel_injected(original, label="foreign"):
                fnnls.fnnls_cholesky = foreign
        assert fnnls.fnnls_cholesky is foreign
    finally:
        fnnls.fnnls_cholesky = original


def test_evidence_gate_rejects_nonfinite():
    for value in (np.nan, np.inf, -np.inf):
        assert not solvers.evidence_delta(value, 10.0)["within_rtol"]
        assert not solvers.evidence_delta(10.0, value)["within_rtol"]


def test_factor_gate_rejects_missing_and_misordered_factor():
    matrix = np.array([[4.0, 1.0, 0.2], [1.0, 3.0, 0.1], [0.2, 0.1, 2.0]])
    factor = {}
    solvers.fnnls_cholesky(matrix, np.array([1.0, 2.0, -1.0]), factor=factor)
    assert not solvers.log_det_factor_check(matrix, {})["within_atol"]
    factor["passive_set"] = factor["passive_set"][::-1]
    assert not solvers.log_det_factor_check(matrix, factor)["within_atol"]


def test_production_fit_identity_candidate_and_red_controls(tiny_s3_pair, monkeypatch):
    import autolens as al
    from autoarray.inversion.inversion import nnls_memo

    monkeypatch.setenv("AUTOARRAY_NNLS_WARM_START", "1")
    _, system = tiny_s3_pair

    def make_fit():
        return al.FitImaging(
            dataset=system.dataset,
            tracer=system.source_only_tracer,
            adapt_images=system.fit.adapt_images,
            settings=system.fit.settings,
            xp=np,
        )

    outside = dict(nnls_memo._nnls_passive_set_memo)
    baseline, memo = evaluate_fit(make_fit, solvers.fnnls_cholesky, {})
    identity, _ = evaluate_fit(make_fit, solvers.fnnls_cholesky, {})
    assert compare(baseline, identity, identity=True)["verdict"] == "PASS"
    warm, _ = evaluate_fit(make_fit, solvers.fnnls_cholesky, memo)
    candidate, _ = evaluate_fit(make_fit, solvers.fnnls_cholesky_permuted, memo)
    assert warm["iterations"]["seed_source"] == "memo"
    assert compare(warm, candidate)["verdict"] == "PASS"
    assert nnls_memo._nnls_passive_set_memo.keys() == outside.keys()
    for key in outside:
        assert nnls_memo._nnls_passive_set_memo[key] is outside[key]

    def wrong_kernel(*args, **kwargs):
        x = solvers.fnnls_cholesky(*args, **kwargs)
        # A real bad solve must fail evidence, not merely a hand-edited report.
        return x + 1.0

    wrong, _ = evaluate_fit(make_fit, wrong_kernel, memo)
    assert not compare(warm, wrong)["gates"]["W3_evidence"]
    wrong_set = copy.deepcopy(candidate)
    wrong_set["passive_set"] = np.array([], dtype=int)
    assert not compare(warm, wrong_set)["gates"]["W2_active_set"]
    wrong_fallback = copy.deepcopy(candidate)
    wrong_fallback["kernel_errors"] = ["LinAlgError"]
    assert not compare(warm, wrong_fallback)["gates"]["W4_fallback"]


def test_factor_residual_rejects_determinant_preserving_wrong_order():
    matrix = np.diag([2.0, 3.0, 7.0])
    factor = {}
    solvers.fnnls_cholesky(matrix, np.ones(3), P_initial=np.ones(3, dtype=bool), factor=factor)
    factor["passive_set"] = np.array([2, 1, 0])
    result = solvers.log_det_factor_check(matrix, factor)
    assert abs(result["delta_nats"]) < 2e-12
    assert result["factor_relative_residual"] > result["factor_rtol"]
    assert not result["within_atol"]
