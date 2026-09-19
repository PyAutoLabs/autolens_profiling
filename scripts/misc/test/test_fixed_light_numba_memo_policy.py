"""Tests for the profiling-only NNLS memo precheck."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPERS = ROOT / "scripts" / "misc"
sys.path.insert(0, str(HELPERS))

from likelihood_breakdown import fixed_light_numba_memo_policy_steps as policy  # noqa: E402


def test_calibration_declaration_and_selection_are_locked_before_holdout():
    assert policy.CALIBRATION == {
        "seed": 278,
        "nearby_n": 8,
        "broad_n": 8,
        "walk_sigma": 0.05,
        "thresholds": [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 2.0],
        "repeats": 3,
        "selection": "minimum max(guarded broad / cold broad, guarded nearby / memo nearby), numerical passing candidates only; ties smaller threshold",
    }
    rows = [
        {"threshold": 0.1, "numerical_passed": True, "broad_ratio": 0.9, "nearby_ratio": 1.1},
        {"threshold": 0.01, "numerical_passed": True, "broad_ratio": 1.0, "nearby_ratio": 1.0},
        {"threshold": 0.0, "numerical_passed": False, "broad_ratio": 0.1, "nearby_ratio": 0.1},
    ]
    assert policy.select_threshold(rows) == 0.01
    assert policy.select_threshold([{**row, "holdout_ratio": 999.0} for row in rows]) == 0.01
    with pytest.raises(ValueError, match="no numerically passing"):
        policy.select_threshold([{**rows[0], "numerical_passed": False}])


def test_build_rows_are_deterministic_disjoint_and_nearby_starts_nonzero():
    calibration = policy.build_rows(seed=278, n=8, nearby=True)
    assert calibration == policy.build_rows(seed=278, n=8, nearby=True)
    holdout = policy.build_rows(seed=1, n=8, nearby=True)
    broad = policy.build_rows(seed=278, n=8, nearby=False)
    assert calibration[0]["offsets"] != {key: 0.0 for key in calibration[0]["offsets"]}
    assert {row["name"] for row in calibration}.isdisjoint(row["name"] for row in holdout)
    assert calibration != holdout
    assert calibration != broad
    assert all(row["clipping"] is False for row in calibration + broad)
    assert all(row["mass_cls"] == "Isothermal" for row in calibration + broad)


def test_actual_cell_smoke_import_uses_canonical_package_path():
    environment = dict(os.environ, AUTOLENS_PROFILING_SMOKE="1")
    result = subprocess.run(
        [sys.executable, "scripts/imaging/likelihood_breakdown/fixed_light_numba_memo_policy.py"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_residual_score_obeys_passive_and_inactive_kkt_signs():
    matrix = np.eye(3)
    vector = np.array([1.0, 2.0, -3.0])
    reconstruction = np.array([1.1, 2.0, 0.0])

    assert policy.residual_precheck_score(
        matrix, vector, reconstruction, np.array([0, 1])
    ) == pytest.approx(0.1 / 3.0)
    assert (
        policy.residual_precheck_score(
            np.zeros((2, 2)), np.zeros(2), np.zeros(2), np.array([], dtype=int)
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "matrix,vector,reconstruction,passive",
    [
        (np.eye(2), np.ones(3), np.ones(2), np.array([0])),
        (np.eye(2), np.ones(2), np.array([np.nan, 0.0]), np.array([0])),
        (np.eye(2), np.ones(2), np.ones(2), np.array([2])),
        (np.full((2, 2), 1.0e308), np.ones(2), np.full(2, 1.0e308), np.array([0])),
    ],
)
def test_residual_score_rejects_incompatible_or_nonfinite_inputs(
    matrix, vector, reconstruction, passive
):
    with pytest.raises(ValueError):
        policy.residual_precheck_score(matrix, vector, reconstruction, passive)


def test_policy_rejects_before_solve_then_refreshes_without_lookahead(monkeypatch):
    from autoarray import Settings
    from autoarray.inversion.inversion import inversion_util, nnls_memo
    from autoarray.util import fnnls

    matrix = np.eye(2)
    settings = Settings(nnls_warm_start_memo=True)
    original_kernel = fnnls.fnnls_cholesky
    seeds = []

    def spy(*args, **kwargs):
        seeds.append(np.asarray(kwargs["P_initial"]).copy())
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(fnnls, "fnnls_cholesky", spy)
    with policy.policy_scope(threshold=0.05, diagnostics=True) as state:
        first = inversion_util.reconstruction_positive_only_from(
            np.array([1.0, -1.0]), matrix, settings=settings, fingerprint="toy"
        )
        assert np.array_equal(first, [1.0, 0.0])
        second = inversion_util.reconstruction_positive_only_from(
            np.array([-1.0, 1.0]), matrix, settings=settings, fingerprint="toy"
        )
        assert seeds[-1].dtype == bool
        assert np.array_equal(second, [0.0, 1.0])
        assert state.records[0]["reason"] == "missing_entry"
        assert state.records[1]["reason"] == "score_above_threshold"
        assert state.records[1]["accepted"] is False
        assert state.rejected == 1
        key = nnls_memo.memo_key(n=2, fingerprint="toy")
        assert np.array_equal(state.reconstruction_cache[key], second)


def test_policy_accepts_same_fingerprint_and_records_only_when_requested():
    from autoarray import Settings
    from autoarray.inversion.inversion import inversion_util

    settings = Settings(nnls_warm_start_memo=True)
    with policy.policy_scope(threshold=0.01) as state:
        for _ in range(2):
            inversion_util.reconstruction_positive_only_from(
                np.array([1.0, -1.0]),
                np.eye(2),
                settings=settings,
                fingerprint="same",
            )
        assert state.accepted == 1
        assert state.records == []


def test_production_postguard_invalidation_also_drops_cached_reconstruction():
    from autoarray import Settings
    from autoarray.inversion.inversion import inversion_util, nnls_memo

    settings = Settings(nnls_warm_start_memo=True, nnls_warm_start_error_tolerance=1.0e-12)
    with policy.policy_scope(threshold=10.0, diagnostics=True) as state:
        inversion_util.reconstruction_positive_only_from(
            np.array([1.0, -1.0]), np.eye(2), settings=settings, fingerprint="guard"
        )
        inversion_util.reconstruction_positive_only_from(
            np.array([-1.0, 1.0]), np.eye(2), settings=settings, fingerprint="guard"
        )
        key = nnls_memo.memo_key(n=2, fingerprint="guard")
        assert state.records[-1]["accepted"] is True
        assert key not in nnls_memo._nnls_passive_set_memo
        assert key not in state.reconstruction_cache


def test_warm_exception_uses_production_cold_retry_and_retains_fresh_cache(monkeypatch):
    from autoarray import Settings
    from autoarray.inversion.inversion import inversion_util, nnls_memo
    from autoarray.util import fnnls

    original_kernel = fnnls.fnnls_cholesky
    attempts = []

    def fail_warm_once(*args, **kwargs):
        is_dense = np.asarray(kwargs["P_initial"]).dtype == bool
        attempts.append("dense" if is_dense else "memo")
        if not is_dense and attempts.count("memo") == 1:
            raise ValueError("synthetic stale seed")
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(fnnls, "fnnls_cholesky", fail_warm_once)
    settings = Settings(nnls_warm_start_memo=True)
    with policy.policy_scope(threshold=10.0, diagnostics=True) as state:
        for _ in range(2):
            inversion_util.reconstruction_positive_only_from(
                np.array([1.0, -1.0]),
                np.eye(2),
                settings=settings,
                fingerprint="retry",
            )
        key = nnls_memo.memo_key(n=2, fingerprint="retry")
        assert attempts == ["dense", "memo", "dense"]
        assert state.records[-1]["accepted"] is True
        assert key in state.reconstruction_cache


def test_prototype_cache_tracks_production_fifo_bound():
    from autoarray import Settings
    from autoarray.inversion.inversion import inversion_util, nnls_memo

    settings = Settings(nnls_warm_start_memo=True)
    with policy.policy_scope(threshold=1.0) as state:
        for index in range(nnls_memo._NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES + 1):
            inversion_util.reconstruction_positive_only_from(
                np.array([1.0, -1.0]),
                np.eye(2),
                settings=settings,
                fingerprint=f"bounded-{index}",
            )
        assert len(state.reconstruction_cache) == nnls_memo._NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES
        assert set(state.reconstruction_cache) == set(nnls_memo._nnls_passive_set_memo)


def test_counterfactual_replays_share_predecision_snapshot_and_restore_actual_state():
    from autoarray import Settings
    from autoarray.inversion.inversion import inversion_util, nnls_memo
    from autoarray.util import fnnls

    settings = Settings(nnls_warm_start_memo=True)
    attempts = []
    original_kernel = fnnls.fnnls_cholesky

    def spy(*args, **kwargs):
        attempts.append("dense" if np.asarray(kwargs["P_initial"]).dtype == bool else "memo")
        return original_kernel(*args, **kwargs)

    original_environment = os.environ.get("AUTOARRAY_NNLS_WARM_START")
    with policy.policy_scope(threshold=0.0, diagnostics=True) as state:
        inversion_util.reconstruction_positive_only_from(
            np.array([1.0, -1.0]), np.eye(2), settings=settings, fingerprint="paired"
        )
        assert state.had_memo_entry is False
        inversion_util.reconstruction_positive_only_from(
            np.array([-1.0, 1.0]), np.eye(2), settings=settings, fingerprint="paired"
        )
        assert state.had_memo_entry is True
        assert state.records[-1]["accepted"] is False

        guarded_outer = inversion_util.reconstruction_positive_only_from
        guarded_get = nnls_memo.passive_set_get
        guarded_memo = dict(nnls_memo._nnls_passive_set_memo)
        guarded_records = list(state.records)
        guarded_counts = (state.accepted, state.rejected)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(fnnls, "fnnls_cholesky", spy)
        try:
            for enabled in (True, True, False, False):
                with state.counterfactual_scope(enabled=enabled):
                    assert os.environ["AUTOARRAY_NNLS_WARM_START"] == ("1" if enabled else "0")
                    inversion_util.reconstruction_positive_only_from(**state.last_solve_kwargs)
        finally:
            monkeypatch.undo()

        assert attempts == ["memo", "memo", "dense", "dense"]
        assert inversion_util.reconstruction_positive_only_from is guarded_outer
        assert nnls_memo.passive_set_get is guarded_get
        assert nnls_memo._nnls_passive_set_memo == guarded_memo
        assert state.records == guarded_records
        assert (state.accepted, state.rejected) == guarded_counts
        assert os.environ["AUTOARRAY_NNLS_WARM_START"] == "1"
    assert os.environ.get("AUTOARRAY_NNLS_WARM_START") == original_environment


def test_counterfactual_requires_a_diagnostic_capture():
    with policy.policy_scope(threshold=1.0) as state:
        with pytest.raises(RuntimeError, match="no diagnostic solve"):
            with state.counterfactual_scope(enabled=True):
                pass


def test_policy_restores_outer_lookup_memo_and_environment_after_exception():
    from autoarray.inversion.inversion import inversion_util, nnls_memo

    original_outer = inversion_util.reconstruction_positive_only_from
    original_get = nnls_memo.passive_set_get
    sentinel = nnls_memo.MemoEntry(np.array([0]), 0.5)
    nnls_memo._nnls_passive_set_memo["sentinel"] = sentinel
    old = os.environ.pop("AUTOARRAY_NNLS_WARM_START", None)
    try:
        with pytest.raises(RuntimeError, match="stop"):
            with policy.policy_scope(0.1):
                assert os.environ["AUTOARRAY_NNLS_WARM_START"] == "1"
                raise RuntimeError("stop")
        assert inversion_util.reconstruction_positive_only_from is original_outer
        assert nnls_memo.passive_set_get is original_get
        assert nnls_memo._nnls_passive_set_memo == {"sentinel": sentinel}
        assert "AUTOARRAY_NNLS_WARM_START" not in os.environ
    finally:
        nnls_memo._nnls_passive_set_memo.pop("sentinel", None)
        if old is not None:
            os.environ["AUTOARRAY_NNLS_WARM_START"] = old
