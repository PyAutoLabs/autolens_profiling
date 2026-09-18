"""Focused tests for phase-5 production NNLS replay helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "ruff.toml").exists()
)
MISC = ROOT / "scripts" / "misc"
if str(MISC) not in sys.path:
    sys.path.insert(0, str(MISC))

from likelihood_breakdown import fixed_light_numba_draws_steps as steps  # noqa: E402

MANIFEST = (
    ROOT
    / "results/breakdown/imaging/"
    / "fixed_light_draws_delaunay_hpc_a100_fp64_fixed_light_draws.json"
)


def test_load_manifest_preserves_rows_and_rejects_incomplete(tmp_path):
    rows, provenance = steps.load_manifest(MANIFEST)
    original = json.loads(MANIFEST.read_text())
    assert rows == original["draws"]
    assert provenance["seed"] == 0
    assert provenance["family_counts"] == {"fiducial": 1, "walk": 16, "random": 24}
    assert len(provenance["sha256"]) == 64

    original["draws"].pop()
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="exactly 41"):
        steps.load_manifest(incomplete)


def test_load_manifest_rejects_changed_finite_offset(tmp_path):
    changed = json.loads(MANIFEST.read_text())
    changed["draws"][1]["offsets"]["einstein_radius"] += 1.0e-12
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="SHA256"):
        steps.load_manifest(path)


def test_memo_scope_restores_state_and_environment_even_after_exception():
    from autoarray.inversion.inversion import nnls_memo

    old_environment = os.environ.get("AUTOARRAY_NNLS_WARM_START")
    sentinel = nnls_memo.MemoEntry(np.array([2]), 0.25)
    snapshot = dict(nnls_memo._nnls_passive_set_memo)
    nnls_memo._nnls_passive_set_memo.clear()
    nnls_memo._nnls_passive_set_memo["before"] = sentinel
    os.environ["AUTOARRAY_NNLS_WARM_START"] = "old"
    try:
        with pytest.raises(RuntimeError), steps.memo_scope(enabled=False):
            assert nnls_memo._nnls_passive_set_memo == {}
            assert os.environ["AUTOARRAY_NNLS_WARM_START"] == "0"
            nnls_memo._nnls_passive_set_memo["inside"] = sentinel
            raise RuntimeError("stop")
        assert nnls_memo._nnls_passive_set_memo == {"before": sentinel}
        assert os.environ["AUTOARRAY_NNLS_WARM_START"] == "old"
    finally:
        nnls_memo._nnls_passive_set_memo.clear()
        nnls_memo._nnls_passive_set_memo.update(snapshot)
        if old_environment is None:
            os.environ.pop("AUTOARRAY_NNLS_WARM_START", None)
        else:
            os.environ["AUTOARRAY_NNLS_WARM_START"] = old_environment


def test_replay_sequence_preserves_order_and_isolates_complete_traversals(monkeypatch):
    seen = []

    class Scope:
        def __init__(self, enabled):
            self.enabled = enabled

        def __enter__(self):
            seen.append(("enter", self.enabled))

        def __exit__(self, *_args):
            seen.append(("exit", self.enabled))

    monkeypatch.setattr(steps, "memo_scope", Scope)
    evaluated = []

    def evaluate(index):
        evaluated.append(index)
        return index + 0.5

    rows = steps.replay_sequence(evaluate, [3, 1, 2], enabled=True)
    assert evaluated == [3, 1, 2]
    assert [row["index"] for row in rows] == [3, 1, 2]
    assert [row["evidence"] for row in rows] == [3.5, 1.5, 2.5]
    assert all(row["elapsed_ms"] >= 0.0 for row in rows)
    assert seen == [("enter", True), ("exit", True)]


def test_observer_distinguishes_retry_and_guard_invalidation(monkeypatch):
    from autoarray.inversion.inversion import inversion_util
    from autoarray.util import fnnls

    original_outer = inversion_util.reconstruction_positive_only_from
    original_kernel = fnnls.fnnls_cholesky

    def dummy_kernel(_matrix, _vector, P_initial, stats, factor=None):
        if np.asarray(P_initial).dtype != bool and 99 in P_initial:
            stats.update(outer_iterations=1, warm_start_errors=9)
            raise ValueError("bad memo seed")
        stats.update(
            outer_iterations=3,
            inner_iterations=2,
            passive_set=np.array([1, 4]),
            n_passive=2,
            warm_start_errors=5,
        )
        return np.array([0.0, 1.0])

    state = {"mode": "retry"}

    def dummy_outer(*_args, **_kwargs):
        stats = {"seed_source": "dense", "warm_start_fallback": False}
        if state["mode"] == "retry":
            try:
                fnnls.fnnls_cholesky(None, None, P_initial=np.array([99]), stats=stats)
            except ValueError:
                result = fnnls.fnnls_cholesky(
                    None, None, P_initial=np.array([True, False]), stats=stats
                )
        else:
            result = fnnls.fnnls_cholesky(None, None, P_initial=np.array([1, 4]), stats=stats)
            stats["seed_source"] = "memo"
            stats["warm_start_fallback"] = True
        return result

    monkeypatch.setattr(fnnls, "fnnls_cholesky", dummy_kernel)
    monkeypatch.setattr(inversion_util, "reconstruction_positive_only_from", dummy_outer)
    try:
        with steps.observe_solver() as calls:
            inversion_util.reconstruction_positive_only_from()
            state["mode"] = "guard"
            inversion_util.reconstruction_positive_only_from()
        assert len(calls[0]["kernel_attempts"]) == 2
        assert calls[0]["kernel_attempts"][0]["error"].startswith("ValueError")
        assert calls[0]["kernel_attempts"][0]["initial_seed_source"] == "memo"
        assert calls[0]["kernel_attempts"][0]["outer_iterations"] == 1
        assert calls[0]["kernel_attempts"][0]["warm_start_errors"] == 9
        assert calls[0]["kernel_attempts"][1]["initial_seed_source"] == "dense"
        assert calls[0]["seed_source"] == "dense"
        assert calls[0]["warm_start_fallback"] is False
        assert calls[1]["seed_source"] == "memo"
        assert calls[1]["warm_start_fallback"] is True
        assert calls[1]["outer_iterations"] == 3
        assert calls[1]["final_passive_set"] == [1, 4]
    finally:
        inversion_util.reconstruction_positive_only_from = original_outer
        fnnls.fnnls_cholesky = original_kernel


def _solution(evidence=10.0, passive_set=(0, 2), reconstruction=(1.0, 2.0)):
    return {
        "evidence": evidence,
        "passive_set": list(passive_set),
        "reconstruction": np.asarray(reconstruction),
    }


def test_compare_solutions_reports_nonfinite_and_active_set_gate_failures():
    active_failure = steps.compare_solutions(_solution(), _solution(passive_set=(0, 1)))
    assert active_failure["active_set_equal"] is False
    assert active_failure["passed"] is False

    nonfinite = steps.compare_solutions(_solution(evidence=np.nan), _solution())
    assert nonfinite["finite"] is False
    assert nonfinite["evidence_absolute_difference"] is None
    assert nonfinite["passed"] is False
    json.dumps(nonfinite, allow_nan=False)


def test_compare_solutions_treats_active_set_as_unordered_unique_membership():
    permuted = steps.compare_solutions(_solution(passive_set=(2, 0)), _solution(passive_set=(0, 2)))
    assert permuted["active_sets_valid"] is True
    assert permuted["active_set_equal"] is True
    assert permuted["passed"] is True

    duplicate = steps.compare_solutions(
        _solution(passive_set=(0, 0, 2)), _solution(passive_set=(0, 2))
    )
    assert duplicate["active_sets_valid"] is False
    assert duplicate["active_set_equal"] is False
    assert duplicate["passed"] is False


def test_compare_solutions_uses_declared_evidence_tolerance_and_reports_reconstruction():
    result = steps.compare_solutions(
        _solution(evidence=10.0 + 5.0e-9, reconstruction=(1.0, 2.0001)),
        _solution(evidence=10.0),
    )
    assert result["evidence_relative_difference"] <= 1.0e-9
    assert result["reconstruction_max_absolute_difference"] == pytest.approx(1.0e-4)
    assert result["passed"] is True

    failed = steps.compare_solutions(_solution(evidence=10.0 + 2.0e-8), _solution())
    assert failed["evidence_passed"] is False
    assert failed["passed"] is False
