"""Unit tests for the ``SEARCHES_BIJECTOR`` / ``SEARCHES_LANE_HISTORY`` /
``SEARCHES_TRACE_PARAMS`` plumbing (W5 Phase 8B, issue #162) and the
``bijector_ab.py`` scorer.

Covers: env-var parsing/validation, the ``_bijector_object`` label -> ``af``
class resolution (including the ``log_reg`` per-path restriction), the
composed ``multi_start_unique_tag`` (none vs log_reg differ; none resolves
identically to the pre-8B tag), the recorded ``multi_start_settings()`` block,
and the ``bijector_ab.score_rows`` verdict logic on synthetic rows (F5 trip,
F4 byte-identity).

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_bijector.py -q
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_misc_dir = str(_ROOT / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)
if str(_ROOT / "scripts" / "misc" / "searches") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "scripts" / "misc" / "searches"))
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import pytest
from searches._samplers import (
    _bijector_object,
    assert_disjoint_output_paths,
    build_multi_start,
    multi_start_bijector,
    multi_start_lane_history,
    multi_start_settings,
    multi_start_trace_param_paths,
    multi_start_unique_tag,
)

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# env-var parsing / validation
# ---------------------------------------------------------------------------


def test_bijector_default_none(monkeypatch):
    monkeypatch.delenv("SEARCHES_BIJECTOR", raising=False)
    assert multi_start_bijector() == "none"


def test_bijector_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("SEARCHES_BIJECTOR", "not-a-bijector")
    with pytest.raises(ValueError):
        multi_start_bijector()


def test_lane_history_default_off(monkeypatch):
    monkeypatch.delenv("SEARCHES_LANE_HISTORY", raising=False)
    assert multi_start_lane_history() is False


def test_lane_history_on(monkeypatch):
    monkeypatch.setenv("SEARCHES_LANE_HISTORY", "1")
    assert multi_start_lane_history() is True


def test_lane_history_anything_else_stays_off(monkeypatch):
    monkeypatch.setenv("SEARCHES_LANE_HISTORY", "true")
    assert multi_start_lane_history() is False


def test_trace_param_paths_default_none(monkeypatch):
    monkeypatch.delenv("SEARCHES_TRACE_PARAMS", raising=False)
    assert multi_start_trace_param_paths() is None


def test_trace_param_paths_parses_comma_list(monkeypatch):
    monkeypatch.setenv("SEARCHES_TRACE_PARAMS", "a.b.c, d.e.f")
    assert multi_start_trace_param_paths() == ["a.b.c", "d.e.f"]


# ---------------------------------------------------------------------------
# _bijector_object — label -> af class resolution
# ---------------------------------------------------------------------------


def test_bijector_object_none_is_bijector_none():
    import autofit as af

    obj = _bijector_object("none", dataset_class="imaging", model_type="knn")
    assert isinstance(obj, af.BijectorNone)


def test_bijector_object_auto_log_is_bijector_auto():
    import autofit as af

    obj = _bijector_object("auto_log", dataset_class="imaging", model_type="knn")
    assert isinstance(obj, af.BijectorAuto)


def test_bijector_object_logit_is_bijector_logit():
    import autofit as af

    obj = _bijector_object("logit", dataset_class="imaging", model_type="knn")
    assert isinstance(obj, af.BijectorLogit)


def test_bijector_object_log_reg_restricts_to_regularization_paths():
    import autofit as af

    obj = _bijector_object("log_reg", dataset_class="imaging", model_type="knn")
    assert isinstance(obj, af.BijectorPerPath)
    assert obj.kind_by_path == {
        "galaxies.source.pixelization.regularization.inner_coefficient": "log",
        "galaxies.source.pixelization.regularization.outer_coefficient": "log",
    }


def test_bijector_object_log_reg_on_mge_is_empty():
    """MGE has no regularization coefficients at all -- the F4 control."""
    import autofit as af

    obj = _bijector_object("log_reg", dataset_class="imaging", model_type="mge")
    assert isinstance(obj, af.BijectorPerPath)
    assert obj.kind_by_path == {}


# ---------------------------------------------------------------------------
# unique_tag composition — the correctness guard
# ---------------------------------------------------------------------------


def test_unique_tag_none_matches_pre_8b_tag(monkeypatch):
    """A ``none``-bijector, unseeded, positions-off cell must resolve to
    EXACTLY the same (``None``) tag as before the bijector arm existed --
    otherwise every pre-existing recorded cell's output path moves."""
    monkeypatch.delenv("SEARCHES_SEED", raising=False)
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    monkeypatch.delenv("SEARCHES_BIJECTOR", raising=False)
    assert multi_start_unique_tag("imaging", "knn") is None


def test_unique_tag_log_reg_differs_from_none(monkeypatch):
    monkeypatch.delenv("SEARCHES_SEED", raising=False)
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    monkeypatch.setenv("SEARCHES_BIJECTOR", "none")
    tag_none = multi_start_unique_tag("imaging", "knn")
    monkeypatch.setenv("SEARCHES_BIJECTOR", "log_reg")
    tag_log_reg = multi_start_unique_tag("imaging", "knn")
    assert tag_none is None
    assert tag_log_reg is not None
    assert tag_log_reg != tag_none


def test_multi_start_search_output_path_and_identifier_differ_none_vs_log_reg(monkeypatch):
    """The load-bearing regression this composition exists to prevent: a
    log_reg arm must NOT resume the none arm's ``.completed`` fit."""
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    monkeypatch.delenv("SEARCHES_SEED", raising=False)
    monkeypatch.setenv("SEARCHES_BIJECTOR", "none")
    search_none = build_multi_start(
        sampler="multi_start_prodigy",
        dataset_class="imaging",
        model_type="knn",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    monkeypatch.setenv("SEARCHES_BIJECTOR", "log_reg")
    search_log_reg = build_multi_start(
        sampler="multi_start_prodigy",
        dataset_class="imaging",
        model_type="knn",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    assert_disjoint_output_paths(search_none, search_log_reg)


# ---------------------------------------------------------------------------
# settings recording
# ---------------------------------------------------------------------------


def test_settings_records_bijector_and_lane_history(monkeypatch):
    monkeypatch.setenv("SEARCHES_BIJECTOR", "log_reg")
    monkeypatch.setenv("SEARCHES_LANE_HISTORY", "1")
    monkeypatch.delenv("SEARCHES_TRACE_PARAMS", raising=False)
    settings = multi_start_settings("multi_start_prodigy", "imaging", "knn", "hst")
    assert settings["bijector"] == "log_reg"
    assert settings["record_lane_nan_history"] is True
    assert "trace_param_indices" not in settings


def test_settings_records_trace_param_indices(monkeypatch):
    monkeypatch.setenv(
        "SEARCHES_TRACE_PARAMS",
        "galaxies.source.pixelization.regularization.inner_coefficient,"
        "galaxies.source.pixelization.regularization.outer_coefficient",
    )
    settings = multi_start_settings("multi_start_prodigy", "imaging", "knn", "hst")
    assert settings["trace_param_indices"] == sorted(settings["trace_param_indices"])
    assert len(settings["trace_param_indices"]) == 2


def test_settings_default_bijector_is_none_string(monkeypatch):
    monkeypatch.delenv("SEARCHES_BIJECTOR", raising=False)
    monkeypatch.delenv("SEARCHES_LANE_HISTORY", raising=False)
    settings = multi_start_settings("multi_start_prodigy", "imaging", "knn", "hst")
    assert settings["bijector"] == "none"
    assert settings["record_lane_nan_history"] is False


# ---------------------------------------------------------------------------
# bijector_ab.py scorer — synthetic rows
# ---------------------------------------------------------------------------


def _synthetic_row(**overrides) -> dict:
    row = dict(
        source=None,
        cell="mge",
        log_det_method=None,
        bijector="none",
        seed=0,
        record_lane_nan_history=True,
        n_starts=2,
        total_steps=5,
        n_resurrections=0,
        n_clipped_lane_steps=0,
        n_value_nan_lane_steps=0,
        clip_rate=0.0,
        first_value_nan_step=None,
        coefficient_at_first_nan=None,
        frac_steps_high_lambda=0.0,
        best_log_posterior=-100.0,
        winning_lane_index=0,
        best_fom=180.0,
        max_log_likelihood=-90.0,
        fom_history_global_best=[200.0, 190.0, 180.0],
        step0_fom=200.0,
        n_lanes_pinned_final=0,
        max_pinned_final_count=0,
        final_params_per_lane=[[1.0, 2.0], [3.0, 4.0]],
        lane_best_params_per_lane=[[1.0, 2.0], [3.0, 4.0]],
        trace_param_indices=None,
        final_d=None,
        final_d_note="n/a",
    )
    row.update(overrides)
    return row


def test_score_rows_f5_trips_on_mismatched_step0_fom():
    import bijector_ab as m

    rows = [
        _synthetic_row(cell="knn", bijector="none", seed=0, step0_fom=200.0),
        _synthetic_row(cell="knn", bijector="log_reg", seed=0, step0_fom=999.0),
    ]
    verdict = m.score_rows(rows)
    assert verdict["halted"] is True
    assert verdict["f5"]["falsified"] is True
    assert len(verdict["f5"]["problems"]) == 1


def test_score_rows_f5_does_not_trip_on_matching_step0_fom():
    import bijector_ab as m

    rows = [
        _synthetic_row(cell="mge", bijector="none", seed=0, step0_fom=200.0),
        _synthetic_row(cell="mge", bijector="log_reg", seed=0, step0_fom=200.0 + 1e-12),
    ]
    verdict = m.score_rows(rows)
    assert verdict["halted"] is False


def test_score_rows_f4_flags_winning_lane_disagreement():
    """The amended F4 (issue #182): the criterion is the winning lane's
    best_fom / max_log_likelihood, not per-lane byte-identity."""
    import bijector_ab as m

    rows = [
        _synthetic_row(cell="mge", bijector="none", seed=0, best_fom=180.0),
        _synthetic_row(cell="mge", bijector="log_reg", seed=0, best_fom=180.5),
    ]
    verdict = m.score_rows(rows)
    assert verdict["halted"] is False
    f4 = verdict["f4_mge_control_and_logit_pathology"]
    assert f4["falsified"] is True
    assert f4["mge_per_seed_equivalence"][0]["agree_within_fp64"] is False


def test_score_rows_f4_tolerates_trailing_bit_drift_in_the_winning_lane():
    """fp64 noise in the last bits is not a bijector effect and must not
    falsify the control (the whole point of the 2026-08-27 amendment)."""
    import bijector_ab as m

    rows = [
        _synthetic_row(
            cell="mge", bijector="none", seed=0, best_fom=180.0, max_log_likelihood=-90.0
        ),
        _synthetic_row(
            cell="mge",
            bijector="log_reg",
            seed=0,
            best_fom=180.0 + 1e-11,
            max_log_likelihood=-90.0 - 1e-11,
            # per-lane vectors differ in a lane that never won — informational
            # only under the amended criterion.
            final_params_per_lane=[[1.0, 2.0], [3.0, 4.5]],
        ),
        _synthetic_row(cell="knn", bijector="logit", seed=0, max_pinned_final_count=0),
    ]
    verdict = m.score_rows(rows)
    f4 = verdict["f4_mge_control_and_logit_pathology"]
    assert f4["falsified"] is False
    # byte-identity DID fail — reported, not scored.
    assert f4["mge_per_seed_byte_identical"][0] is False


def test_score_rows_f4_unscorable_without_matched_mge_seeds():
    import bijector_ab as m

    rows = [_synthetic_row(cell="mge", bijector="none", seed=0)]
    verdict = m.score_rows(rows)
    f4 = verdict["f4_mge_control_and_logit_pathology"]
    assert f4["scorable"] is False
    assert f4["falsified"] is None


def test_score_rows_no_rows_for_a_cell_is_unscorable_not_a_verdict():
    """The #182 repair: absent data used to read as a silent PASS (F1) and a
    silent FAIL (F2) — the same absence, opposite confident answers."""
    import bijector_ab as m

    rows = [_synthetic_row(cell="mge", bijector="none", seed=0)]
    verdict = m.score_rows(rows)
    assert verdict["f1_nan_wall_position"]["falsified"] is None
    assert verdict["f1_nan_wall_position"]["scorable"] is False
    assert verdict["f2_steps_to_reference"]["falsified"] is None
    assert verdict["f2_steps_to_reference"]["scorable"] is False
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["falsified"] is None
    assert set(verdict["unscorable_criteria"]) == {
        "f1_nan_wall_position",
        "f2_steps_to_reference",
        "f3_time_at_high_lambda",
        "f4_mge_control_and_logit_pathology",
    }


def test_score_f1_unscorable_when_no_nan_history_recorded():
    import bijector_ab as m

    rows = [
        _synthetic_row(
            cell="delaunay_adapt_split",
            bijector="none",
            seed=0,
            first_value_nan_step=None,
            n_value_nan_lane_steps=None,
        ),
        _synthetic_row(
            cell="delaunay_adapt_split",
            bijector="log_reg",
            seed=0,
            first_value_nan_step=None,
            n_value_nan_lane_steps=None,
        ),
    ]
    result = m.score_f1(rows)
    assert result["scorable"] is False
    assert result["falsified"] is None


def test_score_f1_stays_conclusive_when_one_limb_fires():
    """A disjunction with a fired limb is settled even if the other is
    unmeasurable — unscorable must not swallow a real result."""
    import bijector_ab as m

    rows = [
        _synthetic_row(
            cell="delaunay_adapt_split",
            bijector="none",
            seed=0,
            first_value_nan_step=100,
            n_value_nan_lane_steps=None,
        ),
        _synthetic_row(
            cell="delaunay_adapt_split",
            bijector="log_reg",
            seed=0,
            first_value_nan_step=200,
            n_value_nan_lane_steps=None,
        ),
    ]
    result = m.score_f1(rows)
    assert result["scorable"] is True
    assert result["falsified"] is True  # log_reg's NaN wall is LATER, not earlier


def test_score_f2_unscorable_rather_than_falsified_without_a_reference():
    """The old code returned falsified=True here: 'never reached the
    reference' silently became 'reached it too slowly'."""
    import bijector_ab as m

    rows = [
        _synthetic_row(cell="knn", bijector="none", seed=0, best_log_posterior=None),
        _synthetic_row(cell="knn", bijector="log_reg", seed=0, best_log_posterior=None),
    ]
    result = m.score_f2(rows)
    assert result["scorable"] is False
    assert result["falsified"] is None


def test_arm_table_has_39_arms_and_unique_config_names():
    import bijector_ab as m

    arms = m.build_arms()
    assert len(arms) == 39
    names = {m.arm_config_name(a) for a in arms}
    assert len(names) == 39


def test_arm_table_counts_per_cell():
    from collections import Counter

    import bijector_ab as m

    arms = m.build_arms()
    counts = Counter(a["cell"] for a in arms)
    assert counts["delaunay_adapt_split"] == 20
    assert counts["knn"] == 15
    assert counts["mge"] == 4
