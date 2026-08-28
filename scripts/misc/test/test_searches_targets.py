"""Unit tests for the Phase 1 targets registry + schema v2 (W4 / issue #161).

Covers: registry uniqueness, ``target_id`` stability/sensitivity (prior /
positions / precision / log_det_method / dataset-file changes DO change it;
sampler/n_live cannot — the function's signature has no such parameters),
``build_for_cell(target=...)`` deriving the right model, a positions-on
target attaching a real ``PositionsLH`` list without any
``SEARCHES_POSITIONS*`` env var set, ``_build_summary``'s schema-v2 block
carrying every v1 key, ``build_readme``'s searches table rendering both v1
and v2 payloads, the MultiStart ``likelihood_evals`` fix, and Kish ESS.

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_targets.py -q
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
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import dataclasses  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402

import autofit as af  # noqa: E402
import pytest  # noqa: E402
from searches import _targets as t  # noqa: E402
from searches._metrics import RunMetrics, _kish_ess, collect_metrics  # noqa: E402
from searches._setup import _mge_model, build_for_cell  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HST_DATASET_PATH = _ROOT / "dataset" / "imaging" / "hst"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_has_32_unique_targets():
    assert len(t.TARGETS) == 32
    names = [target.name for target in t.TARGETS.values()]
    assert len(names) == len(set(names))


def test_registry_key_matches_target_name():
    for key, target in t.TARGETS.items():
        assert target.name == key


def test_registry_covers_expected_model_types():
    model_types = {target.model_type for target in t.TARGETS.values()}
    assert model_types == set(t._REGISTRY_MODEL_TYPES)


def test_registry_every_target_has_a_resolvable_builder():
    for target in t.TARGETS.values():
        assert callable(target.builder)


# ---------------------------------------------------------------------------
# target_id — stability
# ---------------------------------------------------------------------------


def test_target_id_stable_across_independent_model_rebuilds():
    target = t.TARGETS["mge_fp64"]
    model_a = _mge_model(mask_radius=3.5)
    model_b = _mge_model(mask_radius=3.5)
    id_a = t.target_id(target, model_a, _HST_DATASET_PATH)
    id_b = t.target_id(target, model_b, _HST_DATASET_PATH)
    assert id_a == id_b


def test_target_id_signature_has_no_algorithm_knobs():
    """sampler / n_live cannot influence target_id — they are not even
    parameters of the function that computes it."""
    params = set(inspect.signature(t.target_id).parameters)
    assert "sampler" not in params
    assert "n_live" not in params
    # ``positions_setup`` is the resolved positions arm (issue #182) — a
    # recorded, run-defining input, not an algorithm knob.
    assert params == {"target", "model", "dataset_path", "positions_setup"}


# ---------------------------------------------------------------------------
# target_id — sensitivity
# ---------------------------------------------------------------------------


def test_target_id_sensitive_to_positions():
    model = _mge_model(mask_radius=3.5)
    off = t.target_id(t.TARGETS["mge_fp64"], model, _HST_DATASET_PATH)
    on = t.target_id(t.TARGETS["mge_pos_fp64"], model, _HST_DATASET_PATH)
    assert off != on


def _positions_setup(mode="fixed", threshold=0.3, factor=1e8):
    """A resolved positions block of the shape ``_setup.positions_settings()``
    returns and ``_runner`` records as the artifact's top-level ``positions``."""
    from searches._setup import positions_settings_for

    return positions_settings_for(
        enabled=True,
        mode=mode,
        fixed_value=threshold if mode == "fixed" else None,
        factor=factor,
    )


def test_target_id_positions_default_setup_matches_no_setup():
    """The historical default arm (fixed / 0.3 / 1e8) must keep the exact id
    it was recorded with — passing it explicitly changes nothing."""
    model = _mge_model(mask_radius=3.5)
    target = t.TARGETS["mge_pos_fp64"]
    implicit = t.target_id(target, model, _HST_DATASET_PATH)
    explicit = t.target_id(target, model, _HST_DATASET_PATH, _positions_setup())
    assert implicit == explicit


def test_target_id_sensitive_to_positions_factor():
    """The Phase-4 defect (issue #182): f1e5 and f1e8 are different objectives
    and must not share a target_id."""
    model = _mge_model(mask_radius=3.5)
    target = t.TARGETS["mge_pos_fp64"]
    f1e8 = t.target_id(target, model, _HST_DATASET_PATH, _positions_setup(factor=1e8))
    f1e5 = t.target_id(target, model, _HST_DATASET_PATH, _positions_setup(factor=1e5))
    assert f1e8 != f1e5


def test_target_id_sensitive_to_positions_threshold_mode():
    model = _mge_model(mask_radius=3.5)
    target = t.TARGETS["mge_pos_fp64"]
    fixed = t.target_id(target, model, _HST_DATASET_PATH, _positions_setup(mode="fixed"))
    auto = t.target_id(target, model, _HST_DATASET_PATH, _positions_setup(mode="auto"))
    assert fixed != auto


def test_target_id_sensitive_to_positions_threshold_value():
    model = _mge_model(mask_radius=3.5)
    target = t.TARGETS["mge_pos_fp64"]
    t03 = t.target_id(target, model, _HST_DATASET_PATH, _positions_setup(threshold=0.3))
    t05 = t.target_id(target, model, _HST_DATASET_PATH, _positions_setup(threshold=0.5))
    assert t03 != t05


def test_target_id_positions_off_unchanged_by_setup_argument():
    """A positions-OFF row's id must be untouched by the #182 change."""
    model = _mge_model(mask_radius=3.5)
    target = t.TARGETS["mge_fp64"]
    from searches._setup import positions_settings_for

    a = t.target_id(target, model, _HST_DATASET_PATH)
    b = t.target_id(target, model, _HST_DATASET_PATH, positions_settings_for(enabled=False))
    assert a == b


def test_positions_block_refuses_setup_contradicting_target():
    with pytest.raises(ValueError, match="positions='off'"):
        t._positions_block(t.TARGETS["mge_fp64"], _positions_setup())
    from searches._setup import positions_settings_for

    with pytest.raises(ValueError, match="positions='on'"):
        t._positions_block(t.TARGETS["mge_pos_fp64"], positions_settings_for(enabled=False))


def test_positions_block_note_is_not_hashed():
    """The free-text note must never enter the identity (pre-existing rule,
    re-asserted now that the block is caller-supplied)."""
    setup = _positions_setup()
    setup["note"] = "totally different wording"
    model = _mge_model(mask_radius=3.5)
    target = t.TARGETS["mge_pos_fp64"]
    assert t.target_id(target, model, _HST_DATASET_PATH, setup) == t.target_id(
        target, model, _HST_DATASET_PATH, _positions_setup()
    )


def test_target_id_sensitive_to_precision():
    model = _mge_model(mask_radius=3.5)
    fp64 = t.target_id(t.TARGETS["mge_fp64"], model, _HST_DATASET_PATH)
    mp = t.target_id(t.TARGETS["mge_mp"], model, _HST_DATASET_PATH)
    assert fp64 != mp


def test_target_id_sensitive_to_log_det_method():
    model = _mge_model(mask_radius=3.5)
    base = t.TARGETS["mge_fp64"]
    slogdet = dataclasses.replace(base, log_det_method="slogdet")
    id_base = t.target_id(base, model, _HST_DATASET_PATH)
    id_slogdet = t.target_id(slogdet, model, _HST_DATASET_PATH)
    assert id_base != id_slogdet


def test_target_id_sensitive_to_prior_change():
    target = t.TARGETS["mge_fp64"]
    model_a = _mge_model(mask_radius=3.5)
    model_b = _mge_model(mask_radius=3.5)
    model_b.galaxies.lens.mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)
    id_a = t.target_id(target, model_a, _HST_DATASET_PATH)
    id_b = t.target_id(target, model_b, _HST_DATASET_PATH)
    assert id_a != id_b


def test_target_id_ignores_prior_process_local_id():
    """The bug _targets.py exists to avoid: repr(prior) embeds a process-local
    id counter and is NOT deterministic — two freshly-built, parameter-
    identical priors must canonicalise identically."""
    p1 = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    p2 = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    assert repr(p1) != repr(p2)  # the trap: reprs differ (different ids)
    assert t._canonical_prior(p1) == t._canonical_prior(p2)  # the fix: content-equal


def test_target_id_sensitive_to_dataset_file_content(tmp_path):
    target = t.TARGETS["mge_fp64"]
    model = _mge_model(mask_radius=3.5)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "data.fits").write_bytes(b"AAAA")
    (dir_b / "data.fits").write_bytes(b"BBBB")
    id_a = t.target_id(target, model, dir_a)
    id_b = t.target_id(target, model, dir_b)
    assert id_a != id_b


# ---------------------------------------------------------------------------
# build_for_cell(target=...)
# ---------------------------------------------------------------------------


def test_build_for_cell_with_target_model_dim_matches():
    target = t.TARGETS["mge_fp64"]
    _, model, _ = build_for_cell(target=target, use_jax=False)
    assert model.prior_count == 15


def test_build_for_cell_target_field_overridable_by_explicit_kwarg():
    target = t.TARGETS["mge_fp64"]
    # Explicit model_type wins over the target's — proves the override isn't
    # silently ignored.
    _, model, _ = build_for_cell(target=target, model_type="delaunay", use_jax=False)
    assert model.prior_count == 12  # _delaunay_model's dim, not mge's 15


def test_build_for_cell_target_positions_on_without_env(monkeypatch):
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    monkeypatch.delenv("SEARCHES_POSITIONS_THRESHOLD", raising=False)
    monkeypatch.delenv("SEARCHES_POSITIONS_FACTOR", raising=False)
    target = t.TARGETS["mge_pos_fp64"]
    _, _, analysis = build_for_cell(target=target, use_jax=False)
    assert analysis.positions_likelihood_list is not None
    assert len(analysis.positions_likelihood_list) == 1


def test_build_for_cell_target_positions_off_without_env(monkeypatch):
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    target = t.TARGETS["mge_fp64"]
    _, _, analysis = build_for_cell(target=target, use_jax=False)
    assert analysis.positions_likelihood_list is None


# ---------------------------------------------------------------------------
# _build_summary — schema v2 carries every v1 key
# ---------------------------------------------------------------------------

_V1_TOP_LEVEL_KEYS = {
    "sampler",
    "dataset_class",
    "model",
    "instrument",
    "config_name",
    "version",
    "device",
    "use_mixed_precision",
    "sampler_config",
    "positions",
    "log_det_method",
    "model_summary",
    "results",
    "performance",
}


def test_build_summary_v2_has_all_v1_keys(monkeypatch):
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    from searches._runner import _build_summary

    model = _mge_model(mask_radius=3.5)
    metrics = RunMetrics(
        total_wall_s=10.0,
        viz_wall_s=1.0,
        sampler_wall_s=9.0,
        likelihood_evals=100,
        time_per_eval_ms=90.0,
        log_evidence=1.0,
        max_log_likelihood=2.0,
        posterior_samples=5,
        stored_samples=5,
    )

    @dataclasses.dataclass
    class _FakeCLI:
        use_mixed_precision: bool = False

    summary = _build_summary(
        sampler="nautilus",
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        config_name="unit_test",
        cli=_FakeCLI(),
        use_jax=False,
        model=model,
        n_live=200,
        metrics=metrics,
        viz_n_calls=0,
        best_fit="dummy",
    )
    assert _V1_TOP_LEVEL_KEYS <= summary.keys()
    assert summary["schema_version"] == 2
    assert "target" in summary
    assert "algorithm" in summary
    assert "hardware" in summary
    # imaging/mge/hst is TARGETS-registry-covered -> a real, non-null id.
    assert summary["target"]["target_id"] is not None
    assert summary["algorithm"] == {
        "name": "nautilus",
        "config_id": "unit_test",
        "settings": summary["sampler_config"],
        "seed": summary["sampler_config"].get("seed"),
    }
    assert summary["hardware"]["precision"] == "fp64"


def test_build_summary_target_null_for_unregistered_cell(monkeypatch):
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    from searches._runner import _build_summary
    from searches._setup import _point_source_model

    model = _point_source_model()
    metrics = RunMetrics(
        total_wall_s=1.0,
        viz_wall_s=0.0,
        sampler_wall_s=1.0,
        likelihood_evals=10,
        time_per_eval_ms=100.0,
        log_evidence=1.0,
        max_log_likelihood=2.0,
        posterior_samples=1,
        stored_samples=1,
    )

    @dataclasses.dataclass
    class _FakeCLI:
        use_mixed_precision: bool = False

    summary = _build_summary(
        sampler="nautilus",
        dataset_class="point_source",
        model_type="image_plane",
        instrument="simple",
        config_name="unit_test",
        cli=_FakeCLI(),
        use_jax=False,
        model=model,
        n_live=200,
        metrics=metrics,
        viz_n_calls=0,
        best_fit="dummy",
    )
    assert summary["target"]["target_id"] is None
    assert summary["target"]["cell"] == "point_source/image_plane/simple"


# ---------------------------------------------------------------------------
# build_readme — v1 and v2 payloads render side by side
# ---------------------------------------------------------------------------


def test_build_readme_searches_table_renders_v1_and_v2(tmp_path):
    from tooling.build_readme import SearchArtifact, _render_searches_table

    v1_payload = {
        "sampler": "nautilus",
        "dataset_class": "imaging",
        "model": "mge",
        "instrument": "hst",
        "config_name": "v1cell",
        "version": "2026.1.1.1",
        "results": {"max_log_likelihood": 1.0, "log_evidence": 2.0},
        "performance": {
            "total_wall_s": 10.0,
            "likelihood_evals": 100,
            "time_per_eval_ms": 1.0,
        },
    }
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(v1_payload))

    v2_payload = dict(v1_payload)
    v2_payload["config_name"] = "v2cell"
    v2_payload["schema_version"] = 2
    v2_payload["target"] = {"target_id": "sha256:abcdef1234567890"}
    v2_payload["performance"] = dict(v2_payload["performance"], kish_ess=42.0)
    v2_path = tmp_path / "v2.json"
    v2_path.write_text(json.dumps(v2_payload))

    artifacts = [
        SearchArtifact(
            path=v1_path,
            sampler="nautilus",
            cell="imaging/mge/hst",
            config="v1cell",
            version=(2026, 1, 1, 1),
            raw_version="2026.1.1.1",
        ),
        SearchArtifact(
            path=v2_path,
            sampler="nautilus",
            cell="imaging/mge/hst",
            config="v2cell",
            version=(2026, 1, 1, 1),
            raw_version="2026.1.1.1",
        ),
    ]
    table = _render_searches_table(artifacts)
    assert "v1cell" in table
    assert "v2cell" in table
    assert "abcdef12" in table  # v2 row's target_id[7:15]
    assert "—" in table  # v1 row's Target/ESS columns


# ---------------------------------------------------------------------------
# MultiStart likelihood_evals fix
# ---------------------------------------------------------------------------


class _FakeMaxLLSample:
    log_likelihood = -100.0


class _FakeSamples:
    total_samples = 5
    log_evidence = None
    weight_list = None
    parameter_lists = [[1.0], [2.0]]
    max_log_likelihood_sample = _FakeMaxLLSample()


class _FakeResult:
    samples = _FakeSamples()


def test_multi_start_likelihood_evals_fix():
    metrics = collect_metrics(
        result=_FakeResult(),
        total_wall_s=10.0,
        viz_wall_s=1.0,
        is_multi_start=True,
        n_starts=256,
        multi_start_total_steps=178,
    )
    assert metrics.likelihood_evals == 45568
    assert metrics.gradient_evals == 45568
    assert metrics.stored_samples == 5  # unaffected — the raw storage count


def test_multi_start_falls_back_without_captured_total_steps():
    """A MultiStart run whose search_internal capture is unavailable falls
    back to the old (wrong-but-not-crashing) total_samples reading."""
    metrics = collect_metrics(
        result=_FakeResult(),
        total_wall_s=10.0,
        viz_wall_s=1.0,
        is_multi_start=True,
        n_starts=None,
        multi_start_total_steps=None,
    )
    assert metrics.likelihood_evals == 5


def test_nested_sampler_likelihood_evals_unaffected():
    metrics = collect_metrics(result=_FakeResult(), total_wall_s=10.0, viz_wall_s=1.0)
    assert metrics.likelihood_evals == 5
    assert metrics.gradient_evals is None


# ---------------------------------------------------------------------------
# Kish ESS
# ---------------------------------------------------------------------------


def test_kish_ess_uniform_weights_gives_n():
    assert _kish_ess([1.0] * 10) == pytest.approx(10.0)


def test_kish_ess_one_hot_gives_one():
    assert _kish_ess([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_kish_ess_none_for_no_weights():
    assert _kish_ess(None) is None


# ---------------------------------------------------------------------------
# penalty_at_best (Phase 4 / issue #182)
# ---------------------------------------------------------------------------


class _FakeAnalysisWithPositions:
    """Minimal stand-in for an ``AnalysisImaging`` carrying a positions list.

    Only the two attributes ``_penalty_at_best`` touches are provided —
    deliberately, so the test proves the readout's arithmetic and its
    absent-vs-null discipline rather than re-testing PyAutoLens's penalty.
    """

    def __init__(self, penalty, positions_likelihood_list=("sentinel",)):
        self._penalty = penalty
        self.positions_likelihood_list = list(positions_likelihood_list) or None

    def log_likelihood_penalty_from(self, instance):
        if isinstance(self._penalty, Exception):
            raise self._penalty
        return self._penalty


def test_penalty_at_best_decomposes_the_recorded_likelihood():
    from searches._runner import _penalty_at_best

    block = _penalty_at_best(_FakeAnalysisWithPositions(500.0), object(), -1500.0)
    assert block["positions_penalty"] == pytest.approx(500.0)
    assert block["log_likelihood_penalised"] == pytest.approx(-1500.0)
    # The search recorded fit - penalty; the unpenalised fit is that plus back.
    assert block["log_likelihood_unpenalised"] == pytest.approx(-1000.0)
    assert block["delta_log_likelihood"] == pytest.approx(-500.0)


def test_penalty_at_best_zero_penalty_is_recorded_not_omitted():
    """0.0 (best model traced inside the threshold) is a REAL measurement and
    must be distinguishable from 'positions off'."""
    from searches._runner import _penalty_at_best

    block = _penalty_at_best(_FakeAnalysisWithPositions(0.0), object(), -1500.0)
    assert block["positions_penalty"] == pytest.approx(0.0)
    assert block["log_likelihood_unpenalised"] == pytest.approx(-1500.0)


def test_penalty_at_best_none_when_positions_off():
    from searches._runner import _penalty_at_best

    analysis = _FakeAnalysisWithPositions(500.0, positions_likelihood_list=())
    assert _penalty_at_best(analysis, object(), -1500.0) is None


def test_penalty_at_best_none_without_a_best_instance():
    from searches._runner import _penalty_at_best

    assert _penalty_at_best(_FakeAnalysisWithPositions(500.0), None, -1500.0) is None


def test_penalty_at_best_records_the_error_rather_than_killing_the_run():
    from searches._runner import _penalty_at_best

    block = _penalty_at_best(_FakeAnalysisWithPositions(RuntimeError("boom")), object(), -1500.0)
    assert "boom" in block["error"]
    assert "positions_penalty" not in block
