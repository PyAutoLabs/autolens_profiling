"""Unit tests for the ``SEARCHES_NUTS_*`` plumbing (Phase 6, issue #187).

The load-bearing claim under test is the **collision guard**.
``af.BlackJAXNUTS.__identifier_fields__`` is
``(num_warmup, num_samples, num_chains, inverse_mass_matrix)`` — ``seed`` and
``initializer`` are NOT in it. So two arms differing only in seed, or a warm
and a cold arm at identical chains/warmup/samples/mass, would resolve to one
output directory and the second's ``fit()`` would return the first's
``.completed`` result. That is the defect RAL job 340576 exposed for
``log_det_method`` (20 arms, 10 directories), and since warm-vs-cold IS this
cell's experiment, an untagged warm arm would silently report the cold arm's
numbers.

Also covers: env-var parsing/validation (an unrecognised mass mode raises
rather than falling back silently), the recorded ``nuts_settings()`` block,
the warm-start path resolver's refusals, and the eval/ESS substitutions in
``collect_metrics`` that keep a NUTS row comparable with the nested rows.

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_nuts.py -q
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
from searches._metrics import collect_metrics
from searches._samplers import (
    SAMPLER_BUILDERS,
    _resolve_warm_start_files,
    assert_disjoint_output_paths,
    build_nuts,
    nuts_mass_mode,
    nuts_max_doublings,
    nuts_seed,
    nuts_settings,
    nuts_unique_tag,
)

pytestmark = pytest.mark.filterwarnings("ignore")

_NUTS_ENV = (
    "SEARCHES_NUTS_NUM_CHAINS",
    "SEARCHES_NUTS_NUM_WARMUP",
    "SEARCHES_NUTS_NUM_SAMPLES",
    "SEARCHES_NUTS_TARGET_ACCEPT",
    "SEARCHES_NUTS_MASS",
    "SEARCHES_NUTS_WARM_FROM",
    "SEARCHES_NUTS_JITTER",
    "SEARCHES_NUTS_MAX_DOUBLINGS",
    "SEARCHES_SEED",
)

_CELL = dict(
    sampler="nuts",
    dataset_class="imaging",
    model_type="mge",
    instrument="hst",
    config_name="unit_test",
    use_jax=True,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a fully unset NUTS env.

    Without this a stray ``SEARCHES_SEED`` from another test (or the ambient
    shell) would silently change the tag under test.
    """
    for name in _NUTS_ENV:
        monkeypatch.delenv(name, raising=False)
    # Positions compose into the tag too; keep them off so the assertions
    # below are about the NUTS knobs alone.
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)


# ---------------------------------------------------------------------------
# registration + env-var parsing / validation
# ---------------------------------------------------------------------------


def test_nuts_is_registered():
    assert SAMPLER_BUILDERS["nuts"] is build_nuts


def test_mass_mode_defaults_to_diag_and_validates(monkeypatch):
    assert nuts_mass_mode() == "diag"
    monkeypatch.setenv("SEARCHES_NUTS_MASS", "DENSE")  # normalised
    assert nuts_mass_mode() == "dense"
    monkeypatch.setenv("SEARCHES_NUTS_MASS", "identity")
    with pytest.raises(ValueError, match="SEARCHES_NUTS_MASS"):
        nuts_mass_mode()


def test_seed_defaults_to_the_library_value_and_is_concrete(monkeypatch):
    # Unlike multi_start_seed/nautilus_seed, this never returns None:
    # af.BlackJAXNUTS.seed is a plain int.
    assert nuts_seed() == 42
    monkeypatch.setenv("SEARCHES_SEED", "7")
    assert nuts_seed() == 7


def test_max_doublings_defaults_to_the_stan_ceiling(monkeypatch):
    assert nuts_max_doublings() == 10
    monkeypatch.setenv("SEARCHES_NUTS_MAX_DOUBLINGS", "3")
    assert nuts_max_doublings() == 3


# ---------------------------------------------------------------------------
# the recorded settings block
# ---------------------------------------------------------------------------


def test_settings_record_every_knob_including_the_cold_default(monkeypatch):
    monkeypatch.setenv("SEARCHES_NUTS_NUM_CHAINS", "8")
    monkeypatch.setenv("SEARCHES_NUTS_NUM_WARMUP", "200")
    monkeypatch.setenv("SEARCHES_NUTS_NUM_SAMPLES", "300")
    settings = nuts_settings()
    assert settings["num_chains"] == 8
    assert settings["num_warmup"] == 200
    assert settings["num_samples"] == 300
    assert settings["inverse_mass_matrix"] == "diag"
    assert settings["inverse_mass_matrix_kind"] == "diagonal"
    # Always present, including when off: a cold and a warm run of the same
    # cell must never be ambiguous in the artifact.
    assert settings["warm_start"] == {
        "enabled": False,
        "source": None,
        "point": "max_log_likelihood",
        "jitter": None,
        "n_points": None,
    }


def test_profiling_default_is_four_chains_not_the_library_single_chain():
    # A single chain cannot produce the split-R-hat Gate C is stated in terms of.
    assert nuts_settings()["num_chains"] == 4


# ---------------------------------------------------------------------------
# THE COLLISION GUARD — the reason nuts_unique_tag exists
# ---------------------------------------------------------------------------


def test_seed_enters_the_output_path_and_identifier(monkeypatch):
    monkeypatch.setenv("SEARCHES_SEED", "0")
    seed_0 = build_nuts(**_CELL)
    monkeypatch.setenv("SEARCHES_SEED", "1")
    seed_1 = build_nuts(**_CELL)
    # `seed` is not an identifier field, so without the tag these two arms of a
    # reliability scan share one directory and the second resumes the first.
    assert_disjoint_output_paths(seed_0, seed_1)


def test_warm_and_cold_arms_are_disjoint(monkeypatch, tmp_path):
    """The warm/cold A/B is the experiment; it must not collide with itself."""
    monkeypatch.setenv("SEARCHES_SEED", "0")
    cold_tag = nuts_unique_tag()

    warm_dir = tmp_path / "prodigy_fit"
    (warm_dir / "files").mkdir(parents=True)
    monkeypatch.setenv("SEARCHES_NUTS_WARM_FROM", str(warm_dir))
    warm_tag = nuts_unique_tag()

    assert cold_tag.endswith("cold")
    assert "warm" in warm_tag and "cold" not in warm_tag
    assert warm_tag != cold_tag


def test_two_different_warm_sources_get_different_tags(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCHES_SEED", "0")
    tags = set()
    for name in ("source_a", "source_b"):
        source = tmp_path / name
        (source / "files").mkdir(parents=True)
        monkeypatch.setenv("SEARCHES_NUTS_WARM_FROM", str(source))
        tags.add(nuts_unique_tag())
    assert len(tags) == 2


def test_capped_smoke_run_cannot_overwrite_an_uncapped_measurement(monkeypatch):
    """A lowered doubling cap is tagged, so a smoke can never be mistaken for
    (or overwrite) a real uncapped NUTS row."""
    monkeypatch.setenv("SEARCHES_SEED", "0")
    uncapped = nuts_unique_tag()
    monkeypatch.setenv("SEARCHES_NUTS_MAX_DOUBLINGS", "1")
    capped = nuts_unique_tag()
    assert capped != uncapped
    assert "md1" in capped
    # Setting it to the default value EXPLICITLY still tags: the env being set
    # at all is what marks the run as deliberately capped.
    monkeypatch.setenv("SEARCHES_NUTS_MAX_DOUBLINGS", "10")
    assert nuts_unique_tag() != uncapped


def test_mass_mode_is_hashed_by_the_library_not_the_tag(monkeypatch):
    """`inverse_mass_matrix` IS an identifier field, so diag/dense separate on
    their own — the tag deliberately does not duplicate it."""
    monkeypatch.setenv("SEARCHES_SEED", "0")
    diag = build_nuts(**_CELL)
    monkeypatch.setenv("SEARCHES_NUTS_MASS", "dense")
    dense = build_nuts(**_CELL)
    assert diag.unique_tag == dense.unique_tag
    assert diag.paths.identifier != dense.paths.identifier


# ---------------------------------------------------------------------------
# builder guards — every one raises rather than silently degrading
# ---------------------------------------------------------------------------


def test_numpy_config_raises():
    with pytest.raises(ValueError, match="JAX-native"):
        build_nuts(**{**_CELL, "use_jax": False})


def test_warm_start_without_a_model_raises(monkeypatch, tmp_path):
    """The initializer keys start points on the TARGET model's Prior objects.
    Built without one, every lookup would miss and the search would fall back
    to prior defaults — a 'warm' run that was cold, with nothing in the
    artifact to say so."""
    monkeypatch.setenv("SEARCHES_NUTS_WARM_FROM", str(tmp_path))
    with pytest.raises(ValueError, match="without `model`"):
        build_nuts(**_CELL)


def test_mass_result_without_a_warm_source_raises(monkeypatch):
    monkeypatch.setenv("SEARCHES_NUTS_MASS", "result")
    with pytest.raises(ValueError, match="requires SEARCHES_NUTS_WARM_FROM"):
        build_nuts(**_CELL)


# ---------------------------------------------------------------------------
# warm-start path resolution
# ---------------------------------------------------------------------------


def test_warm_path_must_exist(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        _resolve_warm_start_files(tmp_path / "nope")


def test_warm_path_rejects_an_incomplete_fit(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="missing"):
        _resolve_warm_start_files(tmp_path)


def test_warm_path_resolves_a_single_identifier_dir(tmp_path):
    files = tmp_path / "abc123" / "files"
    files.mkdir(parents=True)
    for name in ("model.json", "samples.csv", "samples_info.json"):
        (files / name).write_text("{}")
    assert _resolve_warm_start_files(tmp_path) == files
    # ...and naming the fit dir or the files dir directly works too.
    assert _resolve_warm_start_files(tmp_path / "abc123") == files
    assert _resolve_warm_start_files(files) == files


def test_warm_path_refuses_to_choose_between_two_fits(tmp_path):
    """Silently picking one of several completed fits would warm-start from an
    arm nobody chose."""
    for name in ("abc123", "def456"):
        files = tmp_path / name / "files"
        files.mkdir(parents=True)
        for fname in ("model.json", "samples.csv", "samples_info.json"):
            (files / fname).write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_warm_start_files(tmp_path)


# ---------------------------------------------------------------------------
# eval / ESS accounting (issue #177's error class, applied to NUTS)
# ---------------------------------------------------------------------------


class _FakeSamples:
    def __init__(self, total_samples, weight_list):
        self.total_samples = total_samples
        self.weight_list = weight_list
        self.parameter_lists = [[0.0]] * total_samples
        self.log_evidence = float("nan")

    @property
    def max_log_likelihood_sample(self):
        raise AttributeError


class _FakeResult:
    def __init__(self, samples):
        self.samples = samples


def test_nuts_evals_come_from_integration_steps_not_stored_draws():
    """4 chains x 200 draws = 800 stored, but the trajectories cost 96,000
    leapfrog steps. Reading evals off the stored count would under-report by
    120x and flatter NUTS against every nested row in the same table."""
    result = _FakeResult(_FakeSamples(total_samples=800, weight_list=[1.0] * 800))
    metrics = collect_metrics(
        result=result,
        total_wall_s=600.0,
        viz_wall_s=0.0,
        nuts_logl_evals=96_000,
        nuts_ess=120.0,
    )
    assert metrics.likelihood_evals == 96_000
    assert metrics.gradient_evals == 96_000
    # The stored count is kept, distinctly, rather than overwritten.
    assert metrics.stored_samples == 800


def test_nuts_ess_is_rank_normalised_not_kish():
    """NUTS weights are all 1.0, so Kish degenerates to the raw draw count and
    would report ESS=800 for a chain whose real ESS is 120."""
    result = _FakeResult(_FakeSamples(total_samples=800, weight_list=[1.0] * 800))
    metrics = collect_metrics(
        result=result,
        total_wall_s=600.0,
        viz_wall_s=0.0,
        nuts_logl_evals=96_000,
        nuts_ess=120.0,
    )
    assert metrics.kish_ess == 120.0
    assert metrics.evals_per_ess == pytest.approx(800.0)


def test_other_samplers_are_untouched_by_the_nuts_arguments():
    """Both substitutions are opt-in: a nested row's numbers must not move."""
    result = _FakeResult(_FakeSamples(total_samples=800, weight_list=[1.0] * 800))
    metrics = collect_metrics(result=result, total_wall_s=600.0, viz_wall_s=0.0)
    assert metrics.likelihood_evals == 800
    assert metrics.gradient_evals is None
    assert metrics.kish_ess == pytest.approx(800.0)  # uniform weights -> Kish = n
