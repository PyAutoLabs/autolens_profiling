"""Unit tests for the ``SEARCHES_SMC_*`` plumbing (Phase 7, issue #196).

Three load-bearing claims are under test.

**The collision guard.** ``af.SMC.__identifier_fields__`` is
``(num_particles, kernel, num_mcmc_steps, num_integration_steps, target_ess,
inverse_mass_matrix)`` — ``seed`` and ``initializer`` are NOT in it. Two arms
differing only in seed, or two warm arms from different sources at identical
particle settings, would resolve to one output directory and the second's
``fit()`` would return the first's ``.completed`` result. That is the defect
RAL job 340576 exposed for ``log_det_method`` (20 arms, 10 directories).

**The log-det arm tag, on every builder.** #175 fixed that defect for the
MultiStart path only. ``Nautilus`` / ``NSS`` / ``BlackJAXNUTS`` / ``SMC``
identifier fields carry no log-det field either, so the nested-sampler slogdet
A/B this wave adds would have hit it again. The tag must appear on all five
builders when ``SEARCHES_LOG_DET_METHOD`` is set — and must appear on NONE of
them when it is unset, or every recorded output path moves.

**The eval accounting.** ``af.SMC`` records no evaluation counter and its
``total_samples`` is ``num_particles``, so reading evals off the stored count
is issue #177's error a third time.

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_smc.py -q
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

import numpy as np
import pytest
from searches._metrics import collect_metrics
from searches._samplers import (
    SAMPLER_BUILDERS,
    build_smc,
    log_det_arm_tag,
    multi_start_unique_tag,
    nuts_unique_tag,
    smc_kernel,
    smc_likelihood_evals,
    smc_mass_mode,
    smc_seed,
    smc_settings,
    smc_unique_tag,
)

pytestmark = pytest.mark.filterwarnings("ignore")

_SMC_ENV = (
    "SEARCHES_SMC_NUM_PARTICLES",
    "SEARCHES_SMC_KERNEL",
    "SEARCHES_SMC_NUM_MCMC_STEPS",
    "SEARCHES_SMC_NUM_INTEGRATION_STEPS",
    "SEARCHES_SMC_TARGET_ESS",
    "SEARCHES_SMC_STEP_SIZE",
    "SEARCHES_SMC_WHITEN_INFLATE",
    "SEARCHES_SMC_MAX_STEPS",
    "SEARCHES_SMC_BATCH_SIZE",
    "SEARCHES_SMC_MASS",
    "SEARCHES_SMC_WARM_FROM",
    "SEARCHES_SMC_JITTER",
    "SEARCHES_SMC_WARM_SCALE",
    "SEARCHES_SEED",
)

_CELL = dict(
    sampler="smc",
    dataset_class="imaging",
    model_type="mge",
    instrument="hst",
    config_name="unit_test",
    use_jax=True,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a fully unset SMC env.

    Without this a stray ``SEARCHES_SEED`` from another test (or the ambient
    shell) would silently change the tag under test.
    """
    for name in _SMC_ENV:
        monkeypatch.delenv(name, raising=False)
    # Positions and log_det compose into the tag too; keep them off so the
    # assertions below are about the SMC knobs alone.
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    monkeypatch.delenv("SEARCHES_LOG_DET_METHOD", raising=False)


# ---------------------------------------------------------------------------
# registration + env-var parsing / validation
# ---------------------------------------------------------------------------


def test_smc_is_registered():
    assert SAMPLER_BUILDERS["smc"] is build_smc


def test_kernel_defaults_to_mala_and_validates(monkeypatch):
    assert smc_kernel() == "mala"
    monkeypatch.setenv("SEARCHES_SMC_KERNEL", "HMC")
    assert smc_kernel() == "hmc"
    monkeypatch.setenv("SEARCHES_SMC_KERNEL", "nuts")
    with pytest.raises(ValueError, match="SEARCHES_SMC_KERNEL"):
        smc_kernel()


def test_mass_mode_defaults_to_none_and_validates(monkeypatch):
    assert smc_mass_mode() == "none"
    monkeypatch.setenv("SEARCHES_SMC_MASS", "PRIOR_SCALED")
    assert smc_mass_mode() == "prior_scaled"
    monkeypatch.setenv("SEARCHES_SMC_MASS", "diagonal")
    # `diagonal` is what BlackJAXNUTS accepts and af.SMC rejects; it must not
    # silently resolve to a cold run here either.
    with pytest.raises(ValueError, match="SEARCHES_SMC_MASS"):
        smc_mass_mode()


def test_seed_defaults_to_the_library_value_and_is_concrete(monkeypatch):
    assert smc_seed() == 42
    monkeypatch.setenv("SEARCHES_SEED", "0")
    assert smc_seed() == 0


def test_settings_record_every_knob_including_the_cold_default(monkeypatch):
    monkeypatch.setenv("SEARCHES_SEED", "0")
    settings = smc_settings()
    for key in (
        "num_particles",
        "kernel",
        "num_mcmc_steps",
        "num_integration_steps",
        "target_ess",
        "step_size",
        "whiten_inflate",
        "max_smc_steps",
        "seed",
        "warm_start",
        "positions",
    ):
        assert key in settings, key
    # The warm-start block is present even when cold, so a cold and a warm row
    # of the same cell are never ambiguous in the artifact.
    assert settings["warm_start"]["enabled"] is False
    assert settings["warm_start"]["inverse_mass_matrix_kind"] == "none"
    assert settings["warm_start"]["prior_scale"] is None


# ---------------------------------------------------------------------------
# collision guard (RAL 340576's defect, applied to SMC)
# ---------------------------------------------------------------------------


def test_seed_enters_the_unique_tag(monkeypatch):
    monkeypatch.setenv("SEARCHES_SEED", "0")
    seed_0 = smc_unique_tag()
    monkeypatch.setenv("SEARCHES_SEED", "1")
    assert smc_unique_tag() != seed_0


def test_warm_and_cold_arms_are_disjoint(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCHES_SEED", "0")
    cold_tag = smc_unique_tag()

    warm_dir = tmp_path / "prodigy_fit"
    (warm_dir / "files").mkdir(parents=True)
    monkeypatch.setenv("SEARCHES_SMC_WARM_FROM", str(warm_dir))
    monkeypatch.setenv("SEARCHES_SMC_MASS", "prior_scaled")
    warm_tag = smc_unique_tag()

    assert cold_tag.endswith("cold")
    assert "warm" in warm_tag and "cold" not in warm_tag
    assert warm_tag != cold_tag


def test_two_different_warm_sources_get_different_tags(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCHES_SEED", "0")
    monkeypatch.setenv("SEARCHES_SMC_MASS", "prior_scaled")
    tags = set()
    for name in ("source_a", "source_b"):
        source = tmp_path / name
        (source / "files").mkdir(parents=True)
        monkeypatch.setenv("SEARCHES_SMC_WARM_FROM", str(source))
        tags.add(smc_unique_tag())
    assert len(tags) == 2


def test_kernel_and_mass_mode_are_in_the_tag(monkeypatch, tmp_path):
    """mala_warm / hmc_warm / mala_cold must be three directories, not two."""
    monkeypatch.setenv("SEARCHES_SEED", "0")
    warm_dir = tmp_path / "prodigy_fit"
    (warm_dir / "files").mkdir(parents=True)

    tags = []
    for kernel, mass in (("mala", "prior_scaled"), ("hmc", "prior_scaled"), ("mala", "none")):
        monkeypatch.setenv("SEARCHES_SMC_KERNEL", kernel)
        monkeypatch.setenv("SEARCHES_SMC_MASS", mass)
        if mass == "none":
            monkeypatch.delenv("SEARCHES_SMC_WARM_FROM", raising=False)
        else:
            monkeypatch.setenv("SEARCHES_SMC_WARM_FROM", str(warm_dir))
        tags.append(smc_unique_tag())
    assert len(set(tags)) == 3


# ---------------------------------------------------------------------------
# the log_det arm tag, on EVERY builder (#196; #175 fixed one of five)
# ---------------------------------------------------------------------------


def test_log_det_arm_tag_is_none_when_unset():
    """Byte-identity: an unset env must move no recorded output path."""
    assert log_det_arm_tag() is None


def test_log_det_arm_tag_normalises_case(monkeypatch):
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "Cholesky")
    assert log_det_arm_tag() == "ld_cholesky"


def test_log_det_arm_reaches_every_sampler_tag(monkeypatch):
    """The nested-sampler A/B this wave adds needs the nested builders tagged.

    Before #196 only ``multi_start_unique_tag`` composed it, so a two-arm
    Nautilus cholesky-vs-slogdet job would have produced one output directory
    and two results JSONs with different basenames and identical contents.
    """
    monkeypatch.setenv("SEARCHES_SEED", "0")
    before = {
        "multi_start": multi_start_unique_tag("imaging", "mge"),
        "nuts": nuts_unique_tag(),
        "smc": smc_unique_tag(),
    }
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "slogdet")
    after = {
        "multi_start": multi_start_unique_tag("imaging", "mge"),
        "nuts": nuts_unique_tag(),
        "smc": smc_unique_tag(),
    }
    for name, tag in after.items():
        assert tag is not None and tag.endswith("ld_slogdet"), name
        assert tag != before[name], name


def test_nautilus_and_nss_arms_are_disjoint_under_a_log_det_ab(monkeypatch):
    """The exact submit this wave ships: two Nautilus arms, one env difference."""
    from searches._samplers import arm_unique_tag, build_nautilus
    from searches._setup import positions_arm_tag

    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "cholesky")
    cholesky = arm_unique_tag(positions_arm_tag(), log_det_arm_tag())
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "slogdet")
    slogdet = arm_unique_tag(positions_arm_tag(), log_det_arm_tag())
    assert cholesky != slogdet

    # And through the builder itself, which is what the submit actually calls.
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "cholesky")
    search_a = build_nautilus(
        sampler="nautilus",
        dataset_class="imaging",
        model_type="slam_source_pix_nn",
        instrument="hst",
        config_name="unit_test",
        use_jax=True,
    )
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "slogdet")
    search_b = build_nautilus(
        sampler="nautilus",
        dataset_class="imaging",
        model_type="slam_source_pix_nn",
        instrument="hst",
        config_name="unit_test",
        use_jax=True,
    )
    assert search_a.unique_tag != search_b.unique_tag


# ---------------------------------------------------------------------------
# construction guards
# ---------------------------------------------------------------------------


def test_numpy_config_raises():
    with pytest.raises(ValueError, match="JAX-native"):
        build_smc(**{**_CELL, "use_jax": False})


def test_warm_start_without_a_model_raises(monkeypatch, tmp_path):
    warm_dir = tmp_path / "fit"
    (warm_dir / "files").mkdir(parents=True)
    monkeypatch.setenv("SEARCHES_SMC_WARM_FROM", str(warm_dir))
    with pytest.raises(ValueError, match="without `model`"):
        build_smc(**_CELL)


def test_mass_result_without_a_warm_source_raises(monkeypatch):
    monkeypatch.setenv("SEARCHES_SMC_MASS", "result")
    with pytest.raises(ValueError, match="SEARCHES_SMC_WARM_FROM"):
        build_smc(**_CELL)


def test_mass_prior_scaled_without_a_warm_source_raises(monkeypatch):
    """`prior_scaled` centres on a previous best point; there isn't one cold."""
    monkeypatch.setenv("SEARCHES_SMC_MASS", "prior_scaled")
    with pytest.raises(ValueError, match="SEARCHES_SMC_WARM_FROM"):
        build_smc(**_CELL)


def test_prior_scaled_covariance_is_a_declared_diagonal():
    """It is a REFERENCE WIDTH, not a metric: diagonal, positive, and scaled."""
    import autofit as af
    import autolens as al
    from searches._samplers import smc_prior_scaled_covariance

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.IsothermalSph))
        )
    )
    cov = smc_prior_scaled_covariance(model, 0.1)
    assert cov.ndim == 1  # 1-D => af.SMC reads it as a covariance DIAGONAL
    assert cov.shape == (model.prior_count,)
    assert np.all(cov > 0)
    # Halving the scale quarters the covariance (it is a variance).
    assert np.allclose(smc_prior_scaled_covariance(model, 0.05), cov / 4.0)


# ---------------------------------------------------------------------------
# eval accounting (issue #177's error class, applied to SMC)
# ---------------------------------------------------------------------------


def test_derived_evals_for_the_mala_kernel():
    """256 particles stored; 256 * (1 + 40 * 5 * 1) = 51,456 evaluations."""
    info = {
        "num_particles": 256,
        "n_smc_steps": 40,
        "num_mcmc_steps": 5,
        "num_integration_steps": 8,
        "kernel": "mala",
    }
    assert smc_likelihood_evals(info) == 256 * (1 + 40 * 5 * 1)


def test_derived_evals_charge_hmc_for_its_leapfrog_steps():
    """The HMC arm costs num_integration_steps gradient evals per MCMC step."""
    info = {
        "num_particles": 256,
        "n_smc_steps": 40,
        "num_mcmc_steps": 5,
        "num_integration_steps": 8,
        "kernel": "hmc",
    }
    assert smc_likelihood_evals(info) == 256 * (1 + 40 * 5 * 8)


def test_derived_evals_return_none_on_an_incomplete_record():
    """A missing key must give ``None``, never a silently wrong count."""
    assert smc_likelihood_evals({}) is None
    assert smc_likelihood_evals({"num_particles": 256, "kernel": "mala"}) is None


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


def test_smc_evals_come_from_the_schedule_not_the_particle_count():
    """256 stored particles for 51,456 evaluations — a 201x under-report if
    read off ``total_samples``, which would flatter SMC against every nested
    row in the same table (issue #177's error class)."""
    result = _FakeResult(_FakeSamples(total_samples=256, weight_list=[1.0 / 256] * 256))
    metrics = collect_metrics(
        result=result,
        total_wall_s=600.0,
        viz_wall_s=0.0,
        smc_logl_evals=51_456,
    )
    assert metrics.likelihood_evals == 51_456
    assert metrics.gradient_evals == 51_456
    assert metrics.stored_samples == 256


def test_smc_keeps_the_kish_ess_unlike_nuts():
    """SMC weights are genuine importance weights, so Kish is correct here and
    no substitute is passed. A collapsed cloud must show up as a small ESS."""
    weights = [0.97] + [0.03 / 255] * 255
    result = _FakeResult(_FakeSamples(total_samples=256, weight_list=weights))
    metrics = collect_metrics(
        result=result,
        total_wall_s=600.0,
        viz_wall_s=0.0,
        smc_logl_evals=51_456,
    )
    assert metrics.kish_ess is not None
    assert metrics.kish_ess < 2.0  # not 256


def test_other_samplers_are_untouched_by_the_smc_argument():
    """A nested row's counters must not move because a kwarg was added."""
    result = _FakeResult(_FakeSamples(total_samples=4_000, weight_list=[1.0] * 4_000))
    metrics = collect_metrics(result=result, total_wall_s=600.0, viz_wall_s=0.0)
    assert metrics.likelihood_evals == 4_000
    assert metrics.gradient_evals is None
