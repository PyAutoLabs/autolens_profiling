"""Pure NumPy tests for the four numerical-hazard risk bases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in sys.path:
    sys.path.insert(0, str(_misc))

from hazards._measure import (  # noqa: E402
    epsilon_neighbourhood_measurement,
    error_curve_measurement,
    prior_mass_measurement,
    reachability_measurement,
    wilson_interval,
)


def test_wilson_interval_contains_observed_proportion():
    low, high = wilson_interval(22, 10_000)
    assert low < 0.0022 < high
    assert 0.0 <= low < high <= 1.0


def test_prior_mass_carries_sample_count_seed_and_confidence_interval():
    mask = np.asarray([True, False, False, True, False])
    measurement = prior_mass_measurement(
        mask,
        sample_count=5,
        seed=107,
        prior={"type": "synthetic"},
    )
    assert measurement.basis == "prior_mass"
    assert measurement.value == pytest.approx(0.4)
    assert measurement.details["successes"] == 2
    assert measurement.details["sample_count"] == 5
    assert measurement.details["seed"] == 107


def test_epsilon_neighbourhood_requires_explicit_ball_and_domain():
    measurement = epsilon_neighbourhood_measurement(
        epsilon=1.0e-3,
        mass=np.pi * 1.0e-6 / 4.0,
        centre=(0.0, 0.0),
        domain={"type": "uniform_square", "width": 2.0},
    )
    assert measurement.basis == "epsilon_neighbourhood"
    assert measurement.details["epsilon"] == 1.0e-3
    assert measurement.details["centre"] == [0.0, 0.0]


def test_reachability_deduplicates_paths_without_implying_scientific_impact():
    measurement = reachability_measurement(
        reachable_via=["jax_trace", "jax_trace"],
        blocked_paths={"numpy": "constructor_guard"},
    )
    assert measurement.value == 1.0
    assert measurement.details["reachable_via"] == ["jax_trace"]
    assert measurement.details["blocked_paths"] == {"numpy": "constructor_guard"}


def test_error_curve_reports_the_parameter_at_maximum_relative_error():
    parameter = np.asarray([0.0, 0.5, 0.9])
    reference = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    candidate = np.asarray([[1.0, 0.0], [1.01, 0.0], [1.2, 0.0]])
    measurement = error_curve_measurement(
        parameter,
        reference,
        candidate,
        parameter_name="factor",
    )
    assert measurement.basis == "error_curve"
    assert measurement.value == pytest.approx(0.2)
    assert measurement.details["maximum_parameter"] == 0.9
