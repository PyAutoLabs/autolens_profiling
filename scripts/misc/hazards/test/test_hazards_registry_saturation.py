"""Detector-registry and saturation discovery tests (no JAX)."""

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

from hazards._registry import resolve_subjects  # noqa: E402
from hazards.checks.saturation import (  # noqa: E402
    _truncated_normal_pairs,
    detect_plateaus,
)


def test_registry_resolves_component_matrix_likelihood_and_all_scopes():
    components = resolve_subjects("component")
    matrices = resolve_subjects("matrix")
    likelihoods = resolve_subjects("likelihood")
    assert {subject.subject for subject in components} == {"component"}
    assert {subject.subject for subject in matrices} == {"matrix"}
    assert {subject.subject for subject in likelihoods} == {"likelihood"}
    assert len(resolve_subjects("all")) == len(components) + len(matrices) + len(likelihoods)


def test_plateau_detector_finds_both_edges_without_threshold_special_cases():
    parameter = np.asarray((0.0, 1.0e-6, 5.0e-6, 1.0e-5, 0.5, 0.999, 1.0, 1.2))
    values = np.asarray((0.99999, 0.99999, 0.99999, 0.99998, 0.5, 0.0005, 0.0005, 0.0005))
    plateaus = detect_plateaus(parameter, values)
    assert [(plateau.start, plateau.end) for plateau in plateaus] == [
        (0.0, 5.0e-6),
        (0.999, 1.2),
    ]


@pytest.mark.parametrize(
    ("sigma", "expected"),
    ((0.3, 0.0022), (0.5, 0.051)),
)
def test_truncated_gaussian_prior_mass_matches_seed_result(sigma, expected):
    samples = _truncated_normal_pairs(200_000, sigma, seed=107)
    mass = np.mean(np.sum(samples * samples, axis=1) >= 1.0)
    assert mass == pytest.approx(expected, abs=6.0e-4)


def test_uniform_square_prior_mass_matches_geometric_result():
    rng = np.random.default_rng(107)
    samples = rng.uniform(-1.0, 1.0, size=(200_000, 2))
    mass = np.mean(np.sum(samples * samples, axis=1) >= 1.0)
    assert mass == pytest.approx(1.0 - np.pi / 4.0, abs=1.5e-3)
