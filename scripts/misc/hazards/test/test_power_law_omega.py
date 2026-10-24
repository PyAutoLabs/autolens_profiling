"""Tests for the PowerLaw omega convergence research probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in sys.path:
    sys.path.insert(0, str(_misc))

from hazards.power_law_omega import (  # noqa: E402
    default_prior_annulus_mass,
    omega_exact,
    omega_series,
)


def test_long_series_converges_to_hyp2f1():
    angles = np.linspace(-np.pi, np.pi, 65, endpoint=False)
    eiphi = np.exp(1j * angles)
    exact = omega_exact(eiphi, internal_slope=1.2, factor=0.8)
    series = omega_series(eiphi, internal_slope=1.2, factor=0.8, n_terms=160)
    np.testing.assert_allclose(series, exact, rtol=1.0e-11, atol=1.0e-11)


def test_twenty_terms_is_inaccurate_in_reachable_prior_region():
    angles = np.linspace(-np.pi, np.pi, 65, endpoint=False)
    eiphi = np.exp(1j * angles)
    exact = omega_exact(eiphi, internal_slope=1.2, factor=0.9)
    series = omega_series(eiphi, internal_slope=1.2, factor=0.9, n_terms=20)
    relative_error = np.linalg.norm(series - exact) / np.linalg.norm(exact)
    assert relative_error > 1.0e-2
    assert default_prior_annulus_mass(0.9, 1.0) > 0.0


def test_valid_and_invalid_default_ell_comps_prior_mass_partition():
    valid = default_prior_annulus_mass(0.0, 1.0)
    assert 0.99 < valid < 1.0
    assert np.isclose(valid + (1.0 - valid), 1.0)
