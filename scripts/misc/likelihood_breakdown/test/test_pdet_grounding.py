"""The cells' ``ln det(H)`` is the library's (PyAutoArray#682 pseudo-determinant).

Every likelihood-breakdown cell computes ``ln det(H)`` through
``reconstruction_steps.log_det_regularization_from(reg_reduced, grounding_from(inversion))``.
These tests pin that to ``AbstractInversion.log_det_regularization_matrix_term``,
pin the JAX-free copy in ``fixed_light_system`` to the shared helper, and check
the grounded log det runs under ``jax.jit`` (the cells time it there).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import fixed_light_system, reconstruction_steps  # noqa: E402


@pytest.fixture(scope="module")
def inversion():
    """Three linear lens Gaussians (unregularized) plus an 8x8 Constant-regularized mesh."""
    al = pytest.importorskip("autolens")

    grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.2)
    psf = al.Convolver.from_gaussian(shape_native=(5, 5), sigma=0.3, pixel_scales=0.2)
    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, background_sky_level=0.1, noise_seed=1
    )
    truth = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.Sersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5),
                mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            al.Galaxy(
                redshift=1.0,
                bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0, effective_radius=0.2),
            ),
        ]
    )
    dataset = simulator.via_tracer_from(tracer=truth, grid=grid)
    dataset = dataset.apply_mask(
        mask=al.Mask2D.circular(shape_native=(30, 30), pixel_scales=0.2, radius=2.2)
    )
    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_basis.Basis(
            profile_list=[
                al.lp_linear.Gaussian(centre=(0.0, 0.0), sigma=sigma) for sigma in (0.2, 0.5, 1.0)
            ]
        ),
        mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=(8, 8)),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )
    fit = al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=[lens, source]),
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    return fit.inversion


def test_grounded_log_det_is_the_library_term(inversion):
    grounding = reconstruction_steps.grounding_from(inversion)
    reg_reduced = jnp.asarray(np.asarray(inversion.regularization_matrix_reduced), jnp.float64)
    ours = float(reconstruction_steps.log_det_regularization_from(reg_reduced, grounding))
    assert ours == pytest.approx(float(inversion.log_det_regularization_matrix_term), rel=1e-10)


def test_constant_regularization_is_grounded(inversion):
    """``Constant`` has a constant null mode, so it must ground (unless the library predates #682)."""
    if not hasattr(inversion, "_intrinsic_regularization_blocks"):
        pytest.skip("installed PyAutoArray predates #682 (no pseudo-determinant)")
    keep, log_block_sizes = reconstruction_steps.grounding_from(inversion)
    n = np.asarray(inversion.regularization_matrix_reduced).shape[0]
    assert len(keep) == n - 1
    assert log_block_sizes == pytest.approx(np.log(64))


def test_jax_free_copy_matches_the_shared_helper(inversion):
    shared = reconstruction_steps.grounding_from(inversion)
    copy = fixed_light_system._grounding_from(inversion)
    if shared is None:
        assert copy is None
        return
    assert np.array_equal(shared[0], copy[0])
    assert shared[1] == copy[1]


def test_grounded_log_det_runs_under_jit(inversion):
    grounding = reconstruction_steps.grounding_from(inversion)
    reg_reduced = jnp.asarray(np.asarray(inversion.regularization_matrix_reduced), jnp.float64)
    eager = reconstruction_steps.log_det_regularization_from(reg_reduced, grounding)
    jitted = jax.jit(lambda h: reconstruction_steps.log_det_regularization_from(h, grounding))(
        reg_reduced
    )
    assert float(jitted) == pytest.approx(float(eager), rel=1e-12)
