"""Unit tests for the extraction of the fixed-lens-light system builders.

``FixedLightSystem`` and its two builders moved out of ``active_set_steps`` into
``fixed_light_system`` so a numba CPU cell can build the source-only system
without importing JAX. An extraction is only safe if two things hold, and these
tests are those two things:

1. **Nothing moved.** ``active_set_steps`` re-exports the four names *by
   identity*, and ``fixed_light_cpu_kernels._jacobi_scaled_np`` is an alias of
   the promoted function rather than a second copy — so every existing caller,
   test and pin still reaches the same objects. (The hard gate on this is
   ``test_active_set_steps.py`` passing unchanged; these tests say *why* it
   does.)
2. **The extraction achieved its point.** The new module imports with
   ``sys.modules["jax"] = None``, in a subprocess, which is the only way to
   prove it — an in-process check would pass on an already-imported JAX.

The third thing tested is that ``scaling="numpy"`` is the same arithmetic as
``scaling="jax"``: if it were not, a numba row would be measuring a different
preconditioner to every pinned JAX row it is compared against.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_system.py
"""

from __future__ import annotations

import subprocess
import sys as _sys
import textwrap
from pathlib import Path as _Path

import numpy as np
import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import fixed_light_system as fls  # noqa: E402


def _spd(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A well-conditioned SPD matrix and a data vector, fp64."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n + 8, n))
    Q = A.T @ A + 0.5 * np.eye(n)
    return Q, rng.standard_normal(n)


# ---------------------------------------------------------------------------
# 1. Nothing moved
# ---------------------------------------------------------------------------


def test_active_set_steps_still_re_exports_all_four_names():
    """The four names are reachable off ``active_set_steps``, and are the same objects."""
    from likelihood_breakdown import active_set_steps as ass

    for name in (
        "FixedLightSystem",
        "fixed_light_system_from",
        "jacobi_scaled_np",
        "linear_system_from",
    ):
        assert hasattr(ass, name), f"active_set_steps no longer exports {name}"
        assert getattr(ass, name) is getattr(fls, name), f"{name} is a copy, not the same object"
        assert name in ass.__all__


def test_cpu_kernels_private_name_is_an_alias_of_the_promoted_function():
    """``_jacobi_scaled_np`` is the promoted function itself, not a second copy.

    Two implementations of one preconditioner is how a CPU row and the row it is
    compared against drift apart without either changing.
    """
    from likelihood_breakdown import fixed_light_cpu_kernels as flck

    assert flck._jacobi_scaled_np is fls.jacobi_scaled_np


# ---------------------------------------------------------------------------
# 2. The extraction achieved its point
# ---------------------------------------------------------------------------


def test_the_module_imports_with_jax_blocked():
    """A subprocess that forbids JAX must still import the builders.

    ``sys.modules["jax"] = None`` makes any ``import jax`` raise, so this fails
    loudly if the module — or anything on its import path, including the
    ``likelihood_breakdown`` package ``__init__`` — reaches for JAX at import
    time. Run out of process because an in-process check would pass trivially:
    JAX is already imported by the time this test file is collected.
    """
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_misc)!r})
        sys.modules["jax"] = None
        sys.modules["jax.numpy"] = None

        from likelihood_breakdown import call_accounting, fixed_light_system

        assert "jax" not in [m for m in sys.modules if sys.modules[m] is not None]
        import numpy as np
        Q = np.eye(4) * 4.0
        q = np.arange(4, dtype=float)
        Qs, qs, d = fixed_light_system.jacobi_scaled_np(Q, q)
        assert np.allclose(np.diag(Qs), 1.0)
        assert call_accounting.snapshot() == {{}}
        print("OK")
        """
    )
    out = subprocess.run(
        [_sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    assert "OK" in out.stdout


# ---------------------------------------------------------------------------
# 3. The numpy scaling is the JAX scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jacobi_scaled_np_agrees_with_the_jax_twin(seed):
    """The promoted numpy scaling reproduces ``reconstruction_steps.jacobi_scaled``."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from likelihood_breakdown import reconstruction_steps

    Q, q = _spd(150, seed=seed)

    Q_np, q_np, d_np = fls.jacobi_scaled_np(Q, q)
    Q_jx, q_jx, d_jx = reconstruction_steps.jacobi_scaled(
        jnp.asarray(Q, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)
    )

    assert np.allclose(np.diag(Q_np), 1.0, rtol=0, atol=1e-12)
    assert np.allclose(Q_np, np.asarray(Q_jx, dtype=float), rtol=1e-12, atol=1e-14)
    assert np.allclose(q_np, np.asarray(q_jx, dtype=float), rtol=1e-12, atol=1e-14)
    assert np.allclose(d_np, np.asarray(d_jx, dtype=float), rtol=1e-12, atol=1e-14)


@pytest.fixture(scope="module")
def tiny_fit():
    """The 67-parameter fixture of ``test_active_set_steps.py``: 3 linear Gaussians + 8x8."""
    autolens = pytest.importorskip("autolens")
    al = autolens

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
    return fit, dataset


def test_the_two_scalings_build_the_same_system(tiny_fit):
    """``scaling="numpy"`` and ``scaling="jax"`` are the same system to 1e-12.

    The numba cell builds with ``"numpy"``; every pinned number it is compared
    against was built with ``"jax"``. This is the statement that the swap costs
    nothing.
    """
    fit, dataset = tiny_fit

    jax_system = fls.linear_system_from(fit, dataset, name="jax", scaling="jax")
    np_system = fls.linear_system_from(fit, dataset, name="numpy", scaling="numpy")

    assert np.allclose(np_system.Q, jax_system.Q, rtol=1e-12, atol=1e-14)
    assert np.allclose(np_system.q, jax_system.q, rtol=1e-12, atol=1e-14)
    assert np.allclose(np_system.d_scale, jax_system.d_scale, rtol=1e-12, atol=1e-14)
    assert np_system.n_params == jax_system.n_params
    assert np.array_equal(np_system.edge_zero_mask, jax_system.edge_zero_mask)


def test_an_unknown_scaling_is_an_error_not_a_silent_default(tiny_fit):
    fit, dataset = tiny_fit
    with pytest.raises(ValueError, match="scaling must be"):
        fls.linear_system_from(fit, dataset, name="bad", scaling="float32")


def test_an_unknown_sparse_operator_mode_is_an_error(tiny_fit):
    fit, dataset = tiny_fit
    with pytest.raises(ValueError, match="sparse_operator must be"):
        fls.fixed_light_system_from(fit, dataset, name="bad", sparse_operator="jax")
