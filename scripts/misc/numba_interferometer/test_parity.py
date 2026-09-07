"""Parity pins for the recovered numba w-tilde interferometer likelihood.

Run from the repository root::

    python -m pytest scripts/misc/numba_interferometer/test_parity.py -q

What is pinned
--------------
1. The preload. Both ``NumbaPreload`` routes must reproduce
   ``nufft_precision_operator_via_np_from`` — the array today's
   ``apply_sparse_operator`` builds — elementwise at ``rtol=1e-10``.
2. The dense oracle. On a small mask, the scatter kernel's ``F`` must equal
   ``Mᵀ W~ M`` formed explicitly from ``w_tilde_via_preload_from``. This is the
   pin that has no shared code with the thing it checks.
3. The modern path. ``D``, ``F``, the reconstruction and the log evidence must
   match ``InversionInterferometerSparse`` (the JAX/FFT successor) at
   ``rtol=1e-6``.
4. The preconditions. Multiple mappers and ``over_sample_size != 1`` must raise,
   not silently produce a different ``F``.
5. The control. Scaling the kernel's ``preload * w0 * w1`` product by 1.01 must
   make pins 2 and 3 fail — a pin that cannot fail is not a pin.

The comparator ``InversionInterferometerSparse`` runs its curvature assembly
through ``jax.numpy`` (imported inside ``InterferometerSparseOperator``), so JAX
is exercised on CPU here. The numba pack itself never touches JAX.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def _profiling_root() -> Path:
    for _p in Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()

for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from autoarray import exc  # noqa: E402
from autoarray.inversion.inversion.interferometer import (  # noqa: E402
    inversion_interferometer_util,
)
from autoarray.inversion.inversion.interferometer.sparse import (  # noqa: E402
    InversionInterferometerSparse,
)
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import auto_simulate_if_missing  # noqa: E402
from instruments.interferometer import INSTRUMENTS  # noqa: E402
from numba_interferometer import (  # noqa: E402
    inversion_interferometer_numba_util as numba_util_pack,
)
from numba_interferometer.fit import numba_log_evidence_from  # noqa: E402
from numba_interferometer.inversion import InversionInterferometerNumba  # noqa: E402
from numba_interferometer.preload import NumbaPreload  # noqa: E402

INSTRUMENT = "sma"

# Kept small so the O(N^2 P^2) reference scatter and the dense O(N^2) oracle both
# run in seconds: the pins are algebraic, not statistical, so a production-sized
# mesh would buy nothing but run time.
MESH_PIXELS = 60
ORACLE_MASK_RADIUS = 1.0
ORACLE_MESH_PIXELS = 25

RTOL_PRELOAD = 1.0e-10
RTOL_PARITY = 1.0e-6


def _dataset_path() -> Path:
    dataset_path = Path("dataset") / "interferometer" / INSTRUMENT

    auto_simulate_if_missing(
        dataset_path,
        dataset_type="interferometer",
        instrument=INSTRUMENT,
        workspace_root=_ROOT,
    )

    return dataset_path


def _dataset(mask_radius: float):
    """The SMA dataset with its sparse operator applied, on a mask of ``mask_radius``."""
    cfg = INSTRUMENTS[INSTRUMENT]

    real_space_mask = al.Mask2D.circular(
        shape_native=cfg["real_space_shape"],
        pixel_scales=cfg["pixel_scale"],
        radius=mask_radius,
    )

    dataset_path = _dataset_path()

    dataset = al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=lambda uv_wavelengths, real_space_mask: al.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            chunk_size=cfg.get("transformer_chunk_size"),
        ),
    )

    return dataset.apply_sparse_operator(use_jax=False), dataset_path


def _fit(mask_radius: float, mesh_pixels: int):
    """An ``al.FitInterferometer`` on the eager NumPy path, at the prior medians."""
    dataset, dataset_path = _dataset(mask_radius=mask_radius)

    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

    image_mesh = al.image_mesh.Hilbert(pixels=mesh_pixels, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.real_space_mask, adapt_data=adapt_image
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=ell_comps[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=ell_comps[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=0),
        regularization=al.reg.ConstantSplit(coefficient=1.0),
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

    adapt_images = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={instance.galaxies.source: image_plane_mesh_grid},
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": image_plane_mesh_grid},
    )

    return al.FitInterferometer(
        dataset=dataset,
        tracer=al.Tracer(galaxies=list(instance.galaxies)),
        adapt_images=adapt_images,
        settings=al.Settings(),
        xp=np,
    )


@pytest.fixture(scope="module")
def fit():
    return _fit(mask_radius=INSTRUMENTS[INSTRUMENT]["mask_radius"], mesh_pixels=MESH_PIXELS)


@pytest.fixture(scope="module")
def numba_inversion(fit):
    _log_evidence, inversion = numba_log_evidence_from(fit)
    return inversion


@pytest.fixture(scope="module")
def oracle_fit():
    return _fit(mask_radius=ORACLE_MASK_RADIUS, mesh_pixels=ORACLE_MESH_PIXELS)


def _numba_inversion_from(fit) -> InversionInterferometerNumba:
    """A fresh (uncached) numba inversion for ``fit``.

    Used by the tests that must build ``F`` again after a monkeypatch, since
    ``curvature_matrix`` is a cached property.
    """
    inversion = fit.inversion

    return InversionInterferometerNumba(
        dataset=inversion.dataset,
        linear_obj_list=inversion.linear_obj_list,
        settings=inversion.settings,
        xp=np,
    )


def _modern_preload(dataset) -> np.ndarray:
    mask = dataset.transformer.grid.mask

    return inversion_interferometer_util.nufft_precision_operator_via_np_from(
        noise_map_real=np.asarray(dataset.noise_map.array.real, dtype=np.float64),
        uv_wavelengths=np.asarray(dataset.transformer.uv_wavelengths, dtype=np.float64),
        shape_masked_pixels_2d=mask.shape_native_masked_pixels,
        grid_radians_2d=np.asarray(
            mask.derive_grid.all_false.in_radians.native.array, dtype=np.float64
        ),
    )


# ---------------------------------------------------------------------------
# 1. The preload
# ---------------------------------------------------------------------------


def test_preload_via_numba_matches_modern_np_builder(fit):
    """The recovered numba preload kernel == today's chunked NumPy builder.

    This is the pin that the two names — ``curvature_preload`` (0b90c401) and
    ``nufft_precision_operator`` (main) — really are the same object. The two
    implementations sum the ``K`` visibilities in different orders, so they agree
    to round-off rather than bitwise; at ``K=190`` that is ~5e-12.
    """
    dataset = fit.inversion.dataset

    np.testing.assert_allclose(
        NumbaPreload.via_numba(dataset).curvature_preload,
        _modern_preload(dataset),
        rtol=RTOL_PRELOAD,
        atol=0.0,
    )


def test_preload_from_sparse_operator_matches_modern_np_builder(fit):
    """``from_sparse_operator`` rebuilds the operator's own preload array."""
    dataset = fit.inversion.dataset

    np.testing.assert_allclose(
        NumbaPreload.from_sparse_operator(dataset).curvature_preload,
        _modern_preload(dataset),
        rtol=RTOL_PRELOAD,
        atol=0.0,
    )


def test_dirty_image_is_the_sparse_operators_own_array(fit):
    """``d~`` is read off the sparse operator, never recomputed."""
    dataset = fit.inversion.dataset

    np.testing.assert_array_equal(
        NumbaPreload.from_sparse_operator(dataset).dirty_image,
        np.asarray(dataset.sparse_operator.dirty_image, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# 2. The dense oracle
# ---------------------------------------------------------------------------


def _dense_oracle_curvature(inversion: InversionInterferometerNumba) -> np.ndarray:
    """``F = Mᵀ W~ M`` formed explicitly, with no scatter kernel involved."""
    w_matrix = inversion.numba_preload.w_matrix()
    mapping_matrix = np.asarray(inversion.mapping_matrix)

    return mapping_matrix.T @ w_matrix @ mapping_matrix


def test_curvature_matrix_matches_dense_oracle(oracle_fit):
    """The scatter kernel really does compute ``Mᵀ W~ M``.

    ``w_tilde_via_preload_from`` expands the preload to the dense ``[N_pix, N_pix]``
    ``W~``, so this runs on a deliberately small mask.
    """
    inversion = _numba_inversion_from(oracle_fit)

    np.testing.assert_allclose(
        inversion.curvature_matrix_scatter,
        _dense_oracle_curvature(inversion),
        rtol=RTOL_PARITY,
        atol=0.0,
    )


def test_curvature_matrix_is_symmetric_without_mirroring(numba_inversion):
    """The reference kernel loops the full ``N x N`` pair space, so ``F`` arrives complete.

    Pinned because the imaging numba class and the modern sparse class both *do*
    need a mirroring pass — assuming this one does too (or does not) is exactly the
    convention error this test exists to catch.
    """
    curvature_matrix = numba_inversion.curvature_matrix_scatter

    np.testing.assert_allclose(
        curvature_matrix,
        curvature_matrix.T,
        rtol=RTOL_PARITY,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# 3. The modern path
# ---------------------------------------------------------------------------


def test_comparator_is_the_sparse_inversion(fit):
    """Guards the comparison itself: the fit must be on the JAX/FFT sparse path."""
    assert isinstance(fit.inversion, InversionInterferometerSparse)


def test_data_vector_matches_sparse(fit, numba_inversion):
    np.testing.assert_allclose(
        numba_inversion.data_vector,
        np.asarray(fit.inversion.data_vector),
        rtol=RTOL_PARITY,
        atol=0.0,
    )


def test_curvature_matrix_matches_sparse(fit, numba_inversion):
    np.testing.assert_allclose(
        numba_inversion.curvature_matrix,
        np.asarray(fit.inversion.curvature_matrix),
        rtol=RTOL_PARITY,
        atol=0.0,
    )


def test_reconstruction_matches_sparse(fit, numba_inversion):
    np.testing.assert_allclose(
        numba_inversion.reconstruction,
        np.asarray(fit.inversion.reconstruction),
        rtol=RTOL_PARITY,
        atol=0.0,
    )


def test_log_evidence_matches_fit_figure_of_merit(fit):
    log_evidence, _inversion = numba_log_evidence_from(fit)

    np.testing.assert_allclose(
        log_evidence,
        float(fit.figure_of_merit),
        rtol=RTOL_PARITY,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# 4. The preconditions
# ---------------------------------------------------------------------------


def test_multiple_mappers_raise(fit):
    inversion = fit.inversion

    with pytest.raises(exc.InversionException, match="mappers"):
        InversionInterferometerNumba(
            dataset=inversion.dataset,
            linear_obj_list=list(inversion.linear_obj_list) * 2,
            settings=inversion.settings,
            xp=np,
        )


def test_over_sampled_mapper_raises(fit, monkeypatch):
    """``over_sample_size != 1`` must raise, not agree with the modern path by accident.

    The modern sparse triplets fold ``over_sampler.sub_fraction`` into the mapping
    weights; the recovered kernel does not. The interferometer datasets profiled here
    all use ``over_sample_size = 1``, so the guard is asserted rather than handled.
    """
    inversion = fit.inversion
    mapper = inversion.cls_list_from(cls=Mapper)[0]

    sub_fraction = np.asarray(mapper.over_sampler.sub_fraction.array)

    over_sampled_stub = SimpleNamespace(
        sub_fraction=SimpleNamespace(array=0.25 * np.ones(sub_fraction.shape))
    )

    monkeypatch.setattr(
        Mapper, "over_sampler", property(lambda self: over_sampled_stub), raising=True
    )

    with pytest.raises(exc.InversionException, match="sub_fraction"):
        InversionInterferometerNumba(
            dataset=inversion.dataset,
            linear_obj_list=inversion.linear_obj_list,
            settings=inversion.settings,
            xp=np,
        )


def test_jax_array_module_raises(fit):
    import jax.numpy as jnp

    inversion = fit.inversion

    with pytest.raises(exc.InversionException, match="non-NumPy"):
        InversionInterferometerNumba(
            dataset=inversion.dataset,
            linear_obj_list=inversion.linear_obj_list,
            settings=inversion.settings,
            xp=jnp,
        )


def test_unknown_kernel_raises(fit):
    inversion = fit.inversion

    with pytest.raises(exc.InversionException, match="Unknown curvature kernel"):
        InversionInterferometerNumba(
            dataset=inversion.dataset,
            linear_obj_list=inversion.linear_obj_list,
            settings=inversion.settings,
            xp=np,
            kernel="two_stage",
        )


# ---------------------------------------------------------------------------
# 5. The control
# ---------------------------------------------------------------------------


def _broken_curvature_kernel(scale: float):
    """The reference kernel with its accumulated ``preload * w0 * w1`` product scaled.

    The kernel accumulates ``F[s0, s1] += curvature_preload[Δy, Δx] * w0 * w1``, which
    is linear in that product, so scaling every accumulation by ``scale`` is the same
    break as scaling the constant inside the innermost loop — expressed here without
    copying 80 lines of numba into the test. The unpatched kernel is bound at closure
    creation so ``monkeypatch.setattr`` on the module attribute does not recurse.
    """
    reference = numba_util_pack.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from

    def kernel(**kwargs):
        return scale * reference(**kwargs)

    return kernel


def test_control_broken_kernel_fails_the_dense_oracle_pin(oracle_fit, monkeypatch):
    """The oracle pin must FAIL when the kernel's constant is broken by 1%."""
    monkeypatch.setattr(
        numba_util_pack,
        "curvature_matrix_via_w_tilde_curvature_preload_interferometer_from",
        _broken_curvature_kernel(scale=1.01),
    )

    inversion = _numba_inversion_from(oracle_fit)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            inversion.curvature_matrix_scatter,
            _dense_oracle_curvature(inversion),
            rtol=RTOL_PARITY,
            atol=0.0,
        )


def test_control_broken_kernel_fails_the_sparse_and_evidence_pins(fit, monkeypatch):
    """The F and log-evidence pins must FAIL when the kernel's constant is broken by 1%."""
    monkeypatch.setattr(
        numba_util_pack,
        "curvature_matrix_via_w_tilde_curvature_preload_interferometer_from",
        _broken_curvature_kernel(scale=1.01),
    )

    log_evidence, inversion = numba_log_evidence_from(fit)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            inversion.curvature_matrix,
            np.asarray(fit.inversion.curvature_matrix),
            rtol=RTOL_PARITY,
            atol=0.0,
        )

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            log_evidence,
            float(fit.figure_of_merit),
            rtol=RTOL_PARITY,
            atol=0.0,
        )
