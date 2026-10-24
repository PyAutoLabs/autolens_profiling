"""JAX and numba standard-size cells evaluate the same likelihood (human ruling N1 on #456).

Under ``--standard-size`` every pixelized imaging cell builds the numba production preset of its
instrument. The JAX cells reach it through ``_production_config.standard_jax_cell``; the numba
cells build it from the same parts in their own order (capped adapt image, production
over-sampling with the CPU sparse operator, ``production_image_mesh_grid``, ``production_mesh``,
``production_model``, iid draw 0). This test builds both routes on one small simulated dataset,
for every pixelized likelihood x instrument, and requires the same model (prior count, mesh
vertices, parameters of the evaluated instance) and — numba sparse against the JAX cell's eager
NumPy dense reference — the same log-likelihood to rtol 1e-8. The cell scripts themselves are
only read as text (importing one runs a profile).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT))
import _production_config as prod  # noqa: E402

INSTRUMENTS = ("ao", "euclid", "hst", "jwst")
VARIANTS = ("delaunay", "delaunay_nn", "rectangular")
MASK_RADIUS = 2.4


@pytest.fixture(scope="module")
def cell_inputs(tmp_path_factory):
    """A small simulated lens (enough masked pixels for a 1250-point Hilbert mesh), its S/N-like
    adapt image, and a dataset folder holding the hst positions (the euclid penalty reads it)."""
    import autolens as al
    import numpy as np

    shape, scale = (72, 72), 0.08
    grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=scale)
    psf = al.Convolver.from_gaussian(shape_native=(5, 5), sigma=0.08, pixel_scales=scale)
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6)),
            al.Galaxy(
                redshift=1.0,
                bulge=al.lp.SersicCore(
                    centre=(0.05, 0.05), intensity=1.0, effective_radius=0.3, sersic_index=1.0
                ),
            ),
        ]
    )
    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise_to_data=False
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    mask = al.Mask2D.circular(shape_native=shape, pixel_scales=scale, radius=MASK_RADIUS)
    dataset = dataset.apply_mask(mask=mask)
    signal = np.asarray(tracer.image_2d_from(grid=dataset.grid))
    adapt_image = al.Array2D(values=10.0 * signal / signal.max() + 0.05, mask=dataset.mask)
    folder = tmp_path_factory.mktemp("dataset")
    shutil.copy(ROOT / "dataset/imaging/hst/positions.json", folder / "positions.json")
    return dataset, adapt_image, folder


def numba_route(variant, instrument, dataset, adapt_image, folder):
    """The numba cells' construction, in their order (likelihood_runtime_numba.py)."""
    import autolens as al

    family = prod.STANDARD_FAMILY[variant]
    preset = prod.standard_size_preset(family, instrument)
    adapt = prod.adapt_image_capped(adapt_image, preset)
    data = prod.apply_production_over_sampling(dataset, adapt, preset)  # + CPU sparse operator
    if preset.mesh_kind == "delaunay":
        grid = prod.production_image_mesh_grid(data, adapt, preset, MASK_RADIUS)
        cls = al.mesh.DelaunayNN if variant == "delaunay_nn" else al.mesh.Delaunay
        mesh = prod.production_mesh(preset, n_vertices=int(grid.shape[0]), mesh_class=cls)
        n = int(grid.shape[0])
    else:
        grid = None
        mesh = prod.production_mesh(preset, mesh_class=al.mesh.RectangularBilinearAdaptDensity)
        n = preset.rect_pixels_yx**2
    model = prod.production_model(preset, MASK_RADIUS, mesh)
    instance = prod.iid_instances(model, 6, preset)[0]  # the numba cells' instance 0
    settings, _ = prod.analysis_settings(preset, "off")
    positions = prod.positions_likelihood(
        preset,
        dataset_path=folder,
        tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
    )
    analysis = al.AnalysisImaging(
        dataset=data,
        adapt_images=prod.adapt_images_for(instance, image_plane_mesh_grid=grid, adapt_image=adapt),
        positions_likelihood_list=[positions] if positions is not None else None,
        settings=settings,
        use_jax=False,
    )
    return {"model": model, "instance": instance, "n": n, "analysis": analysis, "data": data}


def jax_route(variant, instrument, dataset, adapt_image, folder):
    """The JAX cells' construction under --standard-size (the block in every JAX script)."""
    import autolens as al

    preset = prod.standard_size_preset(prod.STANDARD_FAMILY[variant], instrument)
    data, adapt, _record = prod.standard_jax_dataset(dataset, adapt_image, preset, None)
    std = prod.standard_jax_cell(
        variant,
        instrument,
        dataset=data,
        adapt_image=adapt,
        dataset_path=folder,
        mask_radius=MASK_RADIUS,
        rect_mesh_cls=al.mesh.RectangularBilinearAdaptDensity,
    )
    std["adapt_images"] = prod.adapt_images_for(
        std["instance"], image_plane_mesh_grid=std["image_plane_mesh_grid"], adapt_image=adapt
    )
    std["data"] = data
    return std


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_jax_and_numba_standard_cells_build_the_same_model(variant, instrument, cell_inputs):
    numba = numba_route(variant, instrument, *cell_inputs)
    jax = jax_route(variant, instrument, *cell_inputs)
    assert jax["n_source_pixels"] == numba["n"] == (1225 if variant == "rectangular" else 1280)
    assert jax["model"].prior_count == numba["model"].prior_count
    assert jax["model"].total_free_parameters == numba["model"].total_free_parameters
    assert [p.__class__ for p in jax["model"].priors_ordered_by_id] == [
        p.__class__ for p in numba["model"].priors_ordered_by_id
    ]
    for (_, a), (_, b) in zip(
        jax["instance"].path_instance_tuples_for_class(float),
        numba["instance"].path_instance_tuples_for_class(float),
        strict=True,
    ):
        assert a == b
    assert (
        jax["preset"].name
        == prod.standard_size_preset(prod.STANDARD_FAMILY[variant], instrument).name
    )
    if instrument in ("jwst", "ao"):
        assert "borrows the hst preset" in jax["record"]["size_override"]
    assert (
        type(jax["data"].sparse_operator).__name__ != type(numba["data"].sparse_operator).__name__
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("instrument", ("hst", "euclid"))
def test_numba_sparse_equals_the_jax_cells_dense_numpy_reference(variant, instrument, cell_inputs):
    numba = numba_route(variant, instrument, *cell_inputs)
    jax = jax_route(variant, instrument, *cell_inputs)
    value = float(numba["analysis"].log_likelihood_function(instance=numba["instance"]))
    reference = prod.numpy_dense_reference(
        jax["data"],
        jax["adapt_images"],
        jax["instance"],
        settings=jax["settings"],
        positions_likelihood_list=jax["positions_likelihood_list"],
    )
    assert value == pytest.approx(reference, rel=1e-8)


def test_every_jax_pixelized_script_takes_the_production_cell_under_standard_size():
    for variant in VARIANTS:
        for measurement in ("runtime", "breakdown"):
            text = (ROOT / f"scripts/imaging/{variant}/likelihood_{measurement}.py").read_text()
            assert f'standard_jax_cell(\n        "{variant}",' in text
            assert "standard_jax_dataset(" in text and "if _cli.standard_size:" in text
            assert "and not _cli.standard_size:" in text  # no fiducial sparse operator first
            assert "settings=_cell_settings" in text
            if measurement == "runtime":
                assert "numpy_dense_reference(" in text and "_parity_reference," in text
                assert '"production_preset"' in text
            else:
                assert 'breakdown_summary["configuration"]["production_preset"]' in text


def test_every_numba_script_builds_the_model_from_the_shared_parts():
    for path in ROOT.glob("scripts/imaging/*/likelihood_*_numba.py"):
        if path.name.endswith("mge_mass.py"):
            continue
        text = path.read_text()
        assert "production_model(preset, mask_radius, mesh" in text
        assert "production_mesh(" in text
        assert "mge_lens_bulge(" not in text and "af.Model(al.mp.Isothermal)" not in text
        if "delaunay" in path.parent.name:
            assert "production_image_mesh_grid(dataset, adapt_image, preset, mask_radius)" in text
