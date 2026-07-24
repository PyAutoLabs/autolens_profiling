"""
Multi Runtime: shared_preloads speed-up
=======================================

Measures the realized end-to-end speed-up of the `FactorGraphModel` shared-state path for
multi-exposure imaging: the same identical exposure is fitted N times with `shared_preloads=False`
(every exposure computes its own image-mesh ray-trace and source-plane mesh) and with
`shared_preloads=True` (the lead exposure ray-traces the source-plane Delaunay mesh once and every
exposure maps its own grid onto it), and the ratio is reported.

Unlike the datacube (`likelihood_runtime/datacube/shared_preloads.py`), only the mesh geometry is
shared — each exposure still builds its own mapping matrix, PSF-blurred mapping matrix, curvature
matrix and regularization matrix, which dominate the imaging inversion budget. The expected speed-up
is therefore modest; the primary win of the shared path is *consistency* (every exposure reconstructs
on the identical source-pixel grid — see PyAutoLens#599 D1). This script records the honest number.

Timing uses `Fitness._vmap` over a batch rather than a single jit on concrete parameters, because a
single jit over fixed parameters can constant-fold the work and report a misleadingly fast time; the
vmapped per-evaluation time is honest (see PyAutoLabs notes on jit const-folding).

Correctness of the imaging `shared_preloads` is asserted separately in
`autolens_workspace_test/scripts/jax_likelihood_functions/multi/shared_preloads.py`; this script is
purely a runtime measurement.

Run from the repo root:

    python likelihood_runtime/multi/shared_preloads.py [--instrument hst|hst_up|euclid|...]
"""

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)


import os
import sys
import time
from pathlib import Path

import autofit as af
import autolens as al
import jax.numpy as jnp
import numpy as np
from autofit.non_linear.fitness import Fitness

sys.path.insert(0, str(_profiling_root()))

# CI lint smoke: verify the import graph + module setup without the heavy timing run.
if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    from simulators.imaging import INSTRUMENTS  # noqa: F401

    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import auto_simulate_if_missing, parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
instrument = _cli.instrument or "hst"

N_EXPOSURES = 4
BATCH_SIZE = 2
N_REPEATS = 5
N_MESH_VERTICES = 1500

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="imaging",
    instrument=instrument,
    workspace_root=_workspace_root,
)


def _dataset():
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=3.5,
    )
    dataset = dataset.apply_mask(mask=mask)
    return dataset.apply_over_sampling(over_sample_size_lp=1, over_sample_size_pixelization=1)


def _adapt_images(dataset):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
    image_mesh = al.image_mesh.Hilbert(pixels=N_MESH_VERTICES, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=adapt_image
    )
    return al.AdaptImages(
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": image_plane_mesh_grid},
    )


def _model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.05, sigma=0.01)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.Delaunay(pixels=N_MESH_VERTICES, zeroed_pixels=0),
        regularization=al.reg.ConstantSplit,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _factor_graph(shared_preloads):
    analysis_list = []
    for _ in range(N_EXPOSURES):
        dataset = _dataset()
        analysis_list.append(
            al.AnalysisImaging(
                dataset=dataset,
                adapt_images=_adapt_images(dataset),
                use_jax=True,
                shared_preloads=shared_preloads,
                raise_inversion_positions_likelihood_exception=False,
            )
        )

    model = _model()
    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model.copy(), analysis=analysis) for analysis in analysis_list
    ]

    return af.FactorGraphModel(*analysis_factor_list, use_jax=True)


def _time_vmap(shared_preloads):
    factor_graph = _factor_graph(shared_preloads)

    fitness = Fitness(
        model=factor_graph.global_prior_model,
        analysis=factor_graph,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    medians = factor_graph.global_prior_model.physical_values_from_prior_medians
    parameters = jnp.array(np.tile(np.asarray(medians), (BATCH_SIZE, 1)))

    fitness._vmap(parameters).block_until_ready()  # warm-up (compile excluded)

    start = time.perf_counter()
    for _ in range(N_REPEATS):
        fitness._vmap(parameters).block_until_ready()
    elapsed = time.perf_counter() - start

    return elapsed / (N_REPEATS * BATCH_SIZE)


if __name__ == "__main__":
    print(
        f"multi shared_preloads runtime  "
        f"(instrument={instrument}, N_EXPOSURES={N_EXPOSURES}, Hilbert+Delaunay mesh)"
    )

    per_eval_unshared = _time_vmap(shared_preloads=False)
    per_eval_shared = _time_vmap(shared_preloads=True)

    print(f"  per-likelihood (unshared): {per_eval_unshared * 1e3:.1f} ms")
    print(f"  per-likelihood (shared):   {per_eval_shared * 1e3:.1f} ms")
    print(f"  speed-up:                  {per_eval_unshared / per_eval_shared:.2f}x")
    print("  (the mesh is the only shared quantity for imaging — per-exposure blurred mapping")
    print("  matrix / curvature / regularization dominate, so expect a modest ratio; the")
    print("  primary win is mesh consistency across exposures — see PyAutoLens#599 D1)")
