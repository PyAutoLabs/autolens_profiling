"""
Numba CPU Profiling: Multiprocessing Scaling of the Pixelization Likelihood
===========================================================================

Measures how numba CPU sparse-operator likelihood **throughput scales with
Python multiprocessing core count** — the ``number_of_cores`` lever the
workspace ``cpu_fast_modeling.py`` path hands to Nautilus — and quantifies the
overheads that limit it.

How PyAutoFit's Nautilus parallelizes today (traced 2026-08-20):

- ``number_of_cores > 1`` builds ``fork_context().Pool(n)`` and passes the Pool
  **object** to nautilus, whose ``Pool.map(fitness.call_wrap, points)`` pickles
  the bound method — and therefore the whole ``Fitness`` (model + analysis +
  dataset + **sparse operator**) — once per chunk, ~4 x cores times per
  ``n_batch=100`` batch.
- nautilus's own ``pool=<int>`` branch would instead cache the likelihood once
  per worker via a Pool ``initializer`` (PyAutoFit's dormant ``SneakierPool``
  is the same idea), sending only the parameter vectors per call.
- ``number_of_cores=1`` builds no pool at all (plain in-process ``map``).

This harness therefore measures, on one dataset + model fiducial (identical to
the ``likelihood_runtime``/``likelihood_breakdown`` numba siblings):

1. **Payload cost** — ``pickle.dumps(fitness)`` size and time, and the sparse
   operator's share of it: the per-chunk overhead the object-pool design pays.
2. **Pool scaling A/B** over ``--cores`` (default ``1,2,4,8``): throughput of
   ``--n-points`` evaluations (default 100 = nautilus ``n_batch``) under

   - (a) serial in-process baseline — what ``number_of_cores=1`` really does;
   - (b) ``nautilus_object_pool`` — ``Pool(P).map(fitness.call_wrap, points)``,
     replicating today's per-chunk-pickling design (default chunksize, as
     ``Pool.map`` computes it exactly like nautilus's call);
   - (c) ``initializer_cached`` — ``Pool(P, initializer=...)`` caching the
     fitness once per worker, mapping a thin module-level wrapper.

   Efficiency = serial_time / (P x parallel_time). The (b) vs (c) gap is the
   attainable gain from worker-side likelihood caching.

Notes:

- All numba kernels on this path are single-threaded; only BLAS/LAPACK (the
  reconstruction solve + Cholesky log-dets, ~13% of a likelihood at euclid)
  respond to ``OMP_NUM_THREADS``. Guidance: 1 BLAS thread per worker, give
  every core to workers. The active env pinning is recorded in the JSON.
- The first likelihood call is warm-up (lazy numba compile; cold-cache
  first-call hazard — see the runtime sibling). Workers forked afterwards
  inherit the compiled kernels.

Output
------
``results/parallel_scaling/imaging/pixelization_numba_scaling_<instrument>_v<version>.{json,png}``

``--mesh`` selects the fiducial: ``rectangular`` (28x28 adaptive rectangular —
``RectangularBilinearAdaptDensity`` by default, ``--rect-mesh rtu`` for the kernel-CDF variant,
the ``pixelization_numba`` siblings) or ``delaunay`` (Hilbert-1250 Delaunay +
ConstantSplit, the ``delaunay_numba`` siblings — the production campaign
fiducial; artifacts are written as ``delaunay_numba_scaling_<instrument>``).

Usage::

    python scripts/imaging/parallel_scaling/pixelization_numba.py --instrument euclid
    python scripts/imaging/parallel_scaling/pixelization_numba.py --mesh delaunay
    python scripts/imaging/parallel_scaling/pixelization_numba.py --cores 1,2,4,8,16,32  # HPC
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

_sys.path.insert(0, str(_profiling_root()))

import argparse
import json
import os

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os
import pickle
import sys as _smoke_sys
import time
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    rect_mesh_classes,
    resolve_output_paths,
)

_cli = parse_profile_cli()

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--cores", type=str, default="1,2,4,8")
_parser.add_argument("--n-points", type=int, default=100)
_parser.add_argument("--map-repeats", type=int, default=3)
_parser.add_argument("--mesh", choices=("rectangular", "delaunay"), default="rectangular")
_extra, _ = _parser.parse_known_args()

CORES_LIST = [int(c) for c in _extra.cores.split(",")]
N_POINTS = _extra.n_points
MAP_REPEATS = _extra.map_repeats
MESH = _extra.mesh

instrument = _cli.instrument or "euclid"  # default; override via --instrument

# ===================================================================
# Setup — identical fiducial to the runtime/breakdown numba siblings
# ===================================================================

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="imaging",
    instrument=instrument,
    workspace_root=_workspace_root,
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)

dataset = dataset.apply_sparse_operator_cpu()

print("\n--- Model construction ---")

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=60,
    centre_prior_is_uniform=True,
)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
_lens_mass_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_mass_ell[0], sigma=0.01)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_mass_ell[1], sigma=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

# Two mesh fiducials, matching the runtime/breakdown siblings exactly:
# rectangular = the pixelization_numba cells; delaunay = the delaunay_numba
# cells (the production campaign fiducial — Hilbert image mesh from the adapt
# image + ConstantSplit). The pool mechanics are mesh-independent; the serial
# per-eval cost (and so the serial fraction each worker amortises) differs.
adapt_images = None

if MESH == "rectangular":
    mesh_pixels_yx = 28
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
    n_source_pixels = mesh_pixels_yx * mesh_pixels_yx
    mesh_label = f"rectangular_{_cli.rect_mesh}_adapt_density_{mesh_pixels_yx}x{mesh_pixels_yx}"

    pixelization = al.Pixelization(
        mesh=rect_mesh_classes(_cli)[0](shape=mesh_shape),
        regularization=al.reg.Constant(coefficient=1.0),
    )
else:
    from _adapt_image_util import adapt_image_for_dataset

    n_mesh_vertices = 1250  # production campaign fiducial (matches delaunay_numba cells)
    mesh_shape = None
    n_source_pixels = n_mesh_vertices
    mesh_label = f"delaunay_hilbert_{n_mesh_vertices}"

    print(f"\n--- Adapt image + Hilbert image mesh ({n_mesh_vertices} vertices, one-off) ---")
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
    image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=adapt_image
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0),
        regularization=al.reg.ConstantSplit(coefficient=1.0),
    )

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

param_vector = np.array(model.physical_values_from_prior_medians)

if MESH == "delaunay":
    _instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    adapt_images = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            _instance.galaxies.source: image_plane_mesh_grid,
        },
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid,
        },
    )

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=al.Settings(use_border_relocator=True),
    use_jax=False,
)

# The exact object nautilus receives: likelihood = fitness.call_wrap, called
# with physical parameter vectors (prior transform happens nautilus-side).
from autofit.non_linear.fitness import Fitness  # noqa: E402
from autofit.non_linear.parallel.context import fork_context  # noqa: E402

fitness = Fitness(
    model=model,
    analysis=analysis,
    paths=None,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

print("\n--- Warm-up (numba compile; workers fork the compiled state) ---")
_t0 = time.perf_counter()
_warm_1 = fitness.call_wrap(param_vector)
_warm_2 = fitness.call_wrap(param_vector)
print(f"  warm-up: {time.perf_counter() - _t0:.3f} s | log_likelihood = {_warm_2}")
if not np.isfinite(float(_warm_1)):
    print("  NOTE: first call non-finite (known cold-numba-cache first-call hazard).")

points = [param_vector.copy() for _ in range(N_POINTS)]

# ===================================================================
# 1. Payload cost — what the object-pool design pickles per chunk
# ===================================================================

print("\n--- Pickle payload (per-chunk cost of the object-pool design) ---")

_t0 = time.perf_counter()
_fitness_bytes = pickle.dumps(fitness)
fitness_pickle_s = time.perf_counter() - _t0
fitness_pickle_mb = len(_fitness_bytes) / 1024**2

_t0 = time.perf_counter()
_operator_bytes = pickle.dumps(dataset.sparse_operator)
operator_pickle_s = time.perf_counter() - _t0
operator_pickle_mb = len(_operator_bytes) / 1024**2

del _fitness_bytes, _operator_bytes

print(f"  fitness pickle:         {fitness_pickle_mb:8.1f} MB in {fitness_pickle_s:.3f} s")
print(f"  sparse-operator share:  {operator_pickle_mb:8.1f} MB in {operator_pickle_s:.3f} s")

# ===================================================================
# 2. Serial in-process baseline (= number_of_cores=1)
# ===================================================================

print(f"\n--- Serial baseline ({N_POINTS} evaluations) ---")

_t0 = time.perf_counter()
serial_results = [fitness.call_wrap(p) for p in points]
serial_time = time.perf_counter() - _t0
serial_per_eval = serial_time / N_POINTS

print(f"  total: {serial_time:.2f} s | per eval: {serial_per_eval:.4f} s")

_reference_ll = float(serial_results[0])

# ===================================================================
# 3. Pool scaling A/B
# ===================================================================

_WORKER_FITNESS = None


def _init_worker(fitness_for_worker):
    """Pool initializer for the ``initializer_cached`` variant: unpickle the
    fitness once per worker (nautilus's ``pool=<int>`` branch's design)."""
    global _WORKER_FITNESS
    _WORKER_FITNESS = fitness_for_worker


def _call_cached(parameters):
    return _WORKER_FITNESS.call_wrap(parameters)


def _corrupt_count(results) -> int:
    """Evaluations that did not reproduce the serial likelihood.

    Every point is the identical parameter vector, so any mismatch (NaN mapped
    to the ``resample_figure_of_merit=-1e99`` sentinel, or a wrong finite
    value) is a corrupted evaluation inside a worker — the per-worker /
    per-unpickling first-call hazard measured by this harness.
    """
    arr = np.array([float(r) for r in results])
    with np.errstate(invalid="ignore"):
        return int(np.sum(~np.isclose(arr, _reference_ll, rtol=1e-6)))


def profile_pool(n_cores: int, variant: str) -> dict:
    """Time ``Pool.map`` over the points for one (cores, variant) cell."""
    t0 = time.perf_counter()
    if variant == "nautilus_object_pool":
        pool = fork_context().Pool(n_cores)
        map_func, map_target = pool.map, fitness.call_wrap
    elif variant == "initializer_cached":
        pool = fork_context().Pool(n_cores, initializer=_init_worker, initargs=(fitness,))
        map_func, map_target = pool.map, _call_cached
    else:
        raise ValueError(variant)
    pool_create_s = time.perf_counter() - t0

    try:
        t0 = time.perf_counter()
        results = map_func(map_target, points)
        first_map_s = time.perf_counter() - t0
        corrupt_first = _corrupt_count(results)

        steady_times = []
        corrupt_steady = []
        for _ in range(MAP_REPEATS):
            t0 = time.perf_counter()
            results = map_func(map_target, points)
            steady_times.append(time.perf_counter() - t0)
            corrupt_steady.append(_corrupt_count(results))
    finally:
        pool.close()
        pool.join()

    steady_s = min(steady_times)
    per_eval = steady_s / N_POINTS
    return {
        "cores": n_cores,
        "variant": variant,
        "pool_create_s": pool_create_s,
        "first_map_s": first_map_s,
        "steady_map_s": steady_s,
        "per_eval_s": per_eval,
        "throughput_evals_per_s": N_POINTS / steady_s,
        "speedup_vs_serial": serial_time / steady_s,
        "efficiency_vs_serial": serial_time / (steady_s * n_cores),
        "corrupt_evals_first_map": corrupt_first,
        "corrupt_evals_steady_maps": corrupt_steady,
    }


print(f"\n--- Pool scaling ({N_POINTS} evals/map, best of {MAP_REPEATS} steady maps) ---")

pool_rows = []
for n_cores in CORES_LIST:
    for variant in ("nautilus_object_pool", "initializer_cached"):
        row = profile_pool(n_cores=n_cores, variant=variant)
        pool_rows.append(row)
        print(
            f"  {variant:<22} P={n_cores:<3} create {row['pool_create_s']:6.2f} s | "
            f"first map {row['first_map_s']:6.2f} s | steady {row['steady_map_s']:6.2f} s | "
            f"speedup {row['speedup_vs_serial']:5.2f}x | eff {row['efficiency_vs_serial']:5.1%}"
        )
        _n_corrupt = row["corrupt_evals_first_map"] + sum(row["corrupt_evals_steady_maps"])
        if _n_corrupt:
            print(
                f"    WARNING: {_n_corrupt} corrupted evaluation(s) "
                f"(first map: {row['corrupt_evals_first_map']}, "
                f"steady: {row['corrupt_evals_steady_maps']}) — worker likelihoods "
                f"that did not reproduce the serial value (returned the resample "
                f"sentinel or a wrong value). See README hazard section."
            )

# ===================================================================
# Summary + JSON + PNG
# ===================================================================

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

scaling_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(dataset.data.shape[0]),
        "mesh": mesh_label,
        "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
        "source_pixels": int(n_source_pixels),
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "lens_light": "mge_60_linear",
        "n_points": N_POINTS,
        "map_repeats": MAP_REPEATS,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    },
    "payload": {
        "fitness_pickle_mb": fitness_pickle_mb,
        "fitness_pickle_s": fitness_pickle_s,
        "sparse_operator_pickle_mb": operator_pickle_mb,
        "sparse_operator_pickle_s": operator_pickle_s,
    },
    "serial": {
        "total_s": serial_time,
        "per_eval_s": serial_per_eval,
    },
    "pools": pool_rows,
}

_basename_stem = "pixelization_numba" if MESH == "rectangular" else "delaunay_numba"
dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "parallel_scaling" / "imaging",
    default_basename=f"{_basename_stem}_scaling_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(scaling_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Plot: throughput + efficiency vs cores, both variants ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

_colors = {"nautilus_object_pool": "#C44E52", "initializer_cached": "#55A868"}
_labels = {
    "nautilus_object_pool": "object pool (nautilus today)",
    "initializer_cached": "initializer-cached workers",
}

for variant in ("nautilus_object_pool", "initializer_cached"):
    rows = [r for r in pool_rows if r["variant"] == variant]
    cores = [r["cores"] for r in rows]
    ax1.plot(
        cores,
        [r["throughput_evals_per_s"] for r in rows],
        "o-",
        color=_colors[variant],
        label=_labels[variant],
    )
    ax2.plot(
        cores,
        [100 * r["efficiency_vs_serial"] for r in rows],
        "o-",
        color=_colors[variant],
        label=_labels[variant],
    )

_serial_throughput = 1.0 / serial_per_eval
ax1.plot(
    CORES_LIST,
    [_serial_throughput * c for c in CORES_LIST],
    "k--",
    alpha=0.5,
    label="ideal (serial x P)",
)
ax1.axhline(_serial_throughput, color="k", ls=":", alpha=0.5, label="serial in-process")

ax1.set_xlabel("cores (P)")
ax1.set_ylabel("throughput (likelihood evals / s)")
ax1.set_title("Throughput")
ax1.legend(fontsize=8)
ax2.set_xlabel("cores (P)")
ax2.set_ylabel("parallel efficiency vs serial (%)")
ax2.set_ylim(0, 110)
ax2.axhline(100, color="k", ls="--", alpha=0.5)
ax2.set_title("Efficiency")
ax2.legend(fontsize=8)

fig.suptitle(
    f"Numba CPU Likelihood Multiprocessing Scaling — {instrument.upper()} — {mesh_label} — "
    f"v{al_version}  ({N_POINTS} evals/map, serial {serial_per_eval:.3f} s/eval)",
    fontsize=11,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Chart saved to:        {chart_path}")
