"""
Profiling: Fixed Lens Light — the CPU kernel rows
=================================================

Phase 2 of the ``fixed-lens-light-profiling`` epic (autolens_profiling#253),
the CPU half of "optimise for the hardware most users actually own".

Phases 0 (#248) and 1 (#251) measured the fixed-lens-light likelihood on an
A100, in JAX. This cell measures the **solve** on a CPU, in the tools a CPU
actually uses, on the same S3 system (MGE-60 converted to regular light profiles
at their solved intensities and subtracted, leaving a source-only inversion):

- **library numpy NNLS** — ``reconstruction_positive_only_from(..., xp=np)``,
  the library's own CPU production kernel (``fnnls_cholesky``, Bro & de Jong
  1997, seeded from the sign of the dense unconstrained solve);
- **scipy unconstrained** — ``cho_factor`` / ``cho_solve``, the floor;
- **numpy certified active set** — restricted Cholesky per pass at phase 0's
  certifying budget. **The CPU production candidate.**

The rows and the reasoning live in
``likelihood_breakdown.fixed_light_cpu_kernels``; this file is the cell that
builds the system and writes the result, exactly as ``fixed_light.py`` and
``fixed_light_library.py`` do — the same dataset, mask, over-sampling, model and
meshes, so the numbers stack with theirs.

Threads
-------

**One thread setting per process.** OpenBLAS / MKL read ``OMP_NUM_THREADS`` &c.
at import, so a setting can only be chosen *before* the process starts. Run the
cell twice::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPROC=1 \\
      JAX_PLATFORMS=cpu python scripts/imaging/likelihood_breakdown/fixed_light_cpu_kernels.py \\
      --mesh rectangular --config-name local_cpu_fp64_fixed_light_kernels_t1

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NPROC=8 \\
      JAX_PLATFORMS=cpu python … --config-name local_cpu_fp64_fixed_light_kernels_tall

Both settings are recorded in every JSON (``threads`` and the ``machine``
block), never inferred from the config name.

Flags
-----

``--mesh {rectangular,delaunay,delaunay_nn}`` (required). ``--pass-budget N``
— the certified row's budget, defaulting to phase 0's certifying budget for the
mesh. ``--source-pixels N`` sweeps the mesh size. ``--n-repeats N`` (default 10)
is the median sample size.

Output
------

``results/breakdown/imaging/fixed_light_cpu_kernels_<mesh>_<config_name>.{json,png}``.
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
    _sys.path.insert(0, str(_misc_dir))


import argparse
import math
import sys

import autofit as af
import autolens as al
import numpy as np

sys.path.insert(0, str(_profiling_root()))

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke). Verifies the import
# graph + module-level setup succeeded without running the profile.
import os as _smoke_os
import sys as _smoke_sys

from likelihood_breakdown import (  # noqa: E402
    active_set_steps,
    fixed_light_cpu_kernels,
    timing,
)

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    delaunay_regularization,
    device_info_dict,
    machine_info_dict,
    parse_profile_cli,
    rect_mesh_classes,
    resolve_output_paths,
)

_cli = parse_profile_cli()

#: Phase 0's per-mesh certifying pass budget (legs 342802-342804), re-measured
#: on the library path by phase 1 (342908-342910) and unchanged.
CERTIFYING_BUDGET = {"rectangular": 7, "delaunay": 2, "delaunay_nn": 2}

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument(
    "--mesh", choices=("rectangular", "delaunay", "delaunay_nn"), required=True
)
_cell_parser.add_argument("--pass-budget", type=int, default=None)
_cell_parser.add_argument("--source-pixels", type=int, default=None)
_cell_parser.add_argument("--n-repeats", type=int, default=10)
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh
PASS_BUDGET = int(
    _cell_args.pass_budget if _cell_args.pass_budget is not None else CERTIFYING_BUDGET[MESH]
)
SOURCE_PIXELS_REQUESTED = _cell_args.source_pixels
N_REPEATS = int(_cell_args.n_repeats)

if PASS_BUDGET < 1:
    raise ValueError(f"--pass-budget must be >= 1 (got {PASS_BUDGET})")
if N_REPEATS < 1:
    raise ValueError(f"--n-repeats must be >= 1 (got {N_REPEATS})")

instrument = "hst"

timer = timing.Timer()

# ===================================================================
# PART A — Setup (identical to fixed_light.py / fixed_light_library.py)
# ===================================================================

print(f"\n--- Dataset loading & masking [{instrument}, mesh={MESH}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = _Path("dataset") / "imaging" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="imaging",
    instrument=instrument,
    workspace_root=_workspace_root,
)

with timer.section("dataset_load"):
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )

with timer.section("mask_and_oversample"):
    mask_radius = 3.5

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=1,
    )

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )

if MESH == "rectangular":
    mesh_pixels_yx = (
        39 if SOURCE_PIXELS_REQUESTED is None else int(round(math.sqrt(SOURCE_PIXELS_REQUESTED)))
    )
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
    n_source_pixels = mesh_pixels_yx * mesh_pixels_yx
    n_mesh_vertices = None
else:
    mesh_pixels_yx = None
    mesh_shape = None
    n_mesh_vertices = 1500 if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    n_source_pixels = n_mesh_vertices

print(f"  Source pixels: {n_source_pixels}")

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

image_plane_mesh_grid = None
if MESH != "rectangular":
    print("\n--- Image mesh construction (Hilbert) ---")
    with timer.section("image_mesh_hilbert"):
        image_mesh = al.image_mesh.Hilbert(
            pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0
        )
        image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
            mask=dataset.mask, adapt_data=adapt_image
        )
    print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

print("\n--- Model construction ---")

with timer.section("model_build"):
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

    if MESH == "rectangular":
        reg_scheme = "constant"
        mesh_obj = rect_mesh_classes(_cli)[1](shape=mesh_shape, weight_power=1.0, weight_floor=0.0)
        regularization = al.reg.Constant(coefficient=1.0)
        reg_provenance = {"scheme": reg_scheme, "coefficient": 1.0}
    else:
        if MESH == "delaunay":
            mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
        else:
            mesh_obj = al.mesh.DelaunayNN(pixels=n_mesh_vertices, areas_factor=0.5, zeroed_pixels=0)
        reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

    pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

with timer.section("instance_from_vector"):
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

tracer = al.Tracer(galaxies=list(instance.galaxies))

_adapt_kwargs = {
    "galaxy_image_dict": {instance.galaxies.source: adapt_image},
    "galaxy_name_image_dict": {"('galaxies', 'source')": adapt_image},
}
if image_plane_mesh_grid is not None:
    _adapt_kwargs["galaxy_image_plane_mesh_grid_dict"] = {
        instance.galaxies.source: image_plane_mesh_grid
    }
    _adapt_kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
        "('galaxies', 'source')": image_plane_mesh_grid
    }
adapt_images = al.AdaptImages(**_adapt_kwargs)

n_image_pixels = dataset.data.shape[0]

_settings = al.Settings(use_border_relocator=True)

print("\n--- S0 fit (eager, numpy) ---")
with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=_settings,
        xp=np,
    )
    log_evidence_s0 = float(fit.figure_of_merit)
print(f"  S0 figure_of_merit = {log_evidence_s0}")

print("\n--- S3: lens light converted to regular profiles + subtracted ---")
with timer.section("s3_system_build"):
    system_s3 = active_set_steps.fixed_light_system_from(
        fit,
        dataset,
        adapt_images=adapt_images,
        settings=_settings,
        name="S3_mge_converted_to_regular",
    )
print(
    f"  S3: n={system_s3.n_params} (mapper {system_s3.n_mapper} + funcs {system_s3.n_funcs}), "
    f"edge-zeroed {int(system_s3.edge_zero_mask.sum())}"
)

# ===================================================================
# PART B — The CPU kernel rows
# ===================================================================

print("\n" + "=" * 70)
print(f"CPU KERNEL ROWS — {MESH} — pass budget {PASS_BUDGET} — {N_REPEATS} repeats")
print("=" * 70)

kernels = fixed_light_cpu_kernels.cpu_kernel_rows(
    system_s3,
    pass_budget=PASS_BUDGET,
    n_repeats=N_REPEATS,
)

print(
    f"  n full {kernels['n_full_system']}, edge-zeroed {kernels['n_edge_zeroed']}, "
    f"solver sees {kernels['n_seen_by_solver']}"
)
print(
    f"  threads: BLAS {kernels['threads']['n_threads_blas']} / "
    f"XLA pool {kernels['threads']['n_threads_jax_pool']}"
)

_flat_rows = {
    "library_numpy_nnls": kernels["rows"]["library_numpy_nnls"],
    "scipy_unconstrained_subset": kernels["rows"]["scipy_unconstrained"]["subset"],
    "scipy_unconstrained_full": kernels["rows"]["scipy_unconstrained"]["full_system"],
    "numpy_certified_active_set": kernels["rows"]["numpy_certified_active_set"],
}

_w = max(len(r["label"]) for r in _flat_rows.values())
for _key, _row in _flat_rows.items():
    print(
        f"  {_row['label']:<{_w}}  {_row['ms']:9.3f} ms   "
        f"Δlog-ev {_row['d_log_evidence_vs_library']:+.6e}  "
        f"neg {_row['n_negative_entries']}"
    )

print("\n  Equalities:")
for _eq in kernels["equalities"]:
    print(f"    [{_eq['status']:>4}] {_eq['check']}  rel {_eq['rel_diff']:.3e}")

# ===================================================================
# JSON + PNG
# ===================================================================

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "machine": machine_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "mesh": MESH,
        "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
        "rect_mesh": _cli.rect_mesh if MESH == "rectangular" else None,
        "source_pixels": int(n_source_pixels),
        "source_pixels_requested": (
            int(SOURCE_PIXELS_REQUESTED) if SOURCE_PIXELS_REQUESTED is not None else None
        ),
        "inversion_path": "dense",
        "pass_budget": PASS_BUDGET,
        "n_repeats": N_REPEATS,
    },
    "regularization": reg_provenance,
    "log_evidence_eager_s0": log_evidence_s0,
    "s3": {
        "n_params": int(system_s3.n_params),
        "n_mapper": int(system_s3.n_mapper),
        "n_funcs": int(system_s3.n_funcs),
        "edge_zeroed_pixels": int(system_s3.edge_zero_mask.sum()),
        "subtracted_light_flux": float(system_s3.subtracted_light_flux),
    },
    "kernels": kernels,
    "note": (
        "Every row here is a SOLVE on the CPU in numpy/scipy — not a library "
        "likelihood call and not a JAX row. Never sum these with the "
        "fixed_light_library cell's whole-call rows, and never quote the "
        "unconstrained row without its Δlog-evidence and negative-pixel count: "
        "it is a different, infeasible minimiser that scores above the "
        "constrained optimum."
    ),
}

# The scatter_back closures are not JSON-serialisable and are an implementation
# detail of the view, not a result; nothing else in ``kernels`` holds a callable.
dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"fixed_light_cpu_kernels_{MESH}_breakdown_{instrument}_v{al_version}",
    cell=f"fixed_light_cpu_kernels_{MESH}",
)
dict_path.write_text(json.dumps(summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

labels = [r["label"] for r in _flat_rows.values()]
times_ms = [r["ms"] for r in _flat_rows.values()]

fig, ax = plt.subplots(figsize=(12, max(3.5, 0.7 * len(labels))))
y_pos = np.arange(len(labels), dtype=float)
bars = ax.barh(y_pos, times_ms, color="#4C72B0", edgecolor="white", height=0.6)
for bar, t in zip(bars, times_ms):
    ax.text(
        bar.get_width() + max(times_ms) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.2f} ms",
        va="center",
        fontsize=8,
    )
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Solve (ms, median of repeats)", fontsize=11)
fig.suptitle(
    f"CPU kernel rows, lens light fixed — {MESH} — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {n_source_pixels} source pixels  |  "
    f"solver sees n={kernels['n_seen_by_solver']}  |  BLAS threads "
    f"{kernels['threads']['n_threads_blas']}  |  active-set budget {PASS_BUDGET} "
    f"(certifies: {kernels['rows']['numpy_certified_active_set']['certified']})",
    fontsize=9,
)
ax.margins(x=0.20)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

print("\nFinished.")
