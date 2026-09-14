"""
JAX Profiling: Fixed Lens Light, Source-Only Inversion — Positivity Alternatives
================================================================================

The measurement cell for the fixed-lens-light line (autolens_profiling#248). It
builds the **same** fiducial imaging model the three pixelized breakdown cells
build — ``imaging/pixelization.py`` (39x39 rectangular bilinear adapt mesh),
``imaging/delaunay.py`` (1500-vertex Hilbert/Delaunay) and
``imaging/delaunay_nn.py`` (the same mesh with Sibson natural-neighbour
interpolation) — and then measures **two linear systems side by side**:

- **S0**, the system the library solves today: the 60 unregularised linear-MGE
  columns share ``F + λH`` with the source pixels, and ``cond(F + λH) ≈ 4e10``;
- **S3**, the system left once that lens light is *fixed*: the linear MGE is
  converted to regular light profiles at its solved intensities, its
  PSF-convolved image is subtracted from the dataset, and what remains is a
  **source-only** inversion with one mapper, no linear func list and
  ``cond(F + λH) ≈ 1e6-1e7``.

Both systems are built by ``scripts/misc/likelihood_breakdown/active_set_steps.py``
(``linear_system_from`` / ``fixed_light_system_from``), which reads them off the
library's own inversion rather than rebuilding them.

What this cell measures
-----------------------

The reconstruction row, four ways, on each system — the dense ``F + λH`` build,
the unconstrained ``cholesky_solve``, the library's own ``nnls_pdip``
(cell-driven, so the PDIP iteration count survives), and — on S3 only — the
**certified active-set** scheme at fixed pass budgets 1..``--pass-budget-max``
(``active_set_masked_jax``, one full-size Cholesky per pass, static shapes, one
compile for every iterate of a search). Each budget records its per-pass primal
and dual violation counts, its certification flag, and its Δlog-evidence against
the PDIP solution. Both log determinants and ``log_evidence_terms`` are measured
on both systems so the evidence a scheme reaches is quoted, never assumed.

Why S3 is the interesting system
--------------------------------

The 2026-09-11 reconstruction split (``results/notes/reconstruction_row_split_2026_09.md``)
resolved the 37 ms reconstruction row into 21-22 PDIP iterations of dense KKT
Cholesky at ~1.7 ms each, and the matrix-free sweep
(``results/notes/matrix_free_pixelized_2026_09.md``) named the conditioning that
forces them. The phase-1 CPU probe (``results/misc/fixed_light_probe/``) then
measured what fixing the lens light does to that: PDIP falls from 21 to 15
iterations (rectangular) and 22 to 17 (Delaunay), and — the real result — the
certified active set, which **never certifies on S0**, certifies on S3 in 7
passes / 8 factorisations (rectangular) and 2 passes / 3 factorisations
(Delaunay). This cell is the A100 timing of that, at the fiducial tier.

Edge zeroing
------------

``inversion.reconstruction`` does **not** solve the full system when
``Settings.use_edge_zeroed_pixels`` is on (the shipped default): it subsets
``F + λH`` and ``D`` to ``inversion.solve_ids_to_keep`` and scatters exact zeros
back. Every scheme here is measured on the **same** edge-zeroed problem: the
active-set fixed set is *seeded* with that mask and those indices are never
released, so a certificate here is a certificate for the problem the library
actually poses. The one exception is the **A2 row** — the plain unconstrained
``cholesky_solve``, which drops positivity *and* the edge-zero set together; it
carries ``edge_zeroing_disabled_note: true`` in the JSON for exactly that reason.

Rows
----

``s0.rows`` / ``s3.rows`` are ``{name: seconds}`` tables in measurement order.
Like the cells' ``steps_reconstruction_sub_rows`` they are an overlapping
comparator table — never a partition of anything, and never summed.

Flags (beyond the shared ``_profile_cli`` set)
----------------------------------------------

``--mesh {rectangular,delaunay,delaunay_nn}`` (required) — which cell's model to
rebuild. ``--pass-budget-max N`` (8) — the largest active-set pass budget timed;
budgets ``1..N`` are each compiled and timed separately, because a fixed budget
is what a search would actually run. ``--source-pixels N`` sweeps the mesh size
(and skips the pins); it is a **cell-local** flag here rather than a
``_profile_cli`` one, because this branch's shared parser does not carry it.
``--no-library-row`` drops the two full-likelihood rows. ``--vmap-batch N`` adds
the batched rows.

Output
------

Results JSON and PNG are written to ``results/breakdown/imaging/`` using the
basename ``fixed_light_<mesh>_breakdown_{instrument}_v{al_version}``, or
``fixed_light_<mesh>_<config_name>`` under ``--config-name``. The A100 legs run
under ``--config-name hpc_a100_fp64_fixed_light``, which is deliberately
**not** one of the config names ``build_readme.py`` surfaces: these rows are a
comparator study for #248, not a dashboard tier.
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


import argparse
import copy
import math
import sys
import time
from pathlib import Path

import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autofit.jax import register_model as _register_model_pytrees

sys.path.insert(0, str(_profiling_root()))

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke). Verifies the import
# graph + module-level setup succeeded without running the profile. Placed
# before every argparse so a smoke run needs no flags at all.
import os as _smoke_os
import sys as _smoke_sys

from likelihood_breakdown import (  # noqa: E402
    active_set_steps,
    reconstruction_steps,
    timing,
)

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _production_config import observe_thread_env as _observe_thread_env  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    delaunay_regularization,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)

_cli = parse_profile_cli()

# Cell-local flags. ``add_help=False`` because ``parse_profile_cli`` already
# owns ``-h`` (it runs first); the flags are documented in the module docstring.
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument(
    "--mesh", choices=("rectangular", "delaunay", "delaunay_nn"), required=True
)
_cell_parser.add_argument("--pass-budget-max", type=int, default=8)
_cell_parser.add_argument("--source-pixels", type=int, default=None)
_cell_parser.add_argument("--library-row", dest="library_row", action="store_true", default=True)
_cell_parser.add_argument("--no-library-row", dest="library_row", action="store_false")
_cell_args, _ = _cell_parser.parse_known_args()

MESH = _cell_args.mesh
PASS_BUDGET_MAX = int(_cell_args.pass_budget_max)
SOURCE_PIXELS_REQUESTED = _cell_args.source_pixels
RUN_LIBRARY_ROW = bool(_cell_args.library_row)

if PASS_BUDGET_MAX < 1:
    raise ValueError(f"--pass-budget-max must be >= 1 (got {PASS_BUDGET_MAX})")

PASS_BUDGETS = tuple(range(1, PASS_BUDGET_MAX + 1))

_vmap_batch = timing.parse_vmap_batch(sys.argv)
if _vmap_batch is not None and _vmap_batch < 1:
    raise ValueError(f"--vmap-batch must be >= 1 (got {_vmap_batch})")

instrument = "hst"  # <-- change this to profile a different instrument

#: Source-pixel count each mesh's cell builds at its fiducial, and the mesh the
#: pinned log-dets below describe.
FIDUCIAL_SOURCE_PIXELS = {"rectangular": 39 * 39, "delaunay": 1500, "delaunay_nn": 1500}

#: The exact dense-Cholesky log determinants measured on euclid-ral-gpu-2 on
#: 2026-09-11 by the reconstruction-split legs (autolens_profiling#243, PR #244),
#: read from ``results/breakdown/imaging/*_hpc_a100_fp64_recon_split.json``
#: (``log_evidence_terms``). They describe **S0** — the system the library
#: solves today — and are this cell's tripwire that its model construction is
#: still the sibling cells' model construction.
PINNED_LOG_DETS = {
    "rectangular": {
        "log_det_curvature_reg": 3888.258089645771,
        "log_det_regularization": 1692.7868170935546,
    },
    "delaunay": {
        "log_det_curvature_reg": 8360.401762997288,
        "log_det_regularization": 7756.614958909804,
    },
    "delaunay_nn": {
        "log_det_curvature_reg": 7224.568777674853,
        "log_det_regularization": 6690.183751527879,
    },
}

#: Wall-clock ceiling for the O(n^3) ``cond(F + λH)`` estimates. They are a
#: recorded diagnostic, not a measurement this cell exists to make, so a backend
#: on which the first one is slow skips the second rather than paying twice.
COND_BUDGET_S = 5.0


# ---------------------------------------------------------------------------
# Profiling helpers — the shared implementations, bound to this cell's Timer
# ---------------------------------------------------------------------------

Timer = timing.Timer
block = timing.block

timer = Timer()
jit_records: dict[str, dict] = {}


def jit_profile(func, label, *args, n_repeats=10):
    """Cell-local binding of ``timing.jit_profile`` (this cell's timer/records)."""
    return timing.jit_profile(
        func,
        label,
        *args,
        n_repeats=n_repeats,
        timer=timer,
        jit_records=jit_records,
    )


def peak_bytes():
    """Device peak bytes, when the backend exposes them (``None`` on CPU)."""
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 — a backend without memory stats is not an error
        return None
    value = stats.get("peak_bytes_in_use")
    return int(value) if value is not None else None


def max_abs_rel(got, reference):
    """``(max |got - ref|, max |got - ref| / max|ref|)`` as plain floats."""
    got_a = np.asarray(got, dtype=float)
    ref_a = np.asarray(reference, dtype=float)
    abs_diff = float(np.max(np.abs(got_a - ref_a)))
    scale = max(float(np.max(np.abs(ref_a))), 1e-300)
    return abs_diff, abs_diff / scale


# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset — identical to the three cells (pixelization.py:275-333 and the
#    same block in delaunay.py / delaunay_nn.py).
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading & masking [{instrument}, mesh={MESH}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / instrument

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

# ---------------------------------------------------------------------------
# 2. Source-pixel count (``--source-pixels`` = the mesh-size sweep)
# ---------------------------------------------------------------------------

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

PIN_IS_FIDUCIAL = n_source_pixels == FIDUCIAL_SOURCE_PIXELS[MESH]

print(f"  Source pixels: {n_source_pixels} (fiducial: {FIDUCIAL_SOURCE_PIXELS[MESH]})")

# ---------------------------------------------------------------------------
# 3. Adapt image + (Delaunay family) the Hilbert image mesh
# ---------------------------------------------------------------------------
# Mirrors pixelization.py:439-446 (rectangular) and delaunay.py:380-398 /
# delaunay_nn.py:492-510 (the image-mesh branch).

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

# ---------------------------------------------------------------------------
# 4. Model construction
# ---------------------------------------------------------------------------
# Deliberately **duplicated** from the three cells rather than lifted into a
# shared ``model_setup`` module, for the reason ``matrix_free.py`` gives at the
# same place: the three constructions differ in the mesh, the regularization,
# the adapt-image dicts and the pins, and each cell's pinned evidence is a
# statement about the exact object graph it builds today. The source lines each
# block mirrors are named above it; if a cell's construction changes, this cell
# must be changed with it (the pinned log-dets above are the tripwire).

print("\n--- Model construction ---")

with timer.section("model_build"):
    # Lens light + mass + shear: identical in all three cells
    # (pixelization.py:351-369, delaunay.py:404-431, delaunay_nn.py:516-542).
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
        # pixelization.py:375-379 — the rectangular cell takes no
        # ``--regularization``; its scheme is fixed ``Constant(1.0)``.
        reg_scheme = "constant"
        reg_coefficient_model = 1.0
        mesh_obj = rect_mesh_classes(_cli)[1](shape=mesh_shape, weight_power=1.0, weight_floor=0.0)
        regularization = al.reg.Constant(coefficient=reg_coefficient_model)
        reg_provenance = {"scheme": reg_scheme, "coefficient": reg_coefficient_model}
    else:
        # delaunay.py:432-437 / delaunay_nn.py:546-553.
        if MESH == "delaunay":
            mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
        else:
            mesh_obj = al.mesh.DelaunayNN(pixels=n_mesh_vertices, areas_factor=0.5, zeroed_pixels=0)
        reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

    pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization: {reg_scheme} ({reg_provenance})")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

tracer = al.Tracer(galaxies=list(instance.galaxies))

# ``AdaptImages`` carries the per-pixel signal both the adaptive rectangular
# mesh and ``AdaptSplit`` weight by; the Delaunay family additionally needs the
# image-plane mesh grid (delaunay.py:474-489).
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
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Source pixels:           {n_source_pixels}")
print(f"  Pass budgets:            {list(PASS_BUDGETS)}")

# ---------------------------------------------------------------------------
# 5. S0 — the system the library solves today (eager FitImaging)
# ---------------------------------------------------------------------------

print("\n--- S0: full FitImaging with linear MGE lens light (eager) ---")

_settings = al.Settings(
    use_border_relocator=True,
    use_mixed_precision=_cli.use_mixed_precision,
)

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=_settings,
        xp=np,
    )
    log_evidence_ref = fit.figure_of_merit

print(f"  inversion class: {type(fit.inversion).__name__}")
print(f"  figure_of_merit (log_evidence) = {log_evidence_ref}")

with timer.section("s0_system_build"):
    system_s0 = active_set_steps.linear_system_from(fit, dataset, name="S0_current_mge60")

print(
    f"  S0: n={system_s0.n_params} (mapper {system_s0.n_mapper} + funcs {system_s0.n_funcs}), "
    f"edge-zeroed {int(system_s0.edge_zero_mask.sum())}"
)

# ---------------------------------------------------------------------------
# 6. S3 — the source-only system left once the lens light is fixed
# ---------------------------------------------------------------------------

print("\n--- S3: lens light converted to regular profiles + subtracted (eager) ---")

with timer.section("s3_system_build"):
    system_s3 = active_set_steps.fixed_light_system_from(
        fit,
        dataset,
        adapt_images=adapt_images,
        settings=_settings,
        name="S3_mge_converted_to_regular",
    )
s3_system_build_s = timer.records[-1][1]

print(
    f"  S3: n={system_s3.n_params} (mapper {system_s3.n_mapper} + funcs {system_s3.n_funcs}), "
    f"edge-zeroed {int(system_s3.edge_zero_mask.sum())}"
)
print(f"  subtracted lens-light flux: {system_s3.subtracted_light_flux:.6e}")

log_evidence_s3_library = float(system_s3.fit.figure_of_merit)
print(f"  S3 figure_of_merit (library) = {log_evidence_s3_library}")

peak_after_setup = peak_bytes()


# ===================================================================
# PART B — The per-system rows
# ===================================================================


def curvature_reg_builder(system):
    """A jit-able ``(operated_mapping_matrix, noise_map) -> F + λH`` for *system*.

    The same library call the dense cells time
    (``pixelization.py:1094-1102``), closed over this system's settings, its
    ``no_regularization_index_list`` and its regularization matrix, so the row
    is the library's own assembly rather than a hand-rolled Gram product. The
    agreement check below is what says so.
    """
    inversion = system.inversion
    settings = system.fit.settings
    no_reg_list = list(inversion.no_regularization_index_list)
    reg_full = jnp.asarray(np.asarray(inversion.regularization_matrix), dtype=jnp.float64)

    def build(operated_mapping_matrix, noise_map):
        curvature = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=operated_mapping_matrix,
            noise_map=noise_map,
            settings=settings,
            add_to_curvature_diag=True,
            no_regularization_index_list=no_reg_list,
            xp=jnp,
        )
        return curvature + reg_full

    return build


def measure_system(system, tag: str) -> dict:
    """Every row and fact this cell records for one linear system."""
    print("\n" + "-" * 70)
    print(f"ROWS — {tag}: {system.name}")
    print("-" * 70)

    rows: dict[str, float] = {}

    curv_reg = jnp.asarray(system.curvature_reg_matrix, dtype=jnp.float64)
    data_vector = jnp.asarray(system.data_vector, dtype=jnp.float64)
    operated = jnp.asarray(system.operated_mapping_matrix, dtype=jnp.float64)
    noise = jnp.asarray(np.asarray(system.dataset.noise_map.array), dtype=jnp.float64)
    Q = jnp.asarray(system.Q, dtype=jnp.float64)
    q = jnp.asarray(system.q, dtype=jnp.float64)

    # -- F + λH build (dense) ------------------------------------------------
    _, _curv_built = jit_profile(
        curvature_reg_builder(system), f"{tag}_curvature_reg_build_jit", operated, noise
    )
    rows["curvature_reg_matrix build (dense)"] = timer.records[-1][1] / 10
    _build_abs, _build_rel = max_abs_rel(_curv_built, system.curvature_reg_matrix)
    print(f"  F+λH build agreement: abs {_build_abs:.3e}  rel {_build_rel:.3e}")

    # -- Cholesky solve (unconstrained) --------------------------------------
    _, _x_cholesky = jit_profile(
        reconstruction_steps.cholesky_solve, f"{tag}_cholesky_solve_jit", curv_reg, data_vector
    )
    rows["Cholesky solve (unconstrained)"] = timer.records[-1][1] / 10
    x_cholesky = np.asarray(_x_cholesky, dtype=float)

    # -- NNLS PDIP (the library's positivity solve, cell-driven) -------------
    _, _nnls_out = jit_profile(reconstruction_steps.nnls_pdip, f"{tag}_nnls_pdip_jit", Q, q)
    _nnls_row_s = timer.records[-1][1] / 10
    rows["NNLS PDIP (cell-driven, max_iter 50)"] = _nnls_row_s

    jit_profile(
        reconstruction_steps.nnls_pdip_one_iteration, f"{tag}_nnls_pdip_one_iteration_jit", Q, q
    )
    _nnls_one_iteration_s = timer.records[-1][1] / 10
    rows["NNLS PDIP one iteration"] = _nnls_one_iteration_s

    _x_pc, _nnls_converged, _nnls_iterations = _nnls_out
    _nnls_iterations = int(_nnls_iterations)
    x_pdip = np.asarray(_x_pc, dtype=float) * system.d_scale

    # The library's own reconstruction of this system, recorded as a
    # **diagnostic, not a pin**. ``inversion.reconstruction`` solves only
    # ``solve_ids_to_keep``; the cell-driven NNLS here solves the full system,
    # so wherever edge zeroing is active the two are different solves and the
    # difference is expected to be large (1.25 on the rectangular mesh in the
    # phase-1 probe). It is ~0 where edge zeroing drops nothing (Delaunay).
    x_library = np.asarray(system.inversion.reconstruction, dtype=float)
    _recon_abs, _ = max_abs_rel(x_pdip, x_library)

    nnls = {
        "iterations": _nnls_iterations,
        "converged": bool(_nnls_converged),
        "ms_per_iteration": ((_nnls_row_s * 1e3 / _nnls_iterations) if _nnls_iterations else None),
        "one_iteration_ms": _nnls_one_iteration_s * 1e3,
        "max_iter": 50,
        "solver_tol": "jaxnnls default",
        "jacobi_preconditioning": True,
        "full_system_pdip_vs_library_reconstruction_max_abs_diff": _recon_abs,
        "full_system_pdip_vs_library_reconstruction_note": (
            "Diagnostic, never a tolerance: the library subsets the solve to "
            "solve_ids_to_keep and this row solves the full system, so the two "
            "differ by construction wherever edge zeroing drops a pixel."
        ),
    }
    print(
        f"  NNLS PDIP: {_nnls_iterations} iterations, "
        f"{nnls['ms_per_iteration']:.3f} ms/iteration, "
        f"full-system vs library (edge-zeroed) reconstruction {_recon_abs:.3e}"
    )

    # -- the two log determinants -------------------------------------------
    creg_reduced = jnp.asarray(system.curv_reg_reduced, dtype=jnp.float64)
    reg_reduced = jnp.asarray(system.reg_reduced, dtype=jnp.float64)

    jit_profile(
        reconstruction_steps.log_det_cholesky, f"{tag}_log_det_curvature_reg_jit", creg_reduced
    )
    rows["Log det Cholesky (F+λH reduced)"] = timer.records[-1][1] / 10

    jit_profile(
        reconstruction_steps.log_det_cholesky, f"{tag}_log_det_regularization_jit", reg_reduced
    )
    rows["Log det Cholesky (H reduced)"] = timer.records[-1][1] / 10

    # -- log_evidence_terms --------------------------------------------------
    # Eager, not jitted: ``reconstruction_steps.log_evidence_terms`` returns
    # Python floats by design ("called eagerly, once, for the JSON"), so a
    # ``jax.jit`` of it would not trace. The row is the eager cost and says so.
    with timer.section(f"{tag}_log_evidence_terms_eager"):
        evidence_terms = system.log_evidence_terms(x_pdip)
    rows["log_evidence_terms (eager, PDIP solution)"] = timer.records[-1][1]

    log_evidence_pdip = float(evidence_terms["log_evidence"])
    log_evidence_cholesky = float(system.log_evidence(x_cholesky))

    # The library's own answer for this system — PDIP on the **edge-zeroed**
    # problem. This, not the full-system PDIP row above, is the same-problem
    # reference every alternative scheme is scored against: the active set is
    # seeded with the edge-zero mask and never releases it, so it is solving
    # this problem. The gap between the two references is the edge-zeroing
    # penalty and is recorded as ``d_log_evidence_full_system_pdip_minus_library``.
    log_evidence_library = float(system.log_evidence(x_library))

    print(f"  log_evidence(library, edge-zeroed) = {log_evidence_library:.6f}")
    print(f"  log_evidence(PDIP, full system)    = {log_evidence_pdip:.6f}")
    print(f"  log_evidence(Cholesky)             = {log_evidence_cholesky:.6f}")

    # -- cond(F + λH), budgeted ---------------------------------------------
    cond = {"value": None, "seconds": None, "skipped": None}
    if measure_system.cond_budget_spent > COND_BUDGET_S:
        cond["skipped"] = (
            f"an earlier cond() took {measure_system.cond_budget_spent:.1f} s, above the "
            f"{COND_BUDGET_S:g} s budget; this one was not attempted"
        )
        print(f"  cond(F+λH) SKIPPED — {cond['skipped']}")
    else:
        _t0 = time.perf_counter()
        cond["value"] = float(jnp.linalg.cond(curv_reg))
        cond["seconds"] = time.perf_counter() - _t0
        measure_system.cond_budget_spent = max(measure_system.cond_budget_spent, cond["seconds"])
        print(f"  cond(F+λH) = {cond['value']:.6e}  ({cond['seconds']:.2f} s)")

    print(f"\n  {tag} rows:")
    _w = max(len(k) for k in rows)
    for _label, _dt in rows.items():
        print(f"    {_label:<{_w}}  {_dt * 1e3:>12.3f} ms")

    return {
        "name": system.name,
        "n_params": int(system.n_params),
        "n_mapper": int(system.n_mapper),
        "n_funcs": int(system.n_funcs),
        "edge_zeroed_active": bool(system.ids_to_keep is not None),
        "edge_zeroed_pixels": int(system.edge_zero_mask.sum()),
        "rows": {k: float(v) for k, v in rows.items()},
        "nnls": nnls,
        "log_evidence_terms": evidence_terms,
        "log_evidence_pdip": log_evidence_pdip,
        "log_evidence_cholesky": log_evidence_cholesky,
        "log_evidence_library_edge_zeroed": log_evidence_library,
        "d_log_evidence_full_system_pdip_minus_library": log_evidence_pdip - log_evidence_library,
        "curvature_reg_build_max_abs_diff_vs_library": _build_abs,
        "curvature_reg_build_max_rel_diff_vs_library": _build_rel,
        "cond": cond,
        "_x_pdip": x_pdip,
        "_x_cholesky": x_cholesky,
        "_x_library": x_library,
    }


#: Running ceiling for the O(n^3) cond() diagnostics (see ``COND_BUDGET_S``).
measure_system.cond_budget_spent = 0.0

s0 = measure_system(system_s0, "s0")
peak_after_s0 = peak_bytes()

s3 = measure_system(system_s3, "s3")
peak_after_s3 = peak_bytes()

s3["subtracted_light_flux"] = float(system_s3.subtracted_light_flux)
s3["system_build_s"] = float(s3_system_build_s)
# ``fit.figure_of_merit`` of the source-only fit, beside the same number
# recomputed by ``reconstruction_steps.log_evidence_terms`` on the library's own
# reconstruction. They must agree; the difference is the recipe check.
s3["log_evidence_figure_of_merit"] = log_evidence_s3_library
s3["d_log_evidence_terms_minus_figure_of_merit"] = (
    s3["log_evidence_library_edge_zeroed"] - log_evidence_s3_library
)
s0["log_evidence_figure_of_merit"] = float(log_evidence_ref)
s0["d_log_evidence_terms_minus_figure_of_merit"] = s0["log_evidence_library_edge_zeroed"] - float(
    log_evidence_ref
)

# ---------------------------------------------------------------------------
# The mapper-block identity — a hard assertion, not a recorded diagnostic
# ---------------------------------------------------------------------------
# Fixing the lens light removes 60 unregularised columns from the linear system.
# It does **not** touch the mapper block: ``F_mapper = Λᵀ N⁻¹ Λ`` depends on the
# mapping matrix and the noise map, neither of which the subtraction changes,
# and ``H`` is the source's regularization matrix either way. So the two
# rank-stripped log-dets must be the *same numbers* on S0 and S3 — and if they
# are not, the S3 fit is not the S0 fit's source inversion and every Δevidence
# in this JSON is comparing two different problems.

_MAPPER_LOGDET_RTOL = 1e-6

_mapper_log_dets = {}
for _key, _matrix_attr in (
    ("log_det_curvature_reg", "curv_reg_reduced"),
    ("log_det_regularization", "reg_reduced"),
):
    _s0_value = float(
        reconstruction_steps.log_det_cholesky(
            jnp.asarray(getattr(system_s0, _matrix_attr), dtype=jnp.float64)
        )
    )
    _s3_value = float(
        reconstruction_steps.log_det_cholesky(
            jnp.asarray(getattr(system_s3, _matrix_attr), dtype=jnp.float64)
        )
    )
    _rel = abs(_s3_value - _s0_value) / max(abs(_s0_value), 1e-300)
    _mapper_log_dets[_key] = {"s0": _s0_value, "s3": _s3_value, "rel_diff": _rel}
    print(f"  mapper-block {_key}: S0 {_s0_value:.9f}  S3 {_s3_value:.9f}  rel {_rel:.3e}")
    if _rel > _MAPPER_LOGDET_RTOL:
        raise AssertionError(
            f"S3's mapper-block {_key} ({_s3_value!r}) does not match S0's ({_s0_value!r}) "
            f"to {_MAPPER_LOGDET_RTOL:g} (relative difference {_rel:.3e}). Fixing the lens "
            f"light must leave the mapper block of F + λH untouched; a mismatch means the "
            f"source-only fit is not the same source inversion, and every Δevidence in this "
            f"run compares two different problems."
        )

# ---------------------------------------------------------------------------
# The certified active set on S3, at fixed pass budgets
# ---------------------------------------------------------------------------
# Seeded with the library's edge-zero mask, which is never released: the
# certificate is then a certificate for the problem the library actually poses.
# Each budget is a separate ``lax.scan`` length, so each is its own compile —
# which is the point: a search runs one fixed budget, every iterate.

print("\n" + "=" * 70)
print("S3 — CERTIFIED ACTIVE SET (pass budgets)")
print("=" * 70)

_Q_s3 = jnp.asarray(system_s3.Q, dtype=jnp.float64)
_q_s3 = jnp.asarray(system_s3.q, dtype=jnp.float64)
_fixed0_s3 = jnp.asarray(system_s3.edge_zero_mask, dtype=bool)
_d_scale_s3 = system_s3.d_scale

#: The same-problem reference every S3 alternative is scored against: the
#: library's own reconstruction of S3, which is PDIP on the **edge-zeroed**
#: problem. The full-system PDIP row is recorded beside it as a second
#: reference, because the gap between the two is the edge-zeroing penalty (+215
#: nats on the 1521-pixel rectangular mesh in the phase-1 probe) and not
#: something any scheme here did.
reference_log_evidence = s3["log_evidence_library_edge_zeroed"]
reference_log_evidence_full_system_pdip = s3["log_evidence_pdip"]

active_set_budgets: list[dict] = []
certified_budget = None

for _budget in PASS_BUDGETS:

    def _masked(Q, q, fixed0, budget=_budget):
        return active_set_steps.active_set_masked_jax(Q, q, fixed0, budget)

    _, _out = jit_profile(_masked, f"s3_active_set_masked_p{_budget}_jit", _Q_s3, _q_s3, _fixed0_s3)
    _per_call = timer.records[-1][1] / 10

    _certified = [bool(c) for c in np.asarray(_out["certified"])]
    _certified_at = next((i + 1 for i, c in enumerate(_certified) if c), None)
    _x_physical = np.asarray(_out["x"], dtype=float) * _d_scale_s3

    _d_evidence = active_set_steps.active_set_evidence_error(
        system_s3, _x_physical, reference_log_evidence=reference_log_evidence
    )
    _d_evidence_clipped = active_set_steps.active_set_evidence_error(
        system_s3, _x_physical, reference_log_evidence=reference_log_evidence, clip=True
    )
    _d_evidence_vs_pdip = active_set_steps.active_set_evidence_error(
        system_s3, _x_physical, reference_log_evidence=reference_log_evidence_full_system_pdip
    )
    _abs_vs_library, _rel_vs_library = max_abs_rel(_x_physical, s3["_x_library"])
    _abs_vs_pdip, _rel_vs_pdip = max_abs_rel(_x_physical, s3["_x_pdip"])

    _entry = {
        "pass_budget": _budget,
        "ms": _per_call * 1e3,
        "n_fixed_per_pass": [int(v) for v in np.asarray(_out["n_fixed"])],
        "n_primal_violations_per_pass": [int(v) for v in np.asarray(_out["n_primal_violations"])],
        "n_dual_violations_per_pass": [int(v) for v in np.asarray(_out["n_dual_violations"])],
        "certified_per_pass": _certified,
        "certified": bool(any(_certified)),
        "certified_at_pass": _certified_at,
        "d_log_evidence_vs_library": _d_evidence,
        "d_log_evidence_vs_library_clipped": _d_evidence_clipped,
        "d_log_evidence_vs_full_system_pdip": _d_evidence_vs_pdip,
        "max_abs_diff_vs_library": _abs_vs_library,
        "max_rel_diff_vs_library": _rel_vs_library,
        "max_abs_diff_vs_full_system_pdip": _abs_vs_pdip,
        "max_rel_diff_vs_full_system_pdip": _rel_vs_pdip,
        "n_negative_entries": int(np.sum(_x_physical < 0.0)),
    }
    active_set_budgets.append(_entry)

    if certified_budget is None and _entry["certified"]:
        certified_budget = _budget

    print(
        f"  budget {_budget}: {_per_call * 1e3:9.3f} ms  certified={_entry['certified']} "
        f"(at pass {_certified_at})  Δlog-evidence vs library {_d_evidence:+.6e} nats  "
        f"primal {_entry['n_primal_violations_per_pass']}  "
        f"dual {_entry['n_dual_violations_per_pass']}"
    )

active_set_block = {
    "reference": (
        "the library's own reconstruction of S3 — PDIP on the edge-zeroed problem, "
        "which is the problem the seeded active set solves"
    ),
    "reference_log_evidence": reference_log_evidence,
    "reference_log_evidence_full_system_pdip": reference_log_evidence_full_system_pdip,
    "edge_zeroing_penalty_nats": (reference_log_evidence_full_system_pdip - reference_log_evidence),
    "fixed0_is_edge_zero_mask": True,
    "n_fixed0": int(system_s3.edge_zero_mask.sum()),
    "tau_rel": active_set_steps.TAU_REL_DEFAULT,
    "pass_budget_max": PASS_BUDGET_MAX,
    "smallest_certifying_budget": certified_budget,
    "budgets": active_set_budgets,
    "note": (
        "Each budget is its own lax.scan length and therefore its own compile — "
        "a search runs one fixed budget on every iterate, which is why the table "
        "is per budget rather than a single run to convergence. The fixed set is "
        "seeded with the library's edge-zero mask and those indices are never "
        "released, so a certified row certifies the problem the library poses."
    ),
}

# ---------------------------------------------------------------------------
# A2 — drop positivity altogether on S3
# ---------------------------------------------------------------------------
# The cheapest row there is: one Cholesky factorisation and two triangular
# solves. It is recorded with ``edge_zeroing_disabled_note`` because it drops
# the edge-zero set along with the constraint — the solve is the full system,
# not the library's subset, so its evidence is not a like-for-like "same problem,
# cheaper solver" number the way the active-set rows are.

_x_a2 = s3["_x_cholesky"]
_a2_abs, _a2_rel = max_abs_rel(_x_a2, s3["_x_pdip"])
_a2_abs_library, _a2_rel_library = max_abs_rel(_x_a2, s3["_x_library"])

a2_block = {
    "row": "Cholesky solve (unconstrained)",
    "ms": s3["rows"]["Cholesky solve (unconstrained)"] * 1e3,
    "log_evidence": s3["log_evidence_cholesky"],
    "d_log_evidence_vs_library": s3["log_evidence_cholesky"] - reference_log_evidence,
    "d_log_evidence_vs_library_clipped": active_set_steps.active_set_evidence_error(
        system_s3, _x_a2, reference_log_evidence=reference_log_evidence, clip=True
    ),
    "d_log_evidence_vs_full_system_pdip": (
        s3["log_evidence_cholesky"] - reference_log_evidence_full_system_pdip
    ),
    "max_abs_diff_vs_library": _a2_abs_library,
    "max_rel_diff_vs_library": _a2_rel_library,
    "max_abs_diff_vs_full_system_pdip": _a2_abs,
    "max_rel_diff_vs_full_system_pdip": _a2_rel,
    "n_negative_entries": int(np.sum(_x_a2 < 0.0)),
    "negative_flux_fraction": float(
        np.sum(np.abs(_x_a2[_x_a2 < 0.0])) / max(float(np.sum(np.abs(_x_a2))), 1e-300)
    ),
    "edge_zeroing_disabled_note": True,
    "note": (
        "The unconstrained solve drops the non-negativity constraint AND the "
        "edge-zero set (it solves the full system, not solve_ids_to_keep). Its "
        "evidence can therefore sit above the constrained optimum: that is an "
        "infeasible point scoring well, not a win."
    ),
}

print("\n--- A2 (S3, positivity dropped) ---")
print(
    f"  {a2_block['ms']:.3f} ms   Δlog-evidence vs library "
    f"{a2_block['d_log_evidence_vs_library']:+.6f} nats   "
    f"{a2_block['n_negative_entries']} negative entries "
    f"({a2_block['negative_flux_fraction'] * 100:.3f} % of |flux|)"
)

# ---------------------------------------------------------------------------
# Batched rows (--vmap-batch N)
# ---------------------------------------------------------------------------
# Identical lanes, exactly as the cells' ``steps_vmap_per_call`` rows: every lane
# is a copy of the same system, so all lanes converge on the same iteration and
# the ``while_loop`` never waits for a straggler. Best case for amortization,
# not a representative one.

vmap_block: dict = {}
vmap_error = None

if _vmap_batch is not None:
    print(f"\n--- Batched rows (--vmap-batch {_vmap_batch}) ---")
    import traceback as _vmap_traceback

    try:
        _vmap_budget = certified_budget if certified_budget is not None else PASS_BUDGET_MAX

        for _tag, _system in (("s0", system_s0), ("s3", system_s3)):
            _Qb = jnp.broadcast_to(
                jnp.asarray(_system.Q, dtype=jnp.float64),
                (_vmap_batch, _system.n_params, _system.n_params),
            )
            _qb = jnp.broadcast_to(
                jnp.asarray(_system.q, dtype=jnp.float64), (_vmap_batch, _system.n_params)
            )

            def _nnls_lane(Qq):
                return reconstruction_steps.nnls_pdip(Qq[0], Qq[1])[0]

            vmap_block[f"{_tag}_nnls_pdip_per_call_s"] = timing.vmap_profile(
                _nnls_lane,
                f"{_tag}_nnls_pdip",
                (_Qb, _qb),
                _vmap_batch,
                timer=timer,
                jit_records=jit_records,
            )

        _Qb3 = jnp.broadcast_to(_Q_s3, (_vmap_batch, system_s3.n_params, system_s3.n_params))
        _qb3 = jnp.broadcast_to(_q_s3, (_vmap_batch, system_s3.n_params))
        _fb3 = jnp.broadcast_to(_fixed0_s3, (_vmap_batch, system_s3.n_params))

        def _masked_lane(args):
            return active_set_steps.active_set_masked_jax(args[0], args[1], args[2], _vmap_budget)[
                "x"
            ]

        vmap_block["s3_active_set_masked_per_call_s"] = timing.vmap_profile(
            _masked_lane,
            f"s3_active_set_masked_p{_vmap_budget}",
            (_Qb3, _qb3, _fb3),
            _vmap_batch,
            timer=timer,
            jit_records=jit_records,
        )
        vmap_block["active_set_pass_budget"] = _vmap_budget
        vmap_block["batch"] = int(_vmap_batch)
        vmap_block["note"] = (
            f"The @vmap {_vmap_batch} rows batch {_vmap_batch} copies of the same system; "
            f"identical lanes converge on the same iteration, so the lax.while_loop never "
            f"runs on for a straggler. Best case for amortization, not a representative one."
        )
    except Exception:  # noqa: BLE001 — a vmap failure must not lose the unbatched rows
        vmap_error = _vmap_traceback.format_exc()
        print("  VMAP FAILED — unbatched rows are unaffected. Traceback:")
        print(vmap_error)

# ---------------------------------------------------------------------------
# The library path — one full likelihood call, S0 and S3
# ---------------------------------------------------------------------------
# What the rows above sit inside. S3's is the source-only tracer on the
# subtracted dataset, dense, so the two numbers differ by the lens-light work
# and by the 60 columns leaving the linear system — which is the whole claim.

library_row: dict = {"status": "skipped" if not RUN_LIBRARY_ROW else "ok"}

if RUN_LIBRARY_ROW:
    print("\n--- Library path (full likelihood, jit) ---")
    import traceback as _library_traceback

    try:
        params_tree_s0 = jax.tree_util.tree_map(jnp.asarray, instance)

        # The S3 instance is the S0 instance with the lens's light removed —
        # the same galaxies ``fixed_light_system_from`` put in its tracer. The
        # adapt images are keyed by name as well as by object, and the name
        # keys are what the analysis path reads, so the deep copy is safe.
        instance_s3 = copy.deepcopy(instance)
        _source_only_galaxies = list(system_s3.source_only_tracer.galaxies)
        instance_s3.galaxies.lens = _source_only_galaxies[0]
        instance_s3.galaxies.source = _source_only_galaxies[1]
        params_tree_s3 = jax.tree_util.tree_map(jnp.asarray, instance_s3)

        for _tag, _ds, _tree in (
            ("s0", dataset, params_tree_s0),
            ("s3", system_s3.dataset, params_tree_s3),
        ):
            _analysis = al.AnalysisImaging(
                dataset=_ds,
                adapt_images=adapt_images,
                settings=_settings,
                use_jax=True,
            )

            def _likelihood(tree, analysis=_analysis):
                return analysis.log_likelihood_function(instance=tree)

            _, _value = jit_profile(_likelihood, f"{_tag}_library_likelihood_jit", _tree)
            library_row[f"{_tag}_ms"] = timer.records[-1][1] / 10 * 1e3
            library_row[f"{_tag}_log_likelihood"] = float(_value)
            print(f"  {_tag}: {library_row[f'{_tag}_ms']:.3f} ms  -> {float(_value):.6f}")

        library_row["ratio_s3_over_s0"] = library_row["s3_ms"] / library_row["s0_ms"]
        library_row["note"] = (
            "Both legs are dense. S3 is the source-only tracer on the "
            "lens-light-subtracted dataset, so it pays no lens-light image and "
            "carries 60 fewer columns in the linear system; it is NOT a "
            "like-for-like likelihood of the same model."
        )
    except Exception:  # noqa: BLE001 — the comparator rows must survive this
        library_row = {"status": "failed", "error": _library_traceback.format_exc()}
        print("  LIBRARY ROW FAILED — every other row is unaffected. Traceback:")
        print(library_row["error"])
else:
    print("\n--- Library path SKIPPED (--no-library-row) ---")

peak_after_all = peak_bytes()

# ===================================================================
# Summary + JSON + PNG
# ===================================================================

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

print("\n" + "=" * 70)
print(f"FIXED-LIGHT BREAKDOWN — {MESH} — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  S0 PDIP iterations: {s0['nnls']['iterations']}   ({s0['n_params']} params)")
print(f"  S3 PDIP iterations: {s3['nnls']['iterations']}   ({s3['n_params']} params)")
print(f"  S3 smallest certifying pass budget: {certified_budget}")
print("=" * 70)
print("  These rows are overlapping comparators on one model evaluation — never a partition,")
print("  never summed. See the module docstring.")

# The reconstruction-row alternatives, in the order the chart draws them.
reconstruction_alternatives: list[tuple[str, float]] = [
    ("PDIP (S0, library today)", s0["rows"]["NNLS PDIP (cell-driven, max_iter 50)"] * 1e3),
    ("PDIP (S3, source-only)", s3["rows"]["NNLS PDIP (cell-driven, max_iter 50)"] * 1e3),
]
for _entry in active_set_budgets:
    _mark = " ✓" if _entry["certified"] else ""
    reconstruction_alternatives.append(
        (f"active set (S3, budget {_entry['pass_budget']}){_mark}", _entry["ms"])
    )
reconstruction_alternatives.append(
    ("Cholesky (S3, A2: no positivity)", s3["rows"]["Cholesky solve (unconstrained)"] * 1e3)
)

_configuration = {
    "pixel_scale_arcsec": pixel_scale,
    "mask_radius_arcsec": mask_radius,
    "image_pixels_masked": int(n_image_pixels),
    "over_sampled_pixels": int(n_over_sampled_pixels),
    "mesh": MESH,
    "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
    "rect_mesh": _cli.rect_mesh if MESH == "rectangular" else None,
    "source_pixels": int(n_source_pixels),
    "source_pixels_requested": (
        int(SOURCE_PIXELS_REQUESTED) if SOURCE_PIXELS_REQUESTED is not None else None
    ),
    "inversion_path": "dense",
    "pass_budget_max": PASS_BUDGET_MAX,
    "thread_env": _observe_thread_env(),
    "memo": "library_default (inert on the JAX path)",
    "over_sample_size_lp_rule": {
        "sub_size_list": [4, 2, 2],
        "radial_list": [0.3, 0.6],
        "centre": [0.0, 0.0],
    },
}

breakdown_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": _configuration,
    "regularization": reg_provenance,
    "log_evidence_eager": float(log_evidence_ref),
    "s0": {k: v for k, v in s0.items() if not k.startswith("_")},
    "s3": {k: v for k, v in s3.items() if not k.startswith("_")},
    "mapper_block_log_dets": _mapper_log_dets,
    "mapper_block_log_det_rtol": _MAPPER_LOGDET_RTOL,
    "active_set": active_set_block,
    "a2": a2_block,
    "vmap16": vmap_block,
    "library_row": library_row,
    "edge_zeroed_pixels": {
        "s0": int(system_s0.edge_zero_mask.sum()),
        "s3": int(system_s3.edge_zero_mask.sum()),
        "note": (
            "The count the library holds at exactly zero via solve_ids_to_keep. "
            "The active-set fixed set is seeded with it and never releases it."
        ),
    },
    "reconstruction_alternatives_ms": dict(reconstruction_alternatives),
    "peak_bytes": {
        "after_setup": peak_after_setup,
        "after_s0": peak_after_s0,
        "after_s3": peak_after_s3,
        "after_all": peak_after_all,
        "note": (
            "jax.devices()[0].memory_stats()['peak_bytes_in_use']; None on a backend "
            "that does not report it (CPU). Process-wide high-water marks."
        ),
    },
    "jit_phases": jit_records,
}

if _vmap_batch is not None:
    breakdown_summary["vmap_batch"] = int(_vmap_batch)
    if vmap_error is not None:
        breakdown_summary["vmap_error"] = vmap_error

_cell_name = f"fixed_light_{MESH}"
if SOURCE_PIXELS_REQUESTED is not None:
    # This branch's ``_profile_cli`` has no ``--source-pixels``, so the ``_n<N>``
    # suffix that keeps a sweep leg's JSON off the fiducial one is applied here.
    _cell_name = f"{_cell_name}_n{int(n_source_pixels)}"

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"{_cell_name}_breakdown_{instrument}_v{al_version}",
    cell=_cell_name,
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [name for name, _ in reconstruction_alternatives]
times_ms = [ms for _, ms in reconstruction_alternatives]
colors = []
for name in labels:
    if name.startswith("PDIP (S0"):
        colors.append("#C44E52")
    elif name.startswith("PDIP (S3"):
        colors.append("#DD8452")
    elif name.startswith("Cholesky"):
        colors.append("#55A868")
    else:
        colors.append("#4C72B0")

fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(labels))))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times_ms, color=colors, edgecolor="white", height=0.6)

for bar, t in zip(bars, times_ms):
    ax.text(
        bar.get_width() + max(times_ms) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.3f} ms",
        va="center",
        fontsize=8,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Time per call (ms)", fontsize=11)
fig.suptitle(
    f"Reconstruction-row alternatives, lens light fixed — {MESH} — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_source_pixels} source pixels  |  S0 {s0['nnls']['iterations']} / "
    f"S3 {s3['nnls']['iterations']} PDIP iterations  |  ✓ = certified",
    fontsize=9,
)
ax.margins(x=0.18)
fig.tight_layout()

fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

# ===================================================================
# Pins — the 2026-09-11 exact S0 log-dets
# ===================================================================
# The pins describe S0 at the fiducial mesh (39x39 / 1500 / 1500) and nothing
# else, so a run with a non-fiducial ``--source-pixels`` skips them and writes
# ``pinned_expected: null``. They are the tripwire on this cell's model
# construction: if it has drifted from the sibling breakdown cells', the
# comparison between S0 and S3 is still internally consistent but no longer
# describes the fiducial the rest of #248 is written about.

if not PIN_IS_FIDUCIAL:
    print(
        f"  Pinned log-det check SKIPPED: --source-pixels {SOURCE_PIXELS_REQUESTED} builds a "
        f"{n_source_pixels}-pixel mesh, not the {FIDUCIAL_SOURCE_PIXELS[MESH]}-pixel fiducial "
        f"the pins describe."
    )
    record_pinned_check(dict_path, None, [])
else:
    _drift_records = []

    for _key, _expected in PINNED_LOG_DETS[MESH].items():
        _record = check_pinned(
            s0["log_evidence_terms"][_key],
            _expected,
            label=f"imaging/fixed_light[{instrument}, {MESH}] S0 {_key}",
            rtol=1e-4,
        )
        if _record is None:
            print(f"  Pin PASSED: S0 {_key} matches {_expected:.6f}")
        else:
            _drift_records.append(_record)

    record_pinned_check(dict_path, PINNED_LOG_DETS[MESH], _drift_records)

print("\nFinished.")
