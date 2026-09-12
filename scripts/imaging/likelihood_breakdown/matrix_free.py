"""
JAX Profiling: Matrix-Free Pixelized Imaging Likelihood — CG + SLQ Reference
===========================================================================

The measurement cell for the matrix-free line (autolens_profiling#247). It
builds the **same** fiducial imaging model the three pixelized breakdown cells
build — ``imaging/pixelization.py`` (39x39 rectangular bilinear adapt mesh),
``imaging/delaunay.py`` (1500-vertex Hilbert/Delaunay) and
``imaging/delaunay_nn.py`` (the same mesh with Sibson natural-neighbour
interpolation) — attaches the w-tilde sparse operator, and then drives every
consumer of ``F + λH`` **without ever forming it**, through the kernels in
``scripts/misc/likelihood_breakdown/matrix_free_steps.py``:

- the linear solve, by preconditioned conjugate gradients on operator products;
- positivity, by a primal-dual interior point whose inner Newton solve is that
  same PCG instead of a dense KKT Cholesky;
- both log determinants, by stochastic Lanczos quadrature with fixed Rademacher
  probes.

What this cell measures
-----------------------

Per-call time and **agreement** for each of those, against the exact dense
comparators measured in the same process: ``reconstruction_steps.cholesky_solve``
(the unconstrained solve), ``reconstruction_steps.nnls_pdip`` (the library's
own positivity solve, cell-driven so the iteration count survives),
``log_det_cholesky`` on the two reduced blocks, and ``log_evidence_terms``. Every
row records the iteration counts the crossover argument turns on — CG iterations
per solve, PDIP iterations and total inner CG iterations — and the SLQ rows
record the estimator's own spread alongside its error.

What this cell deliberately does **not** measure
------------------------------------------------

A faster likelihood at the fiducial tier. The 2026-09-11 reconstruction split
(``results/notes/a100_pixelized_baseline_2026_09.md``, "Reconstruction split")
settled that: at 1500 source pixels the exact unconstrained solve is 1.52-1.58 ms
and the 37 ms reconstruction row is 21-22 PDIP iterations of dense KKT Cholesky,
so **a matrix-free line is not a speed play at n ≈ 1500** and this cell is not
trying to be one. It is the reference implementation and the fiducial row the
N_src sweep (phase 3, ``--source-pixels``) is anchored to; the crossover the
sweep finds is the deliverable.

It also never builds a dense ``F`` on the matrix-free path. The dense
comparators above *are* built, once, eagerly, so the agreement columns exist —
and at large ``--source-pixels`` they are the first thing that stops fitting.
That is not a failure of the run: the dense block is wrapped, and a run whose
comparators OOM still emits every matrix-free row with
``dense_comparators.status: "oom"``.

Positivity
----------

The **unconstrained** PCG solve is the primary path — it is what the crossover
measures, and it is the honest comparator for ``cholesky_solve``. The
matrix-free PDIP is measured beside it as the honest comparator for the 37 ms
NNLS row, not as the headline. A projection / penalty scheme is deliberately not
built: it changes the estimator, not just the solver.

Edge zeroing
------------

``inversion.reconstruction`` does **not** solve the full system when
``Settings.use_edge_zeroed_pixels`` is on (the shipped default): it subsets
``F + λH`` and ``D`` to ``inversion.solve_ids_to_keep`` and scatters exact zeros
back. The kernels here solve the full system, so every comparison in this cell
is against the cell-driven full-system recipe (``cholesky_solve`` /
``jacobi_scaled`` + ``nnls_pdip``), never against ``inversion.reconstruction``.
Whether edge zeroing is active, and how many parameters it drops, is recorded in
``matrix_free.edge_zeroed_active`` / ``edge_zeroed_pixels`` so a row never has to
be re-derived from the settings.

Rows
----

``steps_matrix_free_rows`` is a **list** of ``{name, ms, ...}`` — unlike the
cells' ``steps_reconstruction_sub_rows`` dict, because each row carries its own
facts (iteration counts, residuals, agreement, SLQ spread). Like those sub-rows
it is an overlapping comparator table, never a partition of anything, and never
summed.

Flags (beyond the shared ``_profile_cli`` set)
----------------------------------------------

``--mesh {rectangular,delaunay,delaunay_nn}`` (required) — which cell's model to
rebuild. ``--cg-tol`` (1e-10) the PCG relative-residual target for the
unconstrained solves; the PDIP's inner solves floor it at 1e-8 (see
``PDIP_CG_TOL``). ``--slq-probes`` /
``--slq-steps`` comma lists (``4,8,16,32`` / ``20,40,80``) — every pair is timed,
which is the noise-vs-probes curve. ``--slq-key`` (0) fixes the probes, so a
sweep sees a smooth bias rather than jitter. ``--no-pdip`` drops the positivity
rows. ``--source-pixels N`` sweeps the mesh size (and skips the pins).
``--vmap-batch N`` adds the batched rows.

Output
------

Results JSON and PNG are written to ``results/breakdown/imaging/`` using the
basename ``matrix_free_<mesh>_breakdown_{instrument}_v{al_version}``, or
``matrix_free_<mesh>_<config_name>`` under ``--config-name``.

The results note this cell feeds is
``results/notes/matrix_free_pixelized_2026_09.md`` (written in phase 4 of #247).
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
import math
import sys
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
    matrix_free_steps,
    reconstruction_steps,
    sparse_steps,
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
_cell_parser.add_argument("--cg-tol", type=float, default=1.0e-10)
_cell_parser.add_argument("--slq-probes", default="4,8,16,32")
_cell_parser.add_argument("--slq-steps", default="20,40,80")
_cell_parser.add_argument("--slq-key", type=int, default=0)
_cell_parser.add_argument("--pdip", dest="pdip", action="store_true", default=True)
_cell_parser.add_argument("--no-pdip", dest="pdip", action="store_false")
_cell_args, _ = _cell_parser.parse_known_args()

MESH = _cell_args.mesh
CG_TOL = float(_cell_args.cg_tol)
SLQ_PROBES = tuple(int(t) for t in _cell_args.slq_probes.split(",") if t.strip())
SLQ_STEPS = tuple(int(t) for t in _cell_args.slq_steps.split(",") if t.strip())
SLQ_KEY = int(_cell_args.slq_key)
RUN_PDIP = bool(_cell_args.pdip)

#: The (probes, steps) pair every "default" row is quoted at — the evidence
#: comparison and the pinned SLQ drift. Falls back to the last available pair
#: when a run narrows the lists (a local check with ``--slq-probes 4,8``).
SLQ_DEFAULT_PROBES = 16 if 16 in SLQ_PROBES else SLQ_PROBES[-1]
SLQ_DEFAULT_STEPS = 40 if 40 in SLQ_STEPS else SLQ_STEPS[-1]

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
#: (``log_evidence_terms``). They are what an SLQ estimate is checked against.
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


#: ``{name, ms, ...}`` rows, in measurement order.
matrix_free_rows: list[dict] = []


def add_row(name: str, seconds: float, **facts) -> dict:
    """Append a row, converting to ms and printing it as it lands."""
    row = {"name": name, "ms": float(seconds) * 1e3, **facts}
    matrix_free_rows.append(row)
    extras = "  ".join(f"{k}={v}" for k, v in facts.items() if not isinstance(v, (dict, list)))
    print(f"  ROW {name:<48} {row['ms']:10.3f} ms  {extras}")
    return row


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

# The w-tilde sparse operator is not optional here: ``matrix_free_context_from``
# rides on ``SparseContext``, whose operator supplies the FFT state every matvec
# uses, and the mapper's ``sparse_triplets_curvature`` only exist on an
# ``InversionImagingSparse``. So this cell always attaches it, whatever
# ``--sparse`` says, and records the path as ``matrix_free``.
with timer.section("sparse_operator_build"):
    dataset = dataset.apply_sparse_operator(batch_size=_cli.sparse_batch_size)
sparse_operator_build_s = timer.records[-1][1]

# ---------------------------------------------------------------------------
# 2. Source-pixel count (``--source-pixels`` = the N_src sweep, #247)
# ---------------------------------------------------------------------------

if MESH == "rectangular":
    mesh_pixels_yx = 39 if _cli.source_pixels is None else int(round(math.sqrt(_cli.source_pixels)))
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
    n_source_pixels = mesh_pixels_yx * mesh_pixels_yx
    n_mesh_vertices = None
else:
    mesh_pixels_yx = None
    mesh_shape = None
    n_mesh_vertices = 1500 if _cli.source_pixels is None else int(_cli.source_pixels)
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
# shared ``model_setup`` module. The three constructions differ in the mesh, the
# regularization, the adapt-image dicts and the pins, and each cell's pinned
# evidence is a statement about the exact object graph it builds today. Lifting
# them would have meant re-measuring all three pins to prove nothing moved,
# which is an A100 job, not a local one — #247 phase 2 is not the place to spend
# that. The source lines each block mirrors are named above it; if a cell's
# construction changes, this cell must be changed with it (the pinned log-dets
# below are the tripwire).

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
print(f"  Sparse batch size:       {_cli.sparse_batch_size}")

# ---------------------------------------------------------------------------
# 5. Full-pipeline reference (FitImaging) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitImaging (eager baseline) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(
            use_border_relocator=True,
            use_mixed_precision=_cli.use_mixed_precision,
        ),
        xp=np,
    )
    log_evidence_ref = fit.figure_of_merit

inversion = fit.inversion
print(f"  inversion class: {type(inversion).__name__}")
print(f"  figure_of_merit (log_evidence) = {log_evidence_ref}")

# Edge zeroing: ``inversion.reconstruction`` solves only ``solve_ids_to_keep``
# when it is not None (PyAutoArray abstract.py:514), so the exact comparator for
# the full-system kernels here is the cell-driven recipe, never
# ``inversion.reconstruction``. Recorded either way.
_ids_to_keep = inversion.solve_ids_to_keep
EDGE_ZEROED_ACTIVE = _ids_to_keep is not None
_n_kept = int(np.asarray(_ids_to_keep).shape[0]) if EDGE_ZEROED_ACTIVE else None
EDGE_ZEROED_PIXELS = int(inversion.total_params) - _n_kept if EDGE_ZEROED_ACTIVE else 0
print(
    f"  edge zeroing active: {EDGE_ZEROED_ACTIVE}"
    + (
        f" — {EDGE_ZEROED_PIXELS} of {inversion.total_params} params dropped"
        if EDGE_ZEROED_ACTIVE
        else ""
    )
)

# ---------------------------------------------------------------------------
# 6. Contexts
# ---------------------------------------------------------------------------

print("\n--- Matrix-free context build (eager, one-off) ---")

with timer.section("sparse_context_build"):
    sparse_ctx = sparse_steps.sparse_context_from(inversion=inversion, dataset=dataset)

with timer.section("matrix_free_context_build"):
    ctx = matrix_free_steps.matrix_free_context_from(
        inversion=inversion, dataset=dataset, sparse_ctx=sparse_ctx
    )
    block(ctx.vals)
    block(ctx.reg_bcoo.data)
context_build_s = timer.records[-1][1]

mapper_nnz = int(ctx.vals.shape[0])
reg_nnz = int(ctx.reg_bcoo.data.shape[0])

add_row(
    "context build (eager)",
    context_build_s,
    mapper_nnz=mapper_nnz,
    reg_bcoo_nnz=reg_nnz,
    total_params=int(ctx.total_params),
    mapper_params=int(ctx.mapper_params),
    n_funcs=int(ctx.n_funcs),
    note="one-off setup, not a per-call cost; the BCOO λH is built from the "
    "library's dense regularization matrix (open item for the PyAutoArray phase)",
)

peak_after_setup = peak_bytes()

# ===================================================================
# PART B — The matrix-free rows
# ===================================================================

print("\n" + "=" * 70)
print("MATRIX-FREE ROWS")
print("=" * 70)

data_vector_jnp = jnp.asarray(inversion.data_vector, dtype=jnp.float64)
diag_estimate = matrix_free_steps.jacobi_diag_estimate(ctx)
block(diag_estimate)

#: Iteration cap for every PCG solve. Ten times the Krylov dimension, not one
#: times it: in exact arithmetic CG terminates in ``n`` steps, but ``F + λH``
#: here is ill-conditioned enough (see ``cond_curvature_reg``) that the
#: Lanczos basis loses orthogonality long before that and the ``n``-step bound
#: is simply not attainable in fp64 — a first local run at ``maxiter = n``
#: stopped at a 5e-7 relative residual on a 400-pixel mesh, still short of
#: ``--cg-tol``. Ten ``n`` is the same bound
#: ``test_pcg_matches_cholesky_solve_on_the_inversion`` uses. The cap is not a
#: tuning knob for the timings: ``cg_iterations`` and ``rel_residual`` are
#: recorded per row, so a solve that hits it says so.
CG_MAXITER = 10 * int(ctx.total_params)

#: The PDIP's **inner** solves get the plain Krylov-dimension cap instead, and a
#: floor of 1e-8 on their tolerance. Two reasons, both about what the row means
#: rather than about speed: the inner system is ``Q + diag(z / s)``, a positively
#: shifted and therefore strictly better-conditioned matrix than ``F + λH``, so
#: it does not need the ill-conditioned system's headroom; and the outer PDIP
#: converges on a KKT residual of ``min(n · eps · 5e3, 1e-2)``, so an inner
#: solve tightened past 1e-8 buys the Newton step nothing while multiplying the
#: row's cost by the PDIP iteration count. ``total_cg_iterations`` is recorded,
#: so an inner solve that hits the cap is visible rather than silent.
PDIP_CG_MAXITER = int(ctx.total_params)
PDIP_CG_TOL = max(CG_TOL, 1.0e-8)


def _matvec_1(V):
    return matrix_free_steps.curvature_reg_matvec(ctx, V)


_rng = np.random.default_rng(0)
_V1 = jnp.asarray(_rng.standard_normal((ctx.total_params, 1)), dtype=jnp.float64)
_V32 = jnp.asarray(_rng.standard_normal((ctx.total_params, 32)), dtype=jnp.float64)

jit_profile(_matvec_1, "curvature_reg_matvec_x1", _V1)
add_row("curvature_reg_matvec x1", timer.records[-1][1] / 10, columns=1)

jit_profile(_matvec_1, "curvature_reg_matvec_x32", _V32)
_matvec32_s = timer.records[-1][1] / 10
add_row(
    "curvature_reg_matvec x32",
    _matvec32_s,
    columns=32,
    ms_per_column=_matvec32_s * 1e3 / 32,
    note="one forward blur and one adjoint blur for the whole block, whatever B is",
)

# ---------------------------------------------------------------------------
# Dense comparators (built once, eagerly) — the agreement columns
# ---------------------------------------------------------------------------
# At large --source-pixels this is the first thing that stops fitting, so the
# whole block is guarded: a run whose comparators OOM still emits every
# matrix-free row and says so in ``dense_comparators``.

print("\n--- Dense comparators (exact) ---")

dense_comparators: dict = {"status": "ok"}
DENSE_OK = True
curv_reg_dense = None
diag_exact = None
x_cholesky = None
x_nnls_reference = None
nnls_reference_iterations = None
cond_curvature_reg = None
exact_log_dets = None
exact_evidence_terms = None
nnls_reference_s = None

try:
    with timer.section("dense_comparators_build"):
        curv_reg_dense = jnp.asarray(inversion.curvature_reg_matrix, dtype=jnp.float64)
        creg_reduced = jnp.asarray(inversion.curvature_reg_matrix_reduced, dtype=jnp.float64)
        reg_reduced = jnp.asarray(inversion.regularization_matrix_reduced, dtype=jnp.float64)
        diag_exact = jnp.diag(curv_reg_dense)
        x_cholesky = reconstruction_steps.cholesky_solve(curv_reg_dense, data_vector_jnp)
        block(x_cholesky)

    # The number that explains every agreement column below: CG's residual bound
    # is on ``‖r‖ / ‖D‖``, but the *solution* error it implies is that times the
    # condition number. A solve that reaches a 1e-10 residual on a matrix with
    # κ ~ 1e8 has no right to a 1e-8 solution. Recorded, never asserted; O(n^3),
    # so it rides inside the dense block and disappears with it.
    cond_curvature_reg = float(jnp.linalg.cond(curv_reg_dense))
    print(f"  cond(F+λH) = {cond_curvature_reg:.6e}")

    exact_log_dets = {
        "log_det_curvature_reg": float(reconstruction_steps.log_det_cholesky(creg_reduced)),
        "log_det_regularization": float(reconstruction_steps.log_det_cholesky(reg_reduced)),
    }
    print(f"  exact log dets: {exact_log_dets}")

    # The library's own positivity solve, cell-driven so the iteration count
    # survives — re-measured in this process so the matrix-free/NNLS ratio is a
    # same-run ratio, not a cross-run one.
    _Q_pc, _q_pc, _D_scale = reconstruction_steps.jacobi_scaled(curv_reg_dense, data_vector_jnp)
    if RUN_PDIP:
        _, _nnls_out = jit_profile(reconstruction_steps.nnls_pdip, "nnls_pdip_jit", _Q_pc, _q_pc)
        nnls_reference_s = timer.records[-1][1] / 10
        _x_pc, _nnls_converged, _nnls_iterations = _nnls_out
        nnls_reference_iterations = int(_nnls_iterations)
        x_nnls_reference = _x_pc * _D_scale
        add_row(
            "nnls_pdip (dense KKT Cholesky, reference)",
            nnls_reference_s,
            pdip_iterations=nnls_reference_iterations,
            converged=bool(_nnls_converged),
            ms_per_iteration=nnls_reference_s * 1e3 / max(nnls_reference_iterations, 1),
            note="the exact comparator for the matrix-free PDIP row, same process",
        )

    jit_profile(
        reconstruction_steps.cholesky_solve, "cholesky_solve_jit", curv_reg_dense, data_vector_jnp
    )
    add_row(
        "cholesky solve (unconstrained, reference)",
        timer.records[-1][1] / 10,
        note="one factorisation plus two triangular solves — the number a CG line has to beat",
    )
except Exception as _dense_error:  # noqa: BLE001 — an OOM here must not lose the run
    import traceback as _dense_traceback

    DENSE_OK = False
    dense_comparators = {
        "status": "oom",
        "error": _dense_traceback.format_exc(),
    }
    print("  DENSE COMPARATORS FAILED (out of memory?) — matrix-free rows are unaffected:")
    print(dense_comparators["error"])

peak_after_dense = peak_bytes()

# ---------------------------------------------------------------------------
# PCG solves
# ---------------------------------------------------------------------------

print("\n--- PCG solves (unconstrained) ---")


def _pcg_row(name, jit_label, diag, *, skip_note=None):
    """Time one PCG configuration and record its agreement with the exact solve."""
    if skip_note is not None:
        matrix_free_rows.append({"name": name, "ms": None, "skipped": skip_note})
        print(f"  ROW {name:<48} SKIPPED — {skip_note}")
        return None

    def solve(D):
        return matrix_free_steps.pcg_solve(ctx, D, tol=CG_TOL, maxiter=CG_MAXITER, diag=diag)

    _, out = jit_profile(solve, jit_label, data_vector_jnp)
    per_call = timer.records[-1][1] / 10
    x, iterations, residual = out
    facts = {
        "cg_iterations": int(iterations),
        "rel_residual": float(residual),
        "converged": bool(float(residual) <= CG_TOL),
        "cg_tol": CG_TOL,
        "cg_maxiter": CG_MAXITER,
    }
    if DENSE_OK:
        abs_diff, rel_diff = max_abs_rel(x, x_cholesky)
        facts["max_abs_diff_vs_cholesky_solve"] = abs_diff
        facts["max_rel_diff_vs_cholesky_solve"] = rel_diff
    add_row(name, per_call, **facts)
    return x


x_pcg = _pcg_row(
    "PCG solve (unconstrained, Jacobi est.)", "pcg_solve_jacobi_est_jit", diag_estimate
)

_pcg_row(
    "PCG solve (unconstrained, exact diag)",
    "pcg_solve_exact_diag_jit",
    diag_exact,
    skip_note=None if DENSE_OK else "dense F+λH did not fit; no exact diagonal",
)

_pcg_row("PCG solve (unconstrained, no precond)", "pcg_solve_no_precond_jit", None)

# ---------------------------------------------------------------------------
# Matrix-free PDIP
# ---------------------------------------------------------------------------

if RUN_PDIP:
    print("\n--- Matrix-free PDIP (positivity) ---")

    def _pdip(D):
        return matrix_free_steps.pdip_matrix_free(
            ctx, D, diag=diag_estimate, cg_tol=PDIP_CG_TOL, cg_maxiter=PDIP_CG_MAXITER
        )

    _, _pdip_out = jit_profile(_pdip, "pdip_matrix_free_jit", data_vector_jnp)
    _pdip_s = timer.records[-1][1] / 10
    _x_pdip, _pdip_iterations, _pdip_cg_total, _pdip_converged = _pdip_out
    _pdip_facts = {
        "pdip_iterations": int(_pdip_iterations),
        "total_cg_iterations": int(_pdip_cg_total),
        "converged": bool(_pdip_converged),
        "cg_tol": PDIP_CG_TOL,
        "cg_maxiter": PDIP_CG_MAXITER,
    }
    if DENSE_OK and x_nnls_reference is not None:
        _abs, _rel = max_abs_rel(_x_pdip, x_nnls_reference)
        _pdip_facts["max_abs_diff_vs_nnls_pdip"] = _abs
        _pdip_facts["max_rel_diff_vs_nnls_pdip"] = _rel
        _pdip_facts["ratio_vs_nnls_pdip"] = _pdip_s / nnls_reference_s
    add_row("matrix-free PDIP", _pdip_s, **_pdip_facts)
else:
    print("\n--- Matrix-free PDIP SKIPPED (--no-pdip) ---")

# ---------------------------------------------------------------------------
# SLQ log determinants
# ---------------------------------------------------------------------------

print("\n--- SLQ log determinants ---")

n_reduced = int(ctx.mapper_params)
slq_key = jax.random.PRNGKey(SLQ_KEY)

_reduced_curv_reg_matvec = matrix_free_steps.reduced_matvec_from(ctx)
_reduced_reg_matvec = matrix_free_steps.reduced_matvec_from(
    ctx, matrix_free_steps.bcoo_matvec(ctx.reg_bcoo)
)

#: ``{(target, probes, steps): estimate}`` — the noise-vs-probes curve, and
#: where the default pair's value is read back from for the evidence row.
slq_estimates: dict[tuple, float] = {}

for _target, _matvec, _exact_key in (
    ("F+λH reduced", _reduced_curv_reg_matvec, "log_det_curvature_reg"),
    ("λH reduced", _reduced_reg_matvec, "log_det_regularization"),
):
    for _probes in SLQ_PROBES:
        for _steps in SLQ_STEPS:

            def _slq(key, matvec=_matvec, probes=_probes, steps=_steps):
                return matrix_free_steps.slq_logdet(
                    matvec, n_reduced, n_probes=probes, n_lanczos=steps, key=key
                )

            _label = f"slq_{_exact_key}_p{_probes}_m{_steps}"
            _, _slq_out = jit_profile(_slq, _label, slq_key)
            _slq_s = timer.records[-1][1] / 10
            _estimate, _per_probe, _n_clamped = _slq_out
            _estimate = float(_estimate)
            slq_estimates[(_exact_key, _probes, _steps)] = _estimate

            _facts = {
                "probes": _probes,
                "lanczos_steps": _steps,
                "estimate": _estimate,
                "spread": float(np.std(np.asarray(_per_probe), ddof=1) / math.sqrt(_probes))
                if _probes > 1
                else None,
                "n_clamped": int(_n_clamped),
            }
            if DENSE_OK:
                _facts["error_vs_cholesky"] = _estimate - exact_log_dets[_exact_key]
            add_row(f"SLQ log det ({_target}) p={_probes} m={_steps}", _slq_s, **_facts)

# ---------------------------------------------------------------------------
# Matrix-free evidence terms at the default (probes, steps)
# ---------------------------------------------------------------------------
# Fed the **inversion's own** reconstruction, exactly as the cells'
# ``log_evidence_terms`` is, so every term except the two log-dets is identical
# by construction and ``log_evidence_error`` is purely the SLQ bias — which is
# the number the 0.5-nat go bar reads. The unconstrained-PCG variant below is a
# separate question (what the solve itself changes) and is recorded separately.

print("\n--- Matrix-free evidence terms ---")

evidence_block: dict = {
    "slq_probes": SLQ_DEFAULT_PROBES,
    "slq_lanczos_steps": SLQ_DEFAULT_STEPS,
}

if DENSE_OK:
    rows_curv_jnp, cols_jnp, vals_jnp = (ctx.rows, ctx.cols, ctx.vals)
    _mge_operated = jnp.asarray(
        next(iter(inversion.linear_func_operated_mapping_matrix_dict.values())),
        dtype=jnp.float64,
    )
    data_array = jnp.asarray(dataset.data.array, dtype=jnp.float64)
    noise_jnp = jnp.asarray(dataset.noise_map.array, dtype=jnp.float64)
    blurred_img_jnp = jnp.asarray(fit.blurred_image.array, dtype=jnp.float64)
    inv_recon_jnp = jnp.asarray(inversion.reconstruction, dtype=jnp.float64)

    def _model_data_fn(x):
        """Lens light plus the mapped source, the w-tilde way (sparse_steps)."""
        return blurred_img_jnp + sparse_steps.mapped_reconstructed_operated_data(
            sparse_ctx, x, rows_curv_jnp, cols_jnp, vals_jnp, _mge_operated
        )

    exact_evidence_terms = reconstruction_steps.log_evidence_terms(
        data_array,
        noise_jnp,
        _model_data_fn(inv_recon_jnp),
        inv_recon_jnp,
        jnp.asarray(np.asarray(inversion.mapper_indices)),
        reg_reduced,
        creg_reduced,
    )

    _slq_curv = slq_estimates[("log_det_curvature_reg", SLQ_DEFAULT_PROBES, SLQ_DEFAULT_STEPS)]
    _slq_reg = slq_estimates[("log_det_regularization", SLQ_DEFAULT_PROBES, SLQ_DEFAULT_STEPS)]

    with timer.section("matrix_free_evidence_terms"):
        matrix_free_terms = matrix_free_steps.matrix_free_evidence_terms(
            ctx,
            data_vector_jnp,
            inv_recon_jnp,
            log_det_curvature_reg=_slq_curv,
            log_det_regularization=_slq_reg,
            data=data_array,
            noise_map=noise_jnp,
            model_data_fn=_model_data_fn,
        )

    evidence_block["exact_terms"] = exact_evidence_terms
    evidence_block["matrix_free_terms"] = matrix_free_terms
    evidence_block["term_errors"] = {
        k: matrix_free_terms[k] - exact_evidence_terms[k] for k in exact_evidence_terms
    }
    evidence_block["log_evidence_error"] = (
        matrix_free_terms["log_evidence"] - exact_evidence_terms["log_evidence"]
    )
    evidence_block["go_bar_nats"] = 0.5
    evidence_block["within_go_bar"] = abs(evidence_block["log_evidence_error"]) <= 0.5

    for _term, _err in evidence_block["term_errors"].items():
        print(f"  {_term:<24} error {_err:+.6e}")
    print(f"  log_evidence_error = {evidence_block['log_evidence_error']:+.6f} nats (bar 0.5)")

    # What the unconstrained solve itself changes, kept separate from the
    # estimator's bias above: this reconstruction is a different (unconstrained)
    # minimiser, so its chi-squared and regularization terms legitimately differ.
    if x_pcg is not None:
        _pcg_terms = matrix_free_steps.matrix_free_evidence_terms(
            ctx,
            data_vector_jnp,
            x_pcg,
            log_det_curvature_reg=_slq_curv,
            log_det_regularization=_slq_reg,
            data=data_array,
            noise_map=noise_jnp,
            model_data_fn=_model_data_fn,
        )
        evidence_block["unconstrained_pcg_terms"] = _pcg_terms
        evidence_block["unconstrained_pcg_log_evidence_error"] = (
            _pcg_terms["log_evidence"] - exact_evidence_terms["log_evidence"]
        )
        evidence_block["unconstrained_pcg_note"] = (
            "The unconstrained PCG reconstruction is a different minimiser from the "
            "non-negative one the library returns, so these terms differ by more than "
            "the SLQ bias by design. The go bar reads log_evidence_error above, not this."
        )
else:
    evidence_block["skipped"] = "dense comparators did not fit; no exact terms to compare against"
    print("  SKIPPED — no exact terms (dense comparators OOM).")

peak_after_matrix_free = peak_bytes()

# ---------------------------------------------------------------------------
# Batched rows (--vmap-batch N)
# ---------------------------------------------------------------------------
# Identical lanes, exactly as the cells' ``steps_vmap_per_call`` rows: every lane
# is a copy of the same right-hand side (and the same SLQ key), so they converge
# on the same iteration and the ``while_loop`` never waits for a straggler. That
# makes these a best case for amortization, not a representative one.

vmap_note = None
vmap_error = None

if _vmap_batch is not None:
    print(f"\n--- Batched rows (--vmap-batch {_vmap_batch}) ---")
    import traceback as _vmap_traceback

    try:
        _D_batched = jnp.broadcast_to(data_vector_jnp, (_vmap_batch, ctx.total_params))

        def _pcg_lane(D):
            return matrix_free_steps.pcg_solve(
                ctx, D, tol=CG_TOL, maxiter=CG_MAXITER, diag=diag_estimate
            )[0]

        _pcg_vmap_per_call = timing.vmap_profile(
            _pcg_lane,
            "pcg_solve_jacobi_est",
            _D_batched,
            _vmap_batch,
            timer=timer,
            jit_records=jit_records,
        )
        add_row(
            f"PCG solve @vmap {_vmap_batch} (identical lanes)",
            _pcg_vmap_per_call,
            batch=_vmap_batch,
        )

        _keys_batched = jnp.broadcast_to(slq_key, (_vmap_batch, *slq_key.shape))

        def _slq_lane(key):
            return matrix_free_steps.slq_logdet(
                _reduced_curv_reg_matvec,
                n_reduced,
                n_probes=SLQ_DEFAULT_PROBES,
                n_lanczos=SLQ_DEFAULT_STEPS,
                key=key,
            )[0]

        _slq_vmap_per_call = timing.vmap_profile(
            _slq_lane,
            f"slq_p{SLQ_DEFAULT_PROBES}_m{SLQ_DEFAULT_STEPS}",
            _keys_batched,
            _vmap_batch,
            timer=timer,
            jit_records=jit_records,
        )
        add_row(
            f"SLQ log det (F+λH reduced) p={SLQ_DEFAULT_PROBES} m={SLQ_DEFAULT_STEPS} "
            f"@vmap {_vmap_batch} (identical lanes)",
            _slq_vmap_per_call,
            batch=_vmap_batch,
        )

        vmap_note = (
            f"The @vmap {_vmap_batch} rows batch {_vmap_batch} copies of the same right-hand "
            f"side and the same SLQ key; identical lanes converge on the same CG iteration, so "
            f"the lax.while_loop never runs on for a straggler. Best case for amortization, not "
            f"a representative one."
        )
    except Exception:  # noqa: BLE001 — a vmap failure must not lose the unbatched rows
        vmap_error = _vmap_traceback.format_exc()
        print("  VMAP FAILED — unbatched rows are unaffected. Traceback:")
        print(vmap_error)

# ===================================================================
# Summary + JSON + PNG
# ===================================================================

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

print("\n" + "=" * 70)
print(f"MATRIX-FREE BREAKDOWN — {MESH} — {instrument.upper()} — v{al_version}")
print("=" * 70)
_timed_rows = [r for r in matrix_free_rows if r.get("ms") is not None]
_width = max(len(r["name"]) for r in matrix_free_rows)
for _row in matrix_free_rows:
    if _row.get("ms") is None:
        print(f"  {_row['name']:<{_width}}  {'SKIPPED':>14}")
    else:
        print(f"  {_row['name']:<{_width}}  {_row['ms']:>12.3f} ms")
print("=" * 70)
print("  These rows are overlapping comparators on one model evaluation — never a partition,")
print("  never summed. See the module docstring.")

matrix_free_block = {
    "mesh": MESH,
    "source_pixels": int(n_source_pixels),
    "total_params": int(ctx.total_params),
    "mapper_params": int(ctx.mapper_params),
    "n_funcs": int(ctx.n_funcs),
    "mapper_nnz": mapper_nnz,
    "reg_bcoo_nnz": reg_nnz,
    "add_to_diag_value": float(ctx.add_to_diag_value),
    "cg_tol": CG_TOL,
    "cg_maxiter": CG_MAXITER,
    "pdip_cg_tol": PDIP_CG_TOL,
    "pdip_cg_maxiter": PDIP_CG_MAXITER,
    "slq_probes": list(SLQ_PROBES),
    "slq_lanczos_steps": list(SLQ_STEPS),
    "slq_key": SLQ_KEY,
    "pdip": RUN_PDIP,
    # Edge zeroing: the library's own reconstruction solves only these ids, the
    # kernels here solve the full system, and every comparison above is against
    # the full-system cell-driven recipe for exactly that reason.
    "edge_zeroed_active": EDGE_ZEROED_ACTIVE,
    "edge_zeroed_pixels": EDGE_ZEROED_PIXELS,
    "context_build_s": float(context_build_s),
    # ``cond(F + λH)``, from the dense comparator — the multiplier between a
    # PCG row's ``rel_residual`` and its ``max_rel_diff_vs_cholesky_solve``, and
    # the reason the SLQ log-det of this block needs so many Lanczos steps.
    # ``None`` when the dense comparators did not fit.
    "cond_curvature_reg": cond_curvature_reg,
    "evidence": evidence_block,
    "exact_log_dets": exact_log_dets,
    # Peak device bytes at three points. The eager FitImaging above has already
    # built the dense matrices before the matrix-free block runs, so these are
    # high-water marks of the whole process, not of each block in isolation —
    # read the differences, and read them as an upper bound.
    "peak_bytes": {
        "after_setup": peak_after_setup,
        "after_dense_comparators": peak_after_dense,
        "after_matrix_free": peak_after_matrix_free,
        "note": (
            "jax.devices()[0].memory_stats()['peak_bytes_in_use']; None on a backend "
            "that does not report it (CPU). Process-wide high-water marks."
        ),
    },
}

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
        int(_cli.source_pixels) if _cli.source_pixels is not None else None
    ),
    "inversion_path": "matrix_free",
    "total_params": int(inversion.total_params),
    "sparse_batch_size": int(_cli.sparse_batch_size),
    "sparse_operator_build_s": float(sparse_operator_build_s),
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
    "steps_matrix_free_rows": matrix_free_rows,
    "matrix_free": matrix_free_block,
    "dense_comparators": dense_comparators,
    "jit_phases": jit_records,
}

if _vmap_batch is not None:
    breakdown_summary["vmap_batch"] = int(_vmap_batch)
    if vmap_error is not None:
        breakdown_summary["vmap_error"] = vmap_error
    else:
        breakdown_summary["steps_matrix_free_vmap_note"] = vmap_note

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"matrix_free_{MESH}_breakdown_{instrument}_v{al_version}",
    cell=f"matrix_free_{MESH}",
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [r["name"] for r in _timed_rows]
times = [r["ms"] / 1e3 for r in _timed_rows]

fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(labels))))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color="#4C72B0", edgecolor="white", height=0.6)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t * 1e3:.3f} ms",
        va="center",
        fontsize=8,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Time per call (s)", fontsize=11)
fig.suptitle(
    f"Matrix-Free Pixelized Likelihood — {MESH} — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_source_pixels} source pixels  |  cg_tol {CG_TOL:g}",
    fontsize=9,
)
ax.margins(x=0.18)
fig.tight_layout()

fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

# ===================================================================
# Pins — the 2026-09-11 exact log-dets
# ===================================================================
# The pins describe the fiducial mesh (39x39 / 1500 / 1500) and nothing else, so
# a run with a non-fiducial ``--source-pixels`` skips them and writes
# ``pinned_expected: null``. Two things are recorded when they do run:
#
# - the **in-process Cholesky** log-dets against the pin, through
#   ``check_pinned`` — this is a drift check in the ordinary sense (the exact
#   computation should reproduce the recorded value);
# - the **SLQ (probes, steps) error**, recorded unconditionally rather than
#   compared to a tolerance, because it is an estimate: its deviation is the
#   measurement this cell exists to make, not a fault.

if not PIN_IS_FIDUCIAL:
    print(
        f"  Pinned log-det check SKIPPED: --source-pixels {_cli.source_pixels} builds a "
        f"{n_source_pixels}-pixel mesh, not the {FIDUCIAL_SOURCE_PIXELS[MESH]}-pixel fiducial "
        f"the pins describe."
    )
    record_pinned_check(dict_path, None, [])
elif not DENSE_OK:
    print("  Pinned log-det check SKIPPED: dense comparators did not fit, so nothing exact to pin.")
    record_pinned_check(dict_path, PINNED_LOG_DETS[MESH], [])
else:
    _drift_records = []

    for _key, _expected in PINNED_LOG_DETS[MESH].items():
        _record = check_pinned(
            exact_log_dets[_key],
            _expected,
            label=f"imaging/matrix_free[{instrument}, {MESH}] Cholesky {_key}",
            rtol=1e-4,
        )
        if _record is None:
            print(f"  Pin PASSED: Cholesky {_key} matches {_expected:.6f}")
        else:
            _drift_records.append(_record)

        _slq_value = slq_estimates[(_key, SLQ_DEFAULT_PROBES, SLQ_DEFAULT_STEPS)]
        _drift_records.append(
            {
                "label": (
                    f"imaging/matrix_free[{instrument}, {MESH}] SLQ {_key} "
                    f"p={SLQ_DEFAULT_PROBES} m={SLQ_DEFAULT_STEPS}"
                ),
                "expected": _expected,
                "got": _slq_value,
                "rel_diff": abs(_slq_value - _expected) / max(abs(_expected), 1e-300),
                "abs_diff": _slq_value - _expected,
                "rtol": None,
                "note": (
                    "Stochastic Lanczos quadrature estimate — recorded, never a fault. "
                    "Its deviation from the exact log-det is what this cell measures."
                ),
            }
        )
        print(f"  SLQ {_key}: {_slq_value:.6f} vs pin {_expected:.6f}")

    record_pinned_check(dict_path, PINNED_LOG_DETS[MESH], _drift_records)

print("\nFinished.")
