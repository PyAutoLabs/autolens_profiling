"""
JAX Profiling: Fixed Lens Light — the XLA DEVICE TIMELINE of the production call
================================================================================

Phase 1 of the ``hst-gpu-non-solver-residue`` epic (autolens_profiling#268).

The ``fixed-lens-light-profiling`` epic ended with ~21 of the 25.4 ms certified
Delaunay A100 call at HST N=1500 **not** being the solver. But its largest term
— "~13.9 ms mesh / mapper / weights / imaging / blurring" — is attribution
*arithmetic* across two cells: phase-0 kernel rows subtracted from a phase-1
whole call. Every JAX decomposition this repo owns (``--split-setup``) is ~11
**separately compiled** prefix programs whose rows move with fusion; PyAutoArray
#531 made one of them negative. Nobody has ever measured where the *fused
production program* spends its device time.

This cell measures it. One program, one process, one timeline.

What it does
------------

1. Builds the same S3 system every cell in the family builds (lens light fixed
   after SLaM ``light[1]``, converted to regular profiles and subtracted), and
   times route ``b`` (library PDIP) and route ``d`` (certified active set at the
   PRODUCTION budget 7, PDIP fallback) exactly as ``fixed_light_library.py``
   does — so the whole-call wall reconciles with phase 5's row.
2. Lowers route ``d`` **once** and compiles that one lowering twice: with CUDA
   command buffers as XLA ships them (the production program) and with them
   disabled (the traceable program). Times both.
3. Traces ``--trace-calls`` steady calls of the command-buffers-OFF executable
   with ``jax.profiler``, joins every device kernel to its HLO instruction and
   every instruction to its library source stack, and writes a per-stage
   millisecond table that sums to the wall.
4. Takes the optimized HLO census: how many ``(n,n)`` adds of ``F + lambda*H``
   survive XLA, how many ``[ids][:, ids]`` gathers, and whether the PSF FFT of
   the mapping-matrix cube is compiled once or twice.

The mechanics of the join — the stack frame index, the ``requires`` rules that
tell the two log determinants apart, the mixed-fusion row — are documented in
``likelihood_breakdown.xla_attribution``, not repeated here.

Command buffers: what the stage rows actually attribute
-------------------------------------------------------

**The stage rows attribute the graph-less program.** With XLA's defaults the GPU
module is captured into a CUDA graph and every kernel reports
``hlo_op="command_buffer_1"``, which destroys the join (23 % of device duration
recoverable on the spike, against 100 % with the graphs off). So the traced
executable is compiled with ``xla_gpu_enable_command_buffer=""``.

That is a different program from the one production runs, and this cell refuses
to hide the difference. It times **both** executables — ``wall_ms`` per route
for command buffers ON (the production number, the one comparable to phase 5's
25.39 ms row) and OFF (the traced program) — and records
``command_buffer_delta_ms`` and its percentage. Kernel time is not what the
graphs change; launch overhead is, so the delta lands in the ``device_idle`` and
``host_outside_span`` rows rather than in any stage. Reconciliation is computed
**against the OFF wall**, because that is the program the kernels came from.

A short probe also traces the ON executable and records whether CUPTI still
emits per-kernel events inside the graph (``command_buffer_probe``), so the
size of the delta can be judged rather than assumed.

Flags (beyond the shared ``_profile_cli`` set)
----------------------------------------------

``--mesh {delaunay,delaunay_nn,rectangular}`` (required) — which cell's model to
rebuild.

``--border-relocator {cell,library,off}`` (default ``cell``) — ``cell`` passes
``use_border_relocator=True``, which every profiling cell in this family forces;
``library`` passes ``None`` and takes the shipped config default; ``off`` passes
``False``. The **resolved** boolean is recorded and is part of the output
filename, because on this workspace ``library`` resolves to **True**
(``config/general.yaml`` declares no ``inversion`` block, so autoconf falls
through to the packaged default) — so ``cell`` and ``library`` are the same
program here, and ``off`` is the leg that measures what the relocator costs.

``--trace-calls K`` (default 10) — steady calls taken inside the profiler.
``--source-pixels N`` — mesh size (fiducial 1500 for the Delaunay family).
``--routes b,d`` (default) — which routes are built and timed; the trace is
always taken on route ``d``, which must therefore be selected.
``--command-buffers {off,on}`` (default ``off``) — which executable the trace is
taken on. ``on`` is the diagnostic that shows the join failing; it is not a
configuration in which any stage row should be quoted.
``--trace-dir`` — where the profiler writes its xplane (a temporary directory by
default; traces are large and do not belong in the repo).

The batched mode (phase 2, autolens_profiling#273)
--------------------------------------------------

``--vmap-batch B`` turns this cell into the **matched vmap-vs-scalar**
experiment. Without it nothing below happens and the cell is phase 1 exactly,
down to the output filename.

Production does not run one likelihood at a time. ``Fitness._vmap``
(``PyAutoFit/autofit/non_linear/fitness.py``) is ``jax.vmap(jax.jit(call))``,
and Nautilus drives it with ``use_jax_vmap=True``. The Delaunay qhull callback
is ``vmap_method="sequential"`` (``PyAutoArray .../interpolator/delaunay.py``),
so the host round trip phase 1 measured as a single 5.44 ms device-idle gap
runs **once per lane** and is the one term that cannot amortise over a batch.
Nobody had traced that program.

So this mode builds ``B`` lane parameter trees and runs **two arms on the same
lanes, in one process, under the same solver injection**:

``vmap``
    ``jax.vmap(jax.jit(fn))`` — the production nesting, *not*
    ``timing.vmap_profile``, which nests them the other way
    (``jax.jit(jax.vmap(fn))``) and is therefore a different program.
``scalar``
    ``jax.jit(fn)`` called ``B`` times per batch, each blocked. This stands for
    ``Nautilus(use_jax_vmap=False, use_jax_jit=True)``; plain
    ``use_jax_vmap=False`` is UNJITTED and is not what this arm measures.

``--lanes distinct`` (default) makes lane *k* the seeded random draw *k* of the
phase-3 draw set (``fixed_light_draws_steps.random_draws``) applied to the lens
mass, with the lens light fixed at S3 — a batch of *different* models, which is
what a search evaluates. ``--lanes identical`` broadcasts the fiducial and is
the **control**: identical lanes share a straggler and hide exactly the cost
this experiment is looking for.

``--fallback on`` (default) keeps the ``lax.cond`` PDIP fallback — under
``vmap`` the batched predicate turns it into a ``select`` and both branches
run, so that row is the certified solve *plus* PDIP. ``--fallback off`` is the
cond-free program. Both are rows, never a footnote.

Per lane the two arms' log likelihoods are pinned equal at 1e-9 relative, and
the cell exits non-zero if any lane fails: an unequal lane means the two arms
are not evaluating the same model and no timing comparison between them means
anything.

``--arms {vmap,scalar,both}`` (default ``both``) and ``--draw-seed`` (default
0) complete the set. Every one of these flags is recorded in
``configuration``, because ``parse_known_args`` ignores unknown flags: a
checkout without this mode would run the submit and silently write a phase-1
table.

Output
------

``results/breakdown/imaging/fixed_light_trace_<mesh>[_border_off][_n<N>][_vmap<B>_<lanes>_fb<on|off>]_<config>.{json,png}``.

The ``--config-name`` used for the phase-1 legs
(``local_rtx2060_fp64_fixed_light_trace``, ``hpc_a100_fp64_fixed_light_trace``)
sits **outside** ``build_readme.py``'s ``CONFIG_TAGGED_RE``, so none of this
reaches the dashboard: these are research rows for one note, not a tracked
profiling series.

What may not be quoted from this JSON
-------------------------------------

- A stage row as a *production* millisecond without saying it was measured on
  the command-buffers-OFF program, with ``command_buffer_delta_ms`` beside it.
- ``mixed_fusion.prorated_estimate_ms`` as a measurement. It is an equal split
  of a fused kernel across the stages its constituents span, and it is labelled
  ESTIMATE everywhere it appears.
- Any row at all if ``reconciliation_pct`` is outside 5 % or ``unjoined_ms`` is
  non-zero — those two say the join is incomplete, and an incomplete join
  produces a table that still adds up.
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
import copy
import math
import sys
import tempfile
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
    host_callback_probe,
    library_solver_injection,
    timing,
    xla_attribution,
)
from likelihood_breakdown import (
    fixed_light_draws_steps as flds,
)

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _production_config import observe_thread_env as _observe_thread_env  # noqa: E402
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

#: Phase 3's smallest **zero-fallback** pass budget per mesh — the budget a
#: PRODUCTION run fixes, and therefore the only budget this phase profiles.
#: ``fixed_light_library.CERTIFYING_BUDGET`` (2 for Delaunay) is phase 0's
#: fiducial and falls back on 67.5 % of a graded draw set; it is not a
#: production cost and is not what this cell measures.
PHASE3_SAFE_BUDGET = {"rectangular": 11, "delaunay": 7, "delaunay_nn": 7}

#: Source-pixel count each mesh's cell builds at its fiducial.
FIDUCIAL_SOURCE_PIXELS = {"rectangular": 39 * 39, "delaunay": 1500, "delaunay_nn": 1500}

#: Relative tolerance on the route d == route b equivalence pin. Inherited from
#: ``fixed_light_library.py``; the certified scheme is a different factorisation
#: order to PDIP, not a different answer.
EQUIVALENCE_RTOL = 1.0e-9

#: The command-buffer knob, passed per compilation so both programs live in one
#: process. ``""`` disables every command-buffer command type.
_COMMAND_BUFFER_OPTION = "xla_gpu_enable_command_buffer"

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument(
    "--mesh", choices=("rectangular", "delaunay", "delaunay_nn"), required=True
)
_cell_parser.add_argument("--border-relocator", choices=("cell", "library", "off"), default="cell")
_cell_parser.add_argument("--trace-calls", type=int, default=10)
_cell_parser.add_argument("--source-pixels", type=int, default=None)
_cell_parser.add_argument("--routes", default="b,d")
_cell_parser.add_argument("--command-buffers", choices=("off", "on"), default="off")
_cell_parser.add_argument("--trace-dir", default=None)
_cell_parser.add_argument("--safe-budget", type=int, default=None)
# --- phase 2 (#273): the matched vmap-vs-scalar mode -----------------------
# Every one of these is recorded into configuration when the mode is on, and
# NONE of them appears there when it is off, so a phase-1 JSON written by this
# checkout is key-for-key the JSON phase 1 wrote.
_cell_parser.add_argument("--vmap-batch", type=int, default=None)
_cell_parser.add_argument("--lanes", choices=("distinct", "identical"), default="distinct")
_cell_parser.add_argument("--arms", choices=("vmap", "scalar", "both"), default="both")
_cell_parser.add_argument("--draw-seed", type=int, default=0)
_cell_parser.add_argument("--fallback", choices=("on", "off"), default="on")
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh
BORDER_RELOCATOR_MODE = _cell_args.border_relocator
TRACE_CALLS = int(_cell_args.trace_calls)
TRACE_ON_COMMAND_BUFFERS = _cell_args.command_buffers == "on"

if TRACE_CALLS < 1:
    raise ValueError(f"--trace-calls must be >= 1 (got {TRACE_CALLS})")

#: ``None`` is phase 1. An int turns on the batched experiment (PART V).
VMAP_BATCH = _cell_args.vmap_batch
LANES_MODE = _cell_args.lanes
ARMS = ("vmap", "scalar") if _cell_args.arms == "both" else (_cell_args.arms,)
DRAW_SEED = int(_cell_args.draw_seed)
#: ``True`` is route d (``lax.cond`` PDIP fallback, a ``select`` under vmap);
#: ``False`` is route d0, the cond-free program.
FALLBACK_ON = _cell_args.fallback == "on"

#: Relative tolerance on the per-lane ``vmap == scalar`` log-likelihood pin.
#: The two arms are the same program on the same lane: anything above this is
#: not precision, it is a different model.
LANE_RTOL = 1.0e-9

if VMAP_BATCH is not None and VMAP_BATCH < 1:
    raise ValueError(f"--vmap-batch must be >= 1 when given (got {VMAP_BATCH})")
if VMAP_BATCH is None and _cell_args.arms != "both":
    raise ValueError(
        "--arms only means anything with --vmap-batch; without it this cell runs the "
        "phase-1 single-call trace and would silently ignore the flag"
    )

PASS_BUDGET = int(
    _cell_args.safe_budget if _cell_args.safe_budget is not None else PHASE3_SAFE_BUDGET[MESH]
)

ROUTE_SELECTION = tuple(t.strip() for t in _cell_args.routes.split(",") if t.strip())
_unknown = set(ROUTE_SELECTION) - {"b", "d"}
if _unknown:
    raise ValueError(f"--routes accepts only b and d in this cell (got {sorted(_unknown)})")
if "d" not in ROUTE_SELECTION:
    raise ValueError("--routes must include d: the trace is taken on route d")

SOURCE_PIXELS_REQUESTED = (
    _cell_args.source_pixels if _cell_args.source_pixels is not None else _cli.source_pixels
)

DATASET = "hst"
if _cli.instrument is not None and _cli.instrument != DATASET:
    raise ValueError(
        f"--instrument {_cli.instrument!r} is not this cell's dataset; phase 1 is HST only."
    )

Timer = timing.Timer
block = timing.block

timer = Timer()
jit_records: dict[str, dict] = {}


def jit_profile(func, label, *args, n_repeats=10):
    """Cell-local binding of ``timing.jit_profile`` (this cell's timer/records)."""
    return timing.jit_profile(
        func, label, *args, n_repeats=n_repeats, timer=timer, jit_records=jit_records
    )


def peak_bytes():
    """Device peak bytes, when the backend exposes them (``None`` on CPU)."""
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 — a backend without memory stats is not an error
        return None
    value = stats.get("peak_bytes_in_use")
    return int(value) if value is not None else None


def steady_wall_ms(executable, tree, n_repeats: int) -> float:
    """Mean per-call wall of *n_repeats* steady calls of an already-compiled executable.

    Every leaf of the returned pytree is blocked (``timing.block``), so a
    tuple-returning call cannot be timed asynchronously.
    """
    block(executable(tree))
    start = time.perf_counter()
    for _ in range(n_repeats):
        block(executable(tree))
    return (time.perf_counter() - start) / n_repeats * 1e3


# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================
# Deliberately identical to ``fixed_light_library.py`` PART A, which is itself
# the three pixelized breakdown cells' construction. If a sibling cell's model
# changes, this cell must change with it or its wall no longer reconciles with
# phase 5's row.

print("=" * 70)
print(f"FIXED-LIGHT TRACE — {MESH} — border relocator: {BORDER_RELOCATOR_MODE}")
print("=" * 70)

print(f"\n--- Dataset loading & masking [{DATASET}, mesh={MESH}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[DATASET]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / DATASET

auto_simulate_if_missing(
    dataset_path,
    dataset_type="imaging",
    instrument=DATASET,
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

print(f"  Source pixels: {n_source_pixels} (fiducial: {FIDUCIAL_SOURCE_PIXELS[MESH]})")

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

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization: {reg_scheme} ({reg_provenance})")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

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
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]

TAU_REL = active_set_steps.TAU_REL_DEFAULT

# ---------------------------------------------------------------------------
# The border relocator, as the cells force it and as the library ships it
# ---------------------------------------------------------------------------
# Every profiling cell in this family passes ``use_border_relocator=True``.
# Production passes nothing and takes the config default. They are different
# programs — the relocator is a whole stage of the trace — so the mode is a flag
# and the RESOLVED boolean is recorded, never inferred from the mode name.

_BORDER_RELOCATOR_SETTING = {"cell": True, "library": None, "off": False}[BORDER_RELOCATOR_MODE]
_settings = al.Settings(
    use_border_relocator=_BORDER_RELOCATOR_SETTING,
    use_mixed_precision=_cli.use_mixed_precision,
)
BORDER_RELOCATOR_RESOLVED = bool(_settings.use_border_relocator)

# MEASURED, not assumed: ``library`` resolved to **True** on this workspace
# (autolens_profiling/config/general.yaml declares no ``inversion`` block, so
# autoconf falls through to the packaged default, which is on). The premise
# "the cells force the relocator on, production ships it off" is therefore false
# here, and ``cell`` and ``library`` compile the same program. ``off`` is the
# flag that actually answers what the relocator costs.
if BORDER_RELOCATOR_MODE == "library" and BORDER_RELOCATOR_RESOLVED:
    print(
        "  NOTE: --border-relocator library resolved to True from the config, so this leg is "
        "the same program as --border-relocator cell. Use --border-relocator off for the "
        "relocator-free comparison."
    )

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {DATASET}")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Source pixels:           {n_source_pixels}")
print(f"  Routes:                  {','.join(ROUTE_SELECTION)}")
print(f"  Pass budget (route d):   {PASS_BUDGET}  [phase 3 production budget]")
print(f"  Border relocator:        {BORDER_RELOCATOR_MODE} -> {BORDER_RELOCATOR_RESOLVED}")
print(f"  Trace calls:             {TRACE_CALLS}")
print(f"  tau_rel:                 {TAU_REL:.6e}")
if VMAP_BATCH is None:
    print("  Batched mode:            off (phase-1 single-call trace)")
else:
    print(f"  Batched mode:            --vmap-batch {VMAP_BATCH}  lanes {LANES_MODE}")
    print(f"  Arms:                    {','.join(ARMS)}")
    print(f"  Draw seed:               {DRAW_SEED}")
    print(
        f"  Solver fallback:         {'on (route d, lax.cond -> select under vmap)' if FALLBACK_ON else 'off (route d0, cond-free)'}"
    )

# ---------------------------------------------------------------------------
# S3 — the source-only system every route in this cell fits
# ---------------------------------------------------------------------------

print("\n--- S0 -> S3: lens light converted to regular profiles + subtracted (eager) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=_settings,
        xp=np,
    )
    log_evidence_ref = float(fit.figure_of_merit)

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
log_evidence_s3_library = float(system_s3.fit.figure_of_merit)
print(f"  S3 figure_of_merit (library) = {log_evidence_s3_library}")

peak_after_setup = peak_bytes()

# ===================================================================
# PART B — the library path, one whole likelihood call per route
# ===================================================================

instance_s3 = copy.deepcopy(instance)
_source_only_galaxies = list(system_s3.source_only_tracer.galaxies)
instance_s3.galaxies.lens = _source_only_galaxies[0]
instance_s3.galaxies.source = _source_only_galaxies[1]
params_tree_s3 = jax.tree_util.tree_map(jnp.asarray, instance_s3)


def _likelihood_fn(dataset_for_route, settings_for_route):
    """``params tree -> log likelihood`` through the library's analysis path."""
    analysis = al.AnalysisImaging(
        dataset=dataset_for_route,
        adapt_images=adapt_images,
        settings=settings_for_route,
        use_jax=True,
    )

    def _likelihood(tree):
        return analysis.log_likelihood_function(instance=tree)

    return _likelihood


print("\n" + "=" * 70)
print("LIBRARY PATH — one full likelihood call per route")
print("=" * 70)

#: ``(key, label, injection)``. ``injection`` is ``None`` for the library solver
#: as shipped, or ``(pass_budget, fallback)`` for the harness-injected certified
#: scheme. Route ``d`` runs at the PRODUCTION budget, never phase 0's fiducial.
_ROUTE_SPECS: dict[str, tuple] = {
    "b": ("b_s3_pdip", "S3 PDIP (source-only)", None),
    "d": (
        "d_s3_certified_fallback",
        f"S3 certified active set, budget {PASS_BUDGET}, PDIP fallback",
        (PASS_BUDGET, True),
    ),
}

routes: dict[str, dict] = {}

for _token in ROUTE_SELECTION:
    _key, _label, _injection = _ROUTE_SPECS[_token]
    print(f"\n--- {_key}: {_label} ---")
    _fn = _likelihood_fn(system_s3.dataset, _settings)
    _entry: dict = {"label": _label, "status": "ok"}

    if _injection is None:
        _, _value = jit_profile(_fn, f"{_key}_library_likelihood_jit", params_tree_s3)
        _entry["solver"] = "library"
    else:
        _budget, _fallback = _injection
        with library_solver_injection.certified_solver_injected(
            _budget, fallback=_fallback, tau_rel=TAU_REL
        ) as _counts:
            _, _value = jit_profile(_fn, f"{_key}_library_likelihood_jit", params_tree_s3)
        _entry["solver"] = "harness-injected certified active set"
        _entry["pass_budget"] = int(_budget)
        _entry["fallback"] = bool(_fallback)
        _entry["injected_solver_calls_jax"] = int(_counts["jax"])
        _entry["injected_solver_calls_numpy"] = int(_counts["numpy"])
        if int(_counts["jax"]) == 0:
            raise AssertionError(
                f"{_key}: the injected solver was never called on the JAX path — "
                f"this row would be the library's own PDIP wearing the wrong label."
            )

    _entry["ms"] = timer.records[-1][1] / 10 * 1e3
    _entry["log_likelihood"] = float(_value)
    print(f"  {_entry['ms']:.3f} ms  -> log likelihood {_entry['log_likelihood']:.6f}")
    routes[_key] = _entry

peak_after_routes = peak_bytes()

# ---------------------------------------------------------------------------
# The equivalence pin — route d IS route b, or the trace decomposes a wrong answer
# ---------------------------------------------------------------------------

equivalence_pins: list[dict] = []

if "b" in ROUTE_SELECTION:
    _got = routes["d_s3_certified_fallback"]["log_likelihood"]
    _ref = routes["b_s3_pdip"]["log_likelihood"]
    _rel = abs(_got - _ref) / max(abs(_ref), 1e-300)
    _pin = {
        "pin": "route d (certified, fallback) == route b (library PDIP)",
        "route": "d_s3_certified_fallback",
        "reference": "b_s3_pdip",
        "expectation": "the certified active set returns the library's own positive solution",
        "rtol": EQUIVALENCE_RTOL,
        "got": _got,
        "reference_value": _ref,
        "rel_diff": _rel,
        "status": "PASS" if _rel <= EQUIVALENCE_RTOL else "FAIL",
    }
    equivalence_pins.append(_pin)
    print(f"\n  [{_pin['status']:>7}] {_pin['pin']}  rel {_rel:.3e}")
else:
    print("\n  route b not selected — the d == b equivalence pin is SKIPPED, not passed.")

# ===================================================================
# PART C — one lowering, two executables, one timeline
# ===================================================================

print("\n" + "=" * 70)
print("XLA TRACE — the production program's device timeline")
print("=" * 70)

trace_dir = (
    Path(_cell_args.trace_dir)
    if _cell_args.trace_dir
    else Path(tempfile.mkdtemp(prefix="fixed_light_trace_"))
)
trace_dir.mkdir(parents=True, exist_ok=True)
print(f"  trace log dir: {trace_dir}")

trace_block: dict = {}
census_block: dict = {}
command_buffer_probe: dict = {}
vmap_block: dict | None = None

if VMAP_BATCH is None:
    with library_solver_injection.certified_solver_injected(
        PASS_BUDGET, fallback=True, tau_rel=TAU_REL
    ) as _trace_counts:
        _fn = _likelihood_fn(system_s3.dataset, _settings)

        with timer.section("trace_lower"):
            _lowered = jax.jit(_fn).lower(params_tree_s3)

        # ONE lowering, TWO compilations. The command-buffer knob is a per-compile
        # option, so both programs live in this process and both walls are measured
        # on the same lowering rather than across two runs.
        with timer.section("trace_compile_command_buffers_on"):
            _ex_on = _lowered.compile()
        with timer.section("trace_compile_command_buffers_off"):
            _ex_off = _lowered.compile(compiler_options={_COMMAND_BUFFER_OPTION: ""})

        _value_on = _ex_on(params_tree_s3)
        block(_value_on)
        _value_off = _ex_off(params_tree_s3)
        block(_value_off)

        if int(_trace_counts["jax"]) == 0:
            raise AssertionError(
                "the injected solver was never called on the JAX path while compiling the "
                "traced executable — the trace would decompose the library's own PDIP."
            )

        wall_on_ms = steady_wall_ms(_ex_on, params_tree_s3, TRACE_CALLS)
        wall_off_ms = steady_wall_ms(_ex_off, params_tree_s3, TRACE_CALLS)
        print(f"  command buffers ON  (production program): {wall_on_ms:8.3f} ms/call")
        print(f"  command buffers OFF (traced program):     {wall_off_ms:8.3f} ms/call")
        _delta_ms = wall_off_ms - wall_on_ms
        print(
            f"  delta (OFF - ON):                         {_delta_ms:8.3f} ms "
            f"({100.0 * _delta_ms / wall_on_ms:+.1f} %)"
        )

        _traced_ex = _ex_on if TRACE_ON_COMMAND_BUFFERS else _ex_off
        _traced_wall_ms = wall_on_ms if TRACE_ON_COMMAND_BUFFERS else wall_off_ms

        hlo_text = _traced_ex.as_text()
        module_proto = (
            _traced_ex.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()
        )
        # The optimized HLO and the module proto are written beside the trace, so the
        # census and the stage map can be re-derived from the SAME program later
        # without a 5-minute recompile — and so a disputed census row can be checked
        # against the text that produced it.
        (trace_dir / "hlo_optimized.txt").write_text(hlo_text)
        (trace_dir / "hlo_module.pb").write_bytes(module_proto)
        print(f"  optimized HLO + module proto written to {trace_dir}")

        stack_index = xla_attribution.StackFrameIndex.from_module_proto(module_proto)
        index = xla_attribution.hlo_index(hlo_text, stack_index)
        print(
            f"  optimized HLO: {len(index)} instructions, "
            f"{sum(1 for i in index.values() if i.frames)} resolved to source"
        )

        # The traced calls, and the SAME number of untraced calls immediately after,
        # so the profiler's own overhead is recorded rather than assumed negligible.
        _main_dir = trace_dir / ("traced_on" if TRACE_ON_COMMAND_BUFFERS else "traced_off")
        # The clock starts INSIDE the profiler context. ``jax.profiler.trace``'s own
        # start-up and its serialisation on exit are seconds, not milliseconds: timing
        # the whole context and dividing by K reported 145 ms/call against a 61 ms
        # untraced call on the shakeout, which is the profiler being switched on, not
        # a traced likelihood evaluation.
        with timer.section("trace_calls"):
            with jax.profiler.trace(
                str(_main_dir), create_perfetto_link=False, create_perfetto_trace=False
            ):
                # No warm-up call in here: the executable is already warm (both
                # command-buffer walls were measured above), and an extra call would
                # make the trace hold K+1 executions, which ``split_calls`` would
                # then refuse to split into K.
                _trace_t0 = time.perf_counter()
                for _ in range(TRACE_CALLS):
                    block(_traced_ex(params_tree_s3))
                traced_wall_ms = (time.perf_counter() - _trace_t0) / TRACE_CALLS * 1e3
        untraced_wall_ms = steady_wall_ms(_traced_ex, params_tree_s3, TRACE_CALLS)

        print(f"  traced   wall: {traced_wall_ms:8.3f} ms/call")
        print(f"  untraced wall: {untraced_wall_ms:8.3f} ms/call")

        events = xla_attribution.device_events(_main_dir)
        # Reconciled against the TRACED wall: these kernel durations came from the
        # traced run, and reconciling them against the untraced wall folds the
        # profiler's own overhead into the residual (it made host_outside_span_ms
        # negative on the first shakeout). The untraced wall is the production-
        # comparable number and is recorded beside it with its own percentage.
        trace_block = xla_attribution.attribute(
            events,
            index,
            wall_ms=traced_wall_ms,
            calls=TRACE_CALLS,
            untraced_wall_ms=untraced_wall_ms,
        )
        census_block = xla_attribution.hlo_census(index)

        # The qhull pure_callback is a host round-trip. Look for it by name on the
        # host planes; when the profiler does not name it, the device-idle gap it
        # creates is the only evidence and ``largest_idle_gaps`` above carries it.
        #
        # Host events NEST: the shakeout showed ``pure_callback.4`` (4.03 ms)
        # containing ``_wrapped_callback`` containing ``pure_callback_impl``
        # containing ``scipy_delaunay_tri_only`` (2.51 ms, the actual qhull). Summing
        # every matching event counted the same work five times over (12.78 ms on a
        # 61 ms call). So the events are aggregated BY NAME, and two headline numbers
        # are reported: the outermost ``pure_callback*`` span (the whole round trip)
        # and the qhull call inside it.
        _callback_hosts = [
            e
            for e in xla_attribution.host_events(_main_dir)
            if any(k in e["name"].lower() for k in ("callback", "qhull", "delaunay", "host_send"))
        ]
        _by_name: dict[str, dict] = {}
        for _e in _callback_hosts:
            _row = _by_name.setdefault(_e["name"], {"count": 0, "ms": 0.0})
            _row["count"] += 1
            _row["ms"] += _e["dur_ns"] / 1e6 / TRACE_CALLS

        def _sum_host_ms(predicate) -> float:
            return sum(v["ms"] for k, v in _by_name.items() if predicate(k.lower()))

        host_callback_ms = _sum_host_ms(lambda n: n.startswith("pure_callback"))
        trace_block["host_callback_ms"] = host_callback_ms
        trace_block["host_callback"] = {
            "pure_callback_span_ms": host_callback_ms,
            "qhull_ms": _sum_host_ms(lambda n: "scipy_delaunay" in n),
            "events_by_name": dict(sorted(_by_name.items(), key=lambda kv: -kv[1]["ms"])),
            "note": (
                "Host-plane events, aggregated BY NAME because they NEST: pure_callback.N contains "
                "_wrapped_callback contains pure_callback_impl contains scipy_delaunay_tri_only "
                "(the qhull call). pure_callback_span_ms is the outermost round trip; qhull_ms is "
                "the triangulation inside it. These are HOST milliseconds and are NOT a row of the "
                "device table — the device pays for them as the idle gap at pure_callback.N, which "
                "largest_idle_gaps names."
            ),
        }

        # The command-buffer probe: does CUPTI still emit per-kernel events inside a
        # CUDA graph? Answering this with data is what lets the ON/OFF delta be
        # judged instead of assumed.
        _probe_dir = trace_dir / "probe_command_buffers_on"
        _probe_ex = _ex_off if TRACE_ON_COMMAND_BUFFERS else _ex_on
        block(_probe_ex(params_tree_s3))
        with jax.profiler.trace(
            str(_probe_dir), create_perfetto_link=False, create_perfetto_trace=False
        ):
            for _ in range(3):
                block(_probe_ex(params_tree_s3))
        _probe_events = xla_attribution.device_events(_probe_dir)
        _probe_index = xla_attribution.hlo_index(_probe_ex.as_text())
        _probe_total = sum(e["dur_ns"] for e in _probe_events) or 1
        _probe_joined = sum(e["dur_ns"] for e in _probe_events if e["hlo_op"] in _probe_index)
        command_buffer_probe = {
            "probed_program": "command_buffers_off"
            if TRACE_ON_COMMAND_BUFFERS
            else "command_buffers_on",
            "events": len(_probe_events),
            "kernel_names_present": len({e["name"] for e in _probe_events}) > 1,
            "distinct_hlo_op": len({e["hlo_op"] for e in _probe_events}),
            "joined_duration_pct": 100.0 * _probe_joined / _probe_total,
            "top_hlo_op_by_duration": sorted(
                {
                    str(e["hlo_op"]): sum(
                        x["dur_ns"] for x in _probe_events if x["hlo_op"] == e["hlo_op"]
                    )
                    / 1e6
                    for e in _probe_events
                }.items(),
                key=lambda kv: -kv[1],
            )[:6],
            "note": (
                "CUPTI emits per-kernel events inside a CUDA graph, but their hlo_op stat is the "
                "graph node (command_buffer_N), not the instruction. joined_duration_pct is how "
                "much of the device time the name-based join recovers on that program."
            ),
        }

else:
    # ===================================================================
    # PART V — the matched vmap-vs-scalar experiment (#273)
    # ===================================================================
    # Two arms, the SAME B lanes, one process, one solver injection. The whole
    # point is that the pair is matched: a vmap millisecond and a scalar
    # millisecond measured in different processes, on different models, or
    # under different fallback semantics are not a comparison.

    print("\n" + "-" * 70)
    print(
        f"BATCHED MODE — B={VMAP_BATCH} {LANES_MODE} lanes, arms {','.join(ARMS)}, "
        f"fallback {'on' if FALLBACK_ON else 'off'}"
    )
    print("-" * 70)

    # --- the lanes ---------------------------------------------------------
    # Lane k IS draw k of the phase-3 seeded random family, applied to the lens
    # MASS, with the lens light fixed at S3 — the same construction
    # ``fixed_light_draws.py`` uses (``flds.mass_from(BASE_MASS, offsets)``).
    # The S3 dataset is the FIDUCIAL light-subtracted one and is shared by every
    # lane, exactly as production shares one dataset across a batch.
    BASE_MASS = flds.fiducial_mass_values(instance)

    if LANES_MODE == "distinct":
        lane_draws = flds.random_draws(DRAW_SEED, VMAP_BATCH)
    else:
        lane_draws = [
            flds.Draw(name="fiducial", kind="fiducial", offsets={}, mass_cls="Isothermal")
            for _ in range(VMAP_BATCH)
        ]

    _lens_s3 = instance_s3.galaxies.lens
    # The Galaxy's profiles by the names it holds them under — the same rule
    # ``fixed_light_system._profile_attrs`` uses, because af.Model paths and the
    # adapt-image dictionaries are keyed on those names.
    _LENS_ATTRS = {
        _k: _v
        for _k, _v in vars(_lens_s3).items()
        if not _k.startswith("_") and _k not in {"id", "redshift"}
    }
    if "mass" not in _LENS_ATTRS:
        raise AssertionError(
            "the S3 source-only lens galaxy has no `mass` attribute — the lane "
            "construction would swap a profile that is not there"
        )

    def _lane_tree(draw):
        """The params pytree of one lane: S3 with draw's mass swapped in.

        SHALLOW copies, deliberately. ``copy.deepcopy(instance_s3)`` duplicates
        the source galaxy's ``Pixelization`` and ``Regularization``, which sit in
        the pytree's **static aux data** and compare by object identity — two
        deepcopies of the same instance therefore have unequal treedefs and
        cannot be stacked into a batch at all. Sharing those objects across every
        lane is also the truth of the experiment: production batches one dataset
        and one pixelization over B parameter vectors.
        """
        if not draw.offsets:
            return params_tree_s3
        _inst = copy.copy(instance_s3)
        _inst.galaxies = copy.copy(instance_s3.galaxies)
        _attrs = dict(_LENS_ATTRS)
        _attrs["mass"] = flds.mass_from(BASE_MASS, draw.offsets, mass_cls=draw.mass_cls)
        _inst.galaxies.lens = al.Galaxy(redshift=float(_lens_s3.redshift), **_attrs)
        return jax.tree_util.tree_map(jnp.asarray, _inst)

    with timer.section("vmap_lane_build"):
        lane_trees = [_lane_tree(_d) for _d in lane_draws]
        _fiducial_structure = jax.tree_util.tree_structure(params_tree_s3)
        for _i, _t in enumerate(lane_trees):
            if jax.tree_util.tree_structure(_t) != _fiducial_structure:
                raise AssertionError(
                    f"lane {_i} has a different pytree structure from the fiducial S3 tree — "
                    f"the stacked batch would not be the same model family. (Static aux data "
                    f"such as the Pixelization and Regularization objects compares by IDENTITY, "
                    f"so every lane must SHARE them, not hold a copy.)"
                )
        batched_tree = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lane_trees)

    print(f"  lanes built: {len(lane_trees)} ({LANES_MODE}, seed {DRAW_SEED})")
    for _d in lane_draws[: min(4, len(lane_draws))]:
        _shown = {_p: round(float(_o), 5) for _p, _o in _d.offsets.items()}
        print(f"    {_d.name:<14} {_shown}")
    if len(lane_draws) > 4:
        print(f"    ... {len(lane_draws) - 4} more")

    arm_blocks: dict[str, dict] = {}
    lane_rows: list[dict] = []
    lane_pins: list[dict] = []
    report_rows: list[tuple[int, bool]] = []
    report_source = None

    def _steady_invoke_ms(invoke, n_repeats: int) -> float:
        """Mean per-INVOCATION wall of *n_repeats* blocked invocations.

        One invocation is one batch: a single call for the vmap arm, ``B``
        sequential blocked calls for the scalar arm.
        """
        block(invoke())
        _start = time.perf_counter()
        for _ in range(n_repeats):
            block(invoke())
        return (time.perf_counter() - _start) / n_repeats * 1e3

    def _host_callback_rows(log_dir) -> dict:
        """The phase-1 host-plane aggregation, per traced call.

        Host events NEST (``pure_callback.N`` contains ``_wrapped_callback``
        contains ``pure_callback_impl`` contains the qhull body), so they are
        aggregated BY NAME and two headline numbers are reported rather than a
        sum that counts the same work several times over.
        """
        _hosts = [
            _e
            for _e in xla_attribution.host_events(log_dir)
            if any(
                _k in _e["name"].lower() for _k in ("callback", "qhull", "delaunay", "host_send")
            )
        ]
        _by_name: dict[str, dict] = {}
        for _e in _hosts:
            _row = _by_name.setdefault(_e["name"], {"count": 0, "ms": 0.0})
            _row["count"] += 1
            _row["ms"] += _e["dur_ns"] / 1e6 / TRACE_CALLS
        _span = sum(
            _v["ms"] for _k, _v in _by_name.items() if _k.lower().startswith("pure_callback")
        )
        # The probe replaces the library body, and the profiler names a host event
        # from the CODE OBJECT rather than __name__ -- so under the probe there is
        # no "scipy_delaunay" event at all and a phase-1 matcher reports 0.0 ms of
        # qhull in silence. The probe's own event names are matched here as well,
        # and the finer qhull-vs-tables split comes from the probe's records.
        _body_keys = ("scipy_delaunay", *host_callback_probe.PROBE_HOST_EVENT_FRAGMENTS)
        return {
            "pure_callback_span_ms": _span,
            "qhull_ms": max(
                (
                    _v["ms"]
                    for _k, _v in _by_name.items()
                    if any(_f in _k.lower() for _f in _body_keys)
                ),
                default=0.0,
            ),
            "callback_body_ms_note": (
                "the OUTERMOST matching body event, not a sum: under the probe the body "
                "appears twice (the wrapper and the timed re-implementation it calls) and "
                "summing them would count the same work twice"
            ),
            "events_by_name": dict(sorted(_by_name.items(), key=lambda kv: -kv[1]["ms"])),
            "note": (
                "HOST milliseconds per traced BATCH (the whole invocation), aggregated by name "
                "because the events nest. Not a row of the device table: the device pays for "
                "them as the idle gaps largest_idle_gaps names."
            ),
        }

    def _run_arm(arm: str, scalar_fn, vmap_fn) -> dict:
        """Compile, time and trace one arm on the B lanes.

        *scalar_fn* is ``jax.jit(fn)`` and *vmap_fn* is ``jax.vmap(scalar_fn)``
        — the SAME inner jit object, which is what makes the vmap arm the
        production nesting rather than a look-alike, and what keeps the two arms
        from compiling the scalar program twice.
        """
        print(f"\n  --- arm {arm} ---")

        if arm == "vmap":
            # The PRODUCTION nesting, Fitness._vmap: jax.vmap(jax.jit(call)).
            # NOT timing.vmap_profile, which is jax.jit(jax.vmap(call)).
            _prod = vmap_fn

            def _invoke_prod():
                return _prod(batched_tree)

            # AOT handle for the HLO text/proto and for the command-buffer knob,
            # which is a per-COMPILE option and so is unreachable through the
            # eager production nesting. The two are pinned equal below by wall
            # and by log likelihood; `hlo_source` records that they differ.
            with timer.section("vmap_arm_lower"):
                _lowered = jax.jit(_prod).lower(batched_tree)
            _hlo_source = (
                "jax.jit(jax.vmap(jax.jit(fn))).lower(batched_tree) — an outer jit is the only "
                "handle that yields optimized HLO and a compiler_options knob; the TIMED "
                "production number is wall_production_nesting_ms, from jax.vmap(jax.jit(fn))"
            )

            def _invoke_ex(ex):
                return ex(batched_tree)

            _expected_calls = TRACE_CALLS
            _executions_per_invoke = 1
        else:
            _prod = scalar_fn

            def _invoke_prod():
                return [_prod(_t) for _t in lane_trees]

            with timer.section("scalar_arm_lower"):
                _lowered = scalar_fn.lower(lane_trees[0])
            _hlo_source = "jax.jit(fn).lower(lane_trees[0]) — the same program phase 1 traced"

            def _invoke_ex(ex):
                return [ex(_t) for _t in lane_trees]

            # One traced invocation is B executions of one program, so the trace
            # is split by TRACE_CALLS * B. DECIDED IN CODE: pooling the lanes
            # gives B times the blocks and therefore a per-stage median over
            # every lane, which is what a straggler shows up in; splitting per
            # lane would give B tables of TRACE_CALLS blocks and no pooled
            # median at all.
            _expected_calls = TRACE_CALLS * VMAP_BATCH
            _executions_per_invoke = VMAP_BATCH

        with timer.section(f"{arm}_arm_compile_command_buffers_on"):
            _ex_on = _lowered.compile()
        with timer.section(f"{arm}_arm_compile_command_buffers_off"):
            _ex_off = _lowered.compile(compiler_options={_COMMAND_BUFFER_OPTION: ""})

        _wall_prod = _steady_invoke_ms(_invoke_prod, TRACE_CALLS)
        _wall_on = _steady_invoke_ms(lambda: _invoke_ex(_ex_on), TRACE_CALLS)
        _wall_off = _steady_invoke_ms(lambda: _invoke_ex(_ex_off), TRACE_CALLS)
        print(f"    batch wall, production nesting:      {_wall_prod:9.3f} ms")
        print(f"    batch wall, AOT command buffers ON:  {_wall_on:9.3f} ms")
        print(f"    batch wall, AOT command buffers OFF: {_wall_off:9.3f} ms")

        _hlo_text = _ex_off.as_text()
        _proto = _ex_off.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()
        (trace_dir / f"hlo_optimized_{arm}.txt").write_text(_hlo_text)
        (trace_dir / f"hlo_module_{arm}.pb").write_bytes(_proto)
        _index = xla_attribution.hlo_index(
            _hlo_text, xla_attribution.StackFrameIndex.from_module_proto(_proto)
        )
        print(
            f"    optimized HLO: {len(_index)} instructions, "
            f"{sum(1 for _i in _index.values() if _i.frames)} resolved to source"
        )

        _arm_dir = trace_dir / f"traced_{arm}"
        probe.reset()
        with timer.section(f"trace_calls_{arm}"):
            with jax.profiler.trace(
                str(_arm_dir), create_perfetto_link=False, create_perfetto_trace=False
            ):
                _t0 = time.perf_counter()
                for _ in range(TRACE_CALLS):
                    block(_invoke_ex(_ex_off))
                _traced_batch_ms = (time.perf_counter() - _t0) / TRACE_CALLS * 1e3
        _callback_records = [_r.as_dict() for _r in probe.records]
        _probe_summary = probe.summary()
        _callbacks_per_call = probe.calls / TRACE_CALLS
        _untraced_batch_ms = _steady_invoke_ms(lambda: _invoke_ex(_ex_off), TRACE_CALLS)

        _events = xla_attribution.device_events(_arm_dir)
        _block = xla_attribution.attribute(
            _events,
            _index,
            wall_ms=_traced_batch_ms / _executions_per_invoke,
            calls=_expected_calls,
            untraced_wall_ms=_untraced_batch_ms / _executions_per_invoke,
        )
        # The vmap arm's "call" is the whole batch, so its stage rows are per
        # BATCH; the scalar arm's are already per likelihood. Both are divided
        # to a per-LANE row so the two arms can be put side by side.
        _per_lane_divisor = VMAP_BATCH if arm == "vmap" else 1
        _block["per_stage_ms_per_lane"] = {
            _label: _row["median_ms"] / _per_lane_divisor
            for _label, _row in _block["per_stage_ms"].items()
        }
        _block["per_lane_divisor"] = _per_lane_divisor
        _block["host_callback"] = _host_callback_rows(_arm_dir)
        _block["host_callback_ms"] = _block["host_callback"]["pure_callback_span_ms"]

        _row = {
            "arm": arm,
            "program": (
                "jax.vmap(jax.jit(fn))" if arm == "vmap" else "jax.jit(fn), called B times"
            ),
            "stands_for": (
                "Nautilus(use_jax_vmap=True) — Fitness._vmap"
                if arm == "vmap"
                else "Nautilus(use_jax_vmap=False, use_jax_jit=True). Plain use_jax_vmap=False "
                "is UNJITTED and is NOT this arm"
            ),
            "hlo_source": _hlo_source,
            "batch_wall_ms": _wall_prod,
            "wall_production_nesting_ms": _wall_prod,
            "wall_aot_command_buffers_on_ms": _wall_on,
            "wall_aot_command_buffers_off_ms": _wall_off,
            "command_buffer_delta_ms": _wall_off - _wall_on,
            "production_nesting_vs_aot_on_delta_ms": _wall_prod - _wall_on,
            "production_nesting_vs_aot_on_delta_pct": 100.0 * (_wall_prod - _wall_on) / _wall_on,
            "wall_per_lane_ms": _wall_prod / VMAP_BATCH,
            "wall_per_lane_command_buffers_off_ms": _wall_off / VMAP_BATCH,
            "traced_batch_wall_ms": _traced_batch_ms,
            "untraced_batch_wall_ms": _untraced_batch_ms,
            "traced_wall_per_lane_ms": _traced_batch_ms / VMAP_BATCH,
            "callback_count_per_call": _callbacks_per_call,
            "callback_count_per_lane": _callbacks_per_call / VMAP_BATCH,
            "callback_expectation": (
                "B per batched call — the qhull pure_callback is vmap_method='sequential'"
                if arm == "vmap"
                else "1 per likelihood evaluation, B per batch"
            ),
            "host_qhull_ms": _probe_summary["qhull_ms_total"] / TRACE_CALLS,
            "host_tables_ms": _probe_summary["tables_ms_total"] / TRACE_CALLS,
            "host_qhull_ms_per_lane": _probe_summary["qhull_ms_total"] / TRACE_CALLS / VMAP_BATCH,
            "host_tables_ms_per_lane": _probe_summary["tables_ms_total"] / TRACE_CALLS / VMAP_BATCH,
            "host_callback_probe": _probe_summary,
            "host_callback_records_first": _callback_records[: min(4, len(_callback_records))],
            "device_idle_ms": _block["device_idle_ms"],
            "device_idle_ms_per_lane": _block["device_idle_ms"] / _per_lane_divisor,
            "trace": _block,
            "hlo_census": xla_attribution.hlo_census(_index),
            "trace_split": {
                "expected_calls": _expected_calls,
                "executions_per_invocation": _executions_per_invoke,
                "method": _block["call_split"],
                "note": (
                    "DECIDED IN CODE: the scalar arm's trace is split by trace_calls * B, so "
                    "every lane's execution is its own block and the per-stage median is taken "
                    "over all of them. The vmap arm's is split by trace_calls, one block per "
                    "batched execution, and per_stage_ms_per_lane divides those rows by B."
                ),
            },
        }
        print(
            f"    callbacks/call {_callbacks_per_call:6.2f}  "
            f"qhull {_row['host_qhull_ms']:7.3f} ms  tables {_row['host_tables_ms']:7.3f} ms  "
            f"device_idle {_block['device_idle_ms']:7.3f} ms  "
            f"reconciliation {_block['reconciliation_pct']:+.2f} %"
        )
        return _row

    # One probe and one injection over BOTH arms: the host wrapper and the
    # solver are identical for the pair, or the pair is not matched.
    with (
        host_callback_probe.qhull_probe() as probe,
        library_solver_injection.certified_solver_injected(
            PASS_BUDGET, fallback=FALLBACK_ON, tau_rel=TAU_REL
        ) as _trace_counts,
    ):
        # ONE pair of wrappers for the pins and both arms: a second
        # jax.jit(_fn_v) would be a second compile of the same program.
        _fn_v = _likelihood_fn(system_s3.dataset, _settings)
        _scalar_pin_fn = jax.jit(_fn_v)
        _vmap_pin_fn = jax.vmap(_scalar_pin_fn)

        # --- the per-lane pin, un-timed ------------------------------------
        with timer.section("vmap_lane_log_likelihoods"):
            _ll_vmap = np.asarray(block(_vmap_pin_fn(batched_tree)), dtype=float)
            _ll_scalar = np.asarray(
                [float(block(_scalar_pin_fn(_t))) for _t in lane_trees], dtype=float
            )
        if _ll_vmap.shape != (VMAP_BATCH,):
            raise AssertionError(
                f"the vmap arm returned shape {_ll_vmap.shape}, not ({VMAP_BATCH},) — "
                f"the batch axis is not the lane axis"
            )

        for _k, _draw in enumerate(lane_draws):
            _rel = abs(_ll_vmap[_k] - _ll_scalar[_k]) / max(abs(_ll_scalar[_k]), 1e-300)
            lane_rows.append(
                {
                    "lane": _k,
                    "draw_name": _draw.name,
                    "draw_kind": _draw.kind,
                    "offsets": {_p: float(_o) for _p, _o in _draw.offsets.items()},
                    "log_likelihood_vmap": float(_ll_vmap[_k]),
                    "log_likelihood_scalar": float(_ll_scalar[_k]),
                    "rel_diff": float(_rel),
                    "status": "PASS" if _rel <= LANE_RTOL else "FAIL",
                    # Filled by the reporting pass below when it runs.
                    "certified": None,
                    "pass_at_certification": None,
                    "pdip_iter": None,
                }
            )
        lane_pins = [
            {
                "pin": f"lane {_r['lane']}: vmap == scalar",
                "rtol": LANE_RTOL,
                "got": _r["log_likelihood_vmap"],
                "reference_value": _r["log_likelihood_scalar"],
                "rel_diff": _r["rel_diff"],
                "status": _r["status"],
            }
            for _r in lane_rows
        ]
        _n_failed = sum(1 for _r in lane_rows if _r["status"] == "FAIL")
        print(
            f"\n  per-lane pin |ll_vmap - ll_scalar| / |ll| <= {LANE_RTOL:.0e}: "
            f"{len(lane_rows) - _n_failed}/{len(lane_rows)} PASS  "
            f"(worst rel {max(_r['rel_diff'] for _r in lane_rows):.3e})"
        )

        # --- the arms ------------------------------------------------------
        for _arm_name in ARMS:
            arm_blocks[_arm_name] = _run_arm(_arm_name, _scalar_pin_fn, _vmap_pin_fn)

        if int(_trace_counts["jax"]) == 0:
            raise AssertionError(
                "the injected solver was never called on the JAX path — every row of this "
                "table would be the library's own PDIP wearing the certified label."
            )

    # --- the per-lane certification report, un-timed -----------------------
    # A SEPARATE injection context with the report hook, and therefore SEPARATE
    # jax.jit objects: a jitted function compiled under one injection is cached,
    # so reusing the timed arm's wrapper here would hand back the program
    # compiled WITHOUT the hook (or, run the other way round, would leave the
    # ordered host callback inside every timed call).
    _report_arm = "vmap" if "vmap" in ARMS else "scalar"

    def _collect_report(pass_at, certified):
        report_rows.append((int(pass_at), bool(certified)))

    with library_solver_injection.certified_solver_injected(
        PASS_BUDGET, fallback=FALLBACK_ON, tau_rel=TAU_REL, report=_collect_report
    ) as _report_counts:
        _fn_report = _likelihood_fn(system_s3.dataset, _settings)
        with timer.section("vmap_certification_report"):
            if _report_arm == "vmap":
                block(jax.vmap(jax.jit(_fn_report))(batched_tree))
            else:
                _rep_fn = jax.jit(_fn_report)
                for _t in lane_trees:
                    block(_rep_fn(_t))
    report_source = (
        f"{_report_arm} arm, jax.debug.callback(..., ordered=True) inside the injected "
        f"certified solver; ordered because the unordered batching rule returns the lanes "
        f"permuted"
    )

    if len(report_rows) == len(lane_rows):
        for _r, (_pass_at, _cert) in zip(lane_rows, report_rows):
            _r["certified"] = bool(_cert)
            _r["pass_at_certification"] = int(_pass_at) if _pass_at > 0 else None
    print(
        f"  certification report: {len(report_rows)} rows for {len(lane_rows)} lanes "
        f"({sum(1 for _, _c in report_rows if _c)} certified within budget {PASS_BUDGET})"
    )

    # --- what the phase-1 keys mean in this mode ---------------------------
    # The scalar arm IS the phase-1 program (jax.jit(fn)), so it fills the
    # top-level trace/census/command-buffer keys and the printed stage table
    # below; the vmap arm's own full block lives under `vmap`. When only the
    # vmap arm was asked for, those keys carry the vmap arm and say so.
    _headline = arm_blocks.get("scalar") or arm_blocks["vmap"]
    trace_block = _headline["trace"]
    census_block = _headline["hlo_census"]
    wall_on_ms = _headline["wall_aot_command_buffers_on_ms"]
    wall_off_ms = _headline["wall_aot_command_buffers_off_ms"]
    traced_wall_ms = trace_block["wall_ms"]
    untraced_wall_ms = trace_block["untraced_wall_ms"]
    command_buffer_probe = {
        "note": (
            "Not taken in --vmap-batch mode: the CUPTI-inside-a-CUDA-graph probe is a phase-1 "
            "diagnostic about the JOIN, and this mode already compiles and times both the "
            "command-buffers ON and OFF executable of every arm."
        )
    }

    vmap_block = {
        "batch": int(VMAP_BATCH),
        "lanes_mode": LANES_MODE,
        "arms": list(ARMS),
        "draw_seed": DRAW_SEED,
        "fallback": FALLBACK_ON,
        "headline_arm": "scalar" if "scalar" in arm_blocks else "vmap",
        "lane_rtol": LANE_RTOL,
        "lanes": lane_rows,
        "lane_pins": lane_pins,
        "lanes_failed": sum(1 for _r in lane_rows if _r["status"] == "FAIL"),
        "arm_rows": arm_blocks,
        "certification_report": {
            "source": report_source,
            "rows": len(report_rows),
            "pdip_iter": (
                "NOT AVAILABLE per lane without changing library numerics. The fallback branch "
                "calls the library's own reconstruction_positive_only_from, which returns the "
                "solution alone — jax_nnls.solve_nnls' iteration count is discarded inside the "
                "library's custom_vjp primal. The PDIP while_loop's cost is in the pdip_solve "
                "stage row of each arm's device table instead, which is a max-over-lanes "
                "quantity under vmap because every lane runs the same fixed kernel sequence."
            ),
        },
        "lane_construction": {
            "base_mass": {_k: float(_v) for _k, _v in BASE_MASS.items()},
            "sigma_scale": flds.RANDOM_SIGMA_SCALE,
            "parameters": list(flds.RANDOM_PARAMETERS),
            "note": (
                "Lane k is random draw k of the phase-3 seeded family applied to the lens mass "
                "(Isothermal), with the lens light FIXED at S3 and the S3 light-subtracted "
                "dataset shared by every lane — one dataset, B parameter vectors, as production "
                "batches. 'identical' broadcasts the fiducial and is the CONTROL: identical "
                "lanes share a straggler and hide the cost this experiment looks for."
            ),
        },
    }
    if "vmap" in arm_blocks and "scalar" in arm_blocks:
        _v_ms = arm_blocks["vmap"]["wall_per_lane_ms"]
        _s_ms = arm_blocks["scalar"]["wall_per_lane_ms"]
        vmap_block["matched"] = {
            "vmap_wall_per_lane_ms": _v_ms,
            "scalar_wall_per_lane_ms": _s_ms,
            "speedup_vmap_over_scalar": _s_ms / _v_ms,
            "note": (
                "Both walls are the PRODUCTION nesting of their arm, measured in one process on "
                "the same B lanes under the same solver injection. A speedup below 1 means the "
                "batch is slower per lane than B sequential jitted calls."
            ),
        }
        print(
            f"\n  MATCHED: vmap {_v_ms:.3f} ms/lane vs scalar {_s_ms:.3f} ms/lane "
            f"-> {_s_ms / _v_ms:.3f}x"
        )

peak_after_all = peak_bytes()

if vmap_block is None:
    print(
        "\n  --- per-stage table (ms/call, command buffers "
        f"{'ON' if TRACE_ON_COMMAND_BUFFERS else 'OFF'}) ---"
    )
else:
    print(
        f"\n  --- per-stage table (ms per likelihood, command buffers OFF) — "
        f"{vmap_block['headline_arm']} arm ---"
    )
for _label in trace_block["stage_order"]:
    _row = trace_block["per_stage_ms"][_label]
    print(
        f"    {_label:<34} {_row['median_ms']:8.3f}  [{_row['min_ms']:.3f} .. {_row['max_ms']:.3f}]"
    )
print(f"    {'device_idle':<34} {trace_block['device_idle_ms']:8.3f}")
print(f"    {'-' * 44}")
print(f"    {'sum (kernels + idle)':<34} {trace_block['sum_ms']:8.3f}")
print(f"    {'wall (traced, same executable)':<34} {trace_block['wall_ms']:8.3f}")
print(f"    {'wall (untraced, same executable)':<34} {trace_block['untraced_wall_ms']:8.3f}")
print(f"    {'host outside device span':<34} {trace_block['host_outside_span_ms']:8.3f}")
print(
    f"    reconciliation: {trace_block['reconciliation_pct']:+.2f} % (vs traced wall)  "
    f"| {trace_block['reconciliation_vs_untraced_pct']:+.2f} % (vs untraced)  "
    f"| unjoined {trace_block['unjoined_ms']:.3f} ms  "
    f"| other {trace_block['other_ms']:.3f} ms"
)

print(
    f"    {'host pure_callback (HOST, not a device row)':<34} "
    f"{trace_block['host_callback']['pure_callback_span_ms']:8.3f}  "
    f"(qhull {trace_block['host_callback']['qhull_ms']:.3f})"
)

if vmap_block is not None:
    print("\n  --- matched arms (ms per LANE, command buffers OFF, traced program) ---")
    _stage_labels = sorted(
        {
            _lbl
            for _arm_row in vmap_block["arm_rows"].values()
            for _lbl in _arm_row["trace"]["per_stage_ms_per_lane"]
        },
        key=lambda _lbl: (
            -max(
                _arm_row["trace"]["per_stage_ms_per_lane"].get(_lbl, 0.0)
                for _arm_row in vmap_block["arm_rows"].values()
            )
        ),
    )
    print(f"    {'stage':<34} {'vmap':>10} {'scalar':>10}")
    for _lbl in _stage_labels:
        _cells = []
        for _arm_name in ("vmap", "scalar"):
            _arm_row = vmap_block["arm_rows"].get(_arm_name)
            _cells.append(
                f"{_arm_row['trace']['per_stage_ms_per_lane'][_lbl]:10.3f}"
                if _arm_row and _lbl in _arm_row["trace"]["per_stage_ms_per_lane"]
                else f"{'-':>10}"
            )
        print(f"    {_lbl:<34} {_cells[0]} {_cells[1]}")
    for _label, _key in (
        ("device_idle", "device_idle_ms_per_lane"),
        ("host qhull (probe)", "host_qhull_ms_per_lane"),
        ("host tables (probe)", "host_tables_ms_per_lane"),
        ("qhull callbacks per batch", "callback_count_per_call"),
        ("qhull callbacks per lane", "callback_count_per_lane"),
        ("WALL per lane (production)", "wall_per_lane_ms"),
    ):
        _cells = []
        for _arm_name in ("vmap", "scalar"):
            _arm_row = vmap_block["arm_rows"].get(_arm_name)
            _cells.append(f"{_arm_row[_key]:10.3f}" if _arm_row else f"{'-':>10}")
        print(f"    {_label:<34} {_cells[0]} {_cells[1]}")

print("\n  --- HLO census ---")
print(
    f"    (n,n) add of F + lambda*H at abstract.py:371: "
    f"{census_block['curvature_reg_add_nn']['count']}"
    f"  -> {census_block['curvature_reg_add_nn']['verdict']}"
)
print(
    f"    gathers at abstract.py:613 (edge subset):     "
    f"{census_block['edge_subset_gathers_613']['count']}"
)
print(
    f"    gathers at abstract.py:397 (reduced):         "
    f"{census_block['curvature_reg_reduced_gathers_397']['count']}"
)
print(
    f"    FFTs sourced at convolver.py:                 "
    f"{census_block['fft_convolver_total']['count']}"
    f"  (mapping matrix {census_block['fft_mapping_matrix']['count']}, "
    f"image {census_block['fft_image']['count']})"
)
print(
    f"    cholesky factorizations:                      {census_block['cholesky']['count']}"
    f"  (triangular solves {census_block['triangular_solve']['count']})"
)
_where = census_block["curvature_reg_add_nn"]["where_the_sum_lives"]
print(
    f"    opcodes written at abstract.py:371:           {census_block['curvature_reg_add_nn']['opcodes_at_line_371']}"
)
print(
    f"    producers shared by >1 consumer of F+lambda*H: {_where['producers_shared_by_more_than_one_consumer']}"
)
for _r in _where["consumers_of_F_plus_lambda_H"]:
    print(
        f"        {_r['consumer']:26} <- {_r['producer']:30} {_r['producer_opcode']:12} "
        f"{(_r['producer_source'] or '').split('/')[-1]}"
    )
print(f"    VERDICT: {census_block['curvature_reg_add_nn']['verdict']}")

# ===================================================================
# Summary + JSON + PNG
# ===================================================================

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

trace_summary = {
    "device": device_info_dict(),
    "machine": machine_info_dict(),
    "precision": "mixed" if _cli.use_mixed_precision else "fp64",
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "mesh": MESH,
        "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
        "source_pixels": int(n_source_pixels),
        "source_pixels_requested": (
            int(SOURCE_PIXELS_REQUESTED) if SOURCE_PIXELS_REQUESTED is not None else None
        ),
        "dataset": DATASET,
        "inversion_path": "dense",
        "pass_budget": PASS_BUDGET,
        "pass_budget_basis": (
            "phase 3 (autolens_profiling#255): the smallest ZERO-fallback budget per mesh, the "
            "budget a production run fixes. NOT phase 0's certifying budget (2 for Delaunay), "
            "which falls back on 67.5 % of a graded draw set and is not a production cost."
        ),
        "border_relocator_mode": BORDER_RELOCATOR_MODE,
        "border_relocator": BORDER_RELOCATOR_RESOLVED,
        "border_relocator_note": (
            "'cell' is use_border_relocator=True, which every profiling cell in this family "
            "forces; 'library' passes None and takes the shipped config default, which is what "
            "production runs. The resolved boolean is what this leg actually ran."
        ),
        "routes_selected": list(ROUTE_SELECTION),
        "trace_calls": TRACE_CALLS,
        "command_buffers_traced": "on" if TRACE_ON_COMMAND_BUFFERS else "off",
        "use_mixed_precision": bool(_cli.use_mixed_precision),
        "tau_rel": TAU_REL,
        "regularization": reg_provenance,
        "thread_env": _observe_thread_env(),
        "over_sample_size_lp_rule": {
            "sub_size_list": [4, 2, 2],
            "radial_list": [0.3, 0.6],
            "centre": [0.0, 0.0],
        },
    },
    "solver_injection": {
        "patched": library_solver_injection.LIBRARY_POSITIVE_ONLY_DOTTED,
        "pass_budget": PASS_BUDGET,
        "fallback": True if vmap_block is None else FALLBACK_ON,
        "calls_jax": int(_trace_counts["jax"]),
        "calls_numpy": int(_trace_counts["numpy"]),
        "note": (
            "Routes d and the traced program use a scoped harness monkeypatch of the library's "
            "own positive-only entry point. Everything else in the call — mapper, F + lambda*H, "
            "both log determinants, the evidence — is the library's own code, unmodified."
        ),
    },
    "routes": routes,
    "equivalence_pins": equivalence_pins,
    "equivalence_rtol": EQUIVALENCE_RTOL,
    "reference": {
        "log_evidence_s0_library": log_evidence_ref,
        "log_evidence_s3_library": log_evidence_s3_library,
        "s3": {
            "n_params": int(system_s3.n_params),
            "n_mapper": int(system_s3.n_mapper),
            "n_funcs": int(system_s3.n_funcs),
            "edge_zeroed_pixels": int(system_s3.edge_zero_mask.sum()),
            "subtracted_light_flux": float(system_s3.subtracted_light_flux),
        },
    },
    "command_buffers": {
        "wall_on_ms": wall_on_ms,
        "wall_off_ms": wall_off_ms,
        "command_buffer_delta_ms": wall_off_ms - wall_on_ms,
        "command_buffer_delta_pct": 100.0 * (wall_off_ms - wall_on_ms) / wall_on_ms,
        "traced": "on" if TRACE_ON_COMMAND_BUFFERS else "off",
        "compiler_option": _COMMAND_BUFFER_OPTION,
        "probe": command_buffer_probe,
        "note": (
            "ONE lowering, TWO compilations, one process. wall_on_ms is the PRODUCTION program "
            "(CUDA graphs as XLA ships them) and is the number comparable to phase 5's row; "
            "wall_off_ms is the graph-less program the stage rows attribute. The delta is the "
            "launch overhead the graphs remove — it lands in device_idle and host_outside_span, "
            "not in any stage. Reconciliation is computed against wall_off_ms."
        ),
    },
    "trace": trace_block,
    "hlo_census": census_block,
    "stage_map_provenance": xla_attribution.stage_map_provenance(),
    "trace_timing": {
        "traced_wall_ms": traced_wall_ms,
        "untraced_wall_ms": untraced_wall_ms,
        "profiler_overhead_ms": traced_wall_ms - untraced_wall_ms,
        "profiler_overhead_pct": 100.0 * (traced_wall_ms - untraced_wall_ms) / untraced_wall_ms,
    },
    "jit_phases": jit_records,
    "peak_bytes": {
        "after_setup": peak_after_setup,
        "after_routes": peak_after_routes,
        "after_all": peak_after_all,
        "note": (
            "jax.devices()[0].memory_stats()['peak_bytes_in_use']; None on a backend that does "
            "not report it (CPU). Process-wide high-water marks."
        ),
    },
    "note": (
        "Every stage row is a MEASURED sum of GPU kernel durations joined to library source "
        "through the optimized HLO of ONE program — not arithmetic across compilations. The "
        "rows attribute the command-buffers-OFF program: kernel time is the same as production, "
        "the launch overhead the CUDA graphs remove is not, and command_buffers.* carries both "
        "walls and their delta. sum_ms = kernel rows + measured device_idle_ms; the residual "
        "against the wall is host_outside_span_ms and is reported, not absorbed. "
        "mixed_fusion.prorated_estimate_ms is an ESTIMATE and is never part of the table. "
        "No row may be quoted if reconciliation_pct is outside 5 % or unjoined_ms is non-zero."
    ),
}

if vmap_block is not None:
    # The new flags go in ``configuration`` ONLY in this mode, so a phase-1 JSON
    # written by this checkout is key-for-key the JSON phase 1 wrote — while a
    # batched JSON still carries every flag ``parse_known_args`` would otherwise
    # have let an old checkout swallow in silence.
    trace_summary["configuration"].update(
        {
            "vmap_batch": int(VMAP_BATCH),
            "lanes": LANES_MODE,
            "arms": list(ARMS),
            "fallback": "on" if FALLBACK_ON else "off",
            "draw_seed": DRAW_SEED,
            "lane_rtol": LANE_RTOL,
            "vmap_flags_note": (
                "parse_known_args IGNORES unknown flags, so these five keys are the only "
                "evidence that the checkout which produced this JSON understood --vmap-batch. "
                "A JSON of a batched leg WITHOUT them was written by a phase-1 checkout that "
                "silently swallowed the flags and measured a single call."
            ),
        }
    )
    trace_summary["vmap"] = vmap_block
    trace_summary["headline_arm_note"] = (
        f"`trace`, `hlo_census`, `command_buffers` and `trace_timing` above describe the "
        f"{vmap_block['headline_arm']} arm — the program phase 1 traced. Both arms' full "
        f"blocks, the lane table and the matched comparison are under `vmap`."
    )

# The suffix carries the MODE and the RESOLVED value. Keying it on the resolved
# value alone let the ``cell`` and ``library`` legs write the same filename on a
# workspace where the config default is already True — the second leg silently
# clobbered the first.
_cell_name = f"fixed_light_trace_{MESH}"
if BORDER_RELOCATOR_MODE == "off":
    # ``off`` states its own resolved value; ``_border_off_off`` said it twice.
    _cell_name = f"{_cell_name}_border_off"
elif BORDER_RELOCATOR_MODE == "library":
    # ``library`` does NOT: it resolves from the config, and on this workspace it
    # resolved to True. The resolved value is in the name so a library leg can
    # never be mistaken for, or clobber, the leg it happened to agree with.
    _cell_name = f"{_cell_name}_border_library_{'on' if BORDER_RELOCATOR_RESOLVED else 'off'}"
if SOURCE_PIXELS_REQUESTED is not None:
    _cell_name = f"{_cell_name}_n{int(n_source_pixels)}"
if VMAP_BATCH is not None:
    # The batched legs never collide with the phase-1 files in the same
    # directory: batch size, lane family and fallback semantics are three
    # different programs and all three are in the name.
    _cell_name = (
        f"{_cell_name}_vmap{int(VMAP_BATCH)}_{LANES_MODE}_fb{'on' if FALLBACK_ON else 'off'}"
    )

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"{_cell_name}_trace_{DATASET}_v{al_version}",
    cell=_cell_name,
)
dict_path.write_text(json.dumps(trace_summary, indent=2, default=str))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart -------------------------------------------------------
# Phase 1: the per-stage table, largest first. Batched mode: the two arms side
# by side, per stage, in milliseconds PER LANE — the only basis on which a
# batched program and B sequential ones can be compared at all.

if vmap_block is not None:
    _arm_names = [_a for _a in ("vmap", "scalar") if _a in vmap_block["arm_rows"]]
    _labels = sorted(
        {
            _lbl
            for _a in _arm_names
            for _lbl in vmap_block["arm_rows"][_a]["trace"]["per_stage_ms_per_lane"]
        },
        key=lambda _lbl: max(
            vmap_block["arm_rows"][_a]["trace"]["per_stage_ms_per_lane"].get(_lbl, 0.0)
            for _a in _arm_names
        ),
    )
    _labels = [
        _lbl
        for _lbl in _labels
        if max(
            vmap_block["arm_rows"][_a]["trace"]["per_stage_ms_per_lane"].get(_lbl, 0.0)
            for _a in _arm_names
        )
        > 0.0
    ]
    _labels = _labels + ["device_idle"]

    def _value(arm_name: str, label: str) -> float:
        _row = vmap_block["arm_rows"][arm_name]
        if label == "device_idle":
            return _row["device_idle_ms_per_lane"]
        return _row["trace"]["per_stage_ms_per_lane"].get(label, 0.0)

    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.42 * len(_labels))))
    _y = np.arange(len(_labels), dtype=float)
    _height = 0.38
    _arm_colors = {"vmap": "#4C72B0", "scalar": "#DD8452"}
    _max_ms = 1.0
    for _i, _a in enumerate(_arm_names):
        _vals = [_value(_a, _lbl) for _lbl in _labels]
        _max_ms = max(_max_ms, max(_vals, default=0.0))
        _offset = (_i - (len(_arm_names) - 1) / 2.0) * _height
        ax.barh(
            _y + _offset,
            _vals,
            height=_height,
            color=_arm_colors[_a],
            edgecolor="white",
            label=f"{_a} ({vmap_block['arm_rows'][_a]['wall_per_lane_ms']:.2f} ms/lane wall)",
        )
        for _yy, _v in zip(_y + _offset, _vals):
            if _v > 0.0:
                ax.text(_v + _max_ms * 0.01, _yy, f"{_v:.3f}", va="center", fontsize=7)

    ax.set_yticks(_y)
    ax.set_yticklabels(_labels, fontsize=9)
    ax.set_xlabel("device time per LANE (ms) — command buffers OFF, traced program")
    ax.set_xlim(0, _max_ms * 1.22)
    _cb_ms = [
        f"{_a} {vmap_block['arm_rows'][_a]['callback_count_per_call']:.1f}" for _a in _arm_names
    ]
    ax.set_title(
        f"Fixed-light {MESH} N={n_source_pixels} — vmap vs scalar at B={VMAP_BATCH} "
        f"({LANES_MODE} lanes, fallback {'on' if FALLBACK_ON else 'off'})\n"
        f"budget {PASS_BUDGET}, border relocator {BORDER_RELOCATOR_RESOLVED} | "
        f"qhull callbacks per batched call: {', '.join(_cb_ms)} | "
        f"lanes pinned {len(lane_rows) - vmap_block['lanes_failed']}/{len(lane_rows)}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    print(f"  Chart saved to:        {chart_path}")

    timer.summary()

    if vmap_block["lanes_failed"]:
        _failed = [_r for _r in lane_rows if _r["status"] == "FAIL"]
        for _r in _failed:
            print(
                f"  [   FAIL] lane {_r['lane']} ({_r['draw_name']}): "
                f"vmap {_r['log_likelihood_vmap']!r} vs scalar "
                f"{_r['log_likelihood_scalar']!r}  rel {_r['rel_diff']:.3e}"
            )
        raise AssertionError(
            f"{len(_failed)} of {len(lane_rows)} lanes disagree between the vmap and scalar "
            f"arms by more than {LANE_RTOL:.0e} relative. The two arms are not evaluating the "
            f"same models, so nothing in this table is a comparison. The JSON and PNG were "
            f"written first and hold the evidence."
        )

    sys.exit(0)

_rows = [
    (label, trace_block["per_stage_ms"][label]["median_ms"]) for label in trace_block["stage_order"]
]
_rows.append(("device_idle", trace_block["device_idle_ms"]))
_rows = [(label, ms) for label, ms in _rows if ms > 0.0]
_rows.sort(key=lambda kv: kv[1])

_colors = []
for label, _ in _rows:
    if label in (xla_attribution.OTHER, xla_attribution.UNJOINED):
        _colors.append("#C44E52")  # the rows that say the join is incomplete
    elif label == xla_attribution.MIXED_FUSION:
        _colors.append("#8172B3")
    elif label in ("device_idle", xla_attribution.MEMSET):
        _colors.append("#937860")
    elif label == "pdip_solve":
        _colors.append("#55A868")
    else:
        _colors.append("#4C72B0")

fig, ax = plt.subplots(figsize=(11, max(4.0, 0.34 * len(_rows))))
_y = np.arange(len(_rows), dtype=float)
_bars = ax.barh(_y, [ms for _, ms in _rows], color=_colors, edgecolor="white", height=0.66)
_max_ms = max((ms for _, ms in _rows), default=1.0)
for _bar, (_, _ms) in zip(_bars, _rows):
    ax.text(
        _bar.get_width() + _max_ms * 0.01,
        _bar.get_y() + _bar.get_height() / 2,
        f"{_ms:.3f} ms",
        va="center",
        fontsize=8,
    )

ax.set_yticks(_y)
ax.set_yticklabels([label for label, _ in _rows], fontsize=9)
ax.set_xlabel("device time per likelihood call (ms)")
ax.set_xlim(0, _max_ms * 1.18)
ax.set_title(
    f"Fixed-light {MESH} N={n_source_pixels} — XLA device timeline of the production call\n"
    f"budget {PASS_BUDGET}, border relocator {BORDER_RELOCATOR_RESOLVED}, "
    f"command buffers {'ON' if TRACE_ON_COMMAND_BUFFERS else 'OFF'} | "
    f"wall {trace_block['wall_ms']:.2f} ms, sum {trace_block['sum_ms']:.2f} ms "
    f"({trace_block['reconciliation_pct']:+.1f} %)",
    fontsize=10,
)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
print(f"  Chart saved to:        {chart_path}")

timer.summary()
