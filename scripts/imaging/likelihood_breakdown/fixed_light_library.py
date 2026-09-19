"""
JAX Profiling: Fixed Lens Light — the LIBRARY-PATH cost of each positivity solver
=================================================================================

Phase 1 of the ``fixed-lens-light-profiling`` epic (autolens_profiling#251),
the measurement phase 0 (#248) could not make.

Phase 0 measured **kernels**. With the lens light fixed after SLaM ``light[1]``
— the linear MGE-60 converted to regular light profiles at its solved
intensities and its PSF-convolved image subtracted from the dataset, leaving a
source-only inversion ("S3") — the reconstruction row falls from a 37 ms S0 PDIP
solve to 26-28 ms (S3 PDIP), to **4.2 ms** on Delaunay (certified active set,
pass 2) and **11.0 ms** on rectangular (pass 7), to 1.5 ms if positivity is
dropped altogether. What phase 0 explicitly did **not** measure is the row
production pays: the whole ``FitImaging`` likelihood with those solvers *inside*
it. Its "≈ 24 ms / call certified" figure is arithmetic on measured rows and
says so.

This cell measures it. One number per route, and the number is the whole
``AnalysisImaging.log_likelihood_function`` under ``jax.jit`` — the same call
phase 0's ``library_row`` timed, with the solver swapped underneath.

The routes
----------

======  ==========================================================  ================================
Route   What runs                                                   Solver
======  ==========================================================  ================================
``a``   S0 — the system the library solves today                    library PDIP (``solve_nnls``)
``b``   S3 — source-only, the reference every S3 route is scored     library PDIP
        against
``c``   S3, ``use_positive_only_solver=False``                       ``xp.linalg.solve``
``d``   S3, certified active set at the mesh's certifying budget,     harness injection + PDIP
        PDIP fallback when uncertified                               fallback
``d0``  route ``d`` with the fallback removed                        harness injection only
``e``   route ``d`` at pass budget 1 — the scheme never certifies    harness injection, fallback
        at budget 1, so the fallback always fires                    ALWAYS fires
======  ==========================================================  ================================

Route ``e`` is the honest worst case: what a production "N-pass budget with PDIP
fallback" costs on the evaluation where the budget is exhausted.

**Route ``c`` is never quoted as a bare millisecond.** Dropping positivity also
silently drops edge zeroing (``Inversion.solve_ids_to_keep`` returns ``None`` the
moment ``use_positive_only_solver`` is off, ``abstract.py:540``), and the
unconstrained solution sits at a *higher* evidence than the constrained optimum
because it is a different, infeasible minimiser. Every route-``c`` timing is
written beside its Δlog-evidence against route ``b``, its negative-pixel count
and its negative-flux fraction, and carries ``edge_zeroing_disabled: true``.

No PyAutoArray change
---------------------

The library has **no** active-set solver and this task does not add one. Routes
``d`` / ``d0`` / ``e`` are produced by a scoped harness monkeypatch of the
library's own positive-only entry point
(``likelihood_breakdown.library_solver_injection``); everything else in the
call — mapper, ``F + λH``, both log determinants, the evidence — is the
library's own code, unmodified. The JSON records the patched dotted name under
``solver_injection`` so no reader can mistake these rows for a library feature.

``lax.cond`` under ``vmap``
---------------------------

The fallback is a ``jax.lax.cond``: one branch under ``jit``, but a batched
predicate turns it into ``select`` under ``vmap``, which evaluates **both**. The
batched route-``d`` row is therefore *not* the certified cost — route ``d0``
(fallback removed) is. Both are measured and the JSON says which is which.

Flags (beyond the shared ``_profile_cli`` set)
----------------------------------------------

``--mesh {rectangular,delaunay,delaunay_nn}`` (required) — which cell's model to
rebuild. ``--pass-budget N`` — the fixed active-set pass budget routes ``d``/``d0``
run at; defaults to phase 0's certifying budget for the mesh (rectangular 7,
delaunay 2, delaunay_nn 2). ``--pass-budget-max N`` — the largest budget the
*certification diagnostics* scan (kernel-level, cheap, not a library row);
defaults to ``--pass-budget`` + 1. ``--source-pixels N`` sweeps the mesh size and
skips the pins. ``--no-fallback-row`` drops routes ``d`` and ``e`` (the
``lax.cond`` shapes) and keeps ``d0``. ``--vmap-batch N`` adds the batched rows.

Phase 5 (autolens_profiling#259) adds three more. ``--dataset {hst,euclid}``
(default ``hst``, byte-identical to phases 1-2) selects the ``INSTRUMENTS``
preset: the pixel scale (0.05" / 0.1") and the dataset directory, and **nothing
else** — the 3.5" mask, the ``[4, 2, 2]`` / ``[0.3, 0.6]`` light-profile
over-sampling, the 60 x 1 MGE, the mass and shear priors and the regularization
are held identical across datasets, which is what makes the HST and Euclid
columns of the phase-5 table comparable rather than two different experiments.
The cell also refuses a shared ``--instrument`` that disagrees with it, instead
of profiling HST while its log says Euclid.

``--pass-budget auto`` is the production setting. Route ``d`` runs at the budget
a production search would FIX — phase 3's zero-fallback budget for the mesh,
**11** rectangular / **7** Delaunay (``--safe-budget`` overrides) — while the
certification sweep scans budgets 1..12 and records the smallest budget that
actually certifies at *this* (dataset, mesh, N). Phase 4 showed the two do not
track each other (the certifying budget wanders in 5-10 / 1-2 with no trend in
N), so the certifying budget is a per-configuration fact worth recording —
especially on Euclid, where no sweep has ever run — and never the number to fix.

``--routes a,b,c,d`` selects which routes are built, compiled and timed. An
unnamed route is not run at all, which is what makes a four-route phase-5 leg
about half the wall clock of the six-route default. Unknown tokens are an error;
``--no-fallback-row`` still owns the two ``lax.cond`` rows (``d`` and ``e``).

``--pins {fp64,none}`` (default ``fp64``) and the shared
``--use-mixed-precision`` are the phase-2 pair. ``--use-mixed-precision`` passes
``use_mixed_precision=True`` into ``al.Settings``, which makes the library
accumulate ``A.T A`` in float32 before casting back to float64
(``inversion_util.py:127-136``). ``--pins none`` then **withdraws the
fp64-calibrated verdicts** — the S0 log-det pins, the mapper-block identity and
the equivalence pins — and records every one of them as data instead, because a
pin calibrated in fp64 is not a pin in fp32. Nothing is skipped: each
comparison is still computed and written, with status ``RECORDED``. The active
set's KKT tolerance ``tau_rel`` is likewise **re-derived** under mixed
precision (``eps_float32 * sqrt(image pixels)``, the float32 accumulation floor
of the matrix being certified) and the value used is reported in the JSON.

Phase 2 (autolens_profiling#253) also records a ``machine`` block beside the
existing ``device`` block — CPU model, core count, RAM, kernel, GPU, and **both**
thread-knob families, because they are disjoint: ``NPROC`` sizes XLA's CPU
intra-op pool (the JAX rows) while ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS``
/ ``MKL_NUM_THREADS`` pin BLAS (the numpy rows) and do not touch JAX. A CPU
millisecond is not interpretable without both.

Output
------

``results/breakdown/imaging/fixed_light_library_<mesh>[_<dataset>][_n<N>]_<config_name>.{json,png}``
— the dataset and the built pixel count are appended only when they differ from
phase 1's HST fiducial, so no phase-1 or phase-2 artifact can be clobbered. A leg
for which no pin was ever calibrated (Euclid at any N, HST away from the fiducial
mesh) asserts nothing and writes a ``reference_recorded`` block instead: the log
determinants, the evidences and the certifying budget the pins WOULD have
compared, as data, so a later phase can pin against this leg.
The A100 legs run under ``--config-name hpc_a100_fp64_fixed_light_library``,
deliberately **not** a config name ``build_readme.py`` surfaces: these rows are
the phase-1 comparator study, not a dashboard tier. Phase 2's local legs use
``local_cpu_fp64_fixed_light_library_t1`` / ``_tall``,
``local_rtx2060_fp64_fixed_light_library`` and
``local_rtx2060_mp_fixed_light_library`` on the same convention.
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
    library_solver_injection,
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
    machine_info_dict,
    parse_profile_cli,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)

_cli = parse_profile_cli()

#: The smallest pass budget at which the certified active set certifies on each
#: mesh, measured on euclid-ral-gpu-2 by the phase-0 legs 342802-342804
#: (results/notes/fixed_lens_light_source_only_2026_09.md). Rectangular needs
#: seven because its *dual* violation set drains one or two indices at a time
#: behind the 152 edge-zeroed pixels; the Delaunay family has no edge zeroing
#: and settles after one correction.
CERTIFYING_BUDGET = {"rectangular": 7, "delaunay": 2, "delaunay_nn": 2}

# Cell-local flags. ``add_help=False`` because ``parse_profile_cli`` already
# owns ``-h`` (it runs first); the flags are documented in the module docstring.
_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument(
    "--mesh", choices=("rectangular", "delaunay", "delaunay_nn"), required=True
)
_cell_parser.add_argument("--pass-budget", default=None)
_cell_parser.add_argument("--safe-budget", type=int, default=None)
_cell_parser.add_argument("--pass-budget-max", type=int, default=None)
_cell_parser.add_argument("--source-pixels", type=int, default=None)
_cell_parser.add_argument("--fallback-row", dest="fallback_row", action="store_true", default=True)
_cell_parser.add_argument("--no-fallback-row", dest="fallback_row", action="store_false")
_cell_parser.add_argument("--pins", choices=("fp64", "none"), default="fp64")
_cell_parser.add_argument("--dataset", choices=("hst", "euclid"), default="hst")
_cell_parser.add_argument("--routes", default=None)
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh

#: Phase 3's smallest **zero-fallback** pass budget per mesh
#: (autolens_profiling#255, ``fixed_lens_light_low_likelihood_draws_2026_09.md``),
#: confirmed N-independent from 484 to 3969 source pixels by phase 4 (#257,
#: ``fixed_lens_light_source_pixel_scaling_2026_09.md``). This is the budget a
#: PRODUCTION run fixes. ``CERTIFYING_BUDGET`` above is phase 0's fiducial
#: budget, which falls back on 27.5 % (rectangular) / 67.5 % (Delaunay) of a
#: graded draw set and is therefore NOT a production cost.
PHASE3_SAFE_BUDGET = {"rectangular": 11, "delaunay": 7, "delaunay_nn": 7}

#: Every route this cell can run, in the order the summary prints them.
ALL_ROUTE_KEYS = ("a", "b", "c", "d", "d0", "e")

#: Which dataset the leg fits. Phase 5 (autolens_profiling#259) is the first
#: phase of the epic to leave HST. The switch changes the INSTRUMENTS preset
#: (pixel scale and dataset directory) and NOTHING else: the mask radius, the
#: over-sampling rule, the MGE, the mass/shear priors and the regularization are
#: held byte-identical across datasets, which is what makes the two columns of
#: the phase-5 table comparable. The HST path is unchanged, byte for byte.
DATASET = _cell_args.dataset

# The shared CLI owns an ``--instrument`` this cell never read, so a leg that
# passed ``--instrument euclid`` used to profile HST and say Euclid in its log.
# One lever, and it says so rather than guarding silently.
if _cli.instrument is not None and _cli.instrument != DATASET:
    raise ValueError(
        f"--instrument {_cli.instrument!r} does not select the dataset in this cell; "
        f"use --dataset {{hst,euclid}} (currently {DATASET!r})."
    )

SAFE_BUDGET = int(
    _cell_args.safe_budget if _cell_args.safe_budget is not None else PHASE3_SAFE_BUDGET[MESH]
)
if SAFE_BUDGET < 1:
    raise ValueError(f"--safe-budget must be >= 1 (got {SAFE_BUDGET})")

#: ``fp64`` (default) asserts every pin phase 0 and phase 1 calibrated in fp64.
#: ``none`` is the **mixed-precision** setting: a pin calibrated in fp64 is not
#: a pin in fp32, so the fp64-calibrated *verdicts* are withdrawn — but nothing
#: is skipped. Every comparison is still computed and written to the JSON with
#: status ``RECORDED``, so the leg reports what the lower precision costs
#: instead of asserting a tolerance it was never calibrated against.
PINS_MODE = _cell_args.pins
PINS_ASSERT = PINS_MODE == "fp64"
USE_MIXED_PRECISION = bool(_cli.use_mixed_precision)

if USE_MIXED_PRECISION and PINS_ASSERT:
    print(
        "  NOTE: --use-mixed-precision with --pins fp64. The fp64-calibrated pins "
        "will be ASSERTED against a float32-accumulated curvature matrix; pass "
        "--pins none for the honest mixed-precision leg."
    )


# ``--pass-budget auto`` is the phase-5 production setting: route d runs at the
# budget a production search would FIX (phase 3's zero-fallback budget for the
# mesh, 11 rectangular / 7 Delaunay), and the smallest budget that actually
# certifies at THIS (dataset, mesh, N) is measured by the certification sweep
# below and recorded beside it. The two are different questions and phase 4
# showed they do not track each other: the certifying budget wanders in 5-10
# (rectangular) / 1-2 (Delaunay) with no trend in N, so it is a per-configuration
# fact worth recording — especially on Euclid, where no sweep has ever run — and
# never the number a production run should fix.
def _resolve_pass_budget(raw, mesh, safe_budget):
    """``--pass-budget`` -> (budget routes d/d0 run at, a human-readable mode).

    ``None`` keeps this cell's phase-0 default for the mesh. ``"auto"`` is the
    phase-5 production setting: the budget a production search would FIX, which
    is phase 3's zero-fallback budget, NOT whatever certifies here. Anything
    else is an explicit integer.
    """
    if raw is None:
        return int(
            CERTIFYING_BUDGET[mesh]
        ), "phase0-certifying-budget (this cell's default, unchanged)"
    if str(raw).strip().lower() == "auto":
        return int(safe_budget), (
            f"auto -> phase 3's production safe budget for {mesh} ({int(safe_budget)}); the "
            f"smallest certifying budget at this configuration is measured and recorded, "
            f"not used"
        )
    try:
        return int(raw), "explicit"
    except (TypeError, ValueError):
        raise ValueError(f"--pass-budget takes an integer or 'auto' (got {raw!r})") from None


def _smallest_certifying_budget(entries):
    """The smallest budget in a certification sweep that certified, else None."""
    return next((int(e["pass_budget"]) for e in entries if e.get("certified")), None)


PASS_BUDGET, PASS_BUDGET_MODE = _resolve_pass_budget(_cell_args.pass_budget, MESH, SAFE_BUDGET)

PASS_BUDGET = int(PASS_BUDGET)
#: Under ``auto`` the diagnostics sweep must reach far enough to FIND the
#: certifying budget rather than stopping at the production one, so it scans to
#: 12 — the range phase 4 swept, which covers both safe budgets at every N.
_DEFAULT_PASS_BUDGET_MAX = (
    max(12, PASS_BUDGET) if PASS_BUDGET_MODE.startswith("auto") else (PASS_BUDGET + 1)
)
PASS_BUDGET_MAX = int(
    _cell_args.pass_budget_max
    if _cell_args.pass_budget_max is not None
    else _DEFAULT_PASS_BUDGET_MAX
)
SOURCE_PIXELS_REQUESTED = _cell_args.source_pixels
RUN_FALLBACK_ROWS = bool(_cell_args.fallback_row)


def _parse_routes(raw, fallback_rows):
    """``--routes a,b,c,d`` -> the selected route keys, in canonical order.

    ``None`` keeps this cell's historic selection (every route, minus the
    ``lax.cond`` rows when ``--no-fallback-row`` is given). An explicit list is
    validated against :data:`ALL_ROUTE_KEYS` — an unknown key is an error, never
    a silently dropped row — and is still subject to ``--no-fallback-row``,
    which owns the ``cond`` shapes.
    """
    if raw is None:
        selected = set(ALL_ROUTE_KEYS)
    else:
        wanted = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
        if not wanted:
            raise ValueError("--routes was given but names no routes")
        unknown = [token for token in wanted if token not in ALL_ROUTE_KEYS]
        if unknown:
            raise ValueError(
                f"--routes: unknown route(s) {unknown}; choose from {list(ALL_ROUTE_KEYS)}"
            )
        selected = set(wanted)
    if not fallback_rows:
        selected -= {"d", "e"}
    return tuple(key for key in ALL_ROUTE_KEYS if key in selected)


ROUTE_SELECTION = _parse_routes(_cell_args.routes, RUN_FALLBACK_ROWS)
if not ROUTE_SELECTION:
    raise ValueError("no routes left to run (--routes and --no-fallback-row cancel out)")

if PASS_BUDGET < 1:
    raise ValueError(f"--pass-budget must be >= 1 (got {PASS_BUDGET})")
if PASS_BUDGET_MAX < PASS_BUDGET:
    raise ValueError(
        f"--pass-budget-max ({PASS_BUDGET_MAX}) must be >= --pass-budget ({PASS_BUDGET})"
    )

#: The budget at which the fallback row runs. 1 is not a tuning choice: the
#: scheme has never certified at pass 1 on any mesh at any size phase 0 measured
#: (rectangular primal 26 / dual 20; Delaunay primal 1), so budget 1 is the
#: cheapest configuration in which the fallback is *guaranteed* to fire, which
#: is what makes it a worst-case cost rather than a lucky one. The certification
#: diagnostics below record whether it certified, so the claim is checked, not
#: assumed.
FALLBACK_ROW_BUDGET = 1

_vmap_batch = timing.parse_vmap_batch(sys.argv)
if _vmap_batch is not None and _vmap_batch < 1:
    raise ValueError(f"--vmap-batch must be >= 1 (got {_vmap_batch})")

# The dataset the leg fits, from ``--dataset`` (default hst, unchanged). The
# INSTRUMENTS preset supplies the pixel scale (0.05" hst / 0.1" euclid) and the
# dataset directory; everything else below is held identical across datasets.
instrument = DATASET

#: Source-pixel count each mesh's cell builds at its fiducial, and the mesh the
#: pinned log-dets below describe.
FIDUCIAL_SOURCE_PIXELS = {"rectangular": 39 * 39, "delaunay": 1500, "delaunay_nn": 1500}

#: The exact dense-Cholesky log determinants measured on euclid-ral-gpu-2 on
#: 2026-09-11 by the reconstruction-split legs (autolens_profiling#243, PR #244)
#: and carried forward by the phase-0 cell. They describe **S0**, and they are
#: this cell's tripwire that its model construction is still the sibling cells'.
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

#: Relative tolerance on the equivalence pins between library routes. The
#: certified scheme is a different factorisation order to PDIP, not a different
#: answer, so the log-likelihoods agree to solver round-off; 1e-8 relative is
#: two orders above the 1e-10 nats phase 0 measured on the kernels and well
#: below anything a solver change would move.
EQUIVALENCE_RTOL = 1.0e-8

#: Relative tolerance on the mapper-block log-det identity (phase 0's value).
_MAPPER_LOGDET_RTOL = 1e-6


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
# Deliberately identical to ``fixed_light.py`` PART A, which is itself the three
# pixelized breakdown cells' construction. The pinned S0 log-dets are the
# tripwire on that identity; if a sibling cell's model changes, this cell must
# change with it.

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

# The S0 log-det pins were calibrated on HST at the fiducial mesh and describe
# nothing else. Phase 5 runs Euclid, where NO pin has ever been calibrated, and
# runs HST at N the pins do not describe — so both record instead of asserting,
# and the recorded values are this family's first Euclid reference.
PIN_IS_FIDUCIAL = (n_source_pixels == FIDUCIAL_SOURCE_PIXELS[MESH]) and DATASET == "hst"

print(f"  Source pixels: {n_source_pixels} (fiducial: {FIDUCIAL_SOURCE_PIXELS[MESH]})")
if not PIN_IS_FIDUCIAL:
    print(
        f"  Pins not applicable: dataset={DATASET} (calibrated on hst), "
        f"N={n_source_pixels} (calibrated at {FIDUCIAL_SOURCE_PIXELS[MESH]}). "
        f"The S0 log-dets and evidences are RECORDED below, not asserted."
    )

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

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass)
    field = af.Model(al.MassField, redshift=0.5, shear=shear)

    if MESH == "rectangular":
        reg_scheme = "constant"
        reg_coefficient_model = 1.0
        mesh_obj = rect_mesh_classes(_cli)[1](shape=mesh_shape, weight_power=1.0, weight_floor=0.0)
        regularization = al.reg.Constant(coefficient=reg_coefficient_model)
        reg_provenance = {"scheme": reg_scheme, "coefficient": reg_coefficient_model}
    else:
        if MESH == "delaunay":
            mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
        else:
            mesh_obj = al.mesh.DelaunayNN(pixels=n_mesh_vertices, areas_factor=0.5, zeroed_pixels=0)
        reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

    pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization: {reg_scheme} ({reg_provenance})")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

tracer = al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields])

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
print(f"  Source pixels:           {n_source_pixels}")
print(f"  Dataset:                 {DATASET}")
print(f"  Routes:                  {','.join(ROUTE_SELECTION)}")
print(f"  Pass budget (d / d0):    {PASS_BUDGET}  [{PASS_BUDGET_MODE}]")
print(f"  Safe budget (phase 3):   {SAFE_BUDGET}")
print(f"  Fallback-row budget (e): {FALLBACK_ROW_BUDGET}")
print(f"  Mixed precision:         {USE_MIXED_PRECISION}")
print(f"  Pins mode:               {PINS_MODE}")

# ---------------------------------------------------------------------------
# tau_rel — RE-DERIVED per precision, never inherited
# ---------------------------------------------------------------------------
# The active set's KKT tolerances are tau_g = tau_rel * max|q| (dual) and
# tau_x = tau_rel * max|x| (primal). ``TAU_REL_DEFAULT = 1e-9`` was calibrated
# in fp64, where the curvature matrix itself is accurate to ~1e-16 relative, so
# 1e-9 sits far above the data's own noise floor and a certificate means what it
# says.
#
# Under ``use_mixed_precision`` that is no longer true. The setting makes the
# library accumulate ``A.T A`` in float32 before casting the result back to
# float64 (``inversion_util.py:127-136``): the SOLVE is still fp64 arithmetic,
# but on a matrix whose entries carry float32 accumulation error. A dot product
# over M image rows accumulates a relative error of order ``eps_f32 * sqrt(M)``,
# so the gradient g = Qx - q inherits a floor of about that fraction of max|q|.
# Certifying at 1e-9 against a matrix good to ~1e-5 would be certifying against
# round-off. The tolerance is therefore re-derived from the precision actually
# used, and the value is reported in the JSON rather than left implicit.
_EPS_F32 = float(np.finfo(np.float32).eps)

if USE_MIXED_PRECISION:
    TAU_REL = _EPS_F32 * math.sqrt(float(n_image_pixels))
    TAU_REL_BASIS = (
        f"re-derived for mixed precision: eps_float32 ({_EPS_F32:.3e}) * "
        f"sqrt(image pixels = {int(n_image_pixels)}) = {TAU_REL:.3e}. The library "
        f"accumulates the curvature matrix in float32 under use_mixed_precision "
        f"(inversion_util.py:127-136), so the KKT residual cannot be resolved "
        f"below the float32 accumulation floor of the matrix itself; a pass "
        f"budget certified at the fp64 tolerance would be certifying round-off."
    )
else:
    TAU_REL = active_set_steps.TAU_REL_DEFAULT
    TAU_REL_BASIS = (
        f"fp64 default ({active_set_steps.TAU_REL_DEFAULT:g}), as phases 0 and 1 "
        f"calibrated it on the A100."
    )

print(f"  tau_rel:                 {TAU_REL:.6e}")
print(f"    basis: {TAU_REL_BASIS}")

# ---------------------------------------------------------------------------
# S0 and S3 — the two systems, built eagerly by the phase-0 builders
# ---------------------------------------------------------------------------

_settings = al.Settings(
    use_border_relocator=True,
    use_mixed_precision=_cli.use_mixed_precision,
)

#: Positivity off. ``Inversion.solve_ids_to_keep`` returns ``None`` the moment
#: this is false (abstract.py:540), so edge zeroing goes with it — which is why
#: no route-(c) number is ever quoted without its Δevidence and its negatives.
_settings_positive_negative = al.Settings(
    use_border_relocator=True,
    use_mixed_precision=_cli.use_mixed_precision,
    use_positive_only_solver=False,
)

print("\n--- S0: full FitImaging with linear MGE lens light (eager) ---")

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

print("\n--- S3: lens light converted to regular profiles + subtracted (eager) ---")

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

# ---------------------------------------------------------------------------
# The mapper-block identity — a hard assertion, carried forward from phase 0
# ---------------------------------------------------------------------------
# Fixing the lens light removes 60 unregularised columns from the linear system
# and must leave the mapper block of F + λH untouched. If it does not, S3 is not
# S0's source inversion and every Δ in this JSON compares two different problems.

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
    _status = ("PASS" if _rel <= _MAPPER_LOGDET_RTOL else "FAIL") if PINS_ASSERT else "RECORDED"
    _mapper_log_dets[_key] = {
        "s0": _s0_value,
        "s3": _s3_value,
        "rel_diff": _rel,
        "status": _status,
    }
    print(
        f"  [{_status:>8}] mapper-block {_key}: S0 {_s0_value:.9f}  "
        f"S3 {_s3_value:.9f}  rel {_rel:.3e}"
    )
    if PINS_ASSERT and _rel > _MAPPER_LOGDET_RTOL:
        raise AssertionError(
            f"S3's mapper-block {_key} ({_s3_value!r}) does not match S0's ({_s0_value!r}) "
            f"to {_MAPPER_LOGDET_RTOL:g} (relative difference {_rel:.3e})."
        )

# ===================================================================
# PART B — Certification diagnostics (kernel level, cheap)
# ===================================================================
# The library-path routes below run at a FIXED pass budget and throw the
# certificate away inside the jit. This block is where the certificate is
# actually read: it runs the same scheme, on the same problem the injected
# solver receives, at every budget from 1 to --pass-budget-max, and records the
# per-pass primal/dual violation counts, the certifying pass and the
# Δlog-evidence. It is what turns "route d certified" from an assumption into a
# recorded fact, and what checks that route e's fallback really does fire.
#
# THE PROBLEM THE SOLVER RECEIVES IS NOT THE FULL SYSTEM. Under edge zeroing the
# library subsets F + λH and D to solve_ids_to_keep *before* calling the solver
# and scatters exact zeros back afterwards (abstract.py:607-618). So the fixed0
# mask here is zeros(n_keep): the border pixels are not variables of this QP at
# all. That is the same problem phase 0 certified at full size with those
# indices permanently fixed and never released, index for index.

print("\n" + "=" * 70)
print("CERTIFICATION DIAGNOSTICS — the QP the injected solver receives")
print("=" * 70)

_inv_s3 = system_s3.inversion
_ids_keep = _inv_s3.solve_ids_to_keep
_crm_full = np.asarray(_inv_s3.curvature_reg_matrix, dtype=float)
_dv_full = np.asarray(_inv_s3.data_vector, dtype=float)

if _ids_keep is not None:
    _ids_keep = np.asarray(_ids_keep, dtype=int)
    _crm_solver = _crm_full[_ids_keep][:, _ids_keep]
    _dv_solver = _dv_full[_ids_keep]
else:
    _crm_solver = _crm_full
    _dv_solver = _dv_full

_Q_solver, _q_solver, _dscale_solver = reconstruction_steps.jacobi_scaled(
    jnp.asarray(_crm_solver, dtype=jnp.float64), jnp.asarray(_dv_solver, dtype=jnp.float64)
)
_n_solver = int(_q_solver.shape[0])
_fixed0_solver = jnp.zeros(_n_solver, dtype=bool)

print(
    f"  full system n={system_s3.n_params}, edge-zeroed {int(system_s3.edge_zero_mask.sum())}, "
    f"solver sees n={_n_solver}"
)


def _scatter_back(x_sub) -> np.ndarray:
    """The library's scatter-back: a subset solution into the full parameter vector."""
    x_sub = np.asarray(x_sub, dtype=float)
    if _ids_keep is None:
        return x_sub
    full = np.zeros(system_s3.n_params, dtype=float)
    full[_ids_keep] = x_sub
    return full


_x_library_s3 = np.asarray(system_s3.inversion.reconstruction, dtype=float)
_reference_log_evidence = float(system_s3.log_evidence(_x_library_s3))

certification_budgets: list[dict] = []


def _active_set_at(Q, q, fixed0, budget):
    """``active_set_masked_jax`` at this leg's re-derived ``tau_rel``."""
    return active_set_steps.active_set_masked_jax(Q, q, fixed0, budget, tau_rel=TAU_REL)


for _budget in range(1, PASS_BUDGET_MAX + 1):
    _out = jax.jit(_active_set_at, static_argnums=(3,))(
        _Q_solver, _q_solver, _fixed0_solver, _budget
    )
    _certified = [bool(c) for c in np.asarray(_out["certified"])]
    _certified_at = next((i + 1 for i, c in enumerate(_certified) if c), None)
    _x_full = _scatter_back(np.asarray(_out["x"], dtype=float) * np.asarray(_dscale_solver))
    _abs_lib, _rel_lib = max_abs_rel(_x_full, _x_library_s3)

    _entry = {
        "pass_budget": _budget,
        "n_fixed_per_pass": [int(v) for v in np.asarray(_out["n_fixed"])],
        "n_primal_violations_per_pass": [int(v) for v in np.asarray(_out["n_primal_violations"])],
        "n_dual_violations_per_pass": [int(v) for v in np.asarray(_out["n_dual_violations"])],
        "certified": bool(any(_certified)),
        "certified_at_pass": _certified_at,
        "d_log_evidence_vs_library": float(system_s3.log_evidence(_x_full))
        - _reference_log_evidence,
        "max_abs_diff_vs_library": _abs_lib,
        "max_rel_diff_vs_library": _rel_lib,
        "n_negative_entries": int(np.sum(_x_full < 0.0)),
    }
    certification_budgets.append(_entry)

    print(
        f"  budget {_budget}: certified={_entry['certified']} (at pass {_certified_at})  "
        f"Δlog-ev vs library {_entry['d_log_evidence_vs_library']:+.6e}  "
        f"primal {_entry['n_primal_violations_per_pass']}  "
        f"dual {_entry['n_dual_violations_per_pass']}"
    )

_certifying_budget_measured = _smallest_certifying_budget(certification_budgets)

_route_d_certifies = any(
    e["certified"] for e in certification_budgets if e["pass_budget"] == PASS_BUDGET
)
_fallback_row_certifies = any(
    e["certified"] for e in certification_budgets if e["pass_budget"] == FALLBACK_ROW_BUDGET
)

print(f"  smallest certifying budget (measured here): {_certifying_budget_measured}")
print(f"  route d/d0 budget {PASS_BUDGET} certifies:  {_route_d_certifies}")
print(
    f"  route e budget {FALLBACK_ROW_BUDGET} certifies: {_fallback_row_certifies} "
    f"(the fallback fires when this is False, which is the point of the row)"
)

certification_block = {
    "problem": (
        "the QP the library hands its positive-only solver: F + λH and D already "
        "subset to solve_ids_to_keep under edge zeroing, Jacobi-scaled exactly as "
        "inversion_util.reconstruction_positive_only_from does"
    ),
    "n_full_system": int(system_s3.n_params),
    "n_edge_zeroed": int(system_s3.edge_zero_mask.sum()),
    "n_seen_by_solver": _n_solver,
    "fixed0_is_zeros": True,
    "fixed0_note": (
        "zeros(n_keep), not the edge-zero mask: the library subsets before it calls, "
        "so the border pixels are not variables of this QP. Equivalent index-for-index "
        "to phase 0's full-size scheme with those indices permanently fixed."
    ),
    "tau_rel": TAU_REL,
    "tau_rel_fp64_default": active_set_steps.TAU_REL_DEFAULT,
    "tau_rel_basis": TAU_REL_BASIS,
    "tau_rel_re_derived_for_precision": bool(USE_MIXED_PRECISION),
    "reference_log_evidence": _reference_log_evidence,
    "reference": "the library's own reconstruction of S3 (PDIP on the edge-zeroed problem)",
    "pass_budget_max": PASS_BUDGET_MAX,
    "smallest_certifying_budget": _certifying_budget_measured,
    "phase0_certifying_budget": CERTIFYING_BUDGET[MESH],
    "route_d_pass_budget": PASS_BUDGET,
    "route_d_certifies": bool(_route_d_certifies),
    "route_e_pass_budget": FALLBACK_ROW_BUDGET,
    "route_e_certifies": bool(_fallback_row_certifies),
    "route_e_fallback_fires": bool(not _fallback_row_certifies),
    "budgets": certification_budgets,
}

# ===================================================================
# PART C — Route (c) diagnostics: what dropping positivity buys and costs
# ===================================================================
# Never a bare millisecond. The eager positive-negative fit is built here so the
# negatives and the Δevidence exist alongside the timing, and so the route the
# library actually takes is *verified* rather than asserted in prose.

print("\n" + "=" * 70)
print("ROUTE (c) — positive-negative (use_positive_only_solver=False), eager")
print("=" * 70)

with library_solver_injection.positive_negative_probe() as _pn_counts:
    fit_positive_negative = al.FitImaging(
        dataset=system_s3.dataset,
        tracer=system_s3.source_only_tracer,
        adapt_images=adapt_images,
        settings=_settings_positive_negative,
        xp=np,
    )
    _x_positive_negative = np.asarray(fit_positive_negative.inversion.reconstruction, dtype=float)
    _log_evidence_positive_negative = float(fit_positive_negative.figure_of_merit)

_pn_calls = int(_pn_counts["calls"])
_pn_ids_to_keep = fit_positive_negative.inversion.solve_ids_to_keep

# The phase-0 A2 row: one unconstrained Cholesky of the FULL S3 system. Route
# (c) is the library taking that same route through xp.linalg.solve, so the two
# must agree — that agreement is the pin tying this cell to phase 0's kernels.
_x_a2 = np.asarray(
    reconstruction_steps.cholesky_solve(
        jnp.asarray(system_s3.curvature_reg_matrix, dtype=jnp.float64),
        jnp.asarray(system_s3.data_vector, dtype=jnp.float64),
    ),
    dtype=float,
)
_a2_abs, _a2_rel = max_abs_rel(_x_positive_negative, _x_a2)

_pn_negative = _x_positive_negative < 0.0
positive_negative_block = {
    "route_verified": (
        "reconstruction_positive_negative_from / xp.linalg.solve"
        if _pn_calls > 0
        else "NOT VERIFIED — the positive-negative entry point was never called"
    ),
    "positive_negative_solver_calls": _pn_calls,
    "solve_ids_to_keep_is_none": _pn_ids_to_keep is None,
    "edge_zeroing_disabled": True,
    "edge_zeroing_note": (
        "Inversion.solve_ids_to_keep returns None the moment use_positive_only_solver "
        "is False (abstract.py:540), so turning positivity off silently turns edge "
        "zeroing off with it. This route solves the FULL system; routes a/b/d/e solve "
        "the edge-zeroed one."
    ),
    "log_evidence_eager": _log_evidence_positive_negative,
    "d_log_evidence_vs_library_s3": _log_evidence_positive_negative - _reference_log_evidence,
    "n_negative_entries": int(np.sum(_pn_negative)),
    "negative_flux_fraction": float(
        np.sum(np.abs(_x_positive_negative[_pn_negative]))
        / max(float(np.sum(np.abs(_x_positive_negative))), 1e-300)
    ),
    "max_abs_diff_vs_phase0_a2_cholesky": _a2_abs,
    "max_rel_diff_vs_phase0_a2_cholesky": _a2_rel,
    "phase0_a2_note": (
        "phase 0's A2 row is one unconstrained Cholesky of the full S3 system. Route "
        "(c) is the library reaching the same solution through xp.linalg.solve, so the "
        "two vectors must agree to solver round-off — that agreement is what makes this "
        "row comparable with the phase-0 note's A2 evidence numbers."
    ),
    "warning": (
        "A positive Δlog-evidence here is NOT an improvement: the unconstrained "
        "solution is a different, infeasible minimiser scoring above the constrained "
        "optimum. Never quote this route's milliseconds without these numbers."
    ),
}

if _pn_calls == 0:
    raise AssertionError(
        "use_positive_only_solver=False did not route through "
        "reconstruction_positive_negative_from — route (c) is not what it claims to be."
    )

print(f"  route verified: {positive_negative_block['route_verified']} ({_pn_calls} call(s))")
print(f"  solve_ids_to_keep is None: {_pn_ids_to_keep is None} (edge zeroing off)")
print(
    f"  Δlog-evidence vs library S3: "
    f"{positive_negative_block['d_log_evidence_vs_library_s3']:+.6f} nats"
)
print(
    f"  negatives: {positive_negative_block['n_negative_entries']} entries, "
    f"{positive_negative_block['negative_flux_fraction'] * 100:.4f} % of |flux|"
)
print(f"  vs phase-0 A2 Cholesky: max rel diff {_a2_rel:.3e}")

# ===================================================================
# PART D — The library rows, one per route
# ===================================================================

print("\n" + "=" * 70)
print("LIBRARY PATH — one full likelihood call per route")
print("=" * 70)

params_tree_s0 = jax.tree_util.tree_map(jnp.asarray, instance)

# The S3 instance is the S0 instance with the lens's light removed — the same
# galaxies ``fixed_light_system_from`` put in its tracer. The adapt images are
# keyed by name as well as by object and the name keys are what the analysis
# path reads, so the deep copy is safe.
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


#: ``(key, label, dataset, params tree, settings, injection)`` per route.
#: ``injection`` is ``None`` (library solver as shipped) or
#: ``(pass_budget, fallback)`` for the harness-injected certified scheme.
#: Every route this cell knows, keyed by the ``--routes`` token, in the order
#: the summary prints them. ``ROUTE_SELECTION`` filters this map — a route that
#: is not selected is not built, not compiled and not timed, which is what makes
#: a four-route phase-5 leg roughly half the wall clock of a six-route one.
_ROUTE_SPECS: dict[str, tuple] = {
    "a": ("a_s0_pdip", "S0 PDIP (library today)", dataset, params_tree_s0, _settings, None),
    "b": (
        "b_s3_pdip",
        "S3 PDIP (source-only)",
        system_s3.dataset,
        params_tree_s3,
        _settings,
        None,
    ),
    "c": (
        "c_s3_positive_negative",
        "S3 positive-negative (xp.linalg.solve)",
        system_s3.dataset,
        params_tree_s3,
        _settings_positive_negative,
        None,
    ),
    "d": (
        "d_s3_certified_fallback",
        f"S3 certified active set, budget {PASS_BUDGET}, PDIP fallback",
        system_s3.dataset,
        params_tree_s3,
        _settings,
        (PASS_BUDGET, True),
    ),
    "d0": (
        "d0_s3_certified_no_fallback",
        f"S3 certified active set, budget {PASS_BUDGET}, no fallback",
        system_s3.dataset,
        params_tree_s3,
        _settings,
        (PASS_BUDGET, False),
    ),
    "e": (
        "e_s3_certified_budget1_fallback_fires",
        f"S3 certified active set, budget {FALLBACK_ROW_BUDGET}, fallback FIRES",
        system_s3.dataset,
        params_tree_s3,
        _settings,
        (FALLBACK_ROW_BUDGET, True),
    ),
}

ROUTES: list[tuple] = [_ROUTE_SPECS[_token] for _token in ROUTE_SELECTION]

print(f"  routes selected: {', '.join(spec[0] for spec in ROUTES)}")

routes: dict[str, dict] = {}

for _key, _label, _ds, _tree, _settings_route, _injection in ROUTES:
    print(f"\n--- {_key}: {_label} ---")
    _fn = _likelihood_fn(_ds, _settings_route)
    _entry: dict = {"label": _label, "status": "ok"}

    import traceback as _route_traceback

    try:
        if _injection is None:
            _, _value = jit_profile(_fn, f"{_key}_library_likelihood_jit", _tree)
            _entry["solver"] = "library"
        else:
            _budget, _fallback = _injection
            with library_solver_injection.certified_solver_injected(
                _budget, fallback=_fallback, tau_rel=TAU_REL
            ) as _counts:
                _, _value = jit_profile(_fn, f"{_key}_library_likelihood_jit", _tree)
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
    except Exception:  # noqa: BLE001 — one failed route must not lose the others
        _entry = {
            "label": _label,
            "status": "failed",
            "error": _route_traceback.format_exc(),
        }
        print(f"  ROUTE FAILED — every other route is unaffected.\n{_entry['error']}")

    routes[_key] = _entry

peak_after_routes = peak_bytes()

# ---------------------------------------------------------------------------
# Batched rows (--vmap-batch N)
# ---------------------------------------------------------------------------
# Identical lanes: every lane is a copy of the same parameter point, so no lane
# waits for a straggler. Best case for amortization, not a representative one.
#
# ``lax.cond`` becomes ``select`` under vmap and evaluates BOTH branches, so the
# batched d and e rows carry the certified solve AND the PDIP solve. Route d0
# (no fallback) is the batched certified cost; d and e are the batched cost of
# the fallback *shape*. The JSON note says so, and so does this comment.

vmap_block: dict = {}
vmap_error = None

if _vmap_batch is not None:
    print(f"\n--- Batched library rows (--vmap-batch {_vmap_batch}) ---")
    import traceback as _vmap_traceback

    try:
        for _key, _label, _ds, _tree, _settings_route, _injection in ROUTES:
            if routes.get(_key, {}).get("status") != "ok":
                continue
            _batched = jax.tree_util.tree_map(
                lambda leaf: jnp.broadcast_to(leaf, (_vmap_batch, *leaf.shape)), _tree
            )
            _fn = _likelihood_fn(_ds, _settings_route)
            print(f"  {_key}")
            if _injection is None:
                _per_call = timing.vmap_profile(
                    _fn, _key, _batched, _vmap_batch, timer=timer, jit_records=jit_records
                )
            else:
                _budget, _fallback = _injection
                with library_solver_injection.certified_solver_injected(
                    _budget, fallback=_fallback, tau_rel=TAU_REL
                ):
                    _per_call = timing.vmap_profile(
                        _fn, _key, _batched, _vmap_batch, timer=timer, jit_records=jit_records
                    )
            vmap_block[f"{_key}_per_call_ms"] = _per_call * 1e3
            routes[_key]["vmap_ms"] = _per_call * 1e3

        vmap_block["batch"] = int(_vmap_batch)
        vmap_block["note"] = (
            f"@vmap {_vmap_batch} identical lanes. lax.cond becomes select under vmap and "
            f"evaluates BOTH branches, so the batched d and e rows are the certified solve "
            f"PLUS the library PDIP — the batched certified cost is route d0, which has no "
            f"cond. Read the batched fallback rows as the cost of the fallback shape, never "
            f"as the certified cost."
        )
    except Exception:  # noqa: BLE001 — a vmap failure must not lose the unbatched rows
        vmap_error = _vmap_traceback.format_exc()
        print("  VMAP FAILED — unbatched rows are unaffected. Traceback:")
        print(vmap_error)

peak_after_all = peak_bytes()

# ===================================================================
# PART E — Equivalence pins
# ===================================================================
# The timings are only worth reading if every route computed the same
# likelihood. These are the checks that say so, recorded as data (PASS/FAIL with
# the measured relative difference) rather than raised, so a drifted run still
# writes its JSON and the drift is visible in it.

print("\n" + "=" * 70)
print("EQUIVALENCE PINS")
print("=" * 70)


def _route_value(key):
    entry = routes.get(key, {})
    return entry.get("log_likelihood") if entry.get("status") == "ok" else None


_reference_value = _route_value("b_s3_pdip")

equivalence_pins: list[dict] = []


def _pin(name, key, expectation, rtol=EQUIVALENCE_RTOL, reference=None, reference_name=None):
    got = _route_value(key)
    ref = reference if reference is not None else _reference_value
    entry = {
        "pin": name,
        "route": key,
        "expectation": expectation,
        "reference": reference_name or "b_s3_pdip",
        "rtol": rtol,
    }
    if got is None or ref is None:
        entry["status"] = "SKIPPED"
        entry["reason"] = "one side of the comparison did not run"
    else:
        rel = abs(got - ref) / max(abs(ref), 1e-300)
        entry["got"] = got
        entry["reference_value"] = ref
        entry["rel_diff"] = rel
        entry["status"] = ("PASS" if rel <= rtol else "FAIL") if PINS_ASSERT else "RECORDED"
    equivalence_pins.append(entry)
    print(
        f"  [{entry['status']:>7}] {name}"
        + (
            f"  rel {entry['rel_diff']:.3e}"
            if "rel_diff" in entry
            else f"  ({entry.get('reason')})"
        )
    )
    return entry


if "d" in ROUTE_SELECTION and _route_d_certifies:
    _pin(
        "route d (certified, fallback) == route b (library PDIP)",
        "d_s3_certified_fallback",
        "the certified active set returns the library's own positive solution",
    )
if "d0" in ROUTE_SELECTION and _route_d_certifies:
    _pin(
        "route d0 (certified, no fallback) == route b (library PDIP)",
        "d0_s3_certified_no_fallback",
        "the certified iterate IS the constrained optimum at the certifying budget",
    )
if "e" in ROUTE_SELECTION:
    _pin(
        "route e (budget 1, fallback fires) == route b (library PDIP)",
        "e_s3_certified_budget1_fallback_fires",
        "an uncertified budget must reproduce PDIP exactly — that is what the fallback is",
    )

# S3 == S0 on the likelihood: phase 0's gate, re-measured here on the library
# path rather than on the eager fits.
_s0_value = _route_value("a_s0_pdip")
if _s0_value is not None and _reference_value is not None:
    _rel_s3_s0 = abs(_reference_value - _s0_value) / max(abs(_s0_value), 1e-300)
    _entry = {
        "pin": "route b (S3) == route a (S0) — fixing the light is a re-arrangement",
        "route": "b_s3_pdip",
        "reference": "a_s0_pdip",
        "expectation": "the source-only fit is the same fit, so the likelihood is unchanged",
        "rtol": 1e-6,
        "got": _reference_value,
        "reference_value": _s0_value,
        "rel_diff": _rel_s3_s0,
        "status": ("PASS" if _rel_s3_s0 <= 1e-6 else "FAIL") if PINS_ASSERT else "RECORDED",
    }
    equivalence_pins.append(_entry)
    print(f"  [{_entry['status']:>7}] {_entry['pin']}  rel {_rel_s3_s0:.3e}")

# Route (c) against phase 0's A2 kernel row — the same unconstrained solution
# reached through the library instead of the kernel.
_c_pin = {
    "pin": "route c reconstruction == phase 0's A2 unconstrained Cholesky",
    "route": "c_s3_positive_negative",
    "reference": "reconstruction_steps.cholesky_solve on the same F+λH, D",
    "expectation": "positivity off IS the unconstrained solve; the two must coincide",
    "rtol": 1e-6,
    "rel_diff": _a2_rel,
    "status": ("PASS" if _a2_rel <= 1e-6 else "FAIL") if PINS_ASSERT else "RECORDED",
}
equivalence_pins.append(_c_pin)
print(f"  [{_c_pin['status']:>7}] {_c_pin['pin']}  rel {_a2_rel:.3e}")

# ===================================================================
# PART F — The recorded reference (no pin exists for this configuration)
# ===================================================================
# Phase 5 runs Euclid, where no pin has ever been calibrated, and HST at source
# pixel counts the phase-0 pins do not describe. Nothing is asserted there — and
# nothing is skipped either: the quantities a pin WOULD have compared are
# computed and written as data, so this leg becomes the reference a later phase
# can pin against. The block says plainly that it is a record, not a verdict.

print("\n" + "=" * 70)
print("RECORDED REFERENCE — the values no pin covers at this configuration")
print("=" * 70)

reference_recorded = {
    "status": "RECORDED",
    "asserted": bool(PIN_IS_FIDUCIAL and PINS_ASSERT),
    "reason": (
        "the phase-0 S0 log-det pins were calibrated on hst at the fiducial mesh "
        f"({FIDUCIAL_SOURCE_PIXELS[MESH]} source pixels) in fp64; this leg is "
        f"dataset={DATASET}, {n_source_pixels} source pixels, pins_mode={PINS_MODE}. "
        "A pin calibrated elsewhere is not a pin here, so these are recorded."
        if not (PIN_IS_FIDUCIAL and PINS_ASSERT)
        else "this leg IS the fiducial hst configuration in fp64; the pins above are asserted"
    ),
    "dataset": DATASET,
    "pixel_scale_arcsec": pixel_scale,
    "mesh": MESH,
    "source_pixels": int(n_source_pixels),
    "image_pixels_masked": int(n_image_pixels),
    "log_det_curvature_reg_s0": _mapper_log_dets["log_det_curvature_reg"]["s0"],
    "log_det_curvature_reg_s3": _mapper_log_dets["log_det_curvature_reg"]["s3"],
    "log_det_regularization_s0": _mapper_log_dets["log_det_regularization"]["s0"],
    "log_det_regularization_s3": _mapper_log_dets["log_det_regularization"]["s3"],
    "log_evidence_eager_s0": float(log_evidence_ref),
    "log_evidence_eager_s3": float(log_evidence_s3_library),
    "log_evidence_positive_negative": positive_negative_block["log_evidence_eager"],
    "d_log_evidence_positive_negative_vs_s3": positive_negative_block[
        "d_log_evidence_vs_library_s3"
    ],
    "n_negative_entries_positive_negative": positive_negative_block["n_negative_entries"],
    "route_log_likelihoods": {
        key: entry.get("log_likelihood")
        for key, entry in routes.items()
        if entry.get("status") == "ok"
    },
    "smallest_certifying_budget": _certifying_budget_measured,
    "pass_budget_timed": PASS_BUDGET,
    "safe_budget": SAFE_BUDGET,
    "certifying_budget_headroom": (
        None
        if _certifying_budget_measured is None
        else int(SAFE_BUDGET - _certifying_budget_measured)
    ),
}

for _label, _key in (
    ("S0 log_det_curvature_reg", "log_det_curvature_reg_s0"),
    ("S0 log_det_regularization", "log_det_regularization_s0"),
    ("S0 log evidence (eager)", "log_evidence_eager_s0"),
    ("S3 log evidence (eager)", "log_evidence_eager_s3"),
):
    print(f"  [RECORDED] {_label}: {reference_recorded[_key]!r}")
print(
    f"  [RECORDED] smallest certifying budget {_certifying_budget_measured} vs safe budget "
    f"{SAFE_BUDGET} (headroom {reference_recorded['certifying_budget_headroom']})"
)

# ===================================================================
# Summary + JSON + PNG
# ===================================================================

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

print("\n" + "=" * 70)
print(f"FIXED-LIGHT LIBRARY PATH — {MESH} — {instrument.upper()} — v{al_version}")
print("=" * 70)
_w = max(len(v.get("label", k)) for k, v in routes.items())
for _key, _entry in routes.items():
    if _entry.get("status") != "ok":
        print(f"  {_entry.get('label', _key):<{_w}}  FAILED")
        continue
    _vmap_txt = f"   @vmap{_vmap_batch} {_entry['vmap_ms']:8.3f} ms" if "vmap_ms" in _entry else ""
    print(f"  {_entry['label']:<{_w}}  {_entry['ms']:9.3f} ms{_vmap_txt}")

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
    "dataset": DATASET,
    "pass_budget": PASS_BUDGET,
    "pass_budget_mode": PASS_BUDGET_MODE,
    "safe_budget": SAFE_BUDGET,
    "safe_budget_basis": (
        "phase 3 (autolens_profiling#255): the smallest budget with ZERO fallback over a "
        "graded 41-model draw set, confirmed N-independent from 484 to 3969 source pixels "
        "by phase 4 (#257). This is the budget a production run fixes; the smallest budget "
        "that certifies at this configuration is recorded under certification, never used."
    ),
    "pass_budget_max_diagnostics": PASS_BUDGET_MAX,
    "fallback_row_budget": FALLBACK_ROW_BUDGET,
    "fallback_rows": RUN_FALLBACK_ROWS,
    "routes_selected": list(ROUTE_SELECTION),
    "use_mixed_precision": USE_MIXED_PRECISION,
    "pins_mode": PINS_MODE,
    "tau_rel": TAU_REL,
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
    "machine": machine_info_dict(),
    "precision": {
        "use_mixed_precision": USE_MIXED_PRECISION,
        "what_mixed_precision_changes": (
            "Settings.use_mixed_precision makes the library accumulate the "
            "curvature matrix A.T A in float32 and cast the result back to "
            "float64 (inversion_util.py:127-136). The solve, the log "
            "determinants and the evidence are still fp64 arithmetic — on a "
            "matrix carrying float32 accumulation error."
        ),
        "pins_mode": PINS_MODE,
        "pins_note": (
            "fp64: every pin phases 0 and 1 calibrated on the A100 is asserted. "
            "none: those verdicts are WITHDRAWN, not skipped — every comparison "
            "is still computed and written with status RECORDED, because a pin "
            "calibrated in fp64 is not a pin in fp32."
        ),
        "tau_rel": TAU_REL,
        "tau_rel_basis": TAU_REL_BASIS,
        "float32_eps": _EPS_F32,
    },
    "instrument": instrument,
    "configuration": _configuration,
    "regularization": reg_provenance,
    "solver_injection": (
        f"harness monkeypatch of {library_solver_injection.LIBRARY_POSITIVE_ONLY_DOTTED} "
        f"for the duration of routes d/d0/e — NOT library code. Nothing in PyAutoArray "
        f"was modified; every other part of the likelihood is the library's own path."
    ),
    "log_evidence_eager_s0": float(log_evidence_ref),
    "log_evidence_eager_s3": log_evidence_s3_library,
    "systems": {
        "s0": {
            "n_params": int(system_s0.n_params),
            "n_mapper": int(system_s0.n_mapper),
            "n_funcs": int(system_s0.n_funcs),
            "edge_zeroed_pixels": int(system_s0.edge_zero_mask.sum()),
        },
        "s3": {
            "n_params": int(system_s3.n_params),
            "n_mapper": int(system_s3.n_mapper),
            "n_funcs": int(system_s3.n_funcs),
            "edge_zeroed_pixels": int(system_s3.edge_zero_mask.sum()),
            "subtracted_light_flux": float(system_s3.subtracted_light_flux),
        },
    },
    "mapper_block_log_dets": _mapper_log_dets,
    "mapper_block_log_det_rtol": _MAPPER_LOGDET_RTOL,
    "routes": routes,
    "certification": certification_block,
    "positive_negative": positive_negative_block,
    "reference_recorded": reference_recorded,
    "equivalence_pins": equivalence_pins,
    "equivalence_rtol": EQUIVALENCE_RTOL,
    "vmap": vmap_block,
    "peak_bytes": {
        "after_setup": peak_after_setup,
        "after_routes": peak_after_routes,
        "after_all": peak_after_all,
        "note": (
            "jax.devices()[0].memory_stats()['peak_bytes_in_use']; None on a backend "
            "that does not report it (CPU). Process-wide high-water marks."
        ),
    },
    "jit_phases": jit_records,
    "note": (
        "Every row is ONE WHOLE library likelihood call (AnalysisImaging."
        "log_likelihood_function under jax.jit), not a kernel. Never sum these with "
        "the kernel rows of the phase-0 fixed_light cell, and never quote route c's "
        "milliseconds without its Δlog-evidence and negative-pixel count."
    ),
}

if _vmap_batch is not None:
    breakdown_summary["vmap_batch"] = int(_vmap_batch)
    if vmap_error is not None:
        breakdown_summary["vmap_error"] = vmap_error

# ``fixed_light_library_<mesh>`` is phase 1's name and stays exactly that on the
# HST fiducial, so no phase-1 or phase-2 artifact is ever clobbered. The dataset
# and the built pixel count are appended only when they differ from it.
_cell_name = f"fixed_light_library_{MESH}"
if DATASET != "hst":
    _cell_name = f"{_cell_name}_{DATASET}"
if SOURCE_PIXELS_REQUESTED is not None:
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

_ok = [(k, v) for k, v in routes.items() if v.get("status") == "ok"]
labels = [v["label"] for _, v in _ok]
times_ms = [v["ms"] for _, v in _ok]
vmap_ms = [v.get("vmap_ms") for _, v in _ok]

colors = []
for key, _ in _ok:
    if key.startswith("a_"):
        colors.append("#C44E52")
    elif key.startswith("b_"):
        colors.append("#DD8452")
    elif key.startswith("c_"):
        colors.append("#55A868")
    else:
        colors.append("#4C72B0")

fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(labels))))
y_pos = np.arange(len(labels), dtype=float)
_has_vmap = any(v is not None for v in vmap_ms)
_height = 0.36 if _has_vmap else 0.6

bars = ax.barh(
    y_pos - (_height / 2 if _has_vmap else 0.0),
    times_ms,
    color=colors,
    edgecolor="white",
    height=_height,
    label="single call",
)
for bar, t in zip(bars, times_ms):
    ax.text(
        bar.get_width() + max(times_ms) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.2f} ms",
        va="center",
        fontsize=8,
    )

if _has_vmap:
    _vals = [v if v is not None else 0.0 for v in vmap_ms]
    bars2 = ax.barh(
        y_pos + _height / 2,
        _vals,
        color=colors,
        edgecolor="white",
        height=_height,
        alpha=0.45,
        label=f"@vmap {_vmap_batch} (per call)",
    )
    for bar, t in zip(bars2, _vals):
        if t <= 0.0:
            continue
        ax.text(
            bar.get_width() + max(times_ms) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f} ms",
            va="center",
            fontsize=8,
            alpha=0.8,
        )
    ax.legend(loc="lower right", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Whole library likelihood call (ms)", fontsize=11)
fig.suptitle(
    f"Library-path cost per positivity solver, lens light fixed — {MESH} — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_source_pixels} source pixels  |  active-set budget {PASS_BUDGET} "
    f"(certifies: {_route_d_certifies})",
    fontsize=9,
)
ax.margins(x=0.20)
fig.tight_layout()

fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

# ===================================================================
# Pins — the 2026-09-11 exact S0 log-dets
# ===================================================================
# The pins describe S0 at the fiducial mesh and nothing else, so a run with a
# non-fiducial ``--source-pixels`` skips them and writes ``pinned_expected: null``.

if not PINS_ASSERT:
    # --pins none (the mixed-precision leg). The S0 log-dets are fp64-calibrated
    # baselines; asserting them against a float32-accumulated curvature matrix
    # would report a precision choice as library drift. They are RECORDED
    # instead: every comparison is computed and written under
    # ``pins_recorded``, and ``pinned_drift`` stays empty so PyAutoHeart's
    # profiling-drift scan is not handed a false positive from a leg whose pins
    # were never applicable.
    _recorded = []
    if PIN_IS_FIDUCIAL:
        _s0_terms = system_s0.log_evidence_terms(
            np.asarray(system_s0.inversion.reconstruction, dtype=float)
        )
        for _key, _expected in PINNED_LOG_DETS[MESH].items():
            _got = float(np.asarray(_s0_terms[_key], dtype=float).ravel()[0])
            _rel = abs(_got - _expected) / max(abs(_expected), 1e-300)
            _recorded.append(
                {
                    "label": f"imaging/fixed_light_library[{instrument}, {MESH}] S0 {_key}",
                    "expected_fp64": _expected,
                    "got": _got,
                    "rel_diff": _rel,
                    "fp64_rtol_for_reference": 1e-4,
                    "status": "RECORDED",
                }
            )
            print(
                f"  [RECORDED] S0 {_key}: got {_got:.9f}  fp64 pin {_expected:.9f}  rel {_rel:.3e}"
            )
    else:
        print(
            f"  Pinned log-dets not comparable: --source-pixels "
            f"{SOURCE_PIXELS_REQUESTED} is not the fiducial mesh."
        )
    _data = json.loads(dict_path.read_text())
    _data["pins_recorded"] = {
        "mode": PINS_MODE,
        "reason": (
            "--pins none: fp64-calibrated pins are recorded, not asserted, "
            "because this leg did not run in the precision they were "
            "calibrated in."
        ),
        "s0_log_dets": _recorded,
    }
    dict_path.write_text(json.dumps(_data, indent=2))
    record_pinned_check(dict_path, None, [])
elif not PIN_IS_FIDUCIAL:
    print(
        f"  Pinned log-det check SKIPPED: this leg is dataset={DATASET} at "
        f"{n_source_pixels} source pixels; the pins describe hst at the "
        f"{FIDUCIAL_SOURCE_PIXELS[MESH]}-pixel fiducial. Every value they would have "
        f"compared is written under `reference_recorded` instead."
    )
    record_pinned_check(dict_path, None, [])
else:
    _s0_terms = system_s0.log_evidence_terms(
        np.asarray(system_s0.inversion.reconstruction, dtype=float)
    )
    _drift_records = []

    for _key, _expected in PINNED_LOG_DETS[MESH].items():
        _record = check_pinned(
            _s0_terms[_key],
            _expected,
            label=f"imaging/fixed_light_library[{instrument}, {MESH}] S0 {_key}",
            rtol=1e-4,
        )
        if _record is None:
            print(f"  Pin PASSED: S0 {_key} matches {_expected:.6f}")
        else:
            _drift_records.append(_record)

    record_pinned_check(dict_path, PINNED_LOG_DETS[MESH], _drift_records)

print("\nFinished.")
