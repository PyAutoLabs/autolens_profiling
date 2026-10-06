"""
JAX Profiling: Fixed Lens Light — certified active-set cost on LOW-LIKELIHOOD DRAWS
===================================================================================

Phase 3 of the ``fixed-lens-light-profiling`` epic (autolens_profiling#255).

The question this cell exists for
--------------------------------

Every certified active-set timing in phases 0-2 — 4.2 ms on Delaunay at **pass
2**, 11.0 ms on rectangular at **pass 7** — was measured at the *fiducial*
model, which after SLaM ``light[1]`` is essentially the truth. The certified
scheme's cost **is** its pass count, and a good model may simply have an easy
active set: few negative pixels in the unconstrained solve, a zero set that
barely moves. A non-linear search spends almost every evaluation far from the
truth. If poor models need many more passes, the production win shrinks or
disappears.

So this cell rebuilds the whole S0 → S3 chain at a **graded set of deliberately
poor models** and measures, per draw:

- certified **pass count** (numpy ``active_set_certified``, ``max_passes`` =
  ``--pass-budget-max``) and the certified **ms** at that budget
  (``active_set_masked_jax``, the jit-able twin phases 0-2 timed);
- **PDIP** iteration count and ms (``reconstruction_steps.nnls_pdip``, the
  driver, so the count survives);
- the **seed set** ``|{x_unc < 0}|`` and the final fixed set ``|Z|`` — the
  mechanism behind any pass-count growth;
- the unconstrained solve's **Δlog-evidence** against PDIP and its
  negative-pixel count — phase 0 predicted this A2 error grows with model error;
- the draw's **Δlog L** on S0.

and derives the number the phase exists for: the **fallback rate at fixed pass
budgets 2 / 4 / 6 / 7** — the fraction of draws whose certification needs more
passes than the budget, i.e. how often a production "N-pass budget + PDIP
fallback" actually falls back.

The draw set
------------

Built by ``likelihood_breakdown.fixed_light_draws_steps`` and identical on every
mesh and every device for a given seed (see that module's docstring):

- **one-parameter walks** in ``einstein_radius``, ``ell_comps_0``, ``centre_x``
  and the mass **slope**, each bisected onto Δlog L ≈ −10, −100, −1000, −1e4;
- **``--n-random`` seeded random draws** from SLaM-like priors at 5× the cell's
  own prior σ.

The fiducial mass is ``al.mp.Isothermal``, which has no slope, so the slope walk
**promotes** the mass to ``al.mp.PowerLaw`` at ``slope = 2.0``. That promotion is
pinned, not assumed: the promoted fiducial's S0 ``figure_of_merit`` is compared
against the Isothermal's before any slope row is measured, and the slope rows are
**skipped** if it does not match to ``SLOPE_PROMOTION_RTOL``.

Δlog L is measured on **S0** — ``figure_of_merit(draw) − figure_of_merit
(fiducial)`` — because S0 is the system a search scores. S3 is the re-arrangement
whose *solver* this epic is about, and phases 0-1 pinned the two likelihoods
equal to 1e-14 relative. The fiducial subtracted is the one of the draw's **own
mass family**: the Isothermal and the PowerLaw(2.0) are the same model but not
the same deflection code, and referencing each family to its own fiducial makes
that ~0.2-nat code-path offset cancel instead of biasing every slope row.

What is NOT measured here
-------------------------

No library-path row: these are **kernels**, on the same terms as phase 0's, and
must never be summed into a ``figure_of_merit``. No sparse operator, no JWST, no
PyAutoArray change — the certified scheme is still a harness kernel. No
``@vmap`` row: the pass budget differs per draw, and a batched budget is a
different measurement (phase 1's route ``d0``), not this one.

Flags (beyond the shared ``_profile_cli`` set)
----------------------------------------------

``--mesh {rectangular,delaunay}`` (required). ``--n-random N`` (default 24) and
``--seed N`` (default 0) fix the random family. ``--targets -10,-100,-1000,-1e4``
sets the walk targets. ``--pass-budget-max N`` (default 16) is the ceiling the
certification scan runs to — a draw that has not certified by then is recorded as
``passes_to_certification: null`` and counts as a fallback at **every** budget.
``--budgets 2,4,6,7`` sets the fallback-rate columns. ``--n-repeats N`` (default
5) is the steady-state timing repeat count. ``--walk-max-evals N`` (default 12)
caps the S0 evaluations one bisection target may spend. ``--pins {fp64,none}``
behaves as in the phase-1/2 cells: ``fp64`` asserts, ``none`` records.

Output
------

``results/breakdown/imaging/fixed_light_draws_<mesh>_<config_name>.{json,png}``.
The A100 legs run under ``--config-name hpc_a100_fp64_fixed_light_draws``, the
local CPU leg under ``local_cpu_fp64_fixed_light_draws`` — deliberately **not**
config names ``build_readme.py`` surfaces, on the phase-1/2 convention: these
rows are a comparator study, not a dashboard tier.
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
import time
from pathlib import Path

import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(_profiling_root()))

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke). Verifies the import
# graph + module-level setup succeeded without running the profile. Placed
# before every argparse so a smoke run needs no flags at all.
import os as _smoke_os
import sys as _smoke_sys

from likelihood_breakdown import (  # noqa: E402
    active_set_steps,
    fixed_light_draws_steps,
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

flds = fixed_light_draws_steps
ass = active_set_steps

#: Phase 0's certifying budget per mesh, at the fiducial, on euclid-ral-gpu-2
#: (legs 342802 / 342803, results/notes/fixed_lens_light_source_only_2026_09.md).
#: This cell's whole question is whether these survive a bad model, so they are
#: pinned on the fiducial row and never assumed for a draw.
CERTIFYING_BUDGET_FIDUCIAL = {"rectangular": 7, "delaunay": 2}

#: Phase 0's S3 PDIP iteration count at the fiducial, same legs.
PDIP_ITERATIONS_FIDUCIAL = {"rectangular": 15, "delaunay": 17}

#: Relative tolerance on the Isothermal -> PowerLaw(slope=2) promotion check.
#:
#: The two are the *same* profile analytically, but not the same code:
#: ``al.mp.Isothermal`` has a closed-form deflection and ``al.mp.PowerLaw`` does
#: not, so the promoted fiducial's figure of merit differs by the deflection
#: code path — 6.6e-6 relative (~0.2 nats of 2.9e4) on the Delaunay probe of
#: 2026-09-13. 1e-4 admits that and nothing larger: a promotion that actually
#: changed the *model* would move the likelihood by far more, and the slope rows
#: are then withdrawn rather than measured.
#:
#: The residual 0.2 nats is not carried into the slope rows either. A slope
#: walk's Δlog L is measured against the **PowerLaw** fiducial (``FOM_REFERENCE``
#: below), so the code-path offset cancels exactly and a slope row is a statement
#: about the slope alone.
SLOPE_PROMOTION_RTOL = 1.0e-4

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--mesh", choices=("rectangular", "delaunay"), required=True)
_cell_parser.add_argument("--n-random", type=int, default=24)
_cell_parser.add_argument("--seed", type=int, default=0)
_cell_parser.add_argument("--targets", default="-10,-100,-1000,-1e4")
_cell_parser.add_argument("--budgets", default="2,4,6,7")
_cell_parser.add_argument("--pass-budget-max", type=int, default=16)
_cell_parser.add_argument("--n-repeats", type=int, default=5)
_cell_parser.add_argument("--walk-max-evals", type=int, default=12)
_cell_parser.add_argument("--walk-rtol", type=float, default=0.2)
_cell_parser.add_argument("--source-pixels", type=int, default=None)
_cell_parser.add_argument("--pins", choices=("fp64", "none"), default="fp64")
_cell_parser.add_argument("--no-walks", dest="walks", action="store_false", default=True)
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh
N_RANDOM = int(_cell_args.n_random)
SEED = int(_cell_args.seed)
TARGETS = tuple(float(t) for t in str(_cell_args.targets).split(",") if t.strip())
BUDGETS = tuple(int(b) for b in str(_cell_args.budgets).split(",") if b.strip())
PASS_BUDGET_MAX = int(_cell_args.pass_budget_max)
N_REPEATS = int(_cell_args.n_repeats)
WALK_MAX_EVALS = int(_cell_args.walk_max_evals)
WALK_RTOL = float(_cell_args.walk_rtol)
SOURCE_PIXELS_REQUESTED = _cell_args.source_pixels
RUN_WALKS = bool(_cell_args.walks)
PINS_MODE = _cell_args.pins
PINS_ASSERT = PINS_MODE == "fp64"
USE_MIXED_PRECISION = bool(_cli.use_mixed_precision)

if PASS_BUDGET_MAX < max(BUDGETS):
    raise ValueError(
        f"--pass-budget-max ({PASS_BUDGET_MAX}) must be >= the largest budget ({max(BUDGETS)})"
    )

instrument = "hst"

FIDUCIAL_SOURCE_PIXELS = {"rectangular": 39 * 39, "delaunay": 1500}

#: The exact dense-Cholesky log determinants of S0 measured on euclid-ral-gpu-2,
#: carried from the phase-0/1/2 cells. The tripwire that this cell's model
#: construction is still the sibling cells'.
PINNED_LOG_DETS = {
    "rectangular": {
        "log_det_curvature_reg": 3888.258089645771,
        "log_det_regularization": 1692.7868170935546,
    },
    "delaunay": {
        "log_det_curvature_reg": 8360.401762997288,
        "log_det_regularization": 7756.614958909804,
    },
}

TAU_REL = ass.TAU_REL_DEFAULT

Timer = timing.Timer
block = timing.block
timer = Timer()
jit_records: dict[str, dict] = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================
# Deliberately identical to ``fixed_light.py`` / ``fixed_light_library.py``
# PART A, which is itself the pixelized breakdown cells' construction. The
# pinned S0 log-dets are the tripwire on that identity.

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
    mesh_shape = None
    n_mesh_vertices = 1500 if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    n_source_pixels = n_mesh_vertices

PIN_IS_FIDUCIAL = n_source_pixels == FIDUCIAL_SOURCE_PIXELS[MESH]
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
        mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
        reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

    pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
    source_model = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source_model))

print(f"  Regularization: {reg_scheme} ({reg_provenance})")

with timer.section("instance_from_vector"):
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

source = instance.galaxies.source

_adapt_kwargs = {
    "galaxy_image_dict": {source: adapt_image},
    "galaxy_name_image_dict": {"('galaxies', 'source')": adapt_image},
}
if image_plane_mesh_grid is not None:
    _adapt_kwargs["galaxy_image_plane_mesh_grid_dict"] = {source: image_plane_mesh_grid}
    _adapt_kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
        "('galaxies', 'source')": image_plane_mesh_grid
    }
adapt_images = al.AdaptImages(**_adapt_kwargs)
settings = al.Settings(
    use_border_relocator=True,
    **({"use_mixed_precision": True} if USE_MIXED_PRECISION else {}),
)

n_image_pixels = dataset.data.shape[0]
BASE_MASS = flds.fiducial_mass_values(instance)
print(f"  Fiducial mass: {BASE_MASS}")


# ===================================================================
# PART B — the S0 -> S3 rebuild, once per draw
# ===================================================================


def fit_s0_from(mass_profile):
    """The S0 fit — MGE-60 linear lens light + source mapper — at ``mass_profile``."""
    tracer = al.Tracer(
        galaxies=[flds.lens_galaxy_from(instance, mass_profile), source],
    )
    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )


_s0_eval_count = {"n": 0}


def s0_figure_of_merit(offsets: dict, *, mass_cls: str = "Isothermal") -> float:
    """``figure_of_merit`` of the S0 fit at the fiducial mass plus ``offsets``."""
    _s0_eval_count["n"] += 1
    fit = fit_s0_from(flds.mass_from(BASE_MASS, offsets, mass_cls=mass_cls))
    value = float(fit.figure_of_merit)
    return value if np.isfinite(value) else -np.inf


# --- the fiducial reference -------------------------------------------------

log("fiducial S0 fit")
fit_fiducial = fit_s0_from(flds.mass_from(BASE_MASS, {}, mass_cls="Isothermal"))
FOM_FIDUCIAL = float(fit_fiducial.figure_of_merit)
_s0_eval_count["n"] += 1
log(f"  fiducial S0 figure_of_merit = {FOM_FIDUCIAL:.6f}")

# --- the Isothermal -> PowerLaw(slope=2) promotion check ---------------------
# The slope walk needs a profile that HAS a slope. At slope 2.0 the PowerLaw is
# the Isothermal, so the promotion is checkable rather than assumed: if the two
# figures of merit do not agree, the slope rows measure the promotion, not the
# slope, and are withdrawn.

log("slope promotion check: PowerLaw(slope=2.0) vs Isothermal")
_fom_powerlaw_fiducial = s0_figure_of_merit({}, mass_cls="PowerLaw")
_promotion_rel = abs(_fom_powerlaw_fiducial - FOM_FIDUCIAL) / max(abs(FOM_FIDUCIAL), 1e-300)
SLOPE_PROMOTION_OK = bool(_promotion_rel <= SLOPE_PROMOTION_RTOL)
slope_promotion = {
    "isothermal_figure_of_merit": FOM_FIDUCIAL,
    "power_law_slope2_figure_of_merit": _fom_powerlaw_fiducial,
    "rel_diff": float(_promotion_rel),
    "rtol": SLOPE_PROMOTION_RTOL,
    "status": "PASS" if SLOPE_PROMOTION_OK else "FAIL",
    "consequence": (
        "slope walk measured"
        if SLOPE_PROMOTION_OK
        else "slope walk SKIPPED — the promotion changed the model, so a slope row "
        "would be measuring the promotion"
    ),
}
print(
    f"  [{slope_promotion['status']}] PowerLaw(2.0) == Isothermal  rel {_promotion_rel:.3e}  "
    f"-> {slope_promotion['consequence']}"
)

#: Δlog L is measured against the fiducial **of the draw's own mass family**. The
#: Isothermal and the PowerLaw(2.0) are the same model but not the same code, and
#: the ~0.2-nat deflection-code offset between them is not part of what a slope
#: walk is measuring. Referencing each family to its own fiducial cancels it
#: exactly, so a slope row's Δlog L is a statement about the slope alone.
FOM_REFERENCE = {"Isothermal": FOM_FIDUCIAL, "PowerLaw": _fom_powerlaw_fiducial}
slope_promotion["d_log_l_reference_per_family"] = dict(FOM_REFERENCE)


# ===================================================================
# PART C — the draw set
# ===================================================================

print("\n" + "=" * 70)
print("DRAW SET CONSTRUCTION")
print("=" * 70)


def _evaluate_walk(parameter: str, delta: float) -> float:
    mass_cls = "PowerLaw" if parameter == "slope" else "Isothermal"
    value = s0_figure_of_merit({parameter: delta}, mass_cls=mass_cls) - FOM_REFERENCE[mass_cls]
    print(f"    walk {parameter:<16} delta {delta:+.6g}  dlogL {value:+.6g}", flush=True)
    return value


_walk_parameters = tuple(p for p in flds.WALK_PARAMETERS if p != "slope" or SLOPE_PROMOTION_OK)

with timer.section("draw_set_walks"):
    draws: list = []
    if RUN_WALKS:
        draws += flds.walk_draws(
            _evaluate_walk,
            parameters=_walk_parameters,
            targets=TARGETS,
            max_evals=WALK_MAX_EVALS,
            rtol=WALK_RTOL,
        )

draws += flds.random_draws(SEED, N_RANDOM)
fiducial_draw = flds.Draw(
    name="fiducial", kind="fiducial", offsets={}, achieved_d_log_l=0.0, mass_cls="Isothermal"
)
draw_set = [fiducial_draw] + draws

print(
    f"\n  Draws: {len(draw_set)} ({len(draws)} off-fiducial); "
    f"S0 evaluations spent on construction: {_s0_eval_count['n']}"
)


# ===================================================================
# PART D — per-draw measurement
# ===================================================================

print("\n" + "=" * 70)
print("PER-DRAW MEASUREMENT")
print("=" * 70)

_pdip_compiled = jax.jit(reconstruction_steps.nnls_pdip)
_masked_compiled: dict[int, object] = {}


def _masked_at(budget: int):
    if budget not in _masked_compiled:
        _masked_compiled[budget] = jax.jit(ass.active_set_masked_jax, static_argnums=(3,))
    return _masked_compiled[budget]


def _steady_ms(fn, args, n_repeats: int) -> float:
    """Steady-state per-call milliseconds — one warm call, then ``n_repeats``."""
    block(fn(*args))
    start = time.perf_counter()
    for _ in range(n_repeats):
        block(fn(*args))
    return (time.perf_counter() - start) / n_repeats * 1.0e3


def measure_draw(draw) -> dict:
    """Everything the phase asks of one model, end to end from its offsets."""
    t_start = time.perf_counter()
    row = draw.as_dict()

    mass_profile = flds.mass_from(BASE_MASS, draw.offsets, mass_cls=draw.mass_cls)
    fit = fit_s0_from(mass_profile)
    fom_s0 = float(fit.figure_of_merit)
    row["s0_figure_of_merit"] = fom_s0
    row["d_log_l"] = fom_s0 - FOM_REFERENCE[draw.mass_cls]
    row["d_log_l_reference"] = FOM_REFERENCE[draw.mass_cls]

    system = ass.fixed_light_system_from(fit, dataset, name=f"S3_{draw.name}")
    row["n_params"] = int(system.n_params)
    row["n_edge_zeroed"] = int(system.edge_zero_mask.sum())
    row["subtracted_light_flux"] = float(system.subtracted_light_flux)

    # --- PDIP: the library's own positivity solve -------------------------
    x_pdip, pdip_converged, pdip_iterations = system.pdip()
    row["pdip_iterations"] = int(pdip_iterations)
    row["pdip_converged"] = bool(pdip_converged)
    log_ev_pdip = system.log_evidence(x_pdip)
    row["log_evidence_pdip"] = log_ev_pdip

    _Q = jnp.asarray(system.Q, dtype=jnp.float64)
    _q = jnp.asarray(system.q, dtype=jnp.float64)
    row["pdip_ms"] = _steady_ms(_pdip_compiled, (_Q, _q), N_REPEATS)

    # --- the certified active set -----------------------------------------
    x_unc_scaled = ass.unconstrained_solve(system.Q, system.q, fixed0=system.edge_zero_mask)
    run = ass.active_set_certified(
        system.Q,
        system.q,
        x_unc_scaled,
        policy="free_all",
        max_passes=PASS_BUDGET_MAX,
        tau_rel=TAU_REL,
        fixed0=system.edge_zero_mask,
    )
    row["certified"] = bool(run.certified)
    row["passes_to_certification"] = (
        int(run.passes_to_certification) if run.passes_to_certification is not None else None
    )
    row["passes_run"] = int(run.passes_run)
    row["n_factorisations"] = int(run.n_factorisations)
    row["cycle_detected"] = bool(run.cycle_detected)
    row["seed_set_size"] = int(np.sum(x_unc_scaled < 0.0))
    row["final_fixed_set_size"] = int(run.fixed_set.sum())
    row["final_free_set_size"] = int(run.fixed_set.size - run.fixed_set.sum())

    budget = row["passes_to_certification"] or PASS_BUDGET_MAX
    _mask = jnp.asarray(system.edge_zero_mask)
    masked_fn = _masked_at(budget)
    row["certified_budget_timed"] = int(budget)
    row["certified_ms"] = _steady_ms(masked_fn, (_Q, _q, _mask, int(budget)), N_REPEATS)

    masked = masked_fn(_Q, _q, _mask, int(budget))
    _flags = np.asarray(masked["certified"])
    row["masked_jax_certified_at_pass"] = int(np.argmax(_flags)) + 1 if bool(_flags.any()) else None
    row["n_primal_violations_per_pass"] = [
        int(v) for v in np.asarray(masked["n_primal_violations"])
    ]
    row["n_dual_violations_per_pass"] = [int(v) for v in np.asarray(masked["n_dual_violations"])]
    x_masked = np.asarray(masked["x"], dtype=float) * system.d_scale
    row["d_log_evidence_certified_vs_pdip"] = system.log_evidence(x_masked) - log_ev_pdip

    # --- A2: positivity dropped -------------------------------------------
    x_unconstrained = system.unconstrained(use_edge_zero=False)
    row["unconstrained_n_negative"] = int(np.sum(x_unconstrained < 0.0))
    row["d_log_evidence_unconstrained_vs_pdip"] = system.log_evidence(x_unconstrained) - log_ev_pdip

    row["wall_s"] = time.perf_counter() - t_start
    print(
        f"  {draw.name:<26} dlogL {row['d_log_l']:+11.4g}  passes "
        f"{str(row['passes_to_certification']):>4}  cert {row['certified_ms']:7.3f} ms  "
        f"PDIP {row['pdip_iterations']:>3} it {row['pdip_ms']:7.3f} ms  seed "
        f"{row['seed_set_size']:>4}  A2 {row['d_log_evidence_unconstrained_vs_pdip']:+.4g} nats "
        f"({row['wall_s']:.0f}s)",
        flush=True,
    )
    return row


rows: list[dict] = []
with timer.section("per_draw_measurement"):
    for _draw in draw_set:
        rows.append(measure_draw(_draw))

fiducial_row = rows[0]
off_fiducial = rows[1:]


# ===================================================================
# PART E — the derived tables
# ===================================================================

print("\n" + "=" * 70)
print("DERIVED TABLES")
print("=" * 70)


def _tables_for(subset, label):
    pass_counts = [r["passes_to_certification"] for r in subset]
    abs_d_log_l = [abs(r["d_log_l"]) for r in subset]
    return {
        "label": label,
        "n_draws": len(subset),
        "fallback": flds.fallback_table(pass_counts, BUDGETS),
        "spearman_passes_vs_abs_d_log_l": flds.rank_correlation(
            [p if p is not None else PASS_BUDGET_MAX + 1 for p in pass_counts], abs_d_log_l
        ),
        "spearman_pdip_iterations_vs_abs_d_log_l": flds.rank_correlation(
            [r["pdip_iterations"] for r in subset], abs_d_log_l
        ),
        "spearman_seed_set_vs_abs_d_log_l": flds.rank_correlation(
            [r["seed_set_size"] for r in subset], abs_d_log_l
        ),
        "spearman_a2_error_vs_abs_d_log_l": flds.rank_correlation(
            [abs(r["d_log_evidence_unconstrained_vs_pdip"]) for r in subset], abs_d_log_l
        ),
        "passes": flds.quantile_summary([p for p in pass_counts if p is not None], (0.5, 0.9, 1.0)),
        "pdip_iterations": flds.quantile_summary(
            [r["pdip_iterations"] for r in subset], (0.5, 0.9, 1.0)
        ),
        "certified_ms": flds.quantile_summary([r["certified_ms"] for r in subset], (0.5, 0.9, 1.0)),
        "pdip_ms": flds.quantile_summary([r["pdip_ms"] for r in subset], (0.5, 0.9, 1.0)),
        "n_never_certified": sum(1 for p in pass_counts if p is None),
    }


derived = {
    "all_off_fiducial": _tables_for(off_fiducial, "every draw except the fiducial"),
    "walks": _tables_for([r for r in off_fiducial if r["kind"] == "walk"], "one-parameter walks"),
    "random": _tables_for([r for r in off_fiducial if r["kind"] == "random"], "random draws"),
}

for _key, _table in derived.items():
    print(f"\n  {_table['label']} (n={_table['n_draws']})")
    for _entry in _table["fallback"]:
        print(
            f"    budget {_entry['budget']:>2}: fallback "
            f"{_entry['n_fallback']:>3}/{_entry['n_draws']:<3} "
            f"= {100.0 * (_entry['fallback_rate'] or 0.0):6.2f} %"
        )
    print(
        f"    Spearman passes vs |dlogL| = {_table['spearman_passes_vs_abs_d_log_l']}, "
        f"PDIP it = {_table['spearman_pdip_iterations_vs_abs_d_log_l']}"
    )


# ===================================================================
# PART F — the fiducial pins
# ===================================================================
# The fiducial row must reproduce phase 0 or this cell is measuring a different
# model. Asserted under --pins fp64 (the A100 legs), RECORDED otherwise.

print("\n" + "=" * 70)
print("FIDUCIAL PINS")
print("=" * 70)

fiducial_pins: list[dict] = []


def _fiducial_pin(name, got, expected, *, rtol=None):
    entry = {"pin": name, "got": got, "expected": expected}
    if got is None or expected is None:
        entry["status"] = "SKIPPED"
        entry["reason"] = "one side is not available"
    elif rtol is None:
        entry["status"] = ("PASS" if got == expected else "FAIL") if PINS_ASSERT else "RECORDED"
    else:
        rel = abs(float(got) - float(expected)) / max(abs(float(expected)), 1e-300)
        entry["rel_diff"] = rel
        entry["rtol"] = rtol
        entry["status"] = ("PASS" if rel <= rtol else "FAIL") if PINS_ASSERT else "RECORDED"
    fiducial_pins.append(entry)
    _extra = f"  rel {entry['rel_diff']:.3e}" if "rel_diff" in entry else f"  {got} vs {expected}"
    print(f"  [{entry['status']:>8}] {name}{_extra}")
    return entry


if PIN_IS_FIDUCIAL:
    _fiducial_pin(
        f"fiducial certifies at phase 0's budget ({MESH})",
        fiducial_row["passes_to_certification"],
        CERTIFYING_BUDGET_FIDUCIAL[MESH],
    )
    _fiducial_pin(
        f"fiducial S3 PDIP iterations ({MESH})",
        fiducial_row["pdip_iterations"],
        PDIP_ITERATIONS_FIDUCIAL[MESH],
    )
else:
    print(f"  Fiducial pins SKIPPED: {n_source_pixels} is not the fiducial mesh.")

_fiducial_pin(
    "Isothermal == PowerLaw(slope=2.0) at the fiducial",
    _fom_powerlaw_fiducial,
    FOM_FIDUCIAL,
    rtol=SLOPE_PROMOTION_RTOL,
)


# ===================================================================
# Summary + JSON + PNG
# ===================================================================

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

_configuration = {
    "pixel_scale_arcsec": pixel_scale,
    "mask_radius_arcsec": mask_radius,
    "image_pixels_masked": int(n_image_pixels),
    "mesh": MESH,
    "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
    "source_pixels": int(n_source_pixels),
    "source_pixels_requested": (
        int(SOURCE_PIXELS_REQUESTED) if SOURCE_PIXELS_REQUESTED is not None else None
    ),
    "inversion_path": "dense",
    "targets_d_log_l": list(TARGETS),
    "budgets": list(BUDGETS),
    "pass_budget_max": PASS_BUDGET_MAX,
    "n_random": N_RANDOM,
    "seed": SEED,
    "random_sigma_scale": flds.RANDOM_SIGMA_SCALE,
    "prior_sigma": flds.PRIOR_SIGMA,
    "walk_parameters": list(_walk_parameters),
    "walk_max_evals": WALK_MAX_EVALS,
    "walk_rtol": WALK_RTOL,
    "n_repeats": N_REPEATS,
    "tau_rel": TAU_REL,
    "use_mixed_precision": USE_MIXED_PRECISION,
    "pins_mode": PINS_MODE,
    "thread_env": _observe_thread_env(),
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
    "instrument": instrument,
    "configuration": _configuration,
    "regularization": reg_provenance,
    "fiducial": {
        "s0_figure_of_merit": FOM_FIDUCIAL,
        "certifying_budget_phase0": CERTIFYING_BUDGET_FIDUCIAL.get(MESH),
        "pdip_iterations_phase0": PDIP_ITERATIONS_FIDUCIAL.get(MESH),
    },
    "slope_promotion": slope_promotion,
    "draws": rows,
    "derived": derived,
    "fiducial_pins": fiducial_pins,
    "s0_evaluations_total": int(_s0_eval_count["n"]),
    "jit_phases": jit_records,
    "note": (
        "Every row is a KERNEL measured in this cell's own process on the S3 "
        "system rebuilt at that draw's mass — never a library figure_of_merit, "
        "and never to be summed with one. `d_log_l` is measured on S0. A draw "
        "with `passes_to_certification: null` never certified inside "
        "`pass_budget_max` and counts as a fallback at EVERY budget."
    ),
}

_cell_name = f"fixed_light_draws_{MESH}"
if SOURCE_PIXELS_REQUESTED is not None:
    _cell_name = f"{_cell_name}_n{int(n_source_pixels)}"

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"{_cell_name}_breakdown_{instrument}_v{al_version}",
    cell=_cell_name,
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2, default=float))
print(f"\n  Results dict saved to: {dict_path}")

# --- the two-panel figure ---------------------------------------------------

_kinds = {"walk": ("#C44E52", "one-parameter walk"), "random": ("#4C72B0", "random draw")}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for _ax, (_key, _ylabel) in zip(
    axes,
    (
        ("passes_to_certification", "certified active-set passes"),
        ("pdip_iterations", "PDIP iterations"),
    ),
):
    for _kind, (_color, _label) in _kinds.items():
        _subset = [r for r in off_fiducial if r["kind"] == _kind]
        _x = [max(abs(r["d_log_l"]), 1e-3) for r in _subset]
        _y = [(r[_key] if r[_key] is not None else PASS_BUDGET_MAX + 1) for r in _subset]
        _ax.scatter(_x, _y, s=34, color=_color, alpha=0.8, edgecolor="white", label=_label)
    _ax.scatter(
        [1e-3],
        [fiducial_row[_key] if fiducial_row[_key] is not None else PASS_BUDGET_MAX + 1],
        s=90,
        marker="*",
        color="#55A868",
        edgecolor="black",
        zorder=5,
        label="fiducial",
    )
    if _key == "passes_to_certification":
        for _budget in BUDGETS:
            _ax.axhline(_budget, color="grey", lw=0.7, ls="--", alpha=0.6)
            _ax.text(
                _ax.get_xlim()[0],
                _budget,
                f" budget {_budget}",
                fontsize=7,
                va="bottom",
                color="grey",
            )
    _ax.set_xscale("log")
    _ax.set_xlabel(r"$|\Delta \log L|$ from the fiducial", fontsize=10)
    _ax.set_ylabel(_ylabel, fontsize=10)
    _ax.legend(fontsize=8, loc="upper left")

_fb = derived["all_off_fiducial"]["fallback"]
_fb_txt = "  ".join(f"b{e['budget']}: {100.0 * (e['fallback_rate'] or 0.0):.0f}%" for e in _fb)
fig.suptitle(
    f"Certified active-set cost vs model quality — {MESH} — {instrument.upper()}  "
    f"(fallback rate {_fb_txt})",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Figure saved to:       {chart_path}")

# --- the S0 log-det pins ----------------------------------------------------

if not PINS_ASSERT:
    _recorded = []
    if PIN_IS_FIDUCIAL:
        _system_s0 = ass.linear_system_from(fit_fiducial, dataset, name="S0_current_mge60")
        _s0_terms = _system_s0.log_evidence_terms(
            np.asarray(_system_s0.inversion.reconstruction, dtype=float)
        )
        for _key, _expected in PINNED_LOG_DETS[MESH].items():
            _got = float(np.asarray(_s0_terms[_key], dtype=float).ravel()[0])
            _rel = abs(_got - _expected) / max(abs(_expected), 1e-300)
            _recorded.append(
                {
                    "label": f"imaging/fixed_light_draws[{instrument}, {MESH}] S0 {_key}",
                    "expected_fp64": _expected,
                    "got": _got,
                    "rel_diff": _rel,
                    "fp64_rtol_for_reference": 1e-4,
                    "status": "RECORDED",
                }
            )
            print(f"  [RECORDED] S0 {_key}: got {_got:.9f}  pin {_expected:.9f}  rel {_rel:.3e}")
    _data = json.loads(dict_path.read_text())
    _data["pins_recorded"] = {
        "mode": PINS_MODE,
        "reason": (
            "--pins none: the fp64-calibrated S0 log-det pins and the fiducial "
            "pass-count/PDIP pins are recorded, not asserted."
        ),
        "s0_log_dets": _recorded,
        "fiducial_pins": fiducial_pins,
    }
    dict_path.write_text(json.dumps(_data, indent=2, default=float))
    record_pinned_check(dict_path, None, [])
elif not PIN_IS_FIDUCIAL:
    print(f"  Pinned log-det check SKIPPED: {n_source_pixels} is not the fiducial mesh.")
    record_pinned_check(dict_path, None, [])
else:
    _system_s0 = ass.linear_system_from(fit_fiducial, dataset, name="S0_current_mge60")
    _s0_terms = _system_s0.log_evidence_terms(
        np.asarray(_system_s0.inversion.reconstruction, dtype=float)
    )
    _drift_records = []
    for _key, _expected in PINNED_LOG_DETS[MESH].items():
        _record = check_pinned(
            _s0_terms[_key],
            _expected,
            label=f"imaging/fixed_light_draws[{instrument}, {MESH}] S0 {_key}",
            rtol=1e-4,
        )
        if _record is None:
            print(f"  Pin PASSED: S0 {_key} matches {_expected:.6f}")
        else:
            _drift_records.append(_record)
    record_pinned_check(dict_path, PINNED_LOG_DETS[MESH], _drift_records)

print("\nFinished.")
