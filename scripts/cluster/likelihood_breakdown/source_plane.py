"""
JAX Profiling: Cluster Source-Plane Likelihood — Per-Step Breakdown
====================================================================

Decomposes the cluster point-source **source-plane chi-squared** likelihood
(``FitPositionsSource``) into its pipeline steps and JIT-profiles each one
separately, for the standard cluster model:

 - 2 individually-modelled dPIE main lenses + 10 scaling-tier dPIE members
   (reference-anchored relation) + 1 NFW host halo, all at z = 0.5;
 - 2 point sources at *different* redshifts (z = 1.0, 2.0) — a genuine
   multi-plane system, so ray-tracing pays the per-plane recursion.

This is Lenstool's default likelihood: observed multiple-image positions are
ray-traced *back* to each source's plane and the scatter about the group
centroid is penalised. No lens-equation solve is involved, which is why the
whole thing is orders of magnitude cheaper than the image-plane likelihood
(``image_plane.py`` in this folder profiles that one).

Steps profiled:

1. Multi-plane ray-tracing of the observed positions, per source plane —
   the deflection stack over all 13 mass profiles is one hot spot.
2. Magnification at every observed position via the tracer Hessian
   (``LensCalc.magnification_2d_via_hessian_from``) — the production
   source-plane chi-squared weights each residual by its magnification, so
   image-plane noise maps correctly into the source plane. This evaluates
   the full deflection stack several more times per position and competes
   with step 1 for the budget.
3. Magnification-weighted chi-squared per system: the distance of each
   traced position to the *model's* source centre (name pairing hands the
   fit the ``Point`` profile centre — the barycenter is only a fallback when
   the tracer has no matching profile), chi2_i = dist_i^2 * mag_i^2 / sigma_i^2.
4. Total log-likelihood assembly. The noise normalization uses the
   magnification-scaled effective noise, sum ln(2 pi (sigma_i / mag_i)^2) —
   consistent with the chi-squared's source-plane noise mapping.

The steps close over the (fixed) tracer rather than passing it as a JIT
pytree argument: each step compiles exactly once here, so registration
buys nothing and the closure keeps the decomposition free of pytree
plumbing. (In a model-fit the tracer changes per call — see
``simulators/cluster.py`` for the pytree-argument pattern used there.)

Reference check: the summed per-step log-likelihood is asserted against the
eager ``al.FitPositionsSource(profile=None)`` values at ``rtol=1e-4``, so the
decomposition is provably the production calculation. Note the known JAX
constraint: the *end-to-end* source-plane fit is not JIT-compilable today
(see ``autolens_workspace_test/scripts/CLAUDE.md`` — the source-plane entry
is marked JIT-blocked); the per-step decomposition below sidesteps that by
compiling the numerical core of each step in isolation.

Solved-variant addendum (issue #657 phase 3)
----------------------------------------------

Steps 5-6 add a plain-vs-solved fit-total comparison: step 5 times the
production ``al.FitPositionsSource`` per system (eager, atomic — the same
call PART C already asserts in summed form, just split per system here so
it is comparable to step 6), and step 6 times
``al.FitPositionsSourceSolved`` per system against a second tracer
(``tracer_solved``) built with zero-parameter ``al.ps.PointSolved``
profiles in place of ``al.ps.Point``. This IS the paper's result (Lombardi
2024, arXiv:2406.15280 §5.1) — unlike the image-plane solved extension, the
source-plane analytic β* solve is exactly what the paper derives. The
analytic solve itself (``autolens/point/fit/solved.py``: ``SolvedCentre``)
is profiled as a single atomic fit-total here, not further decomposed into
its own jacobian/precision-tensor sub-steps, mirroring how PART C already
treats the plain ``FitPositionsSource`` as one atomic reference alongside
the fully decomposed steps 1-4. A printed summary reports the per-system
fit-total runtime delta (step 6 − step 5), to be weighed against the 2
fewer free centre parameters per point source (4 total for this cluster's
2 systems) a full model-fit would otherwise sample.

Output
------

Results JSON and PNG are written to ``results/breakdown/cluster/`` using the
basename ``source_plane_breakdown_v{al_version}``.
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


# ---------------------------------------------------------------------------
# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke): verify the import
# graph + module-level setup succeeded without running the full pipeline.
# ---------------------------------------------------------------------------
import os as _smoke_os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import autolens as al
import jax
import jax.numpy as jnp
import numpy as np

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

sys.path.insert(0, str(_profiling_root()))
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

_cli = parse_profile_cli()

_script_dir = Path(__file__).resolve().parent
_workspace_root = _profiling_root()


# ---------------------------------------------------------------------------
# Profiling helpers (house pattern — see likelihood_breakdown/imaging/mge.py)
# ---------------------------------------------------------------------------


class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    def summary(self):
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time lower/compile/first-call/steady-state."""
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return compiled, result


timer = Timer()
likelihood_steps = []  # (label, per_call_seconds)


# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset (auto-simulated on first run via simulators/cluster.py)
# ---------------------------------------------------------------------------
dataset_path = _workspace_root / "dataset" / "cluster" / "simple"

auto_simulate_if_missing(
    dataset_path,
    dataset_type="cluster",
    instrument="simple",
    workspace_root=_workspace_root,
)

dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
print(f"Loaded {len(dataset_list)} point-source systems from {dataset_path}.")

# ---------------------------------------------------------------------------
# 2. Tracer — the standard cluster model, at the simulator truth values.
#    Constants mirror simulators/cluster.py (which mirrors the workspace
#    cluster simulator + its reference-anchored scaling relation).
# ---------------------------------------------------------------------------
redshift_lens = 0.5

main_lens_params = [
    ((0.0, 0.0), 8.0, 20.0, 3.0),
    ((10.0, 8.0), 5.0, 12.0, 1.2),
]
main_lens_galaxies = [
    al.Galaxy(
        redshift=redshift_lens,
        mass=al.mp.dPIEMassB0Sph(centre=centre, ra=ra, rs=rs, b0=b0),
    )
    for centre, ra, rs, b0 in main_lens_params
]

scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
SCALING_B0_REF, SCALING_RS_REF, SCALING_RA, SCALING_EXPONENT = 0.12, 10.0, 0.1, 0.5
_lum_ref = max(scaling_table.luminosities)
scaling_galaxies = [
    al.Galaxy(
        redshift=redshift_lens,
        mass=al.mp.dPIEMassB0Sph(
            centre=tuple(centre),
            ra=SCALING_RA,
            rs=SCALING_RS_REF * (lum / _lum_ref) ** SCALING_EXPONENT,
            b0=SCALING_B0_REF * (lum / _lum_ref) ** SCALING_EXPONENT,
        ),
    )
    for centre, lum in zip(scaling_table.centres, scaling_table.luminosities)
]

host_halo_galaxy = al.Galaxy(
    redshift=redshift_lens,
    dark=al.mp.NFWMCRLudlowSph(
        centre=(0.0, 0.0),
        mass_at_200=10**15.3,
        redshift_object=redshift_lens,
        redshift_source=max(float(d.redshift) for d in dataset_list),
    ),
)

# Point centres at the simulator truth (simulators/cluster.py source_centres) —
# name pairing hands these to the fit as the source-plane reference points.
TRUTH_SOURCE_CENTRES = [(0.3, 0.5), (-0.8, 1.2)]
source_galaxies = [
    al.Galaxy(redshift=float(d.redshift), **{d.name: al.ps.Point(centre=centre)})
    for d, centre in zip(dataset_list, TRUTH_SOURCE_CENTRES)
]

tracer = al.Tracer(
    galaxies=main_lens_galaxies + scaling_galaxies + [host_halo_galaxy] + source_galaxies
)
n_mass_profiles = len(main_lens_galaxies) + len(scaling_galaxies) + 1
print(
    f"Tracer: {len(tracer.planes)} planes, {n_mass_profiles} mass components "
    f"(2 main dPIE + {len(scaling_galaxies)} scaling dPIE + 1 NFW host)."
)

# Solved-variant tracer: `al.ps.PointSolved` (0 params) in place of
# `al.ps.Point(centre=...)` (2 centre params) per source, name-paired the
# same way. Used only by step 6 below.
source_galaxies_solved = [
    al.Galaxy(redshift=float(d.redshift), **{d.name: al.ps.PointSolved()}) for d in dataset_list
]
tracer_solved = al.Tracer(
    galaxies=main_lens_galaxies + scaling_galaxies + [host_halo_galaxy] + source_galaxies_solved
)

positions_list = [np.atleast_2d(np.asarray(d.positions)) for d in dataset_list]
noise_list = [np.asarray(d.positions_noise_map) for d in dataset_list]
plane_indices = [
    tracer.plane_index_via_redshift_from(redshift=float(d.redshift)) for d in dataset_list
]


# ===================================================================
# PART B — Per-step JIT profiling
# ===================================================================

# ---------------------------------------------------------------------------
# Step 1 — multi-plane ray-tracing of the observed positions, per system.
# The recursion walks every plane up to the source's, applying the scaled
# deflections of all 13 mass profiles; this is where the cluster's many-
# profile cost lives.
# ---------------------------------------------------------------------------
traced_per_system = []
for i, (dataset, positions, plane_index) in enumerate(
    zip(dataset_list, positions_list, plane_indices)
):

    def trace_positions(positions_arr, _plane_index=plane_index):
        grid = al.Grid2DIrregular(values=positions_arr, xp=jnp)
        traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)[_plane_index]
        return traced.array if hasattr(traced, "array") else traced

    _, traced = jit_profile(
        trace_positions,
        f"step1_trace_{dataset.name}",
        jnp.array(positions),
    )
    traced_per_system.append(traced)
    likelihood_steps.append(
        (f"1.{i} ray-trace {dataset.name} ({len(positions)} img)", timer.records[-1][1] / 10)
    )

# ---------------------------------------------------------------------------
# Step 2 — magnification at the observed positions via the tracer Hessian.
# The production fit weights each source-plane residual by |mu| so the
# image-plane positional noise maps into the source plane correctly.
# ---------------------------------------------------------------------------
import autogalaxy as ag  # noqa: E402

magnifications_per_system = []
for i, (dataset, positions, plane_index) in enumerate(
    zip(dataset_list, positions_list, plane_indices)
):
    lens_calc = ag.LensCalc.from_tracer(tracer=tracer, use_multi_plane=True, plane_j=plane_index)

    def magnifications_at(positions_arr, _lens_calc=lens_calc):
        # The raw traced array goes straight into the hessian path: wrapping it in
        # Grid2DIrregular here would trip __array__ on the tracer (the hessian
        # slices grid[:, 0] directly, which is fine on a bare jax.Array).
        mags = _lens_calc.magnification_2d_via_hessian_from(grid=positions_arr, xp=jnp)
        mags = mags.array if hasattr(mags, "array") else mags
        return jnp.abs(mags)

    _, mags = jit_profile(
        magnifications_at,
        f"step2_magnification_{dataset.name}",
        jnp.array(positions),
    )
    magnifications_per_system.append(mags)
    likelihood_steps.append(
        (f"2.{i} magnification (hessian) {dataset.name}", timer.records[-1][1] / 10)
    )

# ---------------------------------------------------------------------------
# Step 3 — centroid + magnification-weighted chi-squared per system
# (pure jnp math; tiny next to steps 1-2, but the production formula).
# ---------------------------------------------------------------------------
chi_squared_per_system = []
for i, (dataset, traced, mags, noise, centre) in enumerate(
    zip(
        dataset_list, traced_per_system, magnifications_per_system, noise_list, TRUTH_SOURCE_CENTRES
    )
):

    def weighted_chi_squared(traced_arr, mags_arr, noise_arr, centre_arr):
        distances_sq = jnp.sum((traced_arr - centre_arr) ** 2, axis=1)
        return jnp.sum(distances_sq * mags_arr**2 / noise_arr**2)

    _, chi_sq = jit_profile(
        weighted_chi_squared,
        f"step3_chi2_{dataset.name}",
        jnp.array(traced),
        jnp.array(mags),
        jnp.array(noise),
        jnp.array(centre),
    )
    chi_squared_per_system.append(chi_sq)
    likelihood_steps.append((f"3.{i} weighted chi2 {dataset.name}", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Step 4 — total log-likelihood assembly: -0.5 * (sum chi2 + noise norm).
# ---------------------------------------------------------------------------
noise_norm_terms = jnp.array(
    [
        float(np.sum(np.log(2.0 * np.pi * (np.asarray(noise) / np.asarray(mags)) ** 2)))
        for noise, mags in zip(noise_list, magnifications_per_system)
    ]
)


def log_likelihood_total(chi_squareds, noise_norms):
    return -0.5 * (jnp.sum(chi_squareds) + jnp.sum(noise_norms))


_, log_likelihood = jit_profile(
    log_likelihood_total,
    "step4_log_likelihood",
    jnp.array(chi_squared_per_system),
    noise_norm_terms,
)
likelihood_steps.append(("4 log-likelihood assembly", timer.records[-1][1] / 10))

_n_plain_steps = len(likelihood_steps)

# ---------------------------------------------------------------------------
# Step 5 — FitPositionsSource per system, fit-total (eager). Steps 1-4 above
# decompose the internals of this same call; this atomic per-system timing
# exists purely so the solved-vs-plain runtime delta below (step 6 - step 5)
# is apples-to-apples — steps 1-4's individually-JIT-profiled sub-steps
# aren't directly comparable to step 6's single eager
# FitPositionsSourceSolved timing. PART C's `reference_fit_positions_source`
# assertion further below is unaffected — this is an additive per-system
# timing, not a replacement.
# ---------------------------------------------------------------------------
fit_plain_log_likelihoods = []
fit_plain_total_times = []
for dataset in dataset_list:
    with timer.section(f"step5_fit_total_plain_{dataset.name}"):
        fit_plain = al.FitPositionsSource(
            name=dataset.name,
            data=dataset.positions,
            noise_map=dataset.positions_noise_map,
            tracer=tracer,
            solver=None,
        )
        fit_plain_log_likelihoods.append(float(fit_plain.log_likelihood))
    fit_plain_total_times.append(timer.records[-1][1])
    likelihood_steps.append(
        (f"5.{dataset.name} FitPositionsSource (fit total)", timer.records[-1][1])
    )

# ---------------------------------------------------------------------------
# Step 6 — FitPositionsSourceSolved per system, fit-total (eager, analytic
# β*). `tracer_solved` carries `PointSolved` in place of `Point` so the
# *Solved fit class's name-pairing finds a parameter-free profile — pairing
# a centre-bearing profile with a *Solved fit class raises
# PointProfileMismatchException. Defaults to xp=numpy (no JAX needed).
# ---------------------------------------------------------------------------
fit_solved_log_likelihoods = []
fit_solved_total_times = []
for dataset in dataset_list:
    with timer.section(f"step6_fit_total_solved_{dataset.name}"):
        fit_solved = al.FitPositionsSourceSolved(
            name=dataset.name,
            data=dataset.positions,
            noise_map=dataset.positions_noise_map,
            tracer=tracer_solved,
            solver=None,
        )
        fit_solved_log_likelihoods.append(float(fit_solved.log_likelihood))
    fit_solved_total_times.append(timer.records[-1][1])
    likelihood_steps.append(
        (
            f"6.{dataset.name} FitPositionsSourceSolved (fit total, analytic β*)",
            timer.records[-1][1],
        )
    )

log_likelihood_total_solved = sum(fit_solved_log_likelihoods)
print(
    f"\n  source-plane log likelihood, solved (sum over systems): {log_likelihood_total_solved:.6e}"
)

print("\n--- Solved vs plain: per-system fit-total runtime delta ---")
delta_per_system = {}
for dataset, plain_t, solved_t in zip(dataset_list, fit_plain_total_times, fit_solved_total_times):
    delta = solved_t - plain_t
    pct = 100.0 * delta / plain_t if plain_t else float("nan")
    delta_per_system[dataset.name] = delta
    print(
        f"  {dataset.name}: plain={plain_t:.6f}s  solved={solved_t:.6f}s  "
        f"delta={delta:+.6f}s ({pct:+.1f}%)  -- vs -2 free centre params/source if sampled"
    )
delta_total = sum(delta_per_system.values())
print(f"  TOTAL delta across {len(dataset_list)} systems: {delta_total:+.6f}s")


# ===================================================================
# PART C — Eager reference check (FitPositionsSource, production path)
# ===================================================================
with timer.section("reference_fit_positions_source"):
    reference_log_likelihood = sum(
        float(
            al.FitPositionsSource(
                name=dataset.name,
                data=dataset.positions,
                noise_map=dataset.positions_noise_map,
                tracer=tracer,
                solver=None,
            ).log_likelihood
        )
        for dataset in dataset_list
    )

print(f"\n  step-by-step log likelihood: {float(log_likelihood):.8e}")
print(f"  FitPositionsSource reference: {reference_log_likelihood:.8e}")
assert np.isclose(float(log_likelihood), reference_log_likelihood, rtol=1e-4), (
    "per-step decomposition does not match the production FitPositionsSource value"
)
print("  MATCH (rtol=1e-4)")


# ===================================================================
# PART D — Summary + artifacts
# ===================================================================
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

timer.summary()

print("\n" + "=" * 70)
print(f"PER-STEP BREAKDOWN SUMMARY — CLUSTER SOURCE-PLANE — v{al_version}")
print("=" * 70)
max_label = max(len(label) for label, _ in likelihood_steps)
# `step_total` stays scoped to the original (plain-path) decomposition —
# steps 1-4 — so it keeps meaning "sum of the decomposed likelihood steps"
# for the README auto-table / XLA-fusion-caveat comparison. Steps 5-6
# (plain-vs-solved fit-total comparison) are additive extra measurements,
# not part of that decomposition; their cost is reported separately.
step_total = 0.0
step_total_solved_extra = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    if i <= _n_plain_steps:
        step_total += per_call
    else:
        step_total_solved_extra += per_call
print("-" * 70)
print(f"      {'TOTAL (step-by-step, plain)':<{max_label}}  {step_total:>12.6f} s")
print(f"      {'TOTAL (steps 5-6, extra)':<{max_label}}  {step_total_solved_extra:>12.6f} s")
print("=" * 70)

breakdown_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "configuration": {
        "n_systems": len(dataset_list),
        "n_images_total": int(sum(len(p) for p in positions_list)),
        "n_planes": len(tracer.planes),
        "n_mass_profiles": n_mass_profiles,
        "likelihood": "source_plane (FitPositionsSource)",
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "reference_log_likelihood": reference_log_likelihood,
    "solved": {
        "fit_positions_cls": "FitPositionsSourceSolved",
        "log_likelihood": log_likelihood_total_solved,
        "total_step_solved_extra": step_total_solved_extra,
        "delta_per_system_s": delta_per_system,
        "delta_total_s": delta_total,
        "free_centre_params_removed_per_point_source": 2,
        "free_centre_params_removed_total": 2 * len(dataset_list),
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "cluster",
    default_basename=f"source_plane_breakdown_v{al_version}",
    cell="source_plane",
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]
fig, ax = plt.subplots(figsize=(10, 5))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color="#4C72B0", edgecolor="white", height=0.6)
for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.6f} s",
        va="center",
        fontsize=9,
    )
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Time per call (s)", fontsize=11)
fig.suptitle(f"Cluster source-plane likelihood breakdown — v{al_version}", fontsize=12)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
print(f"  Bar chart saved to:    {chart_path}")
