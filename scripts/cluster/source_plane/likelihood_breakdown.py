"""
JAX Profiling: Cluster Source-Plane Likelihood — Per-Step Breakdown
====================================================================

Decomposes the cluster point-source **source-plane chi-squared** likelihood
into its pipeline steps and JIT-profiles each one separately. The decomposed
class is the solved ``FitPositionsSourceSolved`` (``PointSolved``: the source
centre is solved analytically and marginalised over), the headline of the
cluster runtime row and the fit the workspace cluster search uses (decision D3,
human 2026-10-29); the plain ``FitPositionsSource`` (free ``Point`` centres) is a
sub-row. The model is the standard cluster:

 - 2 individually-modelled dPIE main lenses + 10 scaling-tier dPIE members
   (reference-anchored relation) + 1 NFW host halo, all at z = 0.5;
 - 2 point sources at *different* redshifts (z = 1.0, 2.0) — a genuine
   multi-plane system, so ray-tracing pays the per-plane recursion.

This is Lenstool's default likelihood: observed multiple-image positions are
ray-traced *back* to each source's plane and the scatter about the group
centroid is penalised. No lens-equation solve is involved, which is why the
whole thing is orders of magnitude cheaper than the image-plane likelihood
(``image_plane.py`` in this folder profiles that one).

Steps profiled (headline, ``FitPositionsSourceSolved``):

1. Multi-plane ray-tracing of the observed positions, per source plane —
   the deflection stack over all 13 mass profiles is one hot spot.
2. The lensing Jacobian A_i at every observed position via the tracer
   Hessian (``LensCalc.jacobian_from``) — the production source-plane
   chi-squared (``FitPositionsSource.weighting = "jacobian"``, the default
   since PyAutoLens bdb44a87c) measures each residual in the image-plane noise
   ellipse through it. This evaluates the full deflection stack several more
   times per position and competes with step 1 for the budget.
3. The analytic source centre per system: the precision tensor
   W_i = A_i^-T A_i^-1 / sigma_i^2, beta* = (sum W_i)^-1 sum W_i beta_i and
   the marginalisation term -0.5 ln(det(sum W_i) / (2 pi)^2) (Lombardi 2024
   §5.1, Eq. 58; ``autolens/point/fit/solved.py``: ``SolvedCentre``).
4. Tensor-weighted chi-squared per system about beta*:
   chi2_i = (beta_i - beta*)^T W_i (beta_i - beta*) (Lombardi 2024 Eq. 46;
   Keeton 2001 Eq. 10).
5. Total log-likelihood assembly: -0.5 (sum chi2 + noise norm) plus the
   marginalisation terms. The noise normalization is the model-independent
   observed-plane term sum ln((2 pi)^2 sigma_i^4).

Plain sub-row (``FitPositionsSource``, ``Point`` centres at the simulator
truth): steps 1-2 are shared; P3 is the tensor-weighted chi-squared about the
``Point`` centre (precision included), P4 its assembly (no marginalisation).
Steps 6-7 time each class's eager fit total per system.

The steps close over the (fixed) tracer rather than passing it as a JIT
pytree argument: each step compiles exactly once here, so registration
buys nothing and the closure keeps the decomposition free of pytree
plumbing. (In a model-fit the tracer changes per call — see
``simulators/cluster.py`` for the pytree-argument pattern used there.)

Reference checks: the summed per-step log-likelihood is asserted against the
eager ``al.FitPositionsSourceSolved`` values (and beta* against the fit's
``source_plane_coordinate``), and the plain sub-row against
``al.FitPositionsSource``, at ``rtol=1e-4``, so both decompositions are provably
the production calculation. The end-to-end
source-plane likelihood JIT-compiles today; it is timed by the runtime cell
(``likelihood_runtime.py``, likelihood-lifecycle 2e), which also writes the
cell's ``parity.json``. This decomposition compiles the numerical core of each
step in isolation. The likelihood has no PointSolver, so ``--standard-size``
changes nothing here; the row records it.

Solved vs plain (issue #657 phase 3; headline swapped by D3)
-------------------------------------------------------------

Steps 6-7 compare the two classes' eager fit totals per system: step 6
``al.FitPositionsSourceSolved`` against ``tracer_solved`` (zero-parameter
``al.ps.PointSolved`` profiles in place of ``al.ps.Point``), step 7
``al.FitPositionsSource`` against the truth-centred ``Point`` tracer. The
source-plane analytic β* solve is the paper's result (Lombardi 2024,
arXiv:2406.15280 §5.1), unlike the image-plane solved extension. A printed
summary reports the per-system fit-total delta (step 6 − step 7), to be
weighed against the 2 fewer free centre parameters per point source (4 total
for this cluster's 2 systems) the solved fit leaves out of a model-fit.

Output
------

Results JSON and PNG are named under the legacy ``results/breakdown/cluster/`` folder (placed in the mirrored tree, #442) using the
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
from likelihood_breakdown.provenance import source_revisions, thread_environment  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    render_fit_png,
    resolve_output_paths,
    resources_dict,
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
# same way: the headline FitPositionsSourceSolved fit total (step 6) and fit.png.
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
# Step 2 — the lensing Jacobian at the observed positions via the tracer
# Hessian. The production fit (``weighting = "jacobian"``) maps the
# image-plane positional noise into the source plane through A_i.
# ---------------------------------------------------------------------------
import autogalaxy as ag  # noqa: E402

jacobians_per_system = []
for i, (dataset, positions, plane_index) in enumerate(
    zip(dataset_list, positions_list, plane_indices)
):
    lens_calc = ag.LensCalc.from_tracer(tracer=tracer, use_multi_plane=True, plane_j=plane_index)

    def jacobians_at(positions_arr, _lens_calc=lens_calc):
        # The raw array goes straight into the hessian path: wrapping it in Grid2DIrregular here
        # would trip __array__ on the tracer (the hessian slices grid[:, 0] directly, which is fine
        # on a bare jax.Array). ``jacobian_from`` returns its 2x2 list in (x, y) row/col order;
        # re-index to (y, x) like the positions, as the production fit does
        # (``autolens/point/fit/solved.py``: ``precision_tensor_components_from``).
        (a_xx, a_xy), (a_yx, a_yy) = _lens_calc.jacobian_from(grid=positions_arr, xp=jnp)
        comps = [c.array if hasattr(c, "array") else c for c in (a_yy, a_yx, a_xy, a_xx)]
        a11, a12, a21, a22 = comps
        return jnp.stack([jnp.stack([a11, a12], -1), jnp.stack([a21, a22], -1)], -2)

    _, jacobians = jit_profile(
        jacobians_at,
        f"step2_jacobian_{dataset.name}",
        jnp.array(positions),
    )
    jacobians_per_system.append(jacobians)
    likelihood_steps.append(
        (f"2.{i} lensing jacobian (hessian) {dataset.name}", timer.records[-1][1] / 10)
    )


# ---------------------------------------------------------------------------
# Step 3 — the analytic source centre per system, the headline
# FitPositionsSourceSolved (decision D3, human 2026-10-29): the per-image
# precision tensor W_i = A_i^-T A_i^-1 / sigma_i^2 from step 2's Jacobians,
# beta* = (sum_i W_i)^-1 sum_i W_i beta_hat_i, and the marginalisation term
# -0.5 log(det(sum_i W_i) / (2 pi)^2) of the flat-prior centre
# (``autolens/point/fit/solved.py``: ``SolvedCentre``, ``precision_tensor_components_from``).
# ---------------------------------------------------------------------------
def precision_tensors(jacobian_arr, noise_arr):
    a_inv = jnp.linalg.inv(jacobian_arr)
    return jnp.einsum("nji,njk->nik", a_inv, a_inv) / noise_arr[:, None, None] ** 2


solved_centres_per_system = []
precisions_per_system = []
marginalisation_per_system = []
for i, (dataset, traced, jacobians, noise) in enumerate(
    zip(dataset_list, traced_per_system, jacobians_per_system, noise_list)
):

    def solved_centre(traced_arr, jacobian_arr, noise_arr):
        precision = precision_tensors(jacobian_arr, noise_arr)
        sum_w = jnp.sum(precision, axis=0)
        beta_star = jnp.linalg.solve(sum_w, jnp.einsum("nij,nj->i", precision, traced_arr))
        marginalisation = -0.5 * jnp.log(jnp.linalg.det(sum_w) / (2.0 * jnp.pi) ** 2)
        return beta_star, precision, marginalisation

    _, (beta_star, precision, marginalisation) = jit_profile(
        solved_centre,
        f"step3_solved_centre_{dataset.name}",
        jnp.array(traced),
        jnp.array(jacobians),
        jnp.array(noise),
    )
    solved_centres_per_system.append(beta_star)
    precisions_per_system.append(precision)
    marginalisation_per_system.append(marginalisation)
    likelihood_steps.append(
        (f"3.{i} solved source centre beta* {dataset.name}", timer.records[-1][1] / 10)
    )

# ---------------------------------------------------------------------------
# Step 4 — tensor-weighted chi-squared per system about the solved centre:
# (beta_hat_i - beta*)^T W_i (beta_hat_i - beta*) (pure jnp math; tiny next to
# steps 1-2, but the production formula).
# ---------------------------------------------------------------------------


def weighted_chi_squared_about(traced_arr, precision_arr, centre_arr):
    delta = traced_arr - centre_arr
    return jnp.einsum("ni,nij,nj->", delta, precision_arr, delta)


chi_squared_per_system = []
for i, (dataset, traced, precision, beta_star) in enumerate(
    zip(dataset_list, traced_per_system, precisions_per_system, solved_centres_per_system)
):
    _, chi_sq = jit_profile(
        weighted_chi_squared_about,
        f"step4_chi2_solved_{dataset.name}",
        jnp.array(traced),
        jnp.array(precision),
        jnp.array(beta_star),
    )
    chi_squared_per_system.append(chi_sq)
    likelihood_steps.append(
        (f"4.{i} weighted chi2 about beta* {dataset.name}", timer.records[-1][1] / 10)
    )

# ---------------------------------------------------------------------------
# Step 5 — total log-likelihood assembly: -0.5 * (sum chi2 + noise norm) plus
# the marginalisation terms, with the model-independent normalization
# sum ln((2 pi)^2 sigma_i^4).
# ---------------------------------------------------------------------------
noise_norm_terms = jnp.array(
    [float(np.sum(np.log((2.0 * np.pi) ** 2 * np.asarray(noise) ** 4))) for noise in noise_list]
)


def log_likelihood_total_solved(chi_squareds, noise_norms, marginalisations):
    return -0.5 * (jnp.sum(chi_squareds) + jnp.sum(noise_norms)) + jnp.sum(marginalisations)


_, log_likelihood = jit_profile(
    log_likelihood_total_solved,
    "step5_log_likelihood",
    jnp.array(chi_squared_per_system),
    noise_norm_terms,
    jnp.array(marginalisation_per_system),
)
likelihood_steps.append(
    ("5 log-likelihood assembly (+ marginalisation)", timer.records[-1][1] / 10)
)

_n_headline_steps = len(likelihood_steps)

# ---------------------------------------------------------------------------
# Plain sub-row (FitPositionsSource, free Point centres at the simulator truth):
# steps 1-2 are shared; P3 is the tensor-weighted chi-squared about the Point
# centre (precision included, as the plain fit computes it), P4 its assembly
# (no marginalisation term). Reported as extra steps, outside the headline total.
# ---------------------------------------------------------------------------
plain_chi_squared_per_system = []
for i, (dataset, traced, jacobians, noise, centre) in enumerate(
    zip(dataset_list, traced_per_system, jacobians_per_system, noise_list, TRUTH_SOURCE_CENTRES)
):

    def weighted_chi_squared(traced_arr, jacobian_arr, noise_arr, centre_arr):
        return weighted_chi_squared_about(
            traced_arr, precision_tensors(jacobian_arr, noise_arr), centre_arr
        )

    _, chi_sq = jit_profile(
        weighted_chi_squared,
        f"stepP3_chi2_plain_{dataset.name}",
        jnp.array(traced),
        jnp.array(jacobians),
        jnp.array(noise),
        jnp.array(centre),
    )
    plain_chi_squared_per_system.append(chi_sq)
    likelihood_steps.append(
        (
            f"P3.{i} plain: weighted chi2 about Point centre {dataset.name}",
            timer.records[-1][1] / 10,
        )
    )


def log_likelihood_total_plain(chi_squareds, noise_norms):
    return -0.5 * (jnp.sum(chi_squareds) + jnp.sum(noise_norms))


_, plain_log_likelihood = jit_profile(
    log_likelihood_total_plain,
    "stepP4_log_likelihood_plain",
    jnp.array(plain_chi_squared_per_system),
    noise_norm_terms,
)
likelihood_steps.append(("P4 plain: log-likelihood assembly", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Steps 6-7 — per-system fit totals (eager, atomic): step 6 the headline
# al.FitPositionsSourceSolved against `tracer_solved` (PointSolved, so the
# *Solved class's name pairing finds a parameter-free profile; a centre-bearing
# profile raises PointProfileMismatchException), step 7 the plain
# al.FitPositionsSource. Steps 1-5 / P3-P4 decompose these same calls; the
# atomic timings make the solved-vs-plain delta apples-to-apples. Defaults to
# xp=numpy (no JAX needed).
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

fit_plain_log_likelihoods = []
fit_plain_total_times = []
for dataset in dataset_list:
    with timer.section(f"step7_fit_total_plain_{dataset.name}"):
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
        (f"7.{dataset.name} FitPositionsSource (fit total)", timer.records[-1][1])
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
# PART C — Eager reference checks (the production fit classes)
# ===================================================================
# The references are the eager fit totals of steps 6-7, summed over the systems.
reference_log_likelihood = sum(fit_solved_log_likelihoods)
reference_log_likelihood_plain = sum(fit_plain_log_likelihoods)

print(f"\n  step-by-step log likelihood (solved): {float(log_likelihood):.8e}")
print(f"  FitPositionsSourceSolved reference:    {reference_log_likelihood:.8e}")
assert np.isclose(float(log_likelihood), reference_log_likelihood, rtol=1e-4), (
    "per-step decomposition does not match the production FitPositionsSourceSolved value"
)
for d, beta_star, fit_centre in zip(
    dataset_list,
    solved_centres_per_system,
    [
        al.FitPositionsSourceSolved(
            name=d.name,
            data=d.positions,
            noise_map=d.positions_noise_map,
            tracer=tracer_solved,
            solver=None,
        ).source_plane_coordinate
        for d in dataset_list
    ],
):
    np.testing.assert_allclose(np.asarray(beta_star), np.asarray(fit_centre), rtol=1e-6, atol=1e-9)
print(f"  plain step-by-step log likelihood:      {float(plain_log_likelihood):.8e}")
print(f"  FitPositionsSource reference:           {reference_log_likelihood_plain:.8e}")
assert np.isclose(float(plain_log_likelihood), reference_log_likelihood_plain, rtol=1e-4), (
    "plain sub-row decomposition does not match the production FitPositionsSource value"
)
print("  MATCH (rtol=1e-4), both classes; beta* matches the fit's source_plane_coordinate")


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
# `step_total` is the headline decomposition (steps 1-5, FitPositionsSourceSolved), so it means
# "sum of the decomposed likelihood steps" for the README auto-table / XLA-fusion-caveat
# comparison. The plain sub-row (P3-P4) and the fit totals (6-7) are extra measurements, not
# part of that decomposition; their cost is reported separately.
step_total = 0.0
step_total_extra = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    if i <= _n_headline_steps:
        step_total += per_call
    else:
        step_total_extra += per_call
print("-" * 70)
print(f"      {'TOTAL (step-by-step, solved)':<{max_label}}  {step_total:>12.6f} s")
print(
    f"      {'TOTAL (plain sub-row + fit totals, extra)':<{max_label}}  {step_total_extra:>12.6f} s"
)
print("=" * 70)

breakdown_summary = {
    "autolens_version": al_version,
    "source_revisions": source_revisions(_workspace_root),
    "device": device_info_dict(),
    "resources": resources_dict("jax"),
    "thread_environment": thread_environment(),
    "n_repeats": 10,
    "configuration": {
        "n_systems": len(dataset_list),
        "n_images_total": int(sum(len(p) for p in positions_list)),
        "n_planes": len(tracer.planes),
        "n_mass_profiles": n_mass_profiles,
        "likelihood": "source_plane (FitPositionsSourceSolved)",
        "headline_arm": "solved",
        "sub_row_arm": "plain",
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "standard_size": bool(_cli.standard_size),
        "solver": None,
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "total_step_extra": step_total_extra,
    "log_likelihood": float(log_likelihood),
    "reference_log_likelihood": reference_log_likelihood,
    "headline_arm": "solved",
    "solved": {
        "fit_positions_cls": "FitPositionsSourceSolved",
        "log_likelihood": float(log_likelihood),
        "reference_log_likelihood": reference_log_likelihood,
        "solved_source_centres": {
            d.name: [float(v) for v in np.asarray(c)]
            for d, c in zip(dataset_list, solved_centres_per_system)
        },
        "free_centre_params_removed_per_point_source": 2,
        "free_centre_params_removed_total": 2 * len(dataset_list),
    },
    "plain": {
        "fit_positions_cls": "FitPositionsSource",
        "log_likelihood": float(plain_log_likelihood),
        "reference_log_likelihood": reference_log_likelihood_plain,
        "point_centres": [list(c) for c in TRUTH_SOURCE_CENTRES],
        "delta_solved_minus_plain_per_system_s": delta_per_system,
        "delta_solved_minus_plain_total_s": delta_total,
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "cluster",
    default_basename=f"source_plane_breakdown_v{al_version}",
    cell="source_plane",
    hints=breakdown_summary,
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


# Fit figure (likelihood-lifecycle 1B): every system's eager headline FitPositionsSourceSolved fit
# of the truth lens (PointSolved sources), wrapped in the FitPointDataset the plotter reads, drawn after every timed region and
# kept as fit.png beside this row, the way the cluster viz producer draws all sources together.
# This cell has no PointSolver, so the panel extent comes from the observed positions. Off under
# AUTOLENS_PROFILING_SMOKE and with --no-fit-png.
render_fit_png(
    [
        al.FitPointDataset(
            dataset=dataset,
            tracer=tracer_solved,
            solver=None,
            fit_positions_cls=al.FitPositionsSourceSolved,
        )
        for dataset in dataset_list
    ],
    dict_path.parent,
    dataset_type="cluster",
    cli=_cli,
)
