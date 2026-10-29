"""
JAX Profiling: Cluster Image-Plane Likelihood — Per-Step Breakdown
===================================================================

Decomposes the cluster point-source **image-plane chi-squared** likelihood
into its pipeline steps for the standard cluster model (2 main dPIE + 10
scaling dPIE + NFW host at z = 0.5; 2 point sources at z = 1.0 / 2.0 —
multi-plane). The decomposed class is the solved
``FitPositionsImagePairRepeatSolved`` (``PointSolved``: each source centre is
solved analytically as beta*), the headline of the cluster runtime row
(decision D3, human 2026-10-29: "the cluster default setup should use
solved"); the plain ``FitPositionsImagePairRepeat`` (free ``Point`` centres) is
a sub-row.

Where the source-plane likelihood (``source_plane.py``) only ray-traces the
observed positions *backwards*, the image-plane likelihood **forward-solves
the lens equation** for every source: tile the image plane in triangles,
trace them to the source plane, keep the ones containing the source centre,
subdivide, repeat to sub-pixel precision. That solve dominates everything
else by orders of magnitude and is the reason cluster image-plane fits need
JAX — this script makes its cost (and its one-off JIT compile cost, which a
sampler pays once but a single fit pays in full) visible per source plane.

Steps profiled (headline, ``FitPositionsImagePairRepeatSolved``):

1. The solved source centre beta* per system: the observed images
   back-traced and tensor-weighted by the lensing Jacobian, read from the
   production fit class (eager; a glafic-style extension, **not** a result from
   Lombardi 2024, arXiv:2406.15280 — see ``autolens/point/fit/solved.py``'s
   module docstring for the attribution split).
2. Triangle-tiling PointSolver solve at beta*, JIT-compiled per source plane —
   the dominant step. Lower/compile/first-call/steady-state are reported
   separately so compile amortisation is explicit. ``solver.solve()`` only
   consumes a source-plane coordinate and the tracer's mass profiles, never the
   point-source profile type.
3. ``FitPositionsImagePairRepeatSolved`` per system (eager fit total:
   beta*, the solve, nearest-pair chi-squared).

Plain sub-row (``FitPositionsImagePairRepeat``): P1 the back-traced centroid
of each system's images (its ``Point`` centres), step 4 the plain fit total per
system (eager). A printed summary reports the per-system fit-total delta
(step 3 − step 4), to be weighed against the 2 fewer free centre parameters
per point source (4 total for this cluster's 2 systems) the solved fit leaves
out of a model-fit.

5. FUSED full likelihoods (autolens_profiling#297): the production
   ``AnalysisPoint`` likelihood summed over every system, JIT-compiled end to
   end from a parameterised model instance — ``fused_full_solved_s`` (the
   headline, ``FitPositionsImagePairRepeatSolved``, ``PointSolved``) and
   ``fused_full_plain_s`` (``FitPositionsImagePairRepeat``, free ``Point``
   centres). The model's free parameters enter the compiled function as
   arguments, so constant folding cannot pre-evaluate the solve; each row
   records its lower / compile / first-call / steady phases under
   ``jit_phases``. These two rows — not the eager step totals — are what a
   sampler pays per likelihood call. They are reported next to the steps, not
   inside ``total_step_by_step``. ``fit.png`` draws the solved fused instance.

Without ``--standard-size`` the solver grid (200x200 @ 0.7", precision 0.01") is
the historic tutorial-scale configuration the committed rows used. With
``--standard-size`` (likelihood-lifecycle 2e, autolens_profiling#474, decision D1)
the cell builds the workspace cluster solver the ``cluster-pointsolver-speed``
epic kept (human 2026-10-23): 100x100 @ 1.0", precision 0.001",
``magnification_threshold`` 0.1 and ``residual_threshold`` 10, every setting passed
explicitly (``likelihood_breakdown/cluster_standard.py``: ``STANDARD_SOLVER``), so
this breakdown decomposes the likelihood the cluster runtime cell times. The
cell's ``parity.json`` is the runtime's (``likelihood_runtime.py``), never this
script's.

Output
------

Results JSON and PNG are named under the legacy ``results/breakdown/cluster/`` folder (placed in the mirrored tree, #442) using the
basename ``image_plane_breakdown_v{al_version}``, or
``image_plane_<config_name>`` under ``--config-name``. The JSON records
``config_name``, ``thread_environment``, ``source_revisions`` and ``n_repeats``
alongside the device block.
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
# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
# ---------------------------------------------------------------------------
import os as _smoke_os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import autolens as al
import jax
import jax.numpy as jnp
import jaxlib
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
# Profiling helpers (house pattern)
# ---------------------------------------------------------------------------


class Timer:
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
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)
    lower_s = timer.records[-1][1]

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()
    compile_s = timer.records[-1][1]

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)
    first_call_s = timer.records[-1][1]

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    jit_records[label] = {
        "lower_s": float(lower_s),
        "compile_s": float(compile_s),
        "first_call_s": float(first_call_s),
        "steady_per_call_s": float(per_call),
        "n_repeats": int(n_repeats),
    }
    return compiled, result


timer = Timer()
likelihood_steps = []
jit_records: dict[str, dict] = {}
# Steady-state repeats for every JIT row. The fused rows re-run the whole
# multi-plane forward solve per call, so 3 keeps the compile-dominated cell
# inside minutes; it matches the step-2 solve rows.
N_REPEATS = 3


# ===================================================================
# PART A — Setup
# ===================================================================
dataset_path = _workspace_root / "dataset" / "cluster" / "simple"

auto_simulate_if_missing(
    dataset_path,
    dataset_type="cluster",
    instrument="simple",
    workspace_root=_workspace_root,
)

dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
print(f"Loaded {len(dataset_list)} point-source systems from {dataset_path}.")

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

source_galaxies = [
    al.Galaxy(redshift=float(d.redshift), **{d.name: al.ps.Point(centre=(0.0, 0.0))})
    for d in dataset_list
]

tracer = al.Tracer(
    galaxies=main_lens_galaxies + scaling_galaxies + [host_halo_galaxy] + source_galaxies
)
n_mass_profiles = len(main_lens_galaxies) + len(scaling_galaxies) + 1
print(f"Tracer: {len(tracer.planes)} planes, {n_mass_profiles} mass components.")

# Solved tracer (the headline): `al.ps.PointSolved` (0 params) in place of `al.ps.Point`
# (2 centre params) per source, name-paired the same way. Steps 1 and 3 read it; step
# 2's forward solve does not read the point-source profile type at all.
source_galaxies_solved = [
    al.Galaxy(redshift=float(d.redshift), **{d.name: al.ps.PointSolved()}) for d in dataset_list
]
tracer_solved = al.Tracer(
    galaxies=main_lens_galaxies + scaling_galaxies + [host_halo_galaxy] + source_galaxies_solved
)

positions_list = [np.atleast_2d(np.asarray(d.positions)) for d in dataset_list]
plane_indices = [
    tracer.plane_index_via_redshift_from(redshift=float(d.redshift)) for d in dataset_list
]

# ---------------------------------------------------------------------------
# Step 1 — the headline's solved source centres (decision D3, human
# 2026-10-29): FitPositionsImagePairRepeatSolved reads beta*, the
# tensor-weighted mean of the observed images back-traced to the source plane
# (W_i = A_i^-T A_i^-1 / sigma_i^2 from the lensing Jacobian;
# ``autolens/point/fit/solved.py``: ``SolvedCentre.source_plane_coordinate``).
# Taken from the production fit class itself (eager; no solve happens here).
# ---------------------------------------------------------------------------
solved_centres = []
for dataset in dataset_list:
    with timer.section(f"step1_solved_centre_{dataset.name}"):
        beta_star = al.FitPositionsImagePairRepeatSolved(
            name=dataset.name,
            data=dataset.positions,
            noise_map=dataset.positions_noise_map,
            tracer=tracer_solved,
            solver=None,
        ).source_plane_coordinate
        solved_centres.append(tuple(float(v) for v in beta_star))
    likelihood_steps.append(
        (
            f"1.{dataset.name} solved source centre beta* (back-trace + Jacobian)",
            timer.records[-1][1],
        )
    )

# Plain sub-row: the Point centres of the plain fit (FitPositionsImagePairRepeat) are the
# per-system centroid of the back-traced observed images (truth-adjacent, so the solve is
# realistic); the fused plain model's Point priors sit on them too. Timed as sub-row step P1.
source_centres = []
with timer.section("stepP1_plain_source_centres"):
    for positions, plane_index in zip(positions_list, plane_indices):
        traced = np.asarray(
            tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(positions))[plane_index]
        )
        source_centres.append(tuple(traced.mean(axis=0)))
plain_centres_s = timer.records[-1][1]
tracer_plain = al.Tracer(
    galaxies=main_lens_galaxies
    + scaling_galaxies
    + [host_halo_galaxy]
    + [
        al.Galaxy(redshift=float(d.redshift), **{d.name: al.ps.Point(centre=c)})
        for d, c in zip(dataset_list, source_centres)
    ]
)

# ---------------------------------------------------------------------------
# Step 2 — the triangle-tiling PointSolver forward-solve, per source plane.
# Pytree registration mirrors simulators/cluster.py: the model classes are
# registered via autofit's register_model on an af.Model mirror, and Tracer
# itself via register_instance_pytree (cosmology excluded from flattening).
# ---------------------------------------------------------------------------
import autofit as af  # noqa: E402
from autoarray.abstract_ndarray import register_instance_pytree  # noqa: E402
from autofit.jax import register_model as _register_model_pytrees  # noqa: E402
from autolens.lens.tracer import Tracer  # noqa: E402

_registration_model = af.Collection(
    galaxies=af.Collection(
        af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            mass=af.Model(al.mp.dPIEMassB0Sph, centre=(0.0, 0.0), ra=1.0, rs=10.0, b0=1.0),
        ),
        af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            dark=af.Model(
                al.mp.NFWMCRLudlowSph,
                centre=(0.0, 0.0),
                mass_at_200=10**15.3,
                redshift_object=redshift_lens,
                redshift_source=max(float(d.redshift) for d in dataset_list),
            ),
        ),
        *[
            af.Model(
                al.Galaxy,
                redshift=float(d.redshift),
                **{d.name: af.Model(al.ps.Point, centre=(0.0, 0.0))},
            )
            for d in dataset_list
        ],
    )
)
_register_model_pytrees(_registration_model)
register_instance_pytree(Tracer, no_flatten=("cosmology",))

from likelihood_breakdown.cluster_standard import (  # noqa: E402
    build_solver,
    settings_record,
    solver_settings,
)

# --standard-size: the workspace cluster solver (decision D1); else the historic 200x200 @ 0.7".
SOLVER_SETTINGS = solver_settings(_cli.standard_size)
solver = build_solver(SOLVER_SETTINGS, use_jax=True)

predicted_per_system = []
for dataset, centre in zip(dataset_list, solved_centres):

    def solve(source_plane_coordinate, _z=float(dataset.redshift)):
        return solver.solve(
            tracer=tracer,
            source_plane_coordinate=source_plane_coordinate,
            plane_redshift=_z,
        )

    _, predicted = jit_profile(
        solve, f"step2_solve_{dataset.name}", jnp.array(centre), n_repeats=N_REPEATS
    )
    predicted_per_system.append(predicted)
    likelihood_steps.append(
        (
            f"2.{dataset.name} PointSolver solve at beta* (z={float(dataset.redshift):.1f})",
            timer.records[-1][1] / N_REPEATS,
        )
    )

# ---------------------------------------------------------------------------
# Step 3 — the headline FitPositionsImagePairRepeatSolved per system, timed
# eagerly (beta*, the forward solve at it, nearest-pair chi-squared). The fit
# re-runs steps 1-2 internally, so its time is reported as fit-total; the
# pairing overhead is the difference from steps 1-2. `tracer_solved` carries
# `PointSolved` in place of `Point` so the *Solved fit class's name pairing
# finds a parameter-free profile (a centre-bearing profile raises
# `PointProfileMismatchException`).
# ---------------------------------------------------------------------------
fit_solved_log_likelihoods = []
fit_solved_total_times = []
for dataset in dataset_list:
    with timer.section(f"step3_fit_total_solved_{dataset.name}"):
        fit_solved = al.FitPositionsImagePairRepeatSolved(
            name=dataset.name,
            data=dataset.positions,
            noise_map=dataset.positions_noise_map,
            tracer=tracer_solved,
            solver=solver,
        )
        fit_solved_log_likelihoods.append(float(fit_solved.log_likelihood))
    fit_solved_total_times.append(timer.records[-1][1])
    likelihood_steps.append(
        (
            f"3.{dataset.name} FitPositionsImagePairRepeatSolved (fit total, analytic β*)",
            timer.records[-1][1],
        )
    )

log_likelihood_total_solved = sum(fit_solved_log_likelihoods)
print(
    f"\n  image-plane log likelihood, solved (sum over systems): {log_likelihood_total_solved:.6e}"
)

_n_headline_steps = len(likelihood_steps)

# ---------------------------------------------------------------------------
# Plain sub-row: P1 the back-traced centroids (above), then step 4 the
# production FitPositionsImagePairRepeat per system with free `Point` centres
# on those centroids (`tracer_plain`), eager fit-total like step 3.
# ---------------------------------------------------------------------------
likelihood_steps.append(("P1 plain: Point centres (back-traced centroid)", plain_centres_s))
fit_log_likelihoods = []
fit_total_times = []
for dataset in dataset_list:
    with timer.section(f"step4_fit_total_plain_{dataset.name}"):
        fit = al.FitPositionsImagePairRepeat(
            name=dataset.name,
            data=dataset.positions,
            noise_map=dataset.positions_noise_map,
            tracer=tracer_plain,
            solver=solver,
        )
        fit_log_likelihoods.append(float(fit.log_likelihood))
    fit_total_times.append(timer.records[-1][1])
    likelihood_steps.append(
        (f"4.{dataset.name} FitPositionsImagePairRepeat (fit total)", timer.records[-1][1])
    )

log_likelihood_total = sum(fit_log_likelihoods)
print(f"\n  image-plane log likelihood, plain (sum over systems): {log_likelihood_total:.6e}")

print("\n--- Solved vs plain: per-system fit-total runtime delta ---")
delta_per_system = {}
for dataset, plain_t, solved_t in zip(dataset_list, fit_total_times, fit_solved_total_times):
    delta = solved_t - plain_t
    pct = 100.0 * delta / plain_t if plain_t else float("nan")
    delta_per_system[dataset.name] = delta
    print(
        f"  {dataset.name}: plain={plain_t:.6f}s  solved={solved_t:.6f}s  "
        f"delta={delta:+.6f}s ({pct:+.1f}%)  -- vs -2 free centre params/source if sampled"
    )
delta_total = sum(delta_per_system.values())
print(f"  TOTAL delta across {len(dataset_list)} systems: {delta_total:+.6f}s")


# ---------------------------------------------------------------------------
# Step 5 — FUSED full likelihoods (autolens_profiling#297). The production
# path: one `AnalysisPoint(use_jax=True)` per system, summed the way a
# FactorGraphModel sums its factors, compiled end to end from a model
# instance whose free parameters are JIT *arguments*. Mirrors
# `scripts/point_source_image/image_plane/likelihood_breakdown.py`: priors are tight
# Gaussians centred on this cell's fiducial values, so the prior-median
# instance IS the step 1-4 lens model (main lenses fully free, one shared
# scaling-relation normalisation, free NFW centre). The headline solved model
# carries `PointSolved` (beta* is solved inside every call, as in step 1); the
# plain sub-row model's `Point` centres sit on P1's back-traced centroids.
# Steps 1-4 are not touched by this block.
# ---------------------------------------------------------------------------
from autofit.jax import register_model as _register_fused_pytrees  # noqa: E402


def _gaussian(mean, fraction=0.01, floor=0.01):
    return af.GaussianPrior(mean=float(mean), sigma=max(abs(float(mean)) * fraction, floor))


def _fused_model(*, solved: bool):
    galaxies = {}
    for i, (centre, ra, rs, b0) in enumerate(main_lens_params):
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre.centre_0 = _gaussian(centre[0])
        mass.centre.centre_1 = _gaussian(centre[1])
        mass.ra = _gaussian(ra)
        mass.rs = _gaussian(rs)
        mass.b0 = _gaussian(b0)
        galaxies[f"main_{i}"] = af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)

    scaling_b0_ref = _gaussian(SCALING_B0_REF, floor=0.001)
    for i, (centre, lum) in enumerate(zip(scaling_table.centres, scaling_table.luminosities)):
        ratio = (lum / _lum_ref) ** SCALING_EXPONENT
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre = tuple(centre)
        mass.ra = SCALING_RA
        mass.rs = SCALING_RS_REF * ratio
        mass.b0 = scaling_b0_ref * ratio
        galaxies[f"scaling_{i}"] = af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)

    dark = af.Model(
        al.mp.NFWMCRLudlowSph,
        mass_at_200=10**15.3,
        redshift_object=redshift_lens,
        redshift_source=max(float(d.redshift) for d in dataset_list),
    )
    dark.centre.centre_0 = _gaussian(0.0)
    dark.centre.centre_1 = _gaussian(0.0)
    galaxies["host_halo"] = af.Model(al.Galaxy, redshift=redshift_lens, dark=dark)

    for i, (d, centre) in enumerate(zip(dataset_list, source_centres)):
        if solved:
            point = af.Model(al.ps.PointSolved)
        else:
            point = af.Model(al.ps.Point)
            point.centre.centre_0 = _gaussian(centre[0])
            point.centre.centre_1 = _gaussian(centre[1])
        galaxies[f"source_{i}"] = af.Model(al.Galaxy, redshift=float(d.redshift), **{d.name: point})

    return af.Collection(galaxies=af.Collection(**galaxies))


# The headline (solved, D3) first, then the plain sub-row.
fused_models = {"solved": _fused_model(solved=True), "plain": _fused_model(solved=False)}
fused_fit_cls = {
    "solved": al.FitPositionsImagePairRepeatSolved,
    "plain": al.FitPositionsImagePairRepeat,
}
fused_instances = {}
fused_params = {}
for variant, model in fused_models.items():
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    _register_fused_pytrees(model)
    fused_instances[variant] = instance
    fused_params[variant] = jax.tree_util.tree_map(jnp.asarray, instance)


def _analyses(variant: str, *, use_jax: bool):
    return [
        al.AnalysisPoint(
            dataset=d,
            solver=solver,
            fit_positions_cls=fused_fit_cls[variant],
            use_jax=use_jax,
        )
        for d in dataset_list
    ]


fused_analyses = {variant: _analyses(variant, use_jax=True) for variant in fused_models}
eager_analyses = {variant: _analyses(variant, use_jax=False) for variant in fused_models}


def _fused_full(variant: str):
    analyses = fused_analyses[variant]

    def full(params):
        return sum(analysis.log_likelihood_function(instance=params) for analysis in analyses)

    return full


print("\n--- Step 5: eager controls for the fused rows ---")
fused_eager = {}
for variant in fused_models:
    with timer.section(f"step5_eager_{variant}"):
        fused_eager[variant] = float(
            sum(
                a.log_likelihood_function(instance=fused_instances[variant])
                for a in eager_analyses[variant]
            )
        )
    print(f"  {variant} eager likelihood: {fused_eager[variant]:.12e}")

print("\n--- Step 5: fused full likelihoods (JIT, parameterised instance) ---")
fused_jit = {}
fused_label = {"solved": "fused_full_solved", "plain": "fused_full_plain"}
for variant in fused_models:
    _, value = jit_profile(
        _fused_full(variant), fused_label[variant], fused_params[variant], n_repeats=N_REPEATS
    )
    fused_jit[variant] = float(value)
    print(f"  {variant} fused likelihood: {fused_jit[variant]:.12e}")
    if not np.isfinite(fused_jit[variant]):
        raise AssertionError(f"fused {variant} likelihood is not finite: {fused_jit[variant]}")
    np.testing.assert_allclose(fused_eager[variant], fused_jit[variant], rtol=1.0e-4)

# Solved-variant model positions per system (eager fit on the same instance):
# the forward-solved image positions the nearest-pair chi-squared paired with
# each observed image. A non-finite row would mean the solve lost an image.
fused_solved_positions = {}
for d, analysis in zip(dataset_list, eager_analyses["solved"]):
    fit_positions = analysis.fit_from(instance=fused_instances["solved"]).positions
    model_positions = np.asarray(fit_positions.model_data)
    finite_rows = np.isfinite(model_positions).all(axis=-1)
    fused_solved_positions[d.name] = {
        "observed_images": int(np.atleast_2d(np.asarray(d.positions)).shape[0]),
        "paired_model_positions": model_positions.tolist(),
        "finite_rows": int(finite_rows.sum()),
        "all_finite": bool(finite_rows.all()),
    }
    print(
        f"  {d.name}: {int(finite_rows.sum())}/{model_positions.shape[0]} finite solved positions"
    )

fused_full_plain_s = jit_records["fused_full_plain"]["steady_per_call_s"]
fused_full_solved_s = jit_records["fused_full_solved"]["steady_per_call_s"]


# ===================================================================
# PART B — Summary + artifacts
# ===================================================================
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

timer.summary()

print("\n" + "=" * 70)
print(f"PER-STEP BREAKDOWN SUMMARY — CLUSTER IMAGE-PLANE — v{al_version}")
print("=" * 70)
max_label = max(len(label) for label, _ in likelihood_steps)
# `step_total` is the headline decomposition (steps 1-3,
# FitPositionsImagePairRepeatSolved, D3), so it means "sum of the decomposed
# likelihood steps" for the README auto-table / XLA-fusion-caveat comparison
# against full_pipeline_single_jit. The plain sub-row (P1, step 4) is an
# additive extra measurement, not part of that decomposition; its cost is
# reported separately as `total_step_extra`.
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
print(f"      {'TOTAL (plain sub-row, extra)':<{max_label}}  {step_total_extra:>12.6f} s")
print(f"      {'FUSED full likelihood, solved':<{max_label}}  {fused_full_solved_s:>12.6f} s")
print(f"      {'FUSED full likelihood, plain':<{max_label}}  {fused_full_plain_s:>12.6f} s")
print("=" * 70)

config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_device_fp64"
)

breakdown_summary = {
    "autolens_version": al_version,
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    "source_revisions": source_revisions(_workspace_root),
    "config_name": config_name,
    "device": device_info_dict(),
    "resources": resources_dict("jax"),
    "thread_environment": thread_environment(),
    "n_repeats": N_REPEATS,
    "configuration": {
        "n_systems": len(dataset_list),
        "n_images_total": int(sum(len(p) for p in positions_list)),
        "n_planes": len(tracer.planes),
        "n_mass_profiles": n_mass_profiles,
        "likelihood": "image_plane (FitPositionsImagePairRepeatSolved)",
        "headline_arm": "solved",
        "sub_row_arm": "plain",
        "solver_grid": (
            f"{SOLVER_SETTINGS['shape_native'][0]}x{SOLVER_SETTINGS['shape_native'][1]} "
            f"@ {SOLVER_SETTINGS['pixel_scales']} arcsec"
        ),
        "solver_pixel_scale_precision": SOLVER_SETTINGS["pixel_scale_precision"],
        "solver": settings_record(SOLVER_SETTINGS),
        "standard_size": bool(_cli.standard_size),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "n_repeats": N_REPEATS,
        "fused_model_free_parameters": {
            variant: int(model.prior_count) for variant, model in fused_models.items()
        },
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "total_step_extra": step_total_extra,
    "headline_arm": "solved",
    "fused_full_solved_s": float(fused_full_solved_s),
    "fused_full_plain_s": float(fused_full_plain_s),
    "fused_likelihoods": {
        "solved_eager": fused_eager["solved"],
        "solved_jit": fused_jit["solved"],
        "plain_eager": fused_eager["plain"],
        "plain_jit": fused_jit["plain"],
    },
    "fused_solved_positions": fused_solved_positions,
    "jit_phases": jit_records,
    "log_likelihood": log_likelihood_total_solved,
    "solved": {
        "fit_positions_cls": "FitPositionsImagePairRepeatSolved",
        "log_likelihood": log_likelihood_total_solved,
        "solved_source_centres": {d.name: list(c) for d, c in zip(dataset_list, solved_centres)},
        "free_centre_params_removed_per_point_source": 2,
        "free_centre_params_removed_total": 2 * len(dataset_list),
    },
    "plain": {
        "fit_positions_cls": "FitPositionsImagePairRepeat",
        "log_likelihood": log_likelihood_total,
        "point_centres": {d.name: list(c) for d, c in zip(dataset_list, source_centres)},
        "delta_solved_minus_plain_per_system_s": delta_per_system,
        "delta_solved_minus_plain_total_s": delta_total,
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "cluster",
    default_basename=f"image_plane_breakdown_v{al_version}",
    cell="image_plane",
    hints=breakdown_summary,
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

labels = [label for label, _ in likelihood_steps] + [
    "5. FUSED full likelihood, solved (JIT, all systems)",
    "5. FUSED full likelihood, plain (JIT, all systems)",
]
times = [per_call for _, per_call in likelihood_steps] + [fused_full_solved_s, fused_full_plain_s]
colours = ["#4C72B0"] * len(likelihood_steps) + ["#DD8452", "#DD8452"]
fig, ax = plt.subplots(figsize=(11, 6))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color=colours, edgecolor="white", height=0.6)
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
fig.suptitle(
    f"Cluster image-plane likelihood breakdown ({config_name}) — v{al_version}", fontsize=12
)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
print(f"  Bar chart saved to:    {chart_path}")


# Fit figure (likelihood-lifecycle 1B): every system's eager headline (solved) fit at the solved
# fused instance (the step 1-4 lens model), drawn after every timed region and kept as fit.png beside this row, the
# way the cluster viz producer draws all sources together. Off under AUTOLENS_PROFILING_SMOKE
# and with --no-fit-png.
render_fit_png(
    [a.fit_from(instance=fused_instances["solved"]) for a in eager_analyses["solved"]],
    dict_path.parent,
    dataset_type="cluster",
    cli=_cli,
    solver=solver,
)
