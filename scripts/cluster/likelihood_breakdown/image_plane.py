"""
JAX Profiling: Cluster Image-Plane Likelihood — Per-Step Breakdown
===================================================================

Decomposes the cluster point-source **image-plane chi-squared** likelihood
(``FitPositionsImagePairRepeat``, the model-fit default) into its pipeline
steps for the standard cluster model (2 main dPIE + 10 scaling dPIE + NFW
host at z = 0.5; 2 point sources at z = 1.0 / 2.0 — multi-plane).

Where the source-plane likelihood (``source_plane.py``) only ray-traces the
observed positions *backwards*, the image-plane likelihood **forward-solves
the lens equation** for every source: tile the image plane in triangles,
trace them to the source plane, keep the ones containing the source centre,
subdivide, repeat to sub-pixel precision. That solve dominates everything
else by orders of magnitude and is the reason cluster image-plane fits need
JAX — this script makes its cost (and its one-off JIT compile cost, which a
sampler pays once but a single fit pays in full) visible per source plane.

Steps profiled:

1. Back-trace observed positions → model source centres (setup, eager).
2. Triangle-tiling PointSolver solve, JIT-compiled per source plane — the
   dominant step. Lower/compile/first-call/steady-state are reported
   separately so compile amortisation is explicit.
3. ``FitPositionsImagePairRepeat`` log-likelihood per system (eager,
   nearest-pair chi-squared given the solved positions).

The solver grid below (200x200 @ 0.7", precision 0.01") is the
tutorial-scale configuration of the workspace cluster scripts, chosen so the
one-off compile stays at minutes; production precision (0.001") multiplies
the triangle fan-out, not the structure of the breakdown.

Output
------

Results JSON and PNG are written to ``results/breakdown/cluster/`` using the
basename ``image_plane_breakdown_v{al_version}``.
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
likelihood_steps = []


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

positions_list = [np.atleast_2d(np.asarray(d.positions)) for d in dataset_list]
plane_indices = [
    tracer.plane_index_via_redshift_from(redshift=float(d.redshift)) for d in dataset_list
]

# ---------------------------------------------------------------------------
# Step 1 — model source centres: back-trace the observed images and take the
# per-system centroid (what a model fit derives from its Point centre; here
# the truth-adjacent centroid keeps the solve realistic). Eager + cheap.
# ---------------------------------------------------------------------------
source_centres = []
with timer.section("step1_source_centres"):
    for positions, plane_index in zip(positions_list, plane_indices):
        traced = np.asarray(
            tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(positions))[plane_index]
        )
        source_centres.append(tuple(traced.mean(axis=0)))
likelihood_steps.append(("1 source centres (back-trace)", timer.records[-1][1]))

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

solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.7),
    pixel_scale_precision=0.01,
    use_jax=True,
)

predicted_per_system = []
for dataset, centre in zip(dataset_list, source_centres):

    def solve(source_plane_coordinate, _z=float(dataset.redshift)):
        return solver.solve(
            tracer=tracer,
            source_plane_coordinate=source_plane_coordinate,
            plane_redshift=_z,
        )

    _, predicted = jit_profile(solve, f"step2_solve_{dataset.name}", jnp.array(centre), n_repeats=3)
    predicted_per_system.append(predicted)
    likelihood_steps.append(
        (
            f"2.{dataset.name} PointSolver solve (z={float(dataset.redshift):.1f})",
            timer.records[-1][1] / 3,
        )
    )

# ---------------------------------------------------------------------------
# Step 3 — pairing + chi-squared: the production FitPositionsImagePairRepeat,
# timed eagerly per system (the pairing is trivial next to the solve; the fit
# re-runs the solve internally, so its time is reported as fit-total and the
# pairing overhead is the difference from step 2).
# ---------------------------------------------------------------------------
fit_log_likelihoods = []
for dataset in dataset_list:
    with timer.section(f"step3_fit_total_{dataset.name}"):
        fit = al.FitPositionsImagePairRepeat(
            name=dataset.name,
            data=dataset.positions,
            noise_map=dataset.positions_noise_map,
            tracer=tracer,
            solver=solver,
        )
        fit_log_likelihoods.append(float(fit.log_likelihood))
    likelihood_steps.append(
        (f"3.{dataset.name} FitPositionsImagePairRepeat (fit total)", timer.records[-1][1])
    )

log_likelihood_total = sum(fit_log_likelihoods)
print(f"\n  image-plane log likelihood (sum over systems): {log_likelihood_total:.6e}")


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
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call
print("-" * 70)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print("=" * 70)

breakdown_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "configuration": {
        "n_systems": len(dataset_list),
        "n_images_total": int(sum(len(p) for p in positions_list)),
        "n_planes": len(tracer.planes),
        "n_mass_profiles": n_mass_profiles,
        "likelihood": "image_plane (FitPositionsImagePairRepeat)",
        "solver_grid": "200x200 @ 0.7 arcsec",
        "solver_pixel_scale_precision": 0.01,
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "log_likelihood": log_likelihood_total,
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "cluster",
    default_basename=f"image_plane_breakdown_v{al_version}",
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
fig.suptitle(f"Cluster image-plane likelihood breakdown — v{al_version}", fontsize=12)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
print(f"  Bar chart saved to:    {chart_path}")
