"""
JAX Profiling: Point-Source Image-Plane Likelihood Breakdown
============================================================

Decomposes the production ``FitPositionsImagePairAllSolved`` point-source
likelihood into the analytically solved source centre, every PointSolver
refinement stage, magnification filtering, and the fused likelihood residual.
The same script runs on CPU or GPU; every result records the actual JAX device
and configuration label, because timings are never comparable across devices.

The refinement rows use cumulative prefixes and successive differences. This
keeps each prefix faithful to the production solver while exposing ray tracing,
containment, selection, neighbourhood construction, and up-sampling at every
resolution. XLA may fuse work across a prefix boundary, so a small negative row
is retained as measurement noise. The fused end-to-end likelihood is the
authoritative runtime control.

The primary path uses ``PointSolved`` and
``FitPositionsImagePairAllSolved``. A separately labelled free-centre
``PointFlux`` / ``FitPositionsImagePairAll`` full-likelihood control confirms
that the instrument does not silently conflate the two likelihood variants.
"""

import os
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_MISC = str(_ROOT / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import jax
import jax.numpy as jnp
import jaxlib
import matplotlib
import numpy as np

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from likelihood_breakdown.provenance import source_revisions, thread_environment  # noqa: E402
from likelihood_breakdown.timing import Timer, block, jit_profile  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

_cli = parse_profile_cli()

matplotlib.use("Agg")
import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE  # noqa: E402
from autoarray.structures.triangles.shape import Point  # noqa: E402
from autofit.jax import register_model as register_model_pytrees  # noqa: E402

INSTRUMENT = "simple"
N_REPEATS = 5
PREFIX_STAGES = (
    "ray_trace",
    "containment",
    "selection",
    "neighbourhood",
    "up_sample",
)

timer = Timer()
jit_records: dict[str, dict] = {}


def _mass_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    return mass


def _model(*, solved: bool):
    lens = af.Model(al.Galaxy, redshift=0.5, mass=_mass_model())
    if solved:
        point = af.Model(al.ps.PointSolved)
    else:
        point = af.Model(al.ps.PointFlux)
        point.centre.centre_0 = af.GaussianPrior(mean=0.07, sigma=0.005)
        point.centre.centre_1 = af.GaussianPrior(mean=0.07, sigma=0.005)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


dataset_path = Path("dataset") / "point_source" / INSTRUMENT
auto_simulate_if_missing(
    dataset_path,
    dataset_type="point_source",
    instrument=INSTRUMENT,
    workspace_root=_ROOT,
)
dataset = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)
solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

model_solved = _model(solved=True)
model_plain = _model(solved=False)
instance_solved = model_solved.instance_from_vector(
    vector=model_solved.physical_values_from_prior_medians
)
instance_plain = model_plain.instance_from_vector(
    vector=model_plain.physical_values_from_prior_medians
)
register_model_pytrees(model_solved)
register_model_pytrees(model_plain)
params_solved = jax.tree_util.tree_map(jnp.asarray, instance_solved)
params_plain = jax.tree_util.tree_map(jnp.asarray, instance_plain)

analysis_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
    use_jax=True,
)
analysis_plain = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
)
analysis_solved_eager = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
    use_jax=False,
)
analysis_plain_eager = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=False,
)


def full_solved(params):
    return analysis_solved.log_likelihood_function(instance=params)


def full_plain(params):
    return analysis_plain.log_likelihood_function(instance=params)


def source_centre_prefix(params):
    fit = analysis_solved.fit_from(instance=params).positions
    return jnp.stack(fit.source_plane_coordinate)


def _solver_prefix(params, *, stop_step: int, stop_stage: str):
    fit = analysis_solved.fit_from(instance=params).positions
    shape = Point(*fit.source_plane_coordinate)
    initial_triangles = solver._initial_triangles(jnp)

    for step_number in range(stop_step + 1):
        plane_triangles = solver._plane_triangles(
            tracer=fit.tracer,
            triangles=initial_triangles,
            xp=jnp,
            plane_redshift=fit.plane_redshift,
        )
        if step_number == stop_step and stop_stage == "ray_trace":
            return plane_triangles.triangles

        indexes = plane_triangles.containing_indices(shape=shape)
        if step_number == stop_step and stop_stage == "containment":
            return indexes

        kept_triangles = initial_triangles.for_indexes(indexes=indexes)
        if step_number == stop_step and stop_stage == "selection":
            return kept_triangles.triangles

        neighbourhood = kept_triangles
        for _ in range(solver.neighbor_degree):
            neighbourhood = neighbourhood.neighborhood()
        if step_number == stop_step and stop_stage == "neighbourhood":
            return neighbourhood.triangles

        up_sampled = neighbourhood.up_sample()
        if step_number == stop_step and stop_stage == "up_sample":
            return up_sampled.triangles

        initial_triangles = up_sampled

    raise RuntimeError(f"unknown solver prefix: step={stop_step}, stage={stop_stage}")


def _prefix_function(step_number: int, stage: str):
    def prefix(params):
        return _solver_prefix(params, stop_step=step_number, stop_stage=stage)

    return prefix


def magnification_filter_prefix(params):
    fit = analysis_solved.fit_from(instance=params).positions
    kept_triangles = solver.solve_triangles(
        tracer=fit.tracer,
        shape=Point(*fit.source_plane_coordinate),
        xp=jnp,
        plane_redshift=fit.plane_redshift,
    )
    return solver._filter_low_magnification(
        tracer=fit.tracer,
        points=kept_triangles.means,
        xp=jnp,
        plane_redshift=fit.plane_redshift,
    )


print("\n--- Eager controls ---")
solved_eager = float(analysis_solved_eager.log_likelihood_function(instance=instance_solved))
plain_eager = float(analysis_plain_eager.log_likelihood_function(instance=instance_plain))
print(f"  solved likelihood: {solved_eager:.12f}")
print(f"  plain likelihood:  {plain_eager:.12f}")

print("\n--- Full fused controls ---")
_, solved_result = jit_profile(
    full_solved,
    "full_solved",
    params_solved,
    n_repeats=N_REPEATS,
    timer=timer,
    jit_records=jit_records,
)
_, plain_result = jit_profile(
    full_plain,
    "full_plain_control",
    params_plain,
    n_repeats=N_REPEATS,
    timer=timer,
    jit_records=jit_records,
)
np.testing.assert_allclose(solved_eager, float(solved_result), rtol=1.0e-4)
np.testing.assert_allclose(plain_eager, float(plain_result), rtol=1.0e-4)

print("\n--- Solved-centre and solver prefixes ---")
prefix_order: list[tuple[str, float]] = []
_, source_centre = jit_profile(
    source_centre_prefix,
    "source_centre",
    params_solved,
    n_repeats=N_REPEATS,
    timer=timer,
    jit_records=jit_records,
)
prefix_order.append(("Solved source centre", jit_records["source_centre"]["steady_per_call_s"]))

for step_number in range(solver.n_steps):
    for stage in PREFIX_STAGES:
        label = f"step_{step_number:02d}_{stage}"
        _, _ = jit_profile(
            _prefix_function(step_number, stage),
            label,
            params_solved,
            n_repeats=N_REPEATS,
            timer=timer,
            jit_records=jit_records,
        )
        prefix_order.append(
            (
                f"Step {step_number}: {stage.replace('_', ' ')}",
                jit_records[label]["steady_per_call_s"],
            )
        )

_, filtered_means = jit_profile(
    magnification_filter_prefix,
    "magnification_filter",
    params_solved,
    n_repeats=N_REPEATS,
    timer=timer,
    jit_records=jit_records,
)
prefix_order.append(
    (
        "Magnification filter",
        jit_records["magnification_filter"]["steady_per_call_s"],
    )
)

steps: dict[str, float] = {}
previous_prefix = 0.0
for label, prefix_time in prefix_order:
    steps[label] = float(prefix_time - previous_prefix)
    previous_prefix = prefix_time

full_solved_time = jit_records["full_solved"]["steady_per_call_s"]
steps["Pairing chi-squared / fused-wrapper residual"] = float(full_solved_time - previous_prefix)

print("\n--- Numerical controls ---")
batch_size = 2
batched_params = jax.tree_util.tree_map(
    lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)), params_solved
)
vmap_result = block(jax.jit(jax.vmap(full_solved))(batched_params))
np.testing.assert_allclose(np.asarray(vmap_result), float(solved_result), rtol=1.0e-4)

gradient = block(jax.grad(full_solved)(params_solved))
gradient_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(gradient)]
gradient_finite = all(np.isfinite(leaf).all() for leaf in gradient_leaves)
if not gradient_finite:
    raise AssertionError("solved point-source gradient contains non-finite values")
gradient_l2 = float(np.sqrt(sum(float(np.vdot(leaf, leaf)) for leaf in gradient_leaves)))

fit_eager = analysis_solved_eager.fit_from(instance=instance_solved).positions
shape_eager = Point(*fit_eager.source_plane_coordinate)
step_metadata = []
for step in solver.steps(
    tracer=fit_eager.tracer,
    shape=shape_eager,
    xp=np,
    plane_redshift=fit_eager.plane_redshift,
):
    step_metadata.append(
        {
            "step": int(step.number),
            "initial_shape": list(step.initial_triangles.triangles.shape),
            "plane_shape": list(step.plane_triangles.triangles.shape),
            "filtered_shape": list(step.filtered_triangles.triangles.shape),
            "neighbourhood_shape": list(step.neighbourhood.triangles.shape),
            "up_sampled_shape": list(step.up_sampled.triangles.shape),
            "filtered_finite_triangles": int(
                np.sum(np.isfinite(np.asarray(step.filtered_triangles.triangles)).all(axis=(1, 2)))
            ),
        }
    )

filtered_means_array = np.asarray(filtered_means)
filtered_finite = int(np.sum(np.isfinite(filtered_means_array).all(axis=1)))

al_version = al.__version__
config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_device_fp64"
)
summary = {
    "autolens_version": al_version,
    "package_versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
    # Resolved from the imported modules, not <root>/../PyAuto*: on RAL the
    # libraries live under /mnt/ral/jnightin/PyAuto/ (#297).
    "source_revisions": source_revisions(_ROOT),
    "instrument": INSTRUMENT,
    "config_name": config_name,
    "device": device_info_dict(),
    "thread_environment": thread_environment(),
    "precision": {
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "mixed_precision_requested": bool(_cli.use_mixed_precision),
        "note": "PointSolver has no Settings mixed-precision switch; device task validates precision-specific execution.",
    },
    "configuration": {
        "likelihood": "image_plane_solved (FitPositionsImagePairAllSolved)",
        "plain_control": "image_plane (FitPositionsImagePairAll)",
        "observed_positions": int(dataset.positions.shape[0]),
        "position_noise_sigma": float(dataset.positions_noise_map[0]),
        "dataset_noise_seed": 1,
        "lens_redshift": 0.5,
        "source_redshift": 1.0,
        "solver_grid_shape": [100, 100],
        "solver_grid_pixel_scale": 0.2,
        "solver_pixel_scale_precision": 0.001,
        "magnification_threshold": 0.1,
        "neighbor_degree": int(solver.neighbor_degree),
        "max_containing_size": MAX_CONTAINING_SIZE,
        "n_steps": int(solver.n_steps),
        "n_repeats": N_REPEATS,
    },
    "steps": steps,
    "prefix_steady_per_call_s": dict(prefix_order),
    "total_step_by_step": float(sum(steps.values())),
    "fused_full_solved_s": float(full_solved_time),
    "fused_full_plain_control_s": float(jit_records["full_plain_control"]["steady_per_call_s"]),
    "jit_phases": jit_records,
    "likelihoods": {
        "solved_eager": solved_eager,
        "solved_jit": float(solved_result),
        "plain_eager": plain_eager,
        "plain_jit": float(plain_result),
    },
    "numerical_controls": {
        "vmap_batch": batch_size,
        "vmap_values": np.asarray(vmap_result).tolist(),
        "gradient_all_finite": gradient_finite,
        "gradient_l2": gradient_l2,
        "filtered_finite_positions": filtered_finite,
    },
    "step_metadata": step_metadata,
    "methodology": {
        "prefix_rows": "successive differences of cumulative JIT prefixes",
        "negative_rows": "retained; XLA fusion and timing noise can move work across boundaries",
        "authoritative_runtime": "fused_full_solved_s",
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_image",
    default_basename=f"image_plane_breakdown_{INSTRUMENT}_v{al_version}",
    cell="image_plane",
)
dict_path.write_text(json.dumps(summary, indent=2))

labels = list(steps)
times = [steps[label] for label in labels]
prefix_labels = [label for label, _ in prefix_order] + ["Fused solved likelihood"]
prefix_times = [value for _, value in prefix_order] + [full_solved_time]
fig_height = max(8.0, 0.25 * len(labels))
fig, (prefix_ax, difference_ax) = plt.subplots(
    ncols=2, figsize=(18, fig_height), constrained_layout=True
)
prefix_ax.barh(range(len(prefix_labels)), prefix_times, color="#4C72B0", edgecolor="white")
prefix_ax.set_yticks(range(len(prefix_labels)))
prefix_ax.set_yticklabels(prefix_labels, fontsize=6)
prefix_ax.invert_yaxis()
prefix_ax.set_xlabel("Absolute cumulative-prefix steady-state time [s]")
prefix_ax.set_title("Absolute prefixes (compare these across runs)")

colours = ["#C44E52" if value < 0.0 else "#4C72B0" for value in times]
difference_ax.barh(range(len(labels)), times, color=colours, edgecolor="white")
difference_ax.set_yticks(range(len(labels)))
difference_ax.set_yticklabels(labels, fontsize=6)
difference_ax.invert_yaxis()
difference_ax.axvline(0.0, color="black", linewidth=0.8)
difference_ax.set_xlabel("Successive-difference steady-state time [s]")
difference_ax.set_title("Signed differences (fusion/noise retained)")
fig.suptitle(
    f"Point-source image-plane solved breakdown ({config_name}); "
    f"fused={full_solved_time * 1000:.3f} ms"
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 72)
print("POINT-SOURCE IMAGE-PLANE BREAKDOWN")
print("=" * 72)
for label, value in steps.items():
    print(f"  {label:<54} {value * 1000:10.3f} ms")
print("-" * 72)
print(f"  {'Fused solved likelihood':<54} {full_solved_time * 1000:10.3f} ms")
print(f"  {'Fused plain control':<54} {summary['fused_full_plain_control_s'] * 1000:10.3f} ms")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
timer.summary()
