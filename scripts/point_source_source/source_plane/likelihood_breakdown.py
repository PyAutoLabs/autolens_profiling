"""
JAX Profiling: Point-Source Source-Plane Likelihood Breakdown
=============================================================

Decomposes the production source-plane point-source likelihood — the adopted
PyAutoLens default ``FitPositionsSourceSolved`` (Lombardi 2024 §5.1: solved
source-plane centre, tensor ``jacobian`` weighting, analytic marginalisation) —
into its ray trace, precision tensor, solved centre, chi-squared and
marginalisation term, with a fused end-to-end likelihood control. A separately
labelled free-centre ``PointFlux`` / ``FitPositionsSource`` (scalar
``magnification`` weighting) lane is decomposed the same way, so the
instrument never silently conflates the two likelihood variants. The same
script runs on CPU or GPU; every result records the actual JAX device and
configuration label, because timings are never comparable across devices.

Single-source only: the ``simple`` point-source dataset. Cluster-scale
source-plane profiling lives in ``scripts/cluster/`` and is out of scope here.

Prefix rows
-----------
Each row is a cumulative JIT prefix, ``analysis.fit_from(instance).positions``
followed by an attribute chain cut at the fit object's own boundaries, and
returns the sum of every intermediate reached so far (so XLA cannot dead-code
eliminate an earlier stage). Step rows are successive differences of the
steady per-call prefix times; the final row is the fused likelihood minus the
last prefix. XLA may fuse or common-subexpression-eliminate work across a
prefix boundary, so a small (or negative) row is measurement evidence, not an
error; the fused end-to-end likelihood is the authoritative runtime.

The fit properties are uncached: the plain lane evaluates
``magnifications_at_positions`` (a ``jacfwd`` Hessian) twice per likelihood
(chi-squared map and noise normalisation), and the solved lane rebuilds the
precision tensor three times (solved centre, chi-squared map, marginalisation
term) and ``_beta_hat`` twice. A ``cse_probe`` block times each once versus
repeated inside one JIT; whether XLA merges them is recorded, not assumed.

``solver=None``: the source-plane fit never calls a ``PointSolver``
(``FitPositionsSource`` types it ``Optional``), so no solver is built.

Regression literals
-------------------
``EXPECTED_LOG_LIKELIHOOD_SOLVED`` / ``_PLAIN`` pin the JIT likelihood at the
prior-median parameters on the ``simple`` dataset. Refresh procedure: when a
deliberate library or dataset change moves them, set both to ``None``, run the
cell once (it prints the new values and skips the assertion), paste the
printed values back, and state the cause in the commit message.
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
import time

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
from autofit.jax import register_model as register_model_pytrees  # noqa: E402
from autolens.point.fit.solved import precision_tensor_components_from  # noqa: E402

INSTRUMENT = "simple"
# One call is ~0.3 ms on CPU: 500 steady calls per prefix keeps each mean
# stable to a few microseconds for well under a second of wall time.
N_REPEATS = 500

EXPECTED_LOG_LIKELIHOOD_SOLVED = 0.5986504555536349
EXPECTED_LOG_LIKELIHOOD_PLAIN = -33788.35531560625
LITERAL_RTOL = 1.0e-8

timer = Timer()
LOAD_AVERAGE_START = os.getloadavg()
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


def _array(value):
    return value.array if hasattr(value, "array") else value


dataset_path = Path("dataset") / "point_source" / INSTRUMENT
auto_simulate_if_missing(
    dataset_path,
    dataset_type="point_source",
    instrument=INSTRUMENT,
    workspace_root=_ROOT,
)
dataset = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")

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


def _analysis(fit_positions_cls, *, use_jax: bool):
    return al.AnalysisPoint(
        dataset=dataset,
        solver=None,
        fit_positions_cls=fit_positions_cls,
        use_jax=use_jax,
    )


analysis_solved = _analysis(al.FitPositionsSourceSolved, use_jax=True)
analysis_plain = _analysis(al.FitPositionsSource, use_jax=True)
analysis_solved_eager = _analysis(al.FitPositionsSourceSolved, use_jax=False)
analysis_plain_eager = _analysis(al.FitPositionsSource, use_jax=False)


def full_solved(params):
    return analysis_solved.log_likelihood_function(instance=params)


def full_plain(params):
    return analysis_plain.log_likelihood_function(instance=params)


# --- Cumulative prefixes: solved lane (FitPositionsSourceSolved) -------------


def _solved_fit(params):
    return analysis_solved.fit_from(instance=params).positions


def _solved_stages(fit, stop: str):
    total = jnp.sum(fit._beta_hat.array)
    if stop == "beta_hat":
        return total
    total = total + sum(jnp.sum(w) for w in precision_tensor_components_from(fit, fit.weighting))
    if stop == "precision_tensor":
        return total
    total = total + jnp.sum(jnp.stack(fit.source_plane_coordinate))
    if stop == "source_plane_coordinate":
        return total
    total = total + fit.chi_squared
    if stop == "chi_squared":
        return total
    total = total + fit.marginalization_term
    if stop == "marginalization_term":
        return total
    return total + fit.log_likelihood


SOLVED_PREFIXES = (
    ("beta_hat", "Ray trace: _beta_hat (deflections at observed positions)"),
    ("precision_tensor", "Precision tensor W_i (jacfwd Hessian -> A^-T Theta A^-1)"),
    ("source_plane_coordinate", "Solved centre beta* (rebuilds W_i and _beta_hat)"),
    ("chi_squared", "Chi-squared (rebuilds W_i, beta*, _beta_hat)"),
    ("marginalization_term", "Marginalisation term (rebuilds W_i)"),
    ("log_likelihood", "fit.log_likelihood (re-evaluates every term)"),
)

# --- Cumulative prefixes: plain control lane (FitPositionsSource) ------------


def _plain_fit(params):
    return analysis_plain.fit_from(instance=params).positions


def _plain_stages(fit, stop: str):
    total = jnp.sum(fit.model_data.array)
    if stop == "model_data":
        return total
    total = total + jnp.sum(_array(fit.magnifications_at_positions))
    if stop == "magnifications_at_positions":
        return total
    total = total + jnp.sum(_array(fit.chi_squared_map))
    if stop == "chi_squared_map":
        return total
    return total + fit.log_likelihood


PLAIN_PREFIXES = (
    ("model_data", "Ray trace: model_data (deflections at observed positions)"),
    ("magnifications_at_positions", "Magnifications (jacfwd Hessian)"),
    ("chi_squared_map", "Chi-squared map (rebuilds model_data + magnifications)"),
    ("log_likelihood", "fit.log_likelihood (noise norm re-evaluates magnifications)"),
)


def _prefix(fit_from, stages, stop: str):
    def prefix(params):
        return stages(fit_from(params), stop)

    return prefix


# --- CSE probes: one evaluation versus the repeat count the fit performs -----


def precision_once(params):
    fit = _solved_fit(params)
    return sum(jnp.sum(w) for w in precision_tensor_components_from(fit, fit.weighting))


def precision_thrice(params):
    fit = _solved_fit(params)
    return sum(
        jnp.sum(w) for _ in range(3) for w in precision_tensor_components_from(fit, fit.weighting)
    )


def magnifications_once(params):
    fit = _plain_fit(params)
    return jnp.sum(_array(fit.magnifications_at_positions))


def magnifications_twice(params):
    fit = _plain_fit(params)
    return jnp.sum(_array(fit.magnifications_at_positions)) + jnp.sum(
        _array(fit.magnifications_at_positions)
    )


def _profile(func, label, params):
    """``jit_profile`` (mean steady time) plus a per-call median over N_REPEATS.

    At ~0.3 ms per call the mean is sensitive to a few scheduler hiccups, so the
    step rows use the median; both are written to ``jit_phases``.
    """
    compiled, result = jit_profile(
        func,
        label,
        params,
        n_repeats=N_REPEATS,
        timer=timer,
        jit_records=jit_records,
    )
    samples = np.empty(N_REPEATS)
    for index in range(N_REPEATS):
        start = time.perf_counter()
        block(compiled(params))
        samples[index] = time.perf_counter() - start
    jit_records[label]["median_per_call_s"] = float(np.median(samples))
    jit_records[label]["p10_per_call_s"] = float(np.percentile(samples, 10))
    jit_records[label]["p90_per_call_s"] = float(np.percentile(samples, 90))
    print(f"    -> per-call median: {np.median(samples) * 1000:.4f} ms")
    return compiled, result


def _steady(label: str) -> float:
    return jit_records[label]["median_per_call_s"]


# --- Dispatch floor: what a JIT call costs before any lensing work ------------


def floor_params_pytree(params):
    return sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(params))


def floor_scalar_array(value):
    return value + 1.0


print("\n--- Eager controls ---")
solved_eager = float(analysis_solved_eager.log_likelihood_function(instance=instance_solved))
plain_eager = float(analysis_plain_eager.log_likelihood_function(instance=instance_plain))
print(f"  solved likelihood: {solved_eager:.12f}")
print(f"  plain likelihood:  {plain_eager:.12f}")

print("\n--- Full fused controls ---")
_, solved_result = _profile(full_solved, "full_solved", params_solved)
_, plain_result = _profile(full_plain, "full_plain_control", params_plain)
np.testing.assert_allclose(solved_eager, float(solved_result), rtol=1.0e-4)
np.testing.assert_allclose(plain_eager, float(plain_result), rtol=1.0e-4)
print(f"  solved JIT: {float(solved_result):.12f}")
print(f"  plain JIT:  {float(plain_result):.12f}")


def _walk(lane: str, fit_from, stages, prefixes, params, fused_label: str):
    prefix_order: list[tuple[str, float]] = []
    for stop, row in prefixes:
        label = f"{lane}_{stop}"
        _profile(_prefix(fit_from, stages, stop), label, params)
        prefix_order.append((row, _steady(label)))
    steps: dict[str, float] = {}
    previous = 0.0
    for row, prefix_time in prefix_order:
        steps[row] = float(prefix_time - previous)
        previous = prefix_time
    steps["Fused-wrapper residual (fused - last prefix)"] = float(_steady(fused_label) - previous)
    return prefix_order, steps


print("\n--- Dispatch floor ---")
_profile(floor_params_pytree, "floor_params_pytree_solved", params_solved)
_profile(floor_params_pytree, "floor_params_pytree_plain", params_plain)
_profile(floor_scalar_array, "floor_scalar_array", jnp.asarray(0.0))
dispatch_floor = {
    "params_pytree_solved_s": _steady("floor_params_pytree_solved"),
    "params_pytree_plain_s": _steady("floor_params_pytree_plain"),
    "scalar_array_s": _steady("floor_scalar_array"),
    "reading": "jit(sum of the ModelInstance leaves) and jit(x + 1) on a scalar: the "
    "per-call cost paid before any lensing work (argument flattening + dispatch)",
}

print("\n--- Solved-lane prefixes ---")
prefix_order, steps = _walk(
    "solved", _solved_fit, _solved_stages, SOLVED_PREFIXES, params_solved, "full_solved"
)
print("\n--- Plain-lane prefixes ---")
plain_prefix_order, plain_steps = _walk(
    "plain", _plain_fit, _plain_stages, PLAIN_PREFIXES, params_plain, "full_plain_control"
)

print("\n--- CSE probes ---")
_profile(precision_once, "cse_precision_once", params_solved)
_profile(precision_thrice, "cse_precision_thrice", params_solved)
_profile(magnifications_once, "cse_magnifications_once", params_plain)
_profile(magnifications_twice, "cse_magnifications_twice", params_plain)
cse_probe = {
    "precision_once_s": _steady("cse_precision_once"),
    "precision_thrice_s": _steady("cse_precision_thrice"),
    "precision_thrice_over_once": _steady("cse_precision_thrice") / _steady("cse_precision_once"),
    "magnifications_once_s": _steady("cse_magnifications_once"),
    "magnifications_twice_s": _steady("cse_magnifications_twice"),
    "magnifications_twice_over_once": _steady("cse_magnifications_twice")
    / _steady("cse_magnifications_once"),
    "reading": "ratio ~1 => XLA merged the repeated evaluations inside one JIT; "
    "ratio ~k => each repeat is paid",
}

print("\n--- Numerical controls ---")
batch_size = 2


def _batched(params):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)), params
    )


vmap_solved = block(jax.jit(jax.vmap(full_solved))(_batched(params_solved)))
vmap_plain = block(jax.jit(jax.vmap(full_plain))(_batched(params_plain)))
np.testing.assert_allclose(np.asarray(vmap_solved), float(solved_result), rtol=1.0e-10)
np.testing.assert_allclose(np.asarray(vmap_plain), float(plain_result), rtol=1.0e-10)

for name, expected, value in (
    ("EXPECTED_LOG_LIKELIHOOD_SOLVED", EXPECTED_LOG_LIKELIHOOD_SOLVED, float(solved_result)),
    ("EXPECTED_LOG_LIKELIHOOD_PLAIN", EXPECTED_LOG_LIKELIHOOD_PLAIN, float(plain_result)),
):
    if expected is None:
        print(f"  {name} unset; capture: {name} = {value!r}")
    else:
        np.testing.assert_allclose(value, expected, rtol=LITERAL_RTOL)
literals_checked = (
    EXPECTED_LOG_LIKELIHOOD_SOLVED is not None and EXPECTED_LOG_LIKELIHOOD_PLAIN is not None
)

# An unregistered model makes jax.grad silently all-zero: assert non-zero.
gradient = block(jax.grad(full_solved)(params_solved))
gradient_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(gradient)]
gradient_finite = all(np.isfinite(leaf).all() for leaf in gradient_leaves)
if not gradient_finite:
    raise AssertionError("solved source-plane gradient contains non-finite values")
gradient_l2 = float(np.sqrt(sum(float(np.vdot(leaf, leaf)) for leaf in gradient_leaves)))
if not gradient_l2 > 0.0:
    raise AssertionError("solved source-plane gradient is identically zero (model unregistered?)")
print(f"  grad L2 = {gradient_l2:.6e} over {len(gradient_leaves)} leaves")

print("\n--- Gradient cost ---")
_profile(jax.value_and_grad(full_solved), "value_and_grad_solved", params_solved)
full_solved_time = _steady("full_solved")
gradient_cost = {
    "grad_per_call_s": _steady("value_and_grad_solved"),
    "forward_per_call_s": full_solved_time,
    "grad_over_forward": _steady("value_and_grad_solved") / full_solved_time,
    "note": "jit(jax.value_and_grad(full_solved)); value and gradient in one call",
}

al_version = al.__version__
config_name = _cli.config_name or (
    "local_cpu_fp64" if jax.default_backend() == "cpu" else "unlabelled_device_fp64"
)
free_parameters = {
    "solved": int(model_solved.prior_count),
    "plain_control": int(model_plain.prior_count),
}
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
    # Sub-millisecond rows are sensitive to host contention: record it.
    "host_load_average": {"start": list(LOAD_AVERAGE_START), "end": list(os.getloadavg())},
    "precision": {
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "mixed_precision_requested": bool(_cli.use_mixed_precision),
        "note": "The source-plane fit has no Settings mixed-precision switch.",
    },
    "configuration": {
        "likelihood": "source_plane_solved (FitPositionsSourceSolved + PointSolved)",
        "plain_control": "source_plane (FitPositionsSource + PointFlux)",
        "fit_positions_cls": {
            "solved": al.FitPositionsSourceSolved.__name__,
            "plain_control": al.FitPositionsSource.__name__,
        },
        "weighting": {
            "solved": al.FitPositionsSourceSolved.weighting,
            "plain_control": al.FitPositionsSource.weighting,
        },
        "hessian_method": "jacfwd",
        "solver": None,
        "n_positions": int(dataset.positions.shape[0]),
        "positions_noise_sigma": float(dataset.positions_noise_map[0]),
        "free_parameters": free_parameters,
        "lens_redshift": 0.5,
        "source_redshift": 1.0,
        "n_repeats": N_REPEATS,
    },
    "steps": steps,
    "prefix_steady_per_call_s": dict(prefix_order),
    "total_step_by_step": float(sum(steps.values())),
    "plain_control_steps": plain_steps,
    "plain_control_prefix_steady_per_call_s": dict(plain_prefix_order),
    "fused_full_solved_s": float(full_solved_time),
    "fused_full_plain_control_s": float(_steady("full_plain_control")),
    "dispatch_floor": dispatch_floor,
    "cse_probe": cse_probe,
    "gradient_cost": gradient_cost,
    "jit_phases": jit_records,
    "likelihoods": {
        "solved_eager": solved_eager,
        "solved_jit": float(solved_result),
        "plain_eager": plain_eager,
        "plain_jit": float(plain_result),
        "expected_solved": EXPECTED_LOG_LIKELIHOOD_SOLVED,
        "expected_plain": EXPECTED_LOG_LIKELIHOOD_PLAIN,
        "literals_checked": literals_checked,
        "literal_rtol": LITERAL_RTOL,
    },
    "numerical_controls": {
        "eager_vs_jit_rtol": 1.0e-4,
        "vmap_batch": batch_size,
        "vmap_values_solved": np.asarray(vmap_solved).tolist(),
        "vmap_values_plain": np.asarray(vmap_plain).tolist(),
        "gradient_all_finite": gradient_finite,
        "gradient_l2": gradient_l2,
        "gradient_leaves": len(gradient_leaves),
    },
    "methodology": {
        "prefix_rows": "successive differences of cumulative JIT prefixes (per-call "
        "medians over n_repeats); each prefix returns the sum of every intermediate "
        "reached so far",
        "timing_statistic": "median_per_call_s from jit_phases; steady_per_call_s (mean, "
        "from jit_profile) is also recorded",
        "negative_rows": "retained; XLA fusion, CSE and timing noise can move work across "
        "boundaries (telescoping caveat: the step rows sum to the fused time by "
        "construction, individual rows are not independent stage costs)",
        "repeated_evaluations": "fit properties are uncached: the plain lane evaluates "
        "magnifications_at_positions twice per likelihood (chi-squared map + noise "
        "normalisation); the solved lane rebuilds the precision tensor three times "
        "(solved centre, chi-squared map, marginalisation term) and _beta_hat twice. "
        "Whether XLA CSE merges them is phase-2 hypothesis #1; see cse_probe and the "
        "fused-vs-prefix rows for the observed evidence.",
        "solver": "solver=None; FitPositionsSource never invokes a PointSolver",
        "eager_hessian": "use_jax=False uses the finite-difference Hessian, so eager vs "
        "JIT agree to rtol 1e-4, not bit-for-bit",
        "authoritative_runtime": "fused_full_solved_s",
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_ROOT / "results" / "breakdown" / "point_source_source",
    default_basename=f"source_plane_breakdown_{INSTRUMENT}_v{al_version}",
    cell="source_plane",
)
dict_path.write_text(json.dumps(summary, indent=2))

labels = list(steps)
times = [steps[label] for label in labels]
prefix_labels = [label for label, _ in prefix_order] + ["Fused solved likelihood"]
prefix_times = [value for _, value in prefix_order] + [full_solved_time]
fig, (prefix_ax, difference_ax) = plt.subplots(ncols=2, figsize=(18, 6), constrained_layout=True)
prefix_ax.barh(
    range(len(prefix_labels)), [t * 1000 for t in prefix_times], color="#4C72B0", edgecolor="white"
)
prefix_ax.set_yticks(range(len(prefix_labels)))
prefix_ax.set_yticklabels(prefix_labels, fontsize=8)
prefix_ax.invert_yaxis()
prefix_ax.set_xlabel("Absolute cumulative-prefix steady-state time [ms]")
prefix_ax.set_title("Absolute prefixes (compare these across runs)")

colours = ["#C44E52" if value < 0.0 else "#4C72B0" for value in times]
difference_ax.barh(range(len(labels)), [t * 1000 for t in times], color=colours, edgecolor="white")
difference_ax.set_yticks(range(len(labels)))
difference_ax.set_yticklabels(labels, fontsize=8)
difference_ax.invert_yaxis()
difference_ax.axvline(0.0, color="black", linewidth=0.8)
difference_ax.set_xlabel("Successive-difference steady-state time [ms]")
difference_ax.set_title("Signed differences (fusion/CSE/noise retained)")
fig.suptitle(
    f"Point-source source-plane solved breakdown ({config_name}); "
    f"fused solved={full_solved_time * 1000:.3f} ms, "
    f"plain={summary['fused_full_plain_control_s'] * 1000:.3f} ms"
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\n" + "=" * 78)
print("POINT-SOURCE SOURCE-PLANE BREAKDOWN (solved lane)")
print("=" * 78)
for label, value in steps.items():
    print(f"  {label:<62} {value * 1000:9.4f} ms")
print("-" * 78)
print(f"  {'Fused solved likelihood':<62} {full_solved_time * 1000:9.4f} ms")
print("\nPlain control lane")
for label, value in plain_steps.items():
    print(f"  {label:<62} {value * 1000:9.4f} ms")
print(f"  {'Fused plain control':<62} {summary['fused_full_plain_control_s'] * 1000:9.4f} ms")
print(
    f"\n  Dispatch floor: {json.dumps({k: v for k, v in dispatch_floor.items() if k != 'reading'})}"
)
print(f"  CSE probe: {json.dumps({k: v for k, v in cse_probe.items() if k != 'reading'})}")
print(f"  value_and_grad / forward = {gradient_cost['grad_over_forward']:.2f}")
print(f"  Results JSON: {dict_path}")
print(f"  Results PNG:  {chart_path}")
timer.summary()
