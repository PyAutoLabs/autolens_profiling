"""
JAX Profiling: Point-Source Likelihood — Image-Plane Chi-Squared (Solved)
===========================================================================

Profiles ``AnalysisPoint.log_likelihood_function`` for a lensed point-source
``PointDataset`` using the **analytically-solved image-plane** chi-squared
(``al.FitPositionsImagePairAllSolved``), the sibling of ``image_plane.py``
introduced in issue #657 phase 3.

Instead of sampling the source's centre as 2 free model parameters, this
variant pairs a zero-parameter ``al.ps.PointSolved`` profile with a
``*Solved`` fit class, which analytically solves for the source-plane centre
β* (tensor-weighted precision, Lombardi 2024 arXiv:2406.15280 §5.1) from the
current tracer before handing that centre to the same ``PointSolver``
forward-solve used by the non-solved image-plane path. The image-plane
pairing chi-squared itself is unchanged from ``image_plane.py`` — only the
source of the centre driving the solve differs, and using a centre-free
extension in the image plane at all is **not** from the paper: it mirrors
glafic's source-position optimization convention, so attribute it as an
extension, not a paper result (see ``autolens/point/fit/solved.py`` module
docstring, which makes this attribution split explicit).

Parameter-count delta
----------------------

Swapping ``al.ps.PointFlux`` (used by ``image_plane.py``: 2 centre + 1 flux
= 3 params) for ``al.ps.PointSolved`` (0 params) drops the model from 8 to 5
free parameters here — a delta of 3, not the 2 you would expect from the
paper's centre-only framing. 2 of those 3 are the analytically-solved
source-plane centre; the third is the flux parameter, which this
positions-only harness (``point_dataset_positions_only.json``, no flux
data fitted) never actually used regardless of PointFlux vs PointSolved —
dropping it is a side effect of PointSolved having no flux attribute at
all, not part of the phase-3 analytic-solve story.

Second steady-state entry
--------------------------

In addition to the primary ``FitPositionsImagePairAllSolved`` full-pipeline
JIT/vmap timings (mirroring ``image_plane.py`` exactly), this script also
times ``al.FitPositionsImagePairRepeatSolved`` at the same compiled-input
shape (identical ``params_tree`` / dataset / solver) as a second
steady-state entry — a modest addition since only ``fit_positions_cls``
differs between the two ``AnalysisPoint`` instances. This second entry is
timing-only: it is not part of the vmap / memory / regression-pin machinery
below, which stays scoped to the primary AllSolved fit class.

Pytree-native parameter inputs
------------------------------

As in ``image_plane.py``, this script uses ``af.ModelInstance`` as the JIT
input via PyAutoFit's opt-in pytree registration
(``autofit.jax.register_model``, PRs #1220 / #1221 / #1222).

Three-tier numerical assertions
-------------------------------

1. **eager ≡ JIT**: numpy-path log-likelihood matches single-JIT result.
2. **JIT ≡ vmap**: every entry of the batched vmap output matches the
   single-JIT result.
3. **regression constant**: hardcoded
   ``EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE_SOLVED`` guards against silent
   drift in the underlying solver / chi-squared / analytic-solve stack.
   Captured during authoring against the dataset already committed in this
   worktree (``noise_seed=1`` in ``simulators/point_source.py``); refresh it
   the same way ``image_plane.py`` documents if the dataset or priors
   change.
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


import json

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (Phase 5 / CI lint smoke).
# Verifies the import graph + module-level setup succeeded without running
# the full profiling pipeline. Skipped entirely when the env var is unset.
import os as _smoke_os
import subprocess
import sys
import sys as _smoke_sys
import time
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

# Sweep-driver CLI args (--config-name / --output-dir / --use-mixed-precision).
# Tolerates extra/unknown args via parse_known_args inside the helper.
sys.path.insert(0, str(_profiling_root()))
from simulators.point_source import INSTRUMENTS  # noqa: E402
from vram import (  # noqa: E402
    probe_vmap_memory,
    recommend_batch_size,
    resolve_vmap_batch,
    write_probe_json,
)

from _profile_cli import (
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,  # noqa: E402
    resolve_output_paths,
)

_cli = parse_profile_cli()

matplotlib.use("Agg")
import autofit as af
import autolens as al
import matplotlib.pyplot as plt
from autofit.jax import register_model as _register_model_pytrees

# ---------------------------------------------------------------------------
# Profiling helpers (mirrors imaging/mge.py and image_plane.py)
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
instrument = "simple"


# ===================================================================
# PART A — Setup
# ===================================================================

print(f"\n--- Dataset loading [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
_workspace_root = _profiling_root()
dataset_path = Path("dataset") / "point_source" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="point_source",
    instrument=instrument,
    workspace_root=_workspace_root,
)

with timer.section("dataset_load"):
    dataset = al.from_json(
        file_path=dataset_path / "point_dataset_positions_only.json",
    )

n_observed_positions = dataset.positions.shape[0]
positions_noise_sigma = float(dataset.positions_noise_map[0])

print("\n--- Point solver ---")

with timer.section("solver_build"):
    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)
    solver = al.PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )

print("\n--- Model construction (PointSolved — 0 source params) ---")

with timer.section("model_build"):
    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality. Prior means MUST match the simulator's truth
    # values exactly, otherwise the PointSolver finds fewer image-plane
    # positions than the dataset contains and chi² explodes.
    #
    # Simulator truth (see autolens_workspace_developer/jax_profiling/
    # dataset_setup/point_source.py):
    #   Isothermal at centre=(0, 0), einstein_radius=1.6,
    #   ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45°)
    #            ≈ (0.0526316, 0.0)
    #   source point_0.centre = (0.07, 0.07) — NOT used here: `PointSolved`
    #   has no `centre` at all, so there is nothing to prior-align on the
    #   source side. The analytic solve (see module docstring) recovers the
    #   source-plane centre from the tracer + observed positions instead.
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05263158, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    # `PointSolved` has no `centre` / `flux` attributes, so no priors to set
    # here — that is the whole point of the analytic solve.
    point_0 = af.Model(al.ps.PointSolved)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(
    "  (was 8 with al.ps.PointFlux in image_plane.py: 5 mass + 2 centre + 1 flux; "
    f"PointSolved drops all 3 source params -> {model.total_free_parameters} mass-only. "
    "2 of the 3 are the analytically-solved source centre (paper Sec 5.1 beta*); the "
    "3rd is the flux param this positions-only dataset never fits either way.)"
)

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)


# ---------------------------------------------------------------------------
# Eager baseline — full FitPointDataset (image-plane chi-squared, solved)
# ---------------------------------------------------------------------------

print("\n--- Eager FitPointDataset (image-plane, solved) ---")

analysis_eager = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
    use_jax=False,
)

with timer.section("fit_eager"):
    fit_eager = analysis_eager.fit_from(instance=instance)
    log_likelihood_ref = float(fit_eager.log_likelihood)
    figure_of_merit_ref = float(fit_eager.figure_of_merit)

n_eager_repeats = 10
with timer.section(f"eager_log_likelihood_x{n_eager_repeats}"):
    for _ in range(n_eager_repeats):
        analysis_eager.log_likelihood_function(instance=instance)
eager_per_call = timer.records[-1][1] / n_eager_repeats

print(f"  log_likelihood   = {log_likelihood_ref}")
print(f"  figure_of_merit  = {figure_of_merit_ref}")
print(f"  eager per-call   = {eager_per_call:.6f} s")


# ===================================================================
# PART B — Full-pipeline JIT (FitPositionsImagePairAllSolved)
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT (image-plane, solved)")
print("=" * 70)

analysis_jax = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
    use_jax=True,
)


def full_pipeline_from_params(params_tree):
    return analysis_jax.log_likelihood_function(instance=params_tree)


_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10
print(f"  full log_likelihood = {full_result}")


# ===================================================================
# PART B.2 — Second steady-state entry: FitPositionsImagePairRepeatSolved
# ===================================================================
#
# Same compiled-input shape as PART B (identical params_tree / dataset /
# solver, only fit_positions_cls differs) — a modest addition on top of the
# AllSolved profiling above, since the phase-3 economics question also cares
# about the nearest-pair-with-repeats scheme, not just the all-pairs one.

print("\n" + "=" * 70)
print("SECOND STEADY-STATE (image-plane, solved): FitPositionsImagePairRepeatSolved")
print("=" * 70)

analysis_jax_repeat = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeatSolved,
    use_jax=True,
)


def repeat_pipeline_from_params(params_tree):
    return analysis_jax_repeat.log_likelihood_function(instance=params_tree)


_, repeat_result = jit_profile(repeat_pipeline_from_params, "repeat_fit_class", params_tree)
repeat_fit_class_per_call = timer.records[-1][1] / 10
print(f"  FitPositionsImagePairRepeatSolved log_likelihood = {repeat_result}")


# ===================================================================
# PART B.5 — vmap-probe mode (early exit)
# ===================================================================
#
# When ``--vmap-probe`` is set the script JIT-vmaps the pipeline at the
# configured batch sizes, reads ``compiled.memory_analysis()``, writes a
# ``vmap_probe.json`` with the recommended A100 batch_size, and exits
# before the full vmap timing loop. See ``vram/README.md`` for methodology.
# Scoped to the primary AllSolved fit class only — the repeat-class timing
# above is a bonus entry, not part of this machinery.

if _cli.vmap_probe:
    probe = probe_vmap_memory(
        full_pipeline_from_params,
        params_tree,
        batch_sizes=(1, 4, 16),
        dataset="point_source",
        model="image_plane_solved",
        instrument=instrument,
    )
    recommended = recommend_batch_size(probe)
    probe_path = (
        _cli.output_dir
        or (_workspace_root / "results" / "runtime" / "point_source_image" / "image_plane_solved")
    ) / (
        "vmap_probe_image_plane_solved_sparse.json"
        if _cli.use_sparse_operator
        else "vmap_probe_image_plane_solved.json"
    )
    write_probe_json(probe, recommended, probe_path)
    print(f"\n  vmap_probe samples: {probe.samples}")
    print(f"  per_replica:        {probe.per_replica_mb:.1f} MB / replica")
    print(f"  recommended batch:  {recommended}")
    print(f"  written to:         {probe_path}")
    sys.exit(0)

# ===================================================================
# PART C — vmap over the full pipeline
# ===================================================================

print("\n--- vmap batched evaluation ---")

_batch_resolved, _batch_source = resolve_vmap_batch(
    "point_source",
    "image_plane_solved",
    instrument,
    output_dir=_cli.output_dir
    or (_workspace_root / "results" / "runtime" / "point_source_image" / "image_plane_solved"),
    path="sparse" if _cli.use_sparse_operator else "dense",
    backend=jax.default_backend(),
)
print(f"  vmap batch_size: {_batch_resolved} (source: {_batch_source})")
batch_size = _batch_resolved or 3

batched_params = jax.tree_util.tree_map(
    lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
    params_tree,
)

vmapped_full = jax.jit(jax.vmap(full_pipeline_from_params))

with timer.section("vmap_first_call"):
    result_vmap = vmapped_full(batched_params)
    block(result_vmap)

n_vmap_repeats = 10
with timer.section(f"vmap_steady_x{n_vmap_repeats}"):
    for _ in range(n_vmap_repeats):
        result_vmap = vmapped_full(batched_params)
        block(result_vmap)

vmap_batch_time = timer.records[-1][1] / n_vmap_repeats
vmap_per_call = vmap_batch_time / batch_size
vmap_speedup = full_pipeline_per_call / vmap_per_call

print(f"  batch results = {result_vmap}")
print(f"  vmap batch of {batch_size}:   {vmap_batch_time:.6f} s")
print(f"  vmap per call:         {vmap_per_call:.6f} s")
print(f"  single JIT per call:   {full_pipeline_per_call:.6f} s")
print(f"  vmap speedup:          {vmap_speedup:.1f}x faster per likelihood")


# ===================================================================
# PART D — Three-tier numerical assertions
# ===================================================================
#
# Tier 1: eager (NumPy path) ≡ single JIT
# Tier 2: single JIT ≡ every entry of vmap output
# Tier 3: hardcoded regression constant (deterministic via seeded simulator)

np.testing.assert_allclose(
    log_likelihood_ref,
    float(full_result),
    rtol=1e-4,
    err_msg=(
        f"point_source/image_plane_solved: eager vs JIT mismatch — "
        f"eager={log_likelihood_ref} vs JIT={float(full_result)}"
    ),
)

np.testing.assert_allclose(
    np.array(result_vmap),
    float(full_result),
    rtol=1e-4,
    err_msg="point_source/image_plane_solved: JIT vs vmap mismatch",
)


# ===================================================================
# PART E — Static memory analysis
# ===================================================================

print("\n--- Static memory analysis ---")

lowered_batched = vmapped_full.lower(batched_params)
compiled_batched = lowered_batched.compile()
mem = compiled_batched.memory_analysis()
print(f"  Output size:  {mem.output_size_in_bytes / 1024**2:.3f} MB")
print(f"  Temp size:    {mem.temp_size_in_bytes / 1024**2:.3f} MB")
print(f"  Total:        {(mem.output_size_in_bytes + mem.temp_size_in_bytes) / 1024**2:.3f} MB")


# ===================================================================
# Summary + outputs
# ===================================================================

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD SUMMARY — POINT SOURCE IMAGE-PLANE SOLVED — v{al_version}")
print("=" * 70)
print(f"  Dataset:                    {instrument}")
print(f"  Observed image positions:   {n_observed_positions}")
print(f"  Position noise sigma:       {positions_noise_sigma}")
print(
    f"  Free parameters:            {model.total_free_parameters}  "
    "(image_plane.py: 8; delta=-3 -> -2 solved centre, -1 unused flux)"
)
print(
    "  fit_positions_cls:          FitPositionsImagePairAllSolved (image-plane chi-squared, solved centre)"
)
print("-" * 70)
print(f"  Eager full likelihood:      {eager_per_call:.6f} s/call  ({log_likelihood_ref:.6f})")
print(f"  Full pipeline (JIT):        {full_pipeline_per_call:.6f} s/call")
print(
    f"  Repeat fit class (JIT):     {repeat_fit_class_per_call:.6f} s/call  "
    f"(FitPositionsImagePairRepeatSolved, {float(repeat_result):.6f})"
)
print(f"  vmap per-call (batch={batch_size}):    {vmap_per_call:.6f} s")
print(f"  vmap speedup vs single JIT:           {vmap_speedup:.1f}x")
print("=" * 70)

likelihood_summary = {
    "autolens_version": al_version,
    "dataset": instrument,
    "fit_positions_cls": "FitPositionsImagePairAllSolved",
    "configuration": {
        "observed_image_positions": int(n_observed_positions),
        "positions_noise_sigma": positions_noise_sigma,
        "free_parameters": int(model.total_free_parameters),
        "free_parameters_image_plane_plain": 8,
        "free_parameters_delta": int(model.total_free_parameters) - 8,
        "point_source_profile": "PointSolved",
    },
    "eager_per_call": eager_per_call,
    "eager_log_likelihood": log_likelihood_ref,
    "full_pipeline_single_jit": full_pipeline_per_call,
    "full_pipeline_log_likelihood": float(full_result),
    "repeat_fit_class_steady_state": {
        "fit_positions_cls": "FitPositionsImagePairRepeatSolved",
        "per_call": repeat_fit_class_per_call,
        "log_likelihood": float(repeat_result),
    },
    "vmap": {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    },
}

results_dir = _workspace_root / "results" / "runtime" / "point_source_image" / "image_plane_solved"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"image_plane_solved_summary_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Bar chart ---

labels = [
    "Eager full likelihood",
    "Full pipeline (JIT, AllSolved)",
    "Repeat fit class (JIT, RepeatSolved)",
    f"vmap per-call (batch={batch_size})",
]
times = [eager_per_call, full_pipeline_per_call, repeat_fit_class_per_call, vmap_per_call]
colors = ["#8172B3", "#C44E52", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(10, 4.5))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)

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
    "Point-Source Likelihood — Image-Plane Chi-Squared (Solved)",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {n_observed_positions} positions  |  "
    f"{model.total_free_parameters} free params (was 8)  |  "
    f"vmap speedup: {vmap_speedup:.1f}x",
    fontsize=9,
)
ax.margins(x=0.20)
fig.tight_layout()

chart_path = results_dir / f"image_plane_solved_summary_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Tier 3: regression assertion — deterministic via seeded simulator
# ===================================================================
#
# Simulator truth parameters + seeded noise (noise_seed=1 in
# simulators/point_source.py) make the image-plane log-likelihood
# deterministic; PointSolved's analytic centre solve is itself a
# deterministic function of the tracer + observed positions, so eager, JIT,
# and vmap all agree to float64. Constant captured 2026-07-30 during
# authoring of this script (issue #657 phase 3) against the dataset already
# committed in this worktree (v2026.7.23.1); refresh it the same way
# image_plane.py documents if the dataset or mass priors change.
_pinned_drift: list = []
_pinned_expected = None

EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE_SOLVED = 7.743201200876817
_pinned_expected = EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE_SOLVED

_rec = check_pinned(log_likelihood_ref, _pinned_expected, label="eager", rtol=1e-4)
if _rec is not None:
    _pinned_drift.append(_rec)
_rec = check_pinned(float(full_result), _pinned_expected, label="JIT", rtol=1e-4)
if _rec is not None:
    _pinned_drift.append(_rec)
_rec = check_pinned(np.array(result_vmap), _pinned_expected, label="vmap", rtol=1e-4)
if _rec is not None:
    _pinned_drift.append(_rec)

timer.summary()


# Pinned-value outcome -> result JSON: profiling records and flags drift,
# never adjudicates library correctness (autolens_workspace_test's remit;
# boundary rule in results/notes/design_lock_in.md). PyAutoHeart's vitals
# scan reads the pinned_drift field.
record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")
