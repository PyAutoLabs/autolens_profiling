"""
JAX Profiling: MGE Imaging Likelihood (Step-by-Step)
=====================================================

Profiles each step of the JAX likelihood function for an imaging dataset where
the lens galaxy's light is modelled with a multi-Gaussian expansion (MGE).

Rather than timing the whole likelihood as a single JIT-compiled block (which
hides internal bottlenecks), this script JIT-compiles and times each step of
the pipeline individually:

1. Instance from parameter vector
2. Build Tracer
3. Ray-trace grids through the lens
4. Compute mapping matrix (per-profile images before PSF)
5. Compute blurred mapping matrix (PSF convolution)
6. Compute data vector  (D)
7. Compute curvature matrix  (F)
8. Reconstruction via positive-only NNLS
9. Map reconstruction back to image plane
10. Chi-squared and log likelihood

Note: because the MGE model uses only linear light profiles (lp_linear),
there is no non-linear blurred image or profile-subtracted image step.

Caveat: XLA may fuse operations differently when compiled as one program vs
separate pieces, so per-step timings are approximate. They are still useful
for identifying which step dominates.

All JAX timings use `block_until_ready()` to force synchronous measurement.

Pytree-native parameter inputs (recommended pattern)
----------------------------------------------------

This script uses ``af.ModelInstance`` as the JIT input via PyAutoFit's
opt-in pytree registration (``autofit.jax.register_model(model)``). The
JIT'd closures consume the instance directly, so:

* ``model.instance_from_vector`` is no longer called inside the JIT trace —
  parameter unpacking happens once at registration time and JAX walks the
  pytree on every call.
* Parameter identity is preserved through ``jax.jit`` and ``jax.vmap``;
  XLA cache keys reflect the structured pytree, not a flat vector shape.
* ``vmap`` batching is ``jax.tree_util.tree_map`` over the instance leaves
  — callers no longer have to stack a ``(batch, N)`` array.

New profiling scripts should follow this pattern. The flat-vector path in
``Fitness.call`` / ``model.instance_from_vector(..., xp=jnp)`` remains the
production likelihood entry point and is intentionally untouched here.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al
import autoarray as aa
from autofit.jax import register_model as _register_model_pytrees

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------


# AUTOLENS_PROFILING_SMOKE=1 short-circuit (Phase 5 / CI lint smoke).
# Verifies the import graph + module-level setup succeeded without running
# the full profiling pipeline. Skipped entirely when the env var is unset.
import os as _smoke_os
import sys as _smoke_sys
if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

# Sweep-driver CLI args (--config-name / --output-dir / --use-mixed-precision).
# Tolerates extra/unknown args via parse_known_args inside the helper.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _profile_cli import (  # noqa: E402
    parse_profile_cli,
    device_info_dict,
    resolve_output_paths,
    auto_simulate_if_missing,
)
from simulators.imaging import INSTRUMENTS  # noqa: E402
from vram import (  # noqa: E402
    probe_vmap_memory,
    recommend_batch_size,
    vmap_batch_for,
    write_probe_json,
)
_cli = parse_profile_cli()

instrument = _cli.instrument or "hst"  # default; override via --instrument


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        """Context manager that records wall-clock time for *label*."""
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
    """Call block_until_ready if available (JAX arrays)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time first call and steady-state average.

    Returns the compiled function and its result.
    """
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

# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[1]
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
    dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    if _cli.use_sparse_operator:
        # The pure-MGE-source cell uses LinearObjFuncList for every linear
        # obj, so the inversion factory's all-LinearObjFuncList short-circuit
        # fires and ``InversionImagingMapping`` is still chosen even after
        # ``apply_sparse_operator`` attaches a sparse_operator to the dataset.
        # We still call ``apply_sparse_operator`` for parity with the pix /
        # Delaunay cells so the harness-level cost (kernel construction,
        # serialisation) is included in the timing — but the per-eval cost
        # remains dense. The JSON's ``inversion_path`` records what the flag
        # asked for; downstream synthesis cross-references with the
        # ``InversionImagingSparse``-vs-``InversionImagingMapping`` factory
        # decision when interpreting the MGE row.
        dataset = dataset.apply_sparse_operator(
            use_jax=True, show_progress=False
        )

# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality.
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    # Mass and shear fixed to simulator truth (not GaussianPrior) because
    # tracing GaussianPrior-backed mass params through this script's
    # ``mapping_matrix_from_params`` JIT trigger a pre-existing xp=np/jnp
    # propagation bug in autogalaxy/profiles/mass/total/isothermal.py:108
    # (Isothermal.deflections_yx_2d_from called with xp=np on traced inputs).
    # The bug is specific to this script's MGE-lens-light + over-sampled-LP
    # combination; the likelihood-only imaging/mge_gradients.py uses the
    # same pattern without the failing JIT and works under Option A.
    mass = af.Model(al.mp.Isothermal)
    mass.centre = (0.0, 0.0)
    mass.einstein_radius = 1.6
    mass.ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = 0.05
    shear.gamma_2 = 0.05

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

# ---------------------------------------------------------------------------
# 3. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

# Register every concrete `model.cls` (Galaxy, profile classes, ModelInstance,
# Collection, …) with `jax.tree_util` so the instance can cross JIT/vmap
# boundaries directly. This must happen AFTER the model is built, because
# registration walks the model's class graph.
with timer.section("register_pytrees"):
    _register_model_pytrees(model)

# JIT input: the instance itself, with all parameter leaves promoted to JAX
# arrays. We keep `instance` (the eager NumPy version) around for any
# non-JIT setup that needs to read parameter values directly.
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

tracer = al.Tracer(galaxies=list(instance.galaxies))

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# Key configuration that dictates run time
# ---------------------------------------------------------------------------

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]
n_linear_gaussians = len(tracer.cls_list_from(cls=al.lp_linear.LightProfileLinear))

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Linear Gaussians:        {n_linear_gaussians}")

# ---------------------------------------------------------------------------
# 4. Full-pipeline reference (FitImaging) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitImaging (eager baseline) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    log_evidence_ref = fit.figure_of_merit
    log_likelihood_ref = fit.log_likelihood

print(f"  figure_of_merit (log_evidence) = {log_evidence_ref}")
print(f"  log_likelihood                 = {log_likelihood_ref}")

# ===================================================================
# PART C — Full-pipeline JIT for comparison
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT (for comparison)")
print("=" * 70)

# Build the analysis with ``use_jax=True`` so its ``log_likelihood_function``
# threads ``xp=jnp`` through every internal call (border relocation, profile
# evaluation, inversion, etc.). This is the same wiring that ``Fitness.call``
# uses in production — we just feed it our pytree-native instance directly
# instead of going through ``model.instance_from_vector(parameters, xp=jnp)``.
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

def full_pipeline_from_params(params_tree):
    """Full likelihood from a pytree-shaped ``ModelInstance``.

    No flat-vector unpacking inside the trace — the instance crosses the JIT
    boundary directly, with constants (redshifts, etc.) kept static via the
    ``aux_data`` partition set up by ``autofit.jax.register_model``.
    """
    return analysis.log_likelihood_function(instance=params_tree)

_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_likelihood = {full_result}")

# ===================================================================
# PART C.5 — vmap-probe mode (early exit)
# ===================================================================
#
# When ``--vmap-probe`` is set the script JIT-vmaps the pipeline at two
# batch sizes, reads ``compiled.memory_analysis()`` for each, and writes a
# ``vmap_probe.json`` with the recommended A100 batch_size — then exits
# before the full vmap timing loop. See ``vram/README.md`` for methodology.

if _cli.vmap_probe:
    # mge has cheap XLA compile (~10s); use multi-point fit to catch
    # any rematerialisation non-linearity at the (1, 4, 16) regime.
    probe = probe_vmap_memory(
        full_pipeline_from_params,
        params_tree,
        batch_sizes=(1, 4, 16),
        dataset="imaging",
        model="mge",
        instrument=instrument,
    )
    recommended = recommend_batch_size(probe)
    probe_path = (
        (_cli.output_dir or (_workspace_root / "results" / "likelihood" / "imaging"))
        / "vmap_probe.json"
    )
    write_probe_json(probe, recommended, probe_path)
    print(f"\n  vmap_probe samples: {probe.samples}")
    print(f"  per_replica:        {probe.per_replica_mb:.1f} MB / replica")
    print(f"  recommended batch:  {recommended}")
    print(f"  written to:         {probe_path}")
    sys.exit(0)

# ===================================================================
# PART D — vmap + correctness
# ===================================================================

print("\n--- vmap batched evaluation ---")

batch_size = vmap_batch_for("imaging", "mge", instrument) or 3

# Build the batched pytree: every leaf gets a fresh leading batch axis. No
# flat-vector reshaping required — JAX walks the pytree via the registration
# we added in PART A.
parameters = jax.tree_util.tree_map(
    lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
    params_tree,
)

vmapped_full = jax.jit(jax.vmap(full_pipeline_from_params))

with timer.section("vmap_first_call"):
    result_vmap = vmapped_full(parameters)
    block(result_vmap)

n_vmap_repeats = 10
with timer.section(f"vmap_steady_x{n_vmap_repeats}"):
    for _ in range(n_vmap_repeats):
        result_vmap = vmapped_full(parameters)
        block(result_vmap)

vmap_batch_time = timer.records[-1][1] / n_vmap_repeats
vmap_per_call = vmap_batch_time / batch_size
vmap_speedup = full_pipeline_per_call / vmap_per_call

print(f"  batch results = {result_vmap}")
print(f"  vmap batch of {batch_size}:   {vmap_batch_time:.6f} s")
print(f"  vmap per call:         {vmap_per_call:.6f} s")
print(f"  single JIT per call:   {full_pipeline_per_call:.6f} s")
print(f"  vmap speedup:          {vmap_speedup:.1f}x faster per likelihood")

np.testing.assert_allclose(
    np.array(result_vmap),
    float(full_result),
    rtol=1e-4,
    err_msg="mge: JAX vmap likelihood mismatch",
)
print("  Correctness check PASSED")

# ===================================================================
# PART E — Static memory analysis
# ===================================================================

print("\n--- Static memory analysis ---")

lowered_batched = vmapped_full.lower(parameters)
compiled_batched = lowered_batched.compile()

memory_analysis = compiled_batched.memory_analysis()
print(f"  Output size:  {memory_analysis.output_size_in_bytes / 1024**2:.3f} MB")
print(f"  Temp size:    {memory_analysis.temp_size_in_bytes / 1024**2:.3f} MB")
print(
    f"  Total:        "
    f"{(memory_analysis.output_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024**2:.3f} MB"
)


# ===================================================================
# JAX Likelihood Function Summary
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Instrument:            {instrument}")
print(f"  Pixel scale:           {pixel_scale} arcsec/pixel")
print(f"  Mask radius:           {mask_radius} arcsec")
print(f"  Image pixels (masked): {n_image_pixels}")
print(f"  Over-sampled pixels:   {n_over_sampled_pixels}")
print(f"  Linear Gaussians:      {n_linear_gaussians}")
print("-" * 70)
print(f"      {'Full pipeline (single JIT)':<30}  {full_pipeline_per_call:>12.6f} s")
print(f"      {f'vmap batch={batch_size} (per call)':<30}  {vmap_per_call:>12.6f} s")
print(f"      {f'vmap speedup vs single JIT':<30}  {vmap_speedup:>11.1f}x")
print("=" * 70)

# --- Save results dictionary ---

likelihood_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "linear_gaussians": int(n_linear_gaussians),
        "inversion_path": "sparse" if _cli.use_sparse_operator else "dense",
    },
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    },
    "memory_mb": {
        "output": memory_analysis.output_size_in_bytes / 1024**2,
        "temp": memory_analysis.temp_size_in_bytes / 1024**2,
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "likelihood" / "imaging",
    default_basename=f"mge_likelihood_summary_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")
print(f"  Bar chart path:        {chart_path} (no per-step chart in runtime variant)")


# ===================================================================
# Regression assertion — realistic-scale deterministic likelihood
# ===================================================================
#
# Simulator truth parameters (mass + shear fixed; MGE bulges free around
# default centre/ell_comps priors) put the evaluation point at the
# physically-meaningful truth operating point. Eager, JIT, and vmap all
# agree to ~1e-11 precision.
EXPECTED_LOG_LIKELIHOOD_HST = 27379.38890685539

np.testing.assert_allclose(
    log_likelihood_ref,
    EXPECTED_LOG_LIKELIHOOD_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/mge[{instrument}]: regression — eager log_likelihood drifted "
        f"(got {log_likelihood_ref}, expected {EXPECTED_LOG_LIKELIHOOD_HST})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_HST:.6f}"
)
np.testing.assert_allclose(
    float(full_result),
    EXPECTED_LOG_LIKELIHOOD_HST,
    rtol=1e-4,
    err_msg=f"imaging/mge[{instrument}]: regression — full log_likelihood drifted",
)
np.testing.assert_allclose(
    np.array(result_vmap),
    EXPECTED_LOG_LIKELIHOOD_HST,
    rtol=1e-4,
    err_msg=f"imaging/mge[{instrument}]: regression — vmap log_likelihood drifted",
)
print(f"  Regression assertion PASSED: log_likelihood matches {EXPECTED_LOG_LIKELIHOOD_HST:.6f}")
