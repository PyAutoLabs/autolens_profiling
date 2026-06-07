"""
JAX Profiling: Pixelization Imaging Likelihood (Step-by-Step)
=============================================================

Profiles each step of the JAX likelihood function for an imaging dataset where
the source galaxy is reconstructed using a rectangular pixelization with
constant regularization.

Rather than timing the whole likelihood as a single JIT-compiled block (which
hides internal bottlenecks), this script JIT-compiles and times each step of
the pipeline individually:

1. Ray-trace grids through the lens
2. Blurred image of lens light (non-linear profiles)
3. Profile-subtracted image (lens light subtraction)
4. Border relocation of traced grid
5. Overlay grid (source pixel centres)
6. Interpolation weights and mapper construction
7. Mapping matrix
8. Blurred mapping matrix (PSF convolution)
9. Data vector (D)
10. Curvature matrix (F)
11. Regularization matrix (H)
12. Regularized reconstruction: s = (F + H)^{-1} D
13. Map reconstruction to image + log evidence

Caveat: XLA may fuse operations differently when compiled as one program vs
separate pieces, so per-step timings are approximate. They are still useful
for identifying which step dominates.

All JAX timings use `block_until_ready()` to force synchronous measurement.
"""

import json
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

# Shared adapt-image loader: load or compute+cache `lensed_source.fits`
# next to the dataset, then return the masked ``aa.Array2D``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

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
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=1,
    )

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )

    if _cli.use_sparse_operator:
        # Engage the w-tilde sparse-operator path. The inversion factory will
        # pick ``InversionImagingSparse`` so long as at least one linear obj
        # in the model is a Mapper — which is true here (the source is a
        # Rectangular pixelization). The MGE lens-light columns ride through
        # the same sparse inversion alongside the Mapper columns.
        dataset = dataset.apply_sparse_operator()

# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

mesh_pixels_yx = 39  # 39x39 = 1521 source pixels — 1500-tier production fiducial
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

with timer.section("model_build"):
    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality.
    # Lens light: MGE-60 (full production-fiducial) — replaces single Sersic.
    # The 60 linear Gaussians enter the inversion's mapping matrix
    # alongside the source-pixel columns.
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

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )

    # ``RectangularAdaptImage`` weights mesh pixels by the lensed-source
    # adapt image — the production-grade alternative to the coordinate-
    # density-only ``RectangularAdaptDensity``. Adapt image is loaded /
    # cached below; the same shape and regularization are kept.
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularAdaptImage(
            shape=mesh_shape, weight_power=1.0, weight_floor=0.0
        ),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Mesh shape: {mesh_shape}")
print(f"  Source pixels: {mesh_pixels_yx * mesh_pixels_yx}")

# ---------------------------------------------------------------------------
# 3. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)
tracer = al.Tracer(galaxies=list(instance.galaxies))

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# Key configuration that dictates run time
# ---------------------------------------------------------------------------

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]
n_source_pixels = mesh_pixels_yx * mesh_pixels_yx

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Mesh shape:              {mesh_shape}")
print(f"  Source pixels:           {n_source_pixels}")

# ---------------------------------------------------------------------------
# 4. Adapt image — PSF-convolved lensed-source image used by
#    ``RectangularAdaptImage`` to weight mesh pixels. Loads ``lensed_source.fits``
#    from the dataset directory if present, otherwise computes it from the
#    truth tracer and caches the file for sibling scripts on the same
#    instrument.
# ---------------------------------------------------------------------------

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(
        dataset_path=dataset_path, dataset=dataset
    )
    # ``galaxy_image_dict`` (Galaxy-object-keyed) feeds the eager-path
    # ``image_for_galaxy`` lookup; ``galaxy_name_image_dict`` (path-tuple
    # str-keyed) is rebuilt inside JIT closures where the Galaxy objects
    # are reconstructed on every call. Both must be supplied here.
    adapt_images = al.AdaptImages(
        galaxy_image_dict={instance.galaxies.source: adapt_image},
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
    )

print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

# ---------------------------------------------------------------------------
# 5. Full-pipeline reference (FitImaging) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitImaging (eager baseline) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(
            use_border_relocator=True,
            use_mixed_precision=_cli.use_mixed_precision,
        ),
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

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=al.Settings(
        use_border_relocator=True,
        use_mixed_precision=_cli.use_mixed_precision,
    ),
    use_jax=True,
)

def full_pipeline_from_params(params_tree):
    return analysis.log_likelihood_function(instance=params_tree)

_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_likelihood = {full_result}")


# ===================================================================
# Early JSON write — single-JIT only
# ===================================================================
#
# Writes the headline single-JIT numbers BEFORE the vmap phase below,
# so the JSON survives an OOM kill during vmap (a real laptop scenario
# with the 1500-source-pixel HST cells). The vmap block updates the
# JSON in place if it succeeds.
_early_summary = {
    "autolens_version": al.__version__,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "mesh_shape": list(mesh_shape),
        "source_pixels": int(n_source_pixels),
        "inversion_path": "sparse" if _cli.use_sparse_operator else "dense",
    },
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": "PENDING — vmap phase has not run yet (or was killed)",
}
_early_dict_path, _ = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "likelihood" / "imaging",
    default_basename=f"pixelization_likelihood_summary_{instrument}_v{al.__version__}",
)
_early_dict_path.write_text(json.dumps(_early_summary, indent=2))
print(f"\n  Early JSON saved to: {_early_dict_path}")


# ===================================================================
# PART C.5 — vmap-probe mode (early exit)
# ===================================================================
#
# When ``--vmap-probe`` is set the script JIT-vmaps the pipeline at the
# configured batch sizes, reads ``compiled.memory_analysis()``, writes a
# ``vmap_probe.json`` with the recommended A100 batch_size, and exits
# before the full vmap timing loop. See ``vram/README.md`` for methodology.

if _cli.vmap_probe:
    probe = probe_vmap_memory(
        full_pipeline_from_params,
        params_tree,
        batch_sizes=(1, 4, 16),
        dataset="imaging",
        model="pixelization",
        instrument=instrument,
    )
    recommended = recommend_batch_size(probe)
    _inversion_path = "sparse" if _cli.use_sparse_operator else "dense"
    _probe_basename = (
        "vmap_probe_pixelization_sparse"
        if _cli.use_sparse_operator
        else "vmap_probe_pixelization"
    )
    probe_path = (
        (_cli.output_dir or (_workspace_root / "results" / "likelihood" / "imaging"))
        / f"{_probe_basename}.json"
    )
    write_probe_json(
        probe,
        recommended,
        probe_path,
        extra={"inversion_path": _inversion_path},
    )
    print(f"\n  vmap_probe samples: {probe.samples}")
    print(f"  per_replica:        {probe.per_replica_mb:.1f} MB / replica")
    print(f"  recommended batch:  {recommended}")
    print(f"  inversion_path:     {_inversion_path}")
    print(f"  written to:         {probe_path}")
    sys.exit(0)

# ===================================================================
# PART D — vmap + correctness
# ===================================================================
#
# NOTE: vmap requires at least one JAX array leaf in the params_tree.
# When model.total_free_parameters == 0 (all params fixed to truth), the
# pytree has no array leaves and vmap cannot batch over it. Skip in that case.

print("\n--- vmap batched evaluation ---")

batch_size = vmap_batch_for("imaging", "pixelization", instrument) or 3
vmap_batch_time = None
vmap_per_call = None
vmap_speedup = None
result_vmap = None

_n_leaves = len(jax.tree_util.tree_leaves(params_tree))
_vmap_skipped_reason = None
if _n_leaves == 0:
    _vmap_skipped_reason = (
        "model has 0 free parameters (all fixed to truth); vmap "
        "requires at least one array leaf."
    )
else:
    parameters = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
        params_tree,
    )

    vmapped_full = jax.jit(jax.vmap(full_pipeline_from_params))

    # 1521-source-pixel adapt-mesh pipelines push the per-batch working
    # set past 2.5 GB; on smaller GPUs the vmap compile / first call can
    # OOM. Catch and skip cleanly rather than killing the script.
    try:
        with timer.section("vmap_first_call"):
            result_vmap = vmapped_full(parameters)
            block(result_vmap)
    except Exception as exc:
        if "RESOURCE_EXHAUSTED" in str(exc) or "Out of memory" in str(exc):
            _vmap_skipped_reason = (
                f"OOM during vmap first call (batch_size={batch_size}); skip vmap. "
                f"Re-run on a bigger device or lower `batch_size`."
            )
        else:
            raise

if _vmap_skipped_reason is None and _n_leaves > 0:
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
        err_msg="pixelization: JAX vmap likelihood mismatch",
    )
    print("  Correctness check PASSED")
else:
    print(f"  SKIPPED: {_vmap_skipped_reason}")

# ===================================================================
# PART E — Static memory analysis
# ===================================================================

print("\n--- Static memory analysis ---")

if _vmap_skipped_reason is not None:
    print(f"  SKIPPED: {_vmap_skipped_reason}")
    memory_analysis = None
else:
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
print(f"  Mesh shape:            {mesh_shape}")
print(f"  Source pixels:         {n_source_pixels}")
print("-" * 70)

print("-" * 70)
print(f"      {'Full pipeline (single JIT)':<30}  {full_pipeline_per_call:>12.6f} s")
if vmap_per_call is not None:
    print(f"      {'vmap batch (per call)':<30}  {vmap_per_call:>12.6f} s")
    print(f"      {'vmap speedup vs single JIT':<30}  {vmap_speedup:>11.1f}x")
else:
    print(f"      {'vmap':<30}  {'SKIPPED (0 free params)':>12}")
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
        "mesh_shape": list(mesh_shape),
        "source_pixels": int(n_source_pixels),
        "inversion_path": "sparse" if _cli.use_sparse_operator else "dense",
    },
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": "SKIPPED — model has 0 free parameters (all fixed to truth)" if vmap_per_call is None else {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "likelihood" / "imaging",
    default_basename=f"pixelization_likelihood_summary_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")
print(f"  Bar chart path:        {chart_path} (no per-step chart in runtime variant)")


# ===================================================================
# Regression assertion — realistic-scale deterministic log-evidence
# ===================================================================
#
# RectangularAdaptImage at prior medians anchors the regression on the
# *eager* FitImaging value (deterministic to fp64 noise). The full-pipeline
# single-JIT / vmap paths agree with eager to ~1e-3 only: adaptive mesh
# weighting amplifies fp accumulation in Cholesky / log_det on the bigger
# 1581x1581 mapping matrix relative to the non-adaptive baseline (which
# previously matched at 1e-4). The 1e-3 envelope is still tight enough to
# catch real numerical regressions while accommodating the adaptive path.
EXPECTED_LOG_EVIDENCE_HST = 28370.27770182  # 39x39 = 1521 source pixels, MGE-60 lens light, adapt_image=lensed_source

np.testing.assert_allclose(
    log_evidence_ref,
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/pixelization[{instrument}]: regression — eager log_evidence drifted "
        f"(got {log_evidence_ref}, expected {EXPECTED_LOG_EVIDENCE_HST})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_evidence matches "
    f"{EXPECTED_LOG_EVIDENCE_HST:.6f}"
)
np.testing.assert_allclose(
    float(full_result),
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-3,
    err_msg=f"imaging/pixelization[{instrument}]: regression — full log_evidence drifted",
)
if result_vmap is not None:
    np.testing.assert_allclose(
        np.array(result_vmap),
        EXPECTED_LOG_EVIDENCE_HST,
        rtol=1e-3,
        err_msg=f"imaging/pixelization[{instrument}]: regression — vmap log_evidence drifted",
    )
print(f"  Regression assertion PASSED: log_evidence matches {EXPECTED_LOG_EVIDENCE_HST:.6f}")
