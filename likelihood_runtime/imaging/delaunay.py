"""
JAX Profiling: Delaunay Imaging Likelihood (Step-by-Step)
=========================================================

Profiles each step of the JAX likelihood function for an imaging dataset where
the source galaxy is reconstructed using a Delaunay triangulation mesh with
ConstantSplit regularization.

Key differences from the rectangular pixelization profiling script:

- Mesh vertices are computed in the **image-plane** via an Overlay grid, then
  ray-traced to the source-plane (rectangular computes directly in source-plane).
- Edge points are appended around the mask border and zeroed during inversion.
- Uses **InterpolatorDelaunay** (barycentric interpolation within triangles)
  instead of bilinear interpolation on a rectangular grid.
- Uses **ConstantSplit** regularization (cross-derivative scheme) instead of
  the simpler Constant neighbour-difference scheme.
- Delaunay triangulation itself uses scipy on CPU and cannot be JIT-compiled.

Pipeline steps:

1. Ray-trace data grid to source plane
2. Ray-trace mesh grid (image-plane vertices) to source plane
3. Lens light images (pre-PSF, JIT) + PSF convolution (eager)
4. Profile-subtracted image
5. Border relocation (data grid + mesh grid)
6. Delaunay triangulation + interpolation + mapper
7. Mapping matrix
8. Blurred mapping matrix (PSF convolution)
9. Data vector (D)
10. Curvature matrix (F)
11. Regularization matrix (H) — ConstantSplit scheme
12. Regularized reconstruction: s = (F + H)^{-1} D
13. Map reconstruction to image + log evidence

Caveat: XLA may fuse operations differently when compiled as one program vs
separate pieces, so per-step timings are approximate. They are still useful
for identifying which step dominates.

All JAX timings use `block_until_ready()` to force synchronous measurement.
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
_cli = parse_profile_cli()

instrument = "hst"  # <-- change this to profile a different instrument


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

# ---------------------------------------------------------------------------
# 2. Adapt image + image mesh (Hilbert)
# ---------------------------------------------------------------------------
#
# ``image_mesh.Hilbert`` places the source mesh vertices in the image plane by
# inverse-transform-sampling a Hilbert-curve ordering of the lensed source
# adapt image. The result is a sparser mesh in faint regions and a denser one
# where the source actually lives — production-grade, replaces the
# uniform-coverage ``image_mesh.Overlay`` + circular-edge fallback that
# preceded the Hilbert path. ``zeroed_pixels=0`` because Hilbert's placement
# is data-driven; there are no fixed-position edge points to mask out.

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(
        dataset_path=dataset_path, dataset=dataset
    )

print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

print("\n--- Image mesh construction (Hilbert) ---")

n_mesh_vertices = 1500  # 1500-tier production fiducial

with timer.section("image_mesh_hilbert"):
    image_mesh = al.image_mesh.Hilbert(
        pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=adapt_image
    )

edge_pixels_total = 0
print(f"  Hilbert pixels: {n_mesh_vertices}")
print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

# ---------------------------------------------------------------------------
# 3. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

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

    mesh = al.mesh.Delaunay(
        pixels=n_mesh_vertices,
        zeroed_pixels=0,
    )
    regularization = al.reg.ConstantSplit(coefficient=1.0)
    pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Delaunay pixels: {n_mesh_vertices}")
print(f"  Zeroed edge pixels: {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 4. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

n_pytree_leaves = len(jax.tree_util.tree_leaves(params_tree))
print(f"  Pytree JAX leaves: {n_pytree_leaves}")

tracer = al.Tracer(galaxies=list(instance.galaxies))

# AdaptImages tells FitImaging where mesh vertices live in image-plane
adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={
        instance.galaxies.source: image_plane_mesh_grid,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid,
    },
)

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# Key configuration that dictates run time
# ---------------------------------------------------------------------------

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]
n_source_pixels = n_mesh_vertices

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Delaunay vertices:       {n_source_pixels}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")

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

analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images, use_jax=True)

def full_pipeline_from_params(params_tree):
    return analysis.log_likelihood_function(instance=params_tree)

_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_likelihood = {full_result}")

# ===================================================================
# PART D — vmap + correctness
# ===================================================================

print("\n--- vmap batched evaluation ---")

# WARNING: The vmap compilation for the Delaunay pipeline takes 20+ minutes on CPU.
# The XLA graph for a batched Delaunay inversion (including scipy triangulation,
# border relocation, interpolation, mapping matrix construction, and PSF convolution)
# is extremely large. The single-call JIT above compiles in ~2s and runs in ~1.8s,
# but vmap recompiles the entire graph for batch_size independent evaluations.
#
# This is likely a candidate for optimisation — either via custom_vjp to avoid
# retracing the full pipeline, or by restructuring the Delaunay steps to reduce
# the XLA graph size. For now, skip vmap by default and run it only when explicitly
# requested via DELAUNAY_VMAP=1 environment variable.

import os
run_vmap = os.environ.get("DELAUNAY_VMAP", "0") == "1"

if not run_vmap:
    print("  SKIPPED: vmap compilation takes 20+ minutes for Delaunay pipeline.")
    print("  Set DELAUNAY_VMAP=1 to run this section.")
    vmap_batch_time = None
    vmap_per_call = None
    vmap_speedup = None
else:

    batch_size = 3
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
        err_msg="delaunay: JAX vmap likelihood mismatch",
    )
    print("  Correctness check PASSED")

    # --- Static memory analysis ---

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
print(f"  Delaunay vertices:     {n_source_pixels}")
print(f"  Edge zeroed pixels:    {edge_pixels_total}")
print("-" * 70)

print("-" * 70)
print(f"      {'Full pipeline (single JIT)':<30}  {full_pipeline_per_call:>12.6f} s")
if vmap_per_call is not None:
    print(f"      {'vmap batch (per call)':<30}  {vmap_per_call:>12.6f} s")
    print(f"      {'vmap speedup vs single JIT':<30}  {vmap_speedup:>11.1f}x")
else:
    print(f"      {'vmap':<30}  {'SKIPPED':>12}")
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
        "delaunay_vertices": int(n_source_pixels),
        "edge_zeroed_pixels": int(edge_pixels_total),
    },
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": "SKIPPED — compilation takes 20+ minutes (set DELAUNAY_VMAP=1)",
}

if vmap_per_call is not None:
    likelihood_summary["vmap"] = {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    }

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "likelihood" / "imaging",
    default_basename=f"delaunay_likelihood_summary_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")
print(f"  Bar chart path:        {chart_path} (no per-step chart in runtime variant)")


# ===================================================================
# Regression assertion — realistic-scale deterministic log-evidence
# ===================================================================
#
# Simulator truth parameters via GaussianPrior(mean=truth, sigma=small)
# make the full-pipeline log-evidence deterministic at the prior median.
# Hilbert image_mesh + 1500-pixel Delaunay; rtol=1e-3 for the JIT paths
# matches imaging/pixelization (adaptive meshes amplify fp drift through
# Cholesky / log_det). vmap result asserted only when DELAUNAY_VMAP=1
# (vmap compile takes 20+ min).
EXPECTED_LOG_EVIDENCE_HST = 29110.92085793  # 1500-pixel Hilbert/Delaunay, MGE-60 lens, adapt_image=lensed_source

np.testing.assert_allclose(
    log_evidence_ref,
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/delaunay[{instrument}]: regression — eager log_evidence drifted "
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
    err_msg=f"imaging/delaunay[{instrument}]: regression — full log_evidence drifted",
)
if run_vmap:
    np.testing.assert_allclose(
        np.array(result_vmap),
        EXPECTED_LOG_EVIDENCE_HST,
        rtol=1e-3,
        err_msg=f"imaging/delaunay[{instrument}]: regression — vmap log_evidence drifted",
    )
print(f"  Regression assertion PASSED: log_evidence matches {EXPECTED_LOG_EVIDENCE_HST:.6f}")
