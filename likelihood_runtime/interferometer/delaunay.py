"""
JAX Profiling: Delaunay Interferometer Likelihood
=================================================

Profiles the JAX likelihood function for an interferometer dataset where the
source galaxy is reconstructed using a Delaunay pixelization with cross-
derivative (``ConstantSplit``) regularization, and the lens galaxy is an
Isothermal + ExternalShear.

Mirrors ``likelihood/interferometer/pixelization.py`` (Phase 2) with the
``RectangularUniform`` source replaced by a ``Delaunay`` mesh — matching
``likelihood/imaging/delaunay.py`` so imaging vs interferometer Delaunay
results can be compared side-by-side.

Matches the step-by-step pedagogy of ``likelihood/imaging/delaunay.py``
applied to the visibility-space pipeline. The 11 per-step JIT-profiled stages
map 1:1 onto sections in
``autolens_workspace/scripts/interferometer/features/datacube/likelihood_function.py``
and its single-channel parent
``interferometer/features/pixelization/likelihood_function.py``.

Pipeline steps (matching the imaging-delaunay numbering for cross-reference;
the two lens-light steps from the imaging sibling are dropped since the
interferometer pixelization model has no parametric lens light):

 1. Ray-trace data grid to source plane.
 2. Ray-trace mesh grid (image-plane Overlay vertices) to source plane.
 5. Border relocation (data grid + mesh grid).
 6. Delaunay triangulation + interpolation + mapper.
 7. Mapping matrix.
 8. Transformed mapping matrix (NUFFT) — interferometer-specific. Replaces
    imaging's PSF-convolved blurred mapping matrix; the difference is the
    Fourier transform to visibility space rather than image-space convolution.
 9. Data vector D — visibility-space (real and imaginary components).
 10. Curvature matrix F — real and imaginary curvatures summed.
 11. Regularization matrix H — ConstantSplit (same as imaging).
 12. Reconstruction s = NNLS(F + H, D) (same NNLS path as imaging).
 13. Mapped reconstructed visibilities + log evidence (visibility-space χ²).

Measures:

1. Eager baseline: ``FitInterferometer`` with ``xp=np``, print
   ``figure_of_merit`` / ``log_likelihood``.
2. Per-step JIT profiling: each pipeline stage above gets its own
   ``jit_profile()`` call (lower / compile / first-call / steady-state ×10).
3. Full-pipeline JIT: ``jax.jit(analysis.log_likelihood_function)`` on a
   pytree-registered ``ModelInstance``. Measure lower / compile / first-call /
   steady-state per-call.
4. Batched evaluation (opt-in via ``DELAUNAY_VMAP=1``): ``jax.jit(jax.vmap(...))``.
   Skipped by default because Delaunay vmap compilation can take 20+ minutes
   on CPU due to triangulation + interpolation graph size.
5. Correctness: eager vs JIT log-evidence agreement at ``rtol=1e-4`` for both
   the per-step recomputation and the full pipeline.
6. Static memory analysis of the batched program (only when vmap runs).
7. Results JSON + PNG written to ``results/`` with per-step entries that
   slot into the same bar-chart shape as ``likelihood/imaging/delaunay.py``.

JIT-blocker notes
-----------------

Per-step decomposition risks missing cross-step XLA fusion and hitting
library-level JAX blockers. Caveats from the previous opt-out version that
still apply:

- ``dataset.transformer`` is ``al.TransformerNUFFT`` (nufftax-backed) so
  the setup-time ``apply_sparse_operator → image_from`` call scales to
  ALMA-realistic visibility counts (O((N_pix + N_vis) log N) instead of
  ``TransformerDFT``'s O(N_pix · N_vis), which OOMs at 1M+ visibilities).
  The per-likelihood path itself never touches the transformer once
  ``sparse_operator`` is attached — F is FFT-based, D uses the cached
  dirty image, χ² is ``inversion.fast_chi_squared``. The legacy
  ``TransformerNUFFTPyNUFFT`` is pynufft-based and is not JIT-friendly.
- The visibility-space χ² in step 13 separates the complex visibilities and
  noise into real/imag components inside the JIT body (matching the
  ``pixelization/likelihood_function.py`` reference). Complex-valued JIT
  with autoarray ``Visibilities`` wrappers is avoided.

Pytree-native parameter inputs
------------------------------

Uses ``af.ModelInstance`` as the JIT input via PyAutoFit's opt-in pytree
registration (``autofit.jax.register_model``). Exercises the ``TuplePrior``
pytree support landed in PyAutoFit#1222.
"""

import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autofit.jax import register_model as _register_model_pytrees

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------
# AUTOLENS_PROFILING_SMOKE=1 short-circuit (Phase 5 / CI lint smoke).
# Verifies the import graph + module-level setup succeeded without running
# the full profiling pipeline. Skipped entirely when the env var is unset.
import os as _smoke_os
import sys as _smoke_sys

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

# Sweep-driver CLI args (--config-name / --output-dir / --use-mixed-precision).
# Tolerates extra/unknown args via parse_known_args inside the helper.
from _profile_cli import (
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,  # noqa: E402
    resolve_output_paths,
)
from simulators.interferometer import INSTRUMENTS  # noqa: E402
from vram import (  # noqa: E402
    probe_vmap_memory,
    recommend_batch_size,
    resolve_vmap_batch,
    write_probe_json,
)

_cli = parse_profile_cli()

instrument = _cli.instrument or "sma"  # default; override via --instrument

hilbert_pixels = 1500  # 1500-tier production fiducial (matches imaging/datacube)
regularization_coefficient = 1.0


# ---------------------------------------------------------------------------
# Profiling helpers
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


def block(x):
    """Call block_until_ready if available (JAX arrays)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time lower / compile / first call / steady state."""
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

print(f"\n--- Dataset loading [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[1]
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
dataset_path = Path("dataset") / "interferometer" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="interferometer",
    instrument=instrument,
    workspace_root=_workspace_root,
)

mask_radius = INSTRUMENTS[instrument]["mask_radius"]
transformer_chunk_size = INSTRUMENTS[instrument].get("transformer_chunk_size", None)

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)


def _build_transformer(uv_wavelengths, real_space_mask):
    """Inject per-instrument chunk_size into TransformerNUFFT without needing a
    transformer_kwargs API on Interferometer.from_fits. Required for alma_high
    (5M visibilities) to cap the nufftax gather buffer (PyAutoArray#330)."""
    return al.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=transformer_chunk_size,
    )


with timer.section("dataset_load"):
    dataset = al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=_build_transformer,
    )

with timer.section("apply_sparse_operator"):
    # Precompute the NUFFT precision-matrix preload so per-fit curvature
    # assembly uses the FFT-based sparse path instead of dense DFT for every
    # source pixel. Unblocked by PyAutoArray#316 (the Pmax > 1 extent-indexing
    # fix); on Delaunay this was previously guarded with NotImplementedError.
    dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=True)

n_visibilities = dataset.uv_wavelengths.shape[0]
print(f"  Total visibilities: {n_visibilities}")

# ---------------------------------------------------------------------------
# 2. Adapt image + image mesh (Hilbert)
# ---------------------------------------------------------------------------
#
# ``image_mesh.Hilbert`` adaptively places the source mesh vertices in the
# image plane based on the lensed-source adapt image — denser where the
# source lives, sparser elsewhere. Replaces the regular ``image_mesh.Overlay``
# + circular-edge fallback that preceded this path. ``zeroed_pixels=0``
# because Hilbert's placement is data-driven (no fixed edge points to mask).

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

print("\n--- Image mesh construction (Hilbert) ---")

with timer.section("image_mesh_hilbert"):
    image_mesh = al.image_mesh.Hilbert(pixels=hilbert_pixels, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.real_space_mask, adapt_data=adapt_image
    )

n_mesh_vertices = image_plane_mesh_grid.shape[0]
edge_pixels_total = 0
print(f"  Hilbert pixels: {hilbert_pixels}")
print(f"  Mesh vertices placed: {n_mesh_vertices}")

# ---------------------------------------------------------------------------
# 3. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality.
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

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    mesh = al.mesh.Delaunay(
        pixels=n_mesh_vertices,
        zeroed_pixels=0,
    )
    regularization = al.reg.ConstantSplit(coefficient=regularization_coefficient)
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

# JIT input: the instance itself, with all parameter leaves promoted to JAX
# arrays. The eager NumPy instance is retained for the eager FitInterferometer
# baseline below.
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

tracer = al.Tracer(galaxies=list(instance.galaxies))

# AdaptImages tells FitInterferometer / AnalysisInterferometer where the
# Delaunay mesh vertices live in the image-plane (separate from the source-
# plane vertices that get computed by ray-tracing).
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
# 5. Configuration summary
# ---------------------------------------------------------------------------

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities:            {n_visibilities}")
print(f"  Hilbert pixels:          {hilbert_pixels}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print(f"  Reg. coefficient:        {regularization_coefficient}")

# ---------------------------------------------------------------------------
# 6. Full-pipeline reference (FitInterferometer) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitInterferometer (eager baseline) ---")

with timer.section("fit_interferometer_eager"):
    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(use_mixed_precision=_cli.use_mixed_precision),
        xp=np,
    )
    figure_of_merit_ref = fit.figure_of_merit
    log_likelihood_ref = fit.log_likelihood

print(f"  figure_of_merit = {figure_of_merit_ref}")
print(f"  log_likelihood  = {log_likelihood_ref}")


# ===================================================================
# PART C — Full-pipeline JIT (for comparison)
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT")
print("=" * 70)

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=al.Settings(use_mixed_precision=_cli.use_mixed_precision),
    use_jax=True,
)


def full_pipeline_from_params(params_tree):
    """Full interferometer likelihood from a pytree-shaped ``ModelInstance``.

    No flat-vector unpacking inside the trace — the instance crosses the JIT
    boundary directly, with constants (redshifts, etc.) kept static via the
    ``aux_data`` partition set up by ``autofit.jax.register_model``.
    """
    return analysis.log_likelihood_function(instance=params_tree)


_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_evidence = {full_result}")

# Correctness: for inversion models (pixelization + regularization), the
# analysis "log_likelihood_function" actually returns the log-evidence
# (= figure_of_merit), which includes the regularization/determinant terms.
# Match against figure_of_merit_ref, not log_likelihood_ref.
np.testing.assert_allclose(
    float(full_result),
    float(figure_of_merit_ref),
    rtol=1e-4,
    err_msg="interferometer/delaunay: JIT log-evidence does not match eager figure_of_merit",
)
print("  Eager-vs-JIT correctness PASSED")

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
        batch_sizes=(1,),
        dataset="interferometer",
        model="delaunay",
        instrument=instrument,
    )
    recommended = recommend_batch_size(probe)
    probe_path = (
        _cli.output_dir or (_workspace_root / "results" / "runtime" / "interferometer" / "delaunay")
    ) / (
        "vmap_probe_delaunay_sparse.json"
        if _cli.use_sparse_operator
        else "vmap_probe_delaunay.json"
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

_batch_resolved, _batch_source = resolve_vmap_batch(
    "interferometer",
    "delaunay",
    instrument,
    output_dir=_cli.output_dir
    or (_workspace_root / "results" / "runtime" / "interferometer" / "delaunay"),
    path="sparse" if _cli.use_sparse_operator else "dense",
    backend=jax.default_backend(),
)
print(f"  vmap batch_size: {_batch_resolved} (source: {_batch_source})")
batch_size = _batch_resolved or 3

# Skip vmap if batch resolution (probe JSON / table) returned None for this cell.
_vmap_skipped = _batch_resolved is None

vmap_batch_time = None
vmap_per_call = None
vmap_speedup = None
result_vmap = None
vmapped_full = None
parameters = None

_n_leaves = len(jax.tree_util.tree_leaves(params_tree))
if _vmap_skipped:
    print(
        f"  SKIPPED: batch resolution returned None for this (cell, instrument) — source: {_batch_source}."
    )
elif _n_leaves == 0:
    print(
        "  SKIPPED: model has 0 free parameters (all fixed to truth); "
        "vmap requires at least one array leaf."
    )
else:
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
        err_msg="interferometer/delaunay: JAX vmap likelihood mismatch",
    )
    print("  vmap-vs-single-JIT correctness PASSED")

# ===================================================================
# PART E — Static memory analysis (only if vmap ran)
# ===================================================================

print("\n--- Static memory analysis ---")

if vmapped_full is None:
    print("  SKIPPED: vmap path was not exercised this run.")
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
# JAX Likelihood Function Summary + artefacts
# ===================================================================

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities:            {n_visibilities}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print("-" * 70)
print(f"  Eager log_likelihood:    {log_likelihood_ref}")
print(f"  Eager figure_of_merit:   {figure_of_merit_ref}  (log-evidence)")
print(f"  JIT  log-evidence:       {float(full_result)}")
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

if vmap_per_call is None:
    if _vmap_skipped:
        vmap_payload = "SKIPPED — batch resolution returned None for this (cell, instrument)"
    else:
        vmap_payload = "SKIPPED — model has 0 free parameters (all fixed to truth)"
else:
    vmap_payload = {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    }

likelihood_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "model": "delaunay",
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "real_space_shape": list(real_space_shape),
        "visibilities": int(n_visibilities),
        "hilbert_pixels": int(hilbert_pixels),
        "delaunay_vertices": int(n_mesh_vertices),
        "edge_zeroed_pixels": int(edge_pixels_total),
        "regularization_coefficient": regularization_coefficient,
    },
    "log_likelihood_eager": float(log_likelihood_ref),
    "figure_of_merit_eager": float(figure_of_merit_ref),
    "log_evidence_jit": float(full_result),
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": vmap_payload,
    "memory_mb": None
    if memory_analysis is None
    else {
        "output": memory_analysis.output_size_in_bytes / 1024**2,
        "temp": memory_analysis.temp_size_in_bytes / 1024**2,
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "runtime" / "interferometer" / "delaunay",
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
# Pinned empirically per instrument; ``None`` means "skip the assertion and
# print the value so it can be pasted in here on a clean run". sma was
# bumped to mask_radius=3.5 in 2026-05-21's INSTRUMENTS refactor — the
# old mask_radius=3.0 value no longer applies and needs re-measuring.
_pinned_drift: list = []
_pinned_expected = None

EXPECTED_LOG_EVIDENCE = {
    "sma": None,
    "alma": None,
    "alma_high": None,
}

expected_log_evidence = EXPECTED_LOG_EVIDENCE.get(instrument)
_pinned_expected = expected_log_evidence

if expected_log_evidence is None:
    print(
        f"\n  Regression assertion SKIPPED for [{instrument}] — "
        f"capture this run's eager log_evidence ({figure_of_merit_ref}) "
        f"and paste it into EXPECTED_LOG_EVIDENCE[{instrument!r}]."
    )
else:
    _rec = check_pinned(figure_of_merit_ref, _pinned_expected, label="eager", rtol=1e-4)
    if _rec is not None:
        _pinned_drift.append(_rec)
    _rec = check_pinned(float(full_result), _pinned_expected, label="full", rtol=1e-3)
    if _rec is not None:
        _pinned_drift.append(_rec)
    if result_vmap is not None:
        _rec = check_pinned(np.array(result_vmap), _pinned_expected, label="vmap", rtol=1e-3)
        if _rec is not None:
            _pinned_drift.append(_rec)


# Pinned-value outcome -> result JSON: profiling records and flags drift,
# never adjudicates library correctness (autolens_workspace_test's remit;
# boundary rule in results/notes/design_lock_in.md). PyAutoHeart's vitals
# scan reads the pinned_drift field.
record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")
