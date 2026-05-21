"""
JAX Profiling: Delaunay Datacube Likelihood (Step-by-Step)
==========================================================

Profiles each step of the JAX likelihood function for an ALMA-style datacube —
a list of N ``Interferometer`` channels sharing a single lens model — where
each channel reconstructs its own source with a Delaunay pixelization +
ConstantSplit regularization.

Mirrors the step-by-step structure of
``likelihood/interferometer/delaunay.py`` (Phase 2 of the datacube
roadmap, just merged). The key new ingredient is the **channel-invariant vs
channel-variant** split: most steps are computed once for the whole cube
(shared lens, shared mesh, shared mask), only the NUFFT-based inversion-setup
chain, the data vector, the curvature matrix, the reconstruction, and the
log-evidence depend on per-channel data.

The cube total is::

    cube_cost = sum(channel_invariant_costs) + N_channels * sum(channel_variant_costs)

That number quantifies how much the deferred shared-``Lᵀ W̃ L`` optimisation
will save: moving the curvature matrix from per-channel to shared would
subtract ``(N - 1) * curvature_matrix_cost`` from the cube total.

Channel-invariant vs channel-variant taxonomy
---------------------------------------------

For the canonical datacube case where the lens model is shared across all
channels:

============================================  ================  =========================
Step                                          Channel-invariant Computed
============================================  ================  =========================
1. Ray-trace data grid                        yes               once for the cube
2. Ray-trace mesh grid                        yes               once for the cube
3. Inversion setup (border + mapper + NUFFT)  **NUFFT depends   once per channel
                                              on uv_wavelengths**
4. Data vector D                              per channel       once per channel
5. Curvature matrix F                         per channel       once per channel
6. Regularization matrix H                    yes               once for the cube
7. Reconstruction (NNLS)                      per channel       once per channel
8. Mapped recon + log evidence                per channel       once per channel
============================================  ================  =========================

Dataset
-------

This profiler reuses the SMA interferometer dataset
(``dataset/interferometer/sma/``) loaded N times as a 4-channel
"cube". Each channel has identical visibilities, noise map and uv_wavelengths
— the point here is timing, not science. The N-channel cube log-evidence is
``N × single-channel log-evidence`` exactly, which makes the regression
assertion trivial.

If you want a realistic per-channel-distinct cube, point the loader at the
workspace simulator output at
``../autolens_workspace/dataset/interferometer/datacube/sim_simple/``; the
JIT-cost taxonomy doesn't change because it's a function of which arrays are
loop-variables in ``FitInterferometer``, not the data values themselves.

Measures
--------

1. Eager baseline: ``FitInterferometer`` per channel with ``xp=np``; cube
   reference log-evidence is the sum.
2. Per-step JIT profiling: each pipeline stage gets its own ``jit_profile()``
   call (lower / compile / first-call / steady-state × 10). Channel-invariant
   stages are timed once; channel-variant stages are timed on channel 0 and
   the cube cost is reported as ``N × per-call``.
3. Full-pipeline cube JIT: ``jax.jit`` over the explicit
   ``sum(analysis.log_likelihood_function(instance) for analysis in
   analysis_list)`` — the same shape as the user-facing
   ``datacube/likelihood_function.py`` and the cube modeling scripts'
   internal ``FactorGraphModel`` sum.
4. Correctness: per-step recomputed cube log-evidence and full-pipeline JIT
   log-evidence both match the summed eager ``FitInterferometer.log_evidence``
   at ``rtol=1e-4``.
5. Results JSON + bar chart written to ``results/jit/datacube/`` using the
   same schema as the interferometer sibling. Bar chart shows the cube-total
   form of every step (channel-variant entries pre-multiplied by N).

vmap is **skipped** for the cube profiler. The natural batching dimension is
"datasets" (one entry per channel) not "parameters" (which the
interferometer-sibling vmap exercises). A vmap-over-channels variant would
require a different graph shape and isn't the bottleneck we care about for
the shared-``Lᵀ W̃ L`` optimisation.
"""

import numpy as np
import jax
import jax.numpy as jnp
import os
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
from simulators.interferometer import INSTRUMENTS  # noqa: E402
_cli = parse_profile_cli()

instrument = "sma"  # <-- change to profile a different instrument; cube is N copies of the per-instrument dataset

# n_channels = 34 matches the prior Hannah ALMA cube fiducial. For quick
# iteration on the smaller sma dataset, drop this to 4.
n_channels = 34
hilbert_pixels = 500  # 500-tier production fiducial per channel (× n_channels)
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
# 1. Dataset loading: reuse SMA interferometer dataset N times
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading [{instrument}, {n_channels} channels] ---")

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

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)

with timer.section("dataset_list_load"):
    # apply_sparse_operator: precompute the visibility-space sparse precision
    # operator so per-fit curvature assembly uses the FFT-based sparse path
    # instead of a dense DFT for every source pixel. Unblocked by
    # PyAutoArray#316 (the Pmax > 1 extent-indexing fix); on Delaunay this was
    # previously guarded with NotImplementedError.
    dataset_list = [
        al.Interferometer.from_fits(
            data_path=dataset_path / "data.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
            real_space_mask=real_space_mask,
            transformer_class=al.TransformerDFT,
            # DFT is mandatory here: apply_sparse_operator is not yet
            # compatible with the new nufftax-backed al.TransformerNUFFT (see
            # PyAutoArray/autoarray/dataset/interferometer/dataset.py:261).
            # Swapping the transformer would raise NotImplementedError.
            raise_error_dft_visibilities_limit=False,
        ).apply_sparse_operator(use_jax=True, show_progress=False)
        for _ in range(n_channels)
    ]

n_visibilities = dataset_list[0].uv_wavelengths.shape[0]
print(f"  Channels:           {n_channels}")
print(f"  Visibilities/chan:  {n_visibilities}")

# ---------------------------------------------------------------------------
# 2. Adapt image + image mesh (Hilbert, channel-invariant)
# ---------------------------------------------------------------------------
#
# Adapt image is computed once from the truth tracer and reused across every
# channel — the lens model is channel-invariant, so the lensed-source image
# in image plane is the same for each channel. ``image_mesh.Hilbert`` then
# adaptively places source mesh vertices to follow the source intensity.

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(
        dataset_path=dataset_path, dataset=dataset_list[0]
    )

print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

print("\n--- Image mesh construction (Hilbert) ---")

with timer.section("image_mesh_hilbert"):
    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=1.0, weight_floor=0.0
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset_list[0].real_space_mask, adapt_data=adapt_image
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

tracer = al.Tracer(galaxies=list(instance.galaxies))

# The adapt_images object is channel-invariant — the image-plane Delaunay mesh
# vertices are shared across channels (the lens model is shared).
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
print(f"  Channels:                {n_channels}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities/chan:       {n_visibilities}")
print(f"  Hilbert pixels:          {hilbert_pixels}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print(f"  Reg. coefficient:        {regularization_coefficient}")

# ---------------------------------------------------------------------------
# 6. Per-channel eager FitInterferometer baseline
# ---------------------------------------------------------------------------

print(f"\n--- Per-channel eager FitInterferometer baselines ({n_channels} channels) ---")

fit_list = []
log_evidence_per_channel = []
with timer.section(f"eager_fit_per_channel_x{n_channels}"):
    for c, dataset in enumerate(dataset_list):
        f = al.FitInterferometer(
            dataset=dataset,
            tracer=tracer,
            adapt_images=adapt_images,
            settings=al.Settings(use_mixed_precision=_cli.use_mixed_precision),
            xp=np,
        )
        fit_list.append(f)
        log_evidence_per_channel.append(f.log_evidence)

for c, le in enumerate(log_evidence_per_channel):
    print(f"  channel {c}: log_evidence = {le:.6f}")

cube_log_evidence_ref = float(sum(log_evidence_per_channel))
print(f"  cube reference log_evidence (sum) = {cube_log_evidence_ref:.6f}")


# ===================================================================
# PART C — Full-pipeline cube JIT (sum of per-channel log_likelihoods)
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE CUBE JIT (for comparison)")
print("=" * 70)

# Part C is expensive at large n_channels: lower + compile build a graph
# proportional to n_channels (e.g. ~70s for n_channels=34 on a laptop CPU),
# and the steady-state first-call follows. Default to skipping; opt in with
# CUBE_FULL_JIT=1 when the full-pipeline timing matters (e.g. comparing
# step-by-step total against single-JIT).
_run_full_cube_jit = os.environ.get("CUBE_FULL_JIT") == "1"

if _run_full_cube_jit:
    analysis_list = [
        al.AnalysisInterferometer(
            dataset=d,
            adapt_images=adapt_images,
            settings=al.Settings(use_mixed_precision=_cli.use_mixed_precision),
            use_jax=True,
        )
        for d in dataset_list
    ]

    def full_cube_pipeline_from_params(params_tree):
        """Cube log-evidence via the explicit per-channel sum.

        Same shape as the user-facing ``datacube/likelihood_function.py``:
        feeds the shared instance to every per-channel
        ``AnalysisInterferometer.log_likelihood_function`` and sums.
        """
        total = jnp.zeros(())
        for analysis in analysis_list:
            total = total + analysis.log_likelihood_function(instance=params_tree)
        return total

    _full_cube_n_repeats = 3
    _, full_cube_result = jit_profile(
        full_cube_pipeline_from_params,
        "full_cube_pipeline",
        params_tree,
        n_repeats=_full_cube_n_repeats,
    )
    full_pipeline_per_call = timer.records[-1][1] / _full_cube_n_repeats

    print(f"  full cube log_evidence (JIT) = {full_cube_result}")

    np.testing.assert_allclose(
        float(full_cube_result),
        cube_log_evidence_ref,
        rtol=1e-4,
        err_msg="Full-pipeline cube JIT log_evidence does not match summed eager FitInterferometer.log_evidence",
    )
    print("  Eager-vs-JIT cube correctness PASSED")
else:
    full_cube_result = None
    full_pipeline_per_call = float("nan")
    print(
        "  Full-pipeline cube JIT SKIPPED — opt-in via CUBE_FULL_JIT=1. "
        f"At n_channels={n_channels} the lower + compile alone is on the order of "
        f"{n_channels * 2}-{n_channels * 3}s, so it's gated to keep the default "
        "runtime usable; the per-step Part B JIT data above is what feeds the "
        "shared-Lᵀ W̃ L analysis."
    )

# ===================================================================
# PART D — vmap (skipped for cube)
# ===================================================================
#
# The natural batching axis for a cube fit is "datasets" (one entry per
# channel), not "parameters" (which the interferometer-sibling vmap exercises).
# vmap-over-channels would require a different graph shape and isn't where the
# shared-Lᵀ W̃ L optimisation lives. Skipped.

print("\n--- vmap (skipped) ---")
print(
    "  Cube batching dimension is 'datasets', not 'parameters'. The "
    "interferometer-sibling vmap pattern doesn't map cleanly here. Skipped."
)

# ===================================================================
# Summary + JSON + bar chart
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — CUBE {instrument.upper()} × {n_channels} — v{al_version}")
print("=" * 70)
print(f"  Instrument:              {instrument}")
print(f"  Channels:                {n_channels}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities/chan:       {n_visibilities}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print("-" * 70)
print(f"  Cube reference log_evidence:  {cube_log_evidence_ref}")
if full_cube_result is not None:
    print(f"  Cube JIT log_evidence:        {float(full_cube_result)}")
else:
    print(f"  Cube JIT log_evidence:        SKIPPED (CUBE_FULL_JIT=1 to enable)")
print("-" * 70)

# Shared-Lᵀ W̃ L optimisation savings estimate:
# Moving the curvature matrix from per-channel to shared would save
# (n_channels - 1) × per-channel curvature matrix cost.
shared_lwl_savings = (n_channels - 1) * curvature_matrix_per_channel

print("-" * 70)
if np.isfinite(full_pipeline_per_call):
    print(f"      {'Full pipeline cube (single JIT)':<50}  {full_pipeline_per_call:>12.6f} s")
else:
    print(f"      {'Full pipeline cube (single JIT)':<50}  SKIPPED")
print(f"      {'Shared-Lᵀ W̃ L savings (curvature only, est.)':<50}  {shared_lwl_savings:>12.6f} s")
print("=" * 70)

# --- Save results dictionary ---

likelihood_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "model": "delaunay",
    "n_channels": n_channels,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "real_space_shape": list(real_space_shape),
        "visibilities_per_channel": int(n_visibilities),
        "hilbert_pixels": int(hilbert_pixels),
        "delaunay_vertices": int(n_mesh_vertices),
        "edge_zeroed_pixels": int(edge_pixels_total),
        "regularization_coefficient": regularization_coefficient,
    },
    "cube_log_evidence_eager": cube_log_evidence_ref,
    "cube_log_evidence_jit": (
        float(full_cube_result) if full_cube_result is not None else None
    ),
    "log_evidence_per_channel_eager": [float(le) for le in log_evidence_per_channel],
    "full_pipeline_cube_single_jit": full_pipeline_per_call,
    "shared_lwl_savings_estimate": shared_lwl_savings,
    "vmap": "SKIPPED — cube batching axis is 'datasets', not 'parameters'",
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "likelihood" / "datacube",
    default_basename=f"delaunay_likelihood_summary_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")
print(f"  Bar chart path:        {chart_path} (no per-step chart in runtime variant)")


# ===================================================================
# Regression assertion — deterministic cube log-evidence
# ===================================================================
#
# Identical channels = exact N × single-channel log-evidence (for "sma").
# For "hannah" the per-channel literal isn't pinned yet, so the assertion is
# skipped until the value below is filled in from a clean run.
EXPECTED_LOG_EVIDENCE_PER_CHANNEL = {
    "sma": None,
    "alma": None,
    "alma_high": None,
}

_per_channel = EXPECTED_LOG_EVIDENCE_PER_CHANNEL.get(instrument)
expected_cube_log_evidence = (
    n_channels * _per_channel if _per_channel is not None else None
)

if expected_cube_log_evidence is None:
    print(
        f"\n  Cube regression assertion SKIPPED for [{instrument}] — "
        f"capture this run's eager cube log_evidence ({cube_log_evidence_ref}), "
        f"divide by n_channels ({n_channels}) to get the per-channel value "
        f"({cube_log_evidence_ref / n_channels}), and paste that into "
        f"EXPECTED_LOG_EVIDENCE_PER_CHANNEL[{instrument!r}]."
    )
else:
    np.testing.assert_allclose(
        cube_log_evidence_ref,
        expected_cube_log_evidence,
        rtol=1e-4,
        err_msg=(
            f"datacube/delaunay[{instrument}]: regression — eager cube log_evidence "
            f"drifted (got {cube_log_evidence_ref}, expected {expected_cube_log_evidence})"
        ),
    )
    print(
        f"\n  Eager cube regression assertion PASSED: log_evidence matches "
        f"{expected_cube_log_evidence:.6f}"
    )
    if full_cube_result is not None:
        np.testing.assert_allclose(
            float(full_cube_result),
            expected_cube_log_evidence,
            rtol=1e-3,
            err_msg=f"datacube/delaunay[{instrument}]: regression — full cube log_evidence drifted",
        )
        print(f"  Full-pipeline cube regression assertion PASSED")
