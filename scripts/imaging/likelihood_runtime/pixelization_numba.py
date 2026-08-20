"""
Numba CPU Profiling: Pixelization Imaging Likelihood (Runtime)
==============================================================

Profiles the per-evaluation runtime of the **numba CPU sparse-operator**
likelihood — the path the workspace example
``autolens_workspace/scripts/imaging/features/pixelization/cpu_fast_modeling.py``
uses for fast pixelized-source modeling on CPU-abundant hardware:

- ``dataset.apply_sparse_operator_cpu()`` attaches a ``SparseLinAlgImagingNumba``
  operator (precomputed PSF precision products, numba-jitted kernels).
- ``al.AnalysisImaging(use_jax=False)`` evaluates the likelihood eagerly in
  numpy + numba (no JAX, no JIT tracing).

Timed quantities:

1. One-off ``apply_sparse_operator_cpu()`` setup cost + operator memory.
2. First likelihood call (includes lazy numba compilation; disk-cached).
3. Steady-state per-call average of ``analysis.log_likelihood_function`` —
   a fresh ``FitImaging`` per call, exactly as production sampling does.

The model mirrors the production CPU-fast fiducial: MGE-60 lens light (linear,
held at prior medians), Isothermal + shear mass, and a rectangular
``RectangularAdaptDensity`` source pixelization with constant regularization
(the workspace example's mesh; 28x28 = 784 source pixels).

Notes:

- All numba kernels on this path are single-threaded (no ``prange``); the only
  multi-threaded steps are BLAS/LAPACK (solve, Cholesky) and scipy FFT
  convolution, controlled by ``OMP_NUM_THREADS``.
- ``--use-mixed-precision`` is accepted but ignored: this path is fp64 numpy.

Output
------
``results/runtime/imaging/pixelization_numba/pixelization_numba_likelihood_summary_<instrument>_v<version>.json``
(the runtime variant writes no per-step chart — see the breakdown sibling).
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
import os
import time
from contextlib import contextmanager
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np

_sys.path.insert(0, str(_profiling_root()))

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke). Verifies the import
# graph + module-level setup succeeded without running the full profile.
import os as _smoke_os
import sys as _smoke_sys

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,
    resolve_output_paths,
)

_cli = parse_profile_cli()

instrument = _cli.instrument or "euclid"  # default; override via --instrument

if _cli.use_mixed_precision:
    print("NOTE: --use-mixed-precision has no effect on the numba CPU path (fp64 numpy).")


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


timer = Timer()

# ===================================================================
# PART A — Setup
# ===================================================================

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_workspace_root = _profiling_root()
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

# The one-off numba sparse-operator setup — paid once per dataset per analysis,
# then reused by every likelihood evaluation. Timed + sized because campaigns
# pay it per lens.
with timer.section("apply_sparse_operator_cpu"):
    dataset = dataset.apply_sparse_operator_cpu()

sparse_operator_setup_s = timer.records[-1][1]

_op = dataset.sparse_operator
sparse_operator_nbytes = int(
    _op.psf_precision_operator_sparse.nbytes + _op.indexes.nbytes + _op.lengths.nbytes
)
print(f"  sparse operator memory: {sparse_operator_nbytes / 1024**2:.1f} MB")

# ---------------------------------------------------------------------------
# Model construction — production CPU-fast fiducial
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

mesh_pixels_yx = 28  # workspace cpu_fast_modeling.py mesh: 28x28 = 784 source pixels
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

with timer.section("model_build"):
    # MGE-60 lens light (linear Gaussians — enter the inversion as linear-func
    # columns alongside the source-pixel columns, as in production SLaM CPU
    # stages where the LP-stage MGE rides through as an instance).
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

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

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
print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")

# ===================================================================
# PART B — Likelihood evaluation (numba CPU path)
# ===================================================================

print("\n--- Likelihood evaluation (use_jax=False) ---")

analysis = al.AnalysisImaging(
    dataset=dataset,
    settings=al.Settings(use_border_relocator=True),
    use_jax=False,
)

# Guard: prove the fit actually routes through the numba sparse inversion —
# otherwise this script silently profiles the dense mapping path.
from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)

fit_check = analysis.fit_from(instance=instance)
assert isinstance(fit_check.inversion, InversionImagingSparseNumba), (
    f"Expected InversionImagingSparseNumba, got {type(fit_check.inversion).__name__} — "
    "the numba sparse-operator path is not engaged."
)
print(f"  inversion class: {type(fit_check.inversion).__name__} (numba sparse path confirmed)")
del fit_check

# First call: includes lazy numba compilation of every jitted kernel
# (disk-cached via NUMBA_CACHE_DIR, so a warm cache makes this small).
#
# HAZARD (recorded in the JSON): on a COLD numba cache the very first call of
# ``psf_weighted_data_from`` returns uninitialized-memory garbage (~1e299) with
# identical inputs to the correct second call, making the first likelihood NaN.
# The first call is therefore warm-up only and never enters the timed average.
with timer.section("first_call_incl_numba_compile"):
    log_likelihood_first = analysis.log_likelihood_function(instance=instance)

first_call_s = timer.records[-1][1]
first_call_finite = bool(np.isfinite(float(log_likelihood_first)))
print(f"  log_likelihood (first call) = {log_likelihood_first}")
if not first_call_finite:
    print("  WARNING: first call non-finite (known cold-numba-cache first-call hazard).")

n_repeats = 10

with timer.section(f"steady_x{n_repeats}"):
    log_likelihoods = [
        analysis.log_likelihood_function(instance=instance) for _ in range(n_repeats)
    ]
log_likelihood = log_likelihoods[-1]

per_call = timer.records[-1][1] / n_repeats
print(f"    -> per-call avg: {per_call:.6f} s")

np.testing.assert_allclose(
    np.array([float(ll) for ll in log_likelihoods]),
    float(log_likelihood),
    rtol=1e-8,
    err_msg="numba CPU likelihood is not deterministic across steady-state repeats",
)

# ===================================================================
# Summary + JSON
# ===================================================================

al_version = al.__version__

print("\n" + "=" * 70)
print(f"NUMBA CPU LIKELIHOOD SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Sparse operator setup:      {sparse_operator_setup_s:>12.4f} s (one-off)")
print(f"  Sparse operator memory:     {sparse_operator_nbytes / 1024**2:>12.1f} MB")
print(f"  First call (numba compile): {first_call_s:>12.4f} s")
print(f"  Steady-state per call:      {per_call:>12.6f} s")
print("=" * 70)

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
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "lens_light": "mge_60_linear",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    },
    "full_pipeline_single_jit": per_call,  # headline key aggregate.py reads
    "first_call_incl_numba_compile_s": first_call_s,
    "first_call_finite": first_call_finite,  # False = cold-cache first-call NaN hazard fired
    "sparse_operator_setup_s": sparse_operator_setup_s,
    "sparse_operator_nbytes": sparse_operator_nbytes,
    "log_likelihood": float(log_likelihood),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "runtime" / "imaging" / "pixelization_numba",
    default_basename=f"pixelization_numba_likelihood_summary_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")
print(f"  Bar chart path:        {chart_path} (no per-step chart in runtime variant)")

# ===================================================================
# Pinned-value drift record — profiling records drift, it never adjudicates
# correctness (boundary rule in results/notes/design_lock_in.md).
# ===================================================================

_pinned_drift: list = []

# Pinned 2026-08-20 (v2026.8.17.1). Values vary at the ~1e-9 relative level
# across numba compile sessions (fp reassociation) — rtol=1e-6 accommodates.
EXPECTED_LOG_LIKELIHOOD: dict[str, float] = {
    "euclid": 5860.175003697541,
    "hst": 22803.472837136276,
}

_pinned_expected = EXPECTED_LOG_LIKELIHOOD.get(instrument)

if _pinned_expected is None:
    print(
        f"  Pinned check SKIPPED for {instrument} (no pinned value). "
        f"log_likelihood = {float(log_likelihood)!r}"
    )
else:
    _rec = check_pinned(float(log_likelihood), _pinned_expected, label="numba_cpu", rtol=1e-6)
    if _rec is not None:
        _pinned_drift.append(_rec)

record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")
