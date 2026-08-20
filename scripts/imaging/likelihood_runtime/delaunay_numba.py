"""
Numba CPU Profiling: Delaunay Imaging Likelihood (Runtime)
==========================================================

Profiles the per-evaluation runtime of the **numba CPU sparse-operator**
likelihood for the production Delaunay fiducial — the mesh the large
Euclid-resolution CPU campaigns use:

- Hilbert image-mesh (1250 vertices) placed from the lensed-source adapt image
  (one-off per analysis; passed in via ``al.AdaptImages``).
- ``al.mesh.Delaunay`` source pixelization + ``ConstantSplit`` regularization.
- MGE-60 linear lens light, Isothermal + shear mass.
- ``dataset.apply_sparse_operator_cpu()`` + ``al.AnalysisImaging(use_jax=False)``
  — the numba route of the workspace ``cpu_fast_modeling.py`` example.

Timed quantities (mirrors ``pixelization_numba.py``):

1. One-off ``apply_sparse_operator_cpu()`` setup cost + operator memory, and
   the one-off Hilbert image-mesh placement.
2. First likelihood call (lazy numba compile; warm-up only — on a COLD numba
   cache the first call is additionally known to return garbage, see the
   ``pixelization_numba.py`` hazard note).
3. Steady-state per-call average of ``analysis.log_likelihood_function``.

Output
------
``results/runtime/imaging/delaunay_numba/delaunay_numba_likelihood_summary_<instrument>_v<version>.json``
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

_sys.path.insert(0, str(_profiling_root()))

import json
import os

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os
import sys as _smoke_sys
import time
from contextlib import contextmanager
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
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


class Timer:
    """Accumulates named timing measurements and prints them."""

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
# PART A — Setup (mirrors the JAX delaunay.py fiducial)
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

with timer.section("apply_sparse_operator_cpu"):
    dataset = dataset.apply_sparse_operator_cpu()

sparse_operator_setup_s = timer.records[-1][1]

_op = dataset.sparse_operator
sparse_operator_nbytes = int(
    _op.psf_precision_operator_sparse.nbytes + _op.indexes.nbytes + _op.lengths.nbytes
)
print(f"  sparse operator memory: {sparse_operator_nbytes / 1024**2:.1f} MB")

# ---------------------------------------------------------------------------
# Adapt image + Hilbert image mesh (one-off per analysis — vertices are passed
# to the fit via AdaptImages, not recomputed per evaluation)
# ---------------------------------------------------------------------------

print("\n--- Adapt image + Hilbert image mesh ---")

n_mesh_vertices = 1250  # production campaign fiducial (user-set 2026-08-20)

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

with timer.section("image_mesh_hilbert"):
    image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=adapt_image
    )

hilbert_mesh_s = timer.records[-1][1]
print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

# ---------------------------------------------------------------------------
# Model construction — same fiducial as the JAX delaunay cell
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
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

    mesh = al.mesh.Delaunay(
        pixels=n_mesh_vertices,
        zeroed_pixels=0,
    )
    regularization = al.reg.ConstantSplit(coefficient=1.0)
    pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={
        instance.galaxies.source: image_plane_mesh_grid,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid,
    },
)

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
print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")

# ===================================================================
# PART B — Likelihood evaluation (numba CPU path)
# ===================================================================

print("\n--- Likelihood evaluation (use_jax=False) ---")

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=al.Settings(use_border_relocator=True),
    use_jax=False,
)

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

# First call: warm-up only (lazy numba compile; cold-cache hazard — see the
# pixelization_numba.py sibling).
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

# NOTE: the Delaunay numba likelihood is bistable at the ~1e-8 relative level —
# repeats alternate between two values (observed euclid: 7193.174445 vs
# 7193.174517, i.e. summation-order nondeterminism somewhere in the pipeline).
# Asserted at 1e-7 and the spread recorded in the JSON.
_ll_arr = np.array([float(ll) for ll in log_likelihoods])
log_likelihood_spread = float(np.max(_ll_arr) - np.min(_ll_arr))
np.testing.assert_allclose(
    _ll_arr,
    float(log_likelihood),
    rtol=1e-7,
    err_msg="numba CPU likelihood is not deterministic across steady-state repeats",
)
print(f"  repeat spread: {log_likelihood_spread:.3e}")

# ===================================================================
# Summary + JSON
# ===================================================================

al_version = al.__version__

print("\n" + "=" * 70)
print(f"NUMBA CPU DELAUNAY LIKELIHOOD SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Sparse operator setup:      {sparse_operator_setup_s:>12.4f} s (one-off)")
print(f"  Sparse operator memory:     {sparse_operator_nbytes / 1024**2:>12.1f} MB")
print(f"  Hilbert image mesh:         {hilbert_mesh_s:>12.4f} s (one-off)")
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
        "mesh_shape": [n_mesh_vertices],
        "source_pixels": int(n_source_pixels),
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "mesh": "delaunay_hilbert_1250",
        "regularization": "constant_split",
        "lens_light": "mge_60_linear",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    },
    "full_pipeline_single_jit": per_call,  # headline key aggregate.py reads
    "first_call_incl_numba_compile_s": first_call_s,
    "first_call_finite": first_call_finite,
    "sparse_operator_setup_s": sparse_operator_setup_s,
    "sparse_operator_nbytes": sparse_operator_nbytes,
    "hilbert_image_mesh_s": hilbert_mesh_s,
    "log_likelihood": float(log_likelihood),
    "log_likelihood_repeat_spread": log_likelihood_spread,
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "runtime" / "imaging" / "delaunay_numba",
    default_basename=f"delaunay_numba_likelihood_summary_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")
print(f"  Bar chart path:        {chart_path} (no per-step chart in runtime variant)")

# ===================================================================
# Pinned-value drift record (soft — profiling records drift, never adjudicates
# correctness; boundary rule in results/notes/design_lock_in.md).
# ===================================================================

_pinned_drift: list = []

# Pinned 2026-08-20 (v2026.8.17.1, 1250-vertex fiducial). Delaunay repeats are bistable at the
# ~1e-8 relative level (summation-order nondeterminism) — rtol=1e-6 covers it.
# Captured on a 4-core cloud container (autoarray 2026.8.20.1) after that
# machine reproduced the earlier 1500-vertex euclid pin exactly, so the pins
# are expected hardware-independent.
EXPECTED_LOG_LIKELIHOOD: dict[str, float] = {
    "euclid": 7215.3687893658935,
    "hst": 29090.527192092646,
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
