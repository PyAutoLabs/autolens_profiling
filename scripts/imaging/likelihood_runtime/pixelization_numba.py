"""
Numba CPU Profiling: Pixelization Imaging Likelihood (Runtime)
==============================================================

Profiles the per-evaluation runtime of the **numba CPU sparse-operator**
likelihood — the path production runs on CPU-abundant hardware
(``dataset.apply_sparse_operator_cpu()`` + ``al.AnalysisImaging(use_jax=False)``,
eager numpy + numba, no JAX).

Since 2026-09-08 (autolens_profiling#235) the cell defaults to the **production
configuration** rather than a fiducial of its own: ``--variant production``
resolves the rectangular production stage from ``_production_config`` —
``subhalo_validation``'s ``rect_adapt`` (32x32, ``Adapt`` regularization, S/N-3
capped adapt image, MGE 30x2) for ``--instrument hst``, the same stage on the
Euclid dataset plus the Euclid positions penalty for ``--instrument euclid``,
both with pixelization over-sampling ``4`` where the source S/N exceeds 3 and
``2`` elsewhere. ``--variant legacy`` rebuilds the pre-2026-09-08 cell (28x28
``RectangularBilinearAdaptDensity``, ``Constant(1.0)``, flat
``over_sample_size_pixelization=1``, MGE 60x1) so the historic rows stay
reproducible.

Three things about the protocol matter more than the configuration:

- **Threads are pinned to 1** before numpy imports, as both production submit
  scripts do. A pooled 8-worker Nautilus run gives each worker one core.
- **The instances are an iid stream**, not one prior-median instance repeated.
  Repeating one instance lets the NNLS cross-evaluation warm-start memo seed
  itself from a 100 %-correct previous solve, which production never gets.
- **The memo is off by default and recorded** (``--memo on`` to measure it).

Timed quantities:

1. One-off ``apply_sparse_operator_cpu()`` setup cost + operator memory.
2. Instance 0 — discarded warm-up (lazy numba compile).
3. ``--cold-evals`` cold evaluations — the quantity comparable to PyAutoFit's
   logged "Log Likelihood Function Evaluation Time"
   (``autofit/non_linear/search/updater.py:311-317``), i.e. the production
   ``s/sample x speed-up``.
4. The remaining instances — warm iid evaluations, median and mean.

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

_sys.path.insert(0, str(_profiling_root()))

# The CLI is parsed and the thread environment pinned BEFORE numpy is imported:
# OpenBLAS / MKL read their thread-count variables once, when the shared library
# loads, so pinning after `import numpy` has no effect on the pools the timings
# actually run through. Both modules are stdlib-only at import time.
from _production_config import (  # noqa: E402
    observe_thread_env as _observe_thread_env,
)
from _production_config import (
    pin_thread_env as _pin_thread_env,
)
from _profile_cli import parse_profile_cli as _parse_profile_cli  # noqa: E402

_cli = _parse_profile_cli()

thread_env = _pin_thread_env(1) if _cli.variant == "production" else _observe_thread_env()

import json  # noqa: E402
import os  # noqa: E402

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke). Verifies the import
# graph + module-level setup succeeded without running the full profile.
import os as _smoke_os
import sys as _smoke_sys
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _production_config import (  # noqa: E402
    adapt_image_capped,
    adapt_images_for,
    analysis_settings,
    apply_production_over_sampling,
    iid_instances,
    mge_lens_bulge,
    positions_likelihood,
    preset_for,
    regularization_model,
    timing_summary,
    witness_verdict,
)
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)

instrument = _cli.instrument or "euclid"  # default; override via --instrument

preset = preset_for("pixelization_numba", instrument=instrument, variant=_cli.variant)

print(f"\n--- Preset [{preset.name} / {preset.variant}] ---")
print(f"  {preset.provenance}")
if thread_env["overridden"]:
    print(f"  WARNING: thread env overridden from {thread_env['overridden']} to 1.")

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

    # The source adapt image, capped at the preset's S/N if it has one — the
    # same map that drives the adaptive mesh, the adaptive regularization AND
    # the pixelization over-sampling rule, exactly as production.
    adapt_image = adapt_image_capped(
        adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset),
        preset,
    )

    # Light-profile radial bins, then the S/N-driven pixelization map, then the
    # sparse CPU operator — production's order. `apply_over_sampling` returns a
    # fresh `Imaging` that drops the precomputed operator, so it is re-applied
    # last or the fit silently falls back to the dense inversion.
    dataset = apply_production_over_sampling(dataset, adapt_image, preset)

# The one-off numba sparse-operator setup is folded into the over-sampling
# section above (production re-applies it after every over-sampling change), so
# it is timed here as a fresh re-apply on the finished dataset: the same one-off
# cost, still reported separately because campaigns pay it per lens.
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

mesh_pixels_yx = preset.rect_pixels_yx
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

with timer.section("model_build"):
    # Linear MGE lens light — enters the inversion as linear-func columns
    # alongside the source-pixel columns, as in the production stages where the
    # LP-stage MGE rides through as a frozen instance. Basis structure matters,
    # not just the Gaussian count: production is 20x2 (Euclid) / 30x2 (subhalo),
    # the pre-2026-09-08 cell was 60x1.
    lens_bulge = mge_lens_bulge(preset, mask_radius=mask_radius)

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

    # Production samples the regularization coefficients, so `regularization`
    # comes back as a free `af.Model` under `--variant production` and as the
    # historic fixed instance under `--variant legacy`. A free regularization
    # has to ride inside an `af.Model(al.Pixelization)` for its priors to reach
    # the model.
    regularization = regularization_model(preset)
    mesh = rect_mesh_classes(_cli)[0](shape=mesh_shape)

    if isinstance(regularization, af.Model):
        pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
    else:
        pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

# The instance sequence. Production's Nautilus pool hands each worker draws that
# are, from that worker's point of view, unrelated — so the profiled stream is
# `--n-instances` iid draws from the central 20 % of every prior. The legacy
# variant keeps the single prior-median instance the historic rows repeated.
n_instances = max(int(_cli.n_instances), 2 + int(_cli.cold_evals))
n_cold = int(_cli.cold_evals)

with timer.section("instance_sequence"):
    if preset.sequence == "iid":
        instances = iid_instances(model, n_instances, preset)
    else:
        _median = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
        instances = [_median] * n_instances

instance = instances[0]

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]
n_source_pixels = mesh_pixels_yx * mesh_pixels_yx

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Preset / variant:        {preset.name} / {preset.variant}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Mesh shape:              {mesh_shape}")
print(f"  Source pixels:           {n_source_pixels}")
print(f"  Sequence:                {preset.sequence} x {n_instances} (seed {preset.seed})")
print(f"  Memo:                    {_cli.memo} (resolved below per variant)")
print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")

# ===================================================================
# PART B — Likelihood evaluation (numba CPU path)
# ===================================================================

print("\n--- Likelihood evaluation (use_jax=False) ---")

# `--memo` only applies to the production variant: the legacy variant must
# leave both memo gates untouched (`library_default`), because the historic rows
# it exists to reproduce were measured with the library default (`true`).
memo = _cli.memo if preset.variant == "production" else preset.memo

settings, memo_provenance = analysis_settings(preset, memo)
print(
    f"  memo: requested {memo_provenance['requested']}, "
    f"Settings.nnls_warm_start_memo={memo_provenance['settings_nnls_warm_start_memo']}, "
    f"AUTOARRAY_NNLS_WARM_START={memo_provenance['env_AUTOARRAY_NNLS_WARM_START']}"
)

# The Euclid stage carries a positions penalty; the subhalo `source_pix[2]`
# stage does not. It is a per-evaluation cost (the positions are ray-traced
# every call), so it belongs inside the timed likelihood, not outside it.
positions_lh = positions_likelihood(
    preset,
    dataset_path=dataset_path,
    tracer=al.Tracer(galaxies=list(instance.galaxies)),
)
positions_likelihood_list = [positions_lh] if positions_lh is not None else None
if positions_lh is not None:
    print(
        f"  positions penalty: {len(positions_lh.positions)} images, "
        f"threshold {positions_lh.threshold:.4f}"
    )


def analysis_for(one_instance):
    """A fresh `AnalysisImaging` for one instance of the iid sequence.

    The adapt-image dicts are keyed on the instance's own source-galaxy object,
    so they cannot be shared across the sequence — each instance gets its own
    `AdaptImages`, and therefore its own analysis.
    """
    return al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images_for(one_instance, adapt_image=adapt_image),
        positions_likelihood_list=positions_likelihood_list,
        settings=settings,
        use_jax=False,
    )


analysis = analysis_for(instance)

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

# One timed evaluation per instance of the sequence.
#
# Instance 0 is warm-up only and is discarded: it pays the lazy numba
# compilation of every jitted kernel (disk-cached via NUMBA_CACHE_DIR), and on a
# COLD numba cache the very first call of ``psf_weighted_data_from`` returns
# uninitialized-memory garbage (~1e299) with identical inputs to the correct
# second call, making that likelihood NaN. Instances 1..n_cold are the cold
# evaluations; the rest are warm.
per_eval_s: list[float] = []
log_likelihoods: list[float] = []

for index, one_instance in enumerate(instances):
    _analysis = analysis if index == 0 else analysis_for(one_instance)
    _start = time.perf_counter()
    _ll = _analysis.log_likelihood_function(instance=one_instance)
    per_eval_s.append(time.perf_counter() - _start)
    log_likelihoods.append(float(_ll))
    if index == 0:
        print(f"  [instance 0 warm-up] {per_eval_s[0]:.4f} s (log_likelihood = {_ll})")
    elif index <= n_cold:
        print(f"  [instance {index} cold] {per_eval_s[index]:.4f} s")

first_call_s = per_eval_s[0]
first_call_finite = bool(np.isfinite(log_likelihoods[0]))
if not first_call_finite:
    print("  WARNING: first call non-finite (known cold-numba-cache first-call hazard).")

timing = timing_summary(per_eval_s, n_cold=n_cold)
per_call = timing["warm_iid_median_s"]  # headline: the warm iid MEDIAN

print(f"  cold eval median:   {timing['cold_eval_median_s']:.6f} s (n = {timing['n_cold']})")
print(f"  warm iid median:    {timing['warm_iid_median_s']:.6f} s (n = {timing['n_warm']})")
print(f"  warm iid mean:      {timing['warm_iid_mean_s']:.6f} s")

witness = witness_verdict(timing["cold_eval_median_s"], instrument)
if witness.get("verdict") in ("PASS", "FAIL"):
    print(
        f"  witness vs production job {witness['job']} "
        f"({witness['reference_cold_eval_s']} s, x{witness['factor']}): {witness['verdict']}"
    )

# The pinned log-likelihood is the LAST instance of the sequence, so it is a
# deterministic function of (model, seed, n_instances) rather than of an
# arbitrary repeat.
log_likelihood = log_likelihoods[-1]

if preset.sequence != "iid":
    np.testing.assert_allclose(
        np.array(log_likelihoods[1:]),
        float(log_likelihood),
        rtol=1e-8,
        err_msg="numba CPU likelihood is not deterministic across repeats of one instance",
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
print(f"  Warm-up (numba compile):    {first_call_s:>12.4f} s (discarded)")
print(f"  Cold eval median:           {timing['cold_eval_median_s']:>12.6f} s")
print(f"  Warm iid median per call:   {per_call:>12.6f} s")
print(f"  Warm iid mean per call:     {timing['warm_iid_mean_s']:>12.6f} s")
print("=" * 70)

likelihood_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        **preset.as_json(),
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "mesh_shape": list(mesh_shape),
        "rect_mesh": _cli.rect_mesh,
        "source_pixels": int(n_source_pixels),
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "n_instances": n_instances,
        "thread_env": thread_env,
        "memo_provenance": memo_provenance,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    },
    # Headline key aggregate.py reads. Since 2026-09-08 it is the warm iid
    # MEDIAN, not the arithmetic mean of ten repeats of one instance — the
    # pre-#235 rows are a different quantity and are not comparable.
    "full_pipeline_single_jit": per_call,
    **timing,
    "witness": witness,
    "first_call_incl_numba_compile_s": first_call_s,
    "first_call_finite": first_call_finite,  # False = cold-cache first-call NaN hazard fired
    "sparse_operator_setup_s": sparse_operator_setup_s,
    "sparse_operator_nbytes": sparse_operator_nbytes,
    "log_likelihood": float(log_likelihood),
    "log_likelihood_sequence": log_likelihoods,
}

# `--variant legacy` writes to its own basename so a legacy row can never
# overwrite (or be mistaken for) the production row of the same instrument.
_variant_suffix = "" if preset.variant == "production" else f"_{preset.variant}"

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "runtime" / "imaging" / "pixelization_numba",
    default_basename=(
        f"pixelization_numba_likelihood_summary_{instrument}{_variant_suffix}_v{al_version}"
    ),
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")
print(f"  Bar chart path:        {chart_path} (no per-step chart in runtime variant)")

# ===================================================================
# Pinned-value drift record — profiling records drift, it never adjudicates
# correctness (boundary rule in results/notes/design_lock_in.md).
# ===================================================================

_pinned_drift: list = []

# Keyed by variant, then --rect-mesh, then instrument.
#
# Every pre-2026-09-08 pin is GONE, not moved: autolens_profiling#235 replaced
# this cell's fiducial with the production configuration (mesh, over-sampling,
# regularization, MGE basis, instance sequence all changed), so the old values
# describe a model that no longer exists here. They are recorded in
# `results/notes/production_representative_cells.md` for the historic rows.
#
# The `legacy` pins are the historic values, which the legacy variant no longer
# reproduces exactly either: the light-profile radial bins moved from [4, 2, 1]
# to [4, 2, 2] repo-wide (sub-size 1 causes gradient issues), which shifts the
# likelihood. They were re-measured on 2026-09-08 with that change in place.
#
# Values vary at the ~1e-9 relative level across numba compile sessions (fp
# reassociation) — rtol=1e-6 accommodates. A missing entry resolves to None and
# skips the check, printing the measured value to paste back in.
# Pinned 2026-09-08 from this cell's first production run per instrument on the
# local WSL host (fp64 numba sparse path, threads pinned to 1, memo off,
# --n-instances 20 --cold-evals 3, seed 235): the value is the LAST instance of
# the seeded iid sequence, so it is a deterministic function of
# (model, seed, n_instances). Libraries: PyAutoArray 47a00e8c, PyAutoFit
# 08207bad0, PyAutoGalaxy ec5ce75d, PyAutoLens 08a05858a, PyAutoNerves 0e7163b,
# autolens 2026.8.17.1.
#
# Euclid RE-PINNED 2026-09-08 (autolens_profiling#237): the Euclid presets'
# light-profile radial bins moved from `[4, 2, 2]` to `[4, 4, 2]`, following
# euclid_strong_lens_modeling_pipeline#56 (`util.py:931-937`) — sub-size 2 in
# the 0.1-0.3" annulus under-integrates a compact source. Over-sampled pixels
# 15424 -> 15664. A control run of `likelihood_runtime/delaunay_numba.py` with
# the Euclid presets put back to `[4, 2, 2]`, on the same libraries, re-PASSED
# its old pin unchanged, so the whole move is the bin change, none of it drift. HST is
# untouched (subhalo_validation still runs `[4, 2, 2]`).
EXPECTED_LOG_LIKELIHOOD: dict[str, dict[str, dict[str, float]]] = {
    "production": {
        "bilinear": {
            "euclid": 4243.160082020453,
            "hst": 22677.756578185603,
        },
    },
    "legacy": {},
}

_pinned_expected = (
    EXPECTED_LOG_LIKELIHOOD.get(preset.variant, {}).get(_cli.rect_mesh, {}).get(instrument)
)

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
