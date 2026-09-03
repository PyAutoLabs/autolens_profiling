"""
Numba CPU Profiling: Pixelization Imaging Likelihood — fixed MGE lens *mass*
===========================================================================

The sibling of ``pixelization_numba.py`` for the model the fixed-geometry deflection
memo (PyAutoGalaxy#601) exists for: the SLaM ``mass_light_dark`` shape.

``pixelization_numba.py`` gives the lens a **light-only** MGE (``lp_linear.Gaussian``),
so its deflections come from one ``Isothermal`` and the MGE never enters the ray-trace.
Here the same 30-component MGE is a **light-and-mass** basis
(``lmp_linear.Gaussian``) whose centre, ``ell_comps``, ``intensity`` and ``sigma`` are
all fixed — exactly as ``autogalaxy/analysis/chaining_util.py::mass_light_dark_basis_from``
fixes them from the light-stage instance — and the whole stack shares **one** free
``mass_to_light_ratio``. A dark ``NFWSph`` plus an ``ExternalShear`` complete the mass
model, and the source is the same rectangular pixelization.

Every likelihood evaluation therefore ray-traces 30 Gaussians whose geometry never
moves, which is the case
``autogalaxy/profiles/mass/abstract/deflections_memo.py`` memoises.

What is measured
----------------

Five consecutive ``analysis.log_likelihood_function`` calls with the memo on and five
with ``AUTOGALAXY_DEFLECTIONS_MEMO=0``, plus the assertion that the two agree — the memo
rescales an exactly-linear factor, so the likelihood must not move.

This cell keeps its **own** pin. It does not read, run or re-pin
``pixelization_numba.py``, whose hst bilinear pin (27661.910133665442) is a different
model and is left untouched.

Output
------
``results/runtime/imaging/pixelization_numba_mge_mass/pixelization_numba_mge_mass_likelihood_summary_<instrument>_v<version>.json``
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

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os  # noqa: E402
import sys as _smoke_sys  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402
from autogalaxy.profiles.mass.abstract import deflections_memo as _memo  # noqa: E402
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)

_cli = parse_profile_cli()

instrument = _cli.instrument or "hst"

MASK_RADIUS = 3.5
TOTAL_GAUSSIANS = 30
SIGMA_MIN = 0.01
MESH_PIXELS_YX = 28
N_REPEATS = 5

# The log-likelihood this model produces, per (rect mesh, instrument). Filled from the
# first run of a new key (the run prints the value and says so); checked at rtol 1e-6
# thereafter. Values move at ~1e-9 relative across numba compile sessions.
EXPECTED_LOG_LIKELIHOOD: dict[str, dict[str, float]] = {
    "bilinear": {"hst": -56107.564075886374},
}

# ===================================================================
# PART A — Setup (identical to pixelization_numba.py)
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

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=MASK_RADIUS,
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
dataset = dataset.apply_sparse_operator_cpu()

# ---------------------------------------------------------------------------
# Model construction — the SLaM mass_light_dark shape
# ---------------------------------------------------------------------------

print("\n--- Model construction (fixed MGE light-and-mass + free ratio) ---")

log10_sigma_list = np.linspace(np.log10(SIGMA_MIN), np.log10(MASK_RADIUS), TOTAL_GAUSSIANS)

_bulge_ell = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)

gaussian_list = af.Collection(af.Model(al.lmp_linear.Gaussian) for _ in range(TOTAL_GAUSSIANS))

# One shared free parameter across the whole basis: the mass-to-light ratio.
mass_to_light_ratio = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)

for index, gaussian in enumerate(gaussian_list):
    gaussian.centre = (0.0, 0.0)
    gaussian.ell_comps = (float(_bulge_ell[0]), float(_bulge_ell[1]))
    gaussian.sigma = float(10 ** log10_sigma_list[index])
    gaussian.intensity = 1.0
    gaussian.mass_to_light_ratio = mass_to_light_ratio

lens_bulge = af.Model(al.lp_basis.Basis, profile_list=gaussian_list)

dark = af.Model(al.mp.NFWSph)
dark.centre = (0.0, 0.0)
dark.kappa_s = af.GaussianPrior(mean=0.2, sigma=0.01)
dark.scale_radius = 10.0

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, dark=dark, shear=shear)

pixelization = al.Pixelization(
    mesh=rect_mesh_classes(_cli)[0](shape=(MESH_PIXELS_YX, MESH_PIXELS_YX)),
    regularization=al.reg.Constant(coefficient=1.0),
)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

_lens_instance = instance.galaxies.lens
_basis_instance = _lens_instance.bulge

print(f"  MGE components:        {len(_basis_instance.profile_list)}")
print(f"  Shared ratio:          {_basis_instance.profile_list[0].mass_to_light_ratio!r}")
print(f"  OMP_NUM_THREADS:       {os.environ.get('OMP_NUM_THREADS', '(unset)')}")

# ===================================================================
# PART B — memo on vs memo off
# ===================================================================

analysis = al.AnalysisImaging(
    dataset=dataset,
    settings=al.Settings(use_border_relocator=True),
    use_jax=False,
)

from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)

_fit_check = analysis.fit_from(instance=instance)
assert isinstance(_fit_check.inversion, InversionImagingSparseNumba), (
    f"Expected InversionImagingSparseNumba, got {type(_fit_check.inversion).__name__} — "
    "the numba sparse-operator path is not engaged."
)
del _fit_check


def _measure(memo_enabled: bool) -> tuple[float, float, list]:
    """Median and total per-call time over ``N_REPEATS`` calls, plus the likelihoods.

    The numba kernels and the memo are both warmed by a call that is not timed, so the
    two legs differ only in the memo.
    """
    os.environ["AUTOGALAXY_DEFLECTIONS_MEMO"] = "1" if memo_enabled else "0"
    _memo.memo_clear()

    analysis.log_likelihood_function(instance=instance)

    samples = []
    likelihoods = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        likelihoods.append(float(analysis.log_likelihood_function(instance=instance)))
        samples.append(time.perf_counter() - start)

    return float(statistics.median(samples)), float(sum(samples)), likelihoods


# The numba kernels compile on the first call of the process; that call is inside the
# `off` leg's untimed warm-up, and the `on` leg runs afterwards, so neither leg pays it.
off_median_s, off_total_s, off_likelihoods = _measure(memo_enabled=False)
on_median_s, on_total_s, on_likelihoods = _measure(memo_enabled=True)

log_likelihood = on_likelihoods[-1]
max_relative = float(
    max(abs(on - off) / max(abs(off), 1e-300) for on, off in zip(on_likelihoods, off_likelihoods))
)

memo_stats = _memo.memo_stats()

print("\n" + "=" * 74)
print(f"FIXED-MGE-MASS NUMBA CPU LIKELIHOOD — {instrument.upper()} — v{al.__version__}")
print("=" * 74)
print(f"  memo OFF, per call (median of {N_REPEATS}):  {off_median_s:>10.4f} s")
print(f"  memo ON,  per call (median of {N_REPEATS}):  {on_median_s:>10.4f} s")
print(f"  speed-up:                              {off_median_s / on_median_s:>10.2f}x")
print(f"  log_likelihood (memo on):              {log_likelihood!r}")
print(f"  log_likelihood (memo off):             {off_likelihoods[-1]!r}")
print(f"  max relative difference:               {max_relative:.3e}")
print(
    f"  memo entries / bytes:                  {memo_stats['entries']} / "
    f"{memo_stats['bytes'] / 1024**2:.2f} MB (cap "
    f"{_memo.memo_max_bytes() / 1024**2:.0f} MB)"
)
print("=" * 74)

assert max_relative < 1e-10, (
    f"memo changed the log likelihood by {max_relative:.3e} relative "
    f"({on_likelihoods[-1]!r} vs {off_likelihoods[-1]!r})"
)

# ===================================================================
# Write + pinned-value drift record
# ===================================================================

al_version = al.__version__

summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": MASK_RADIUS,
        "image_pixels_masked": int(dataset.data.shape[0]),
        "over_sampled_pixels": int(dataset.grids.lp.over_sampled.shape[0]),
        "mesh_shape": [MESH_PIXELS_YX, MESH_PIXELS_YX],
        "rect_mesh": _cli.rect_mesh,
        "source_pixels": MESH_PIXELS_YX**2,
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "lens_light": f"mge_{TOTAL_GAUSSIANS}_lmp_linear_fixed_geometry",
        "lens_mass": "mge_basis + NFWSph + ExternalShear",
        "free_parameters": int(model.total_free_parameters),
        "n_repeats": N_REPEATS,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    },
    "full_pipeline_single_jit": on_median_s,
    "deflections_memo_off_s": off_median_s,
    "deflections_memo_on_s": on_median_s,
    "deflections_memo_speedup": off_median_s / on_median_s,
    "deflections_memo_total_off_s": off_total_s,
    "deflections_memo_total_on_s": on_total_s,
    "deflections_memo_max_relative_likelihood_difference": max_relative,
    "deflections_memo_stats": memo_stats,
    "log_likelihood": log_likelihood,
    "log_likelihood_memo_off": off_likelihoods[-1],
}

dict_path, _chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "runtime" / "imaging" / "pixelization_numba_mge_mass",
    default_basename=(f"pixelization_numba_mge_mass_likelihood_summary_{instrument}_v{al_version}"),
)
dict_path.write_text(json.dumps(summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

_pinned_drift: list = []
_pinned_expected = EXPECTED_LOG_LIKELIHOOD.get(_cli.rect_mesh, {}).get(instrument)

if _pinned_expected is None:
    print(
        f"  Pinned check SKIPPED for {_cli.rect_mesh}/{instrument} (no pinned value). "
        f"log_likelihood = {log_likelihood!r} — paste it into "
        f"EXPECTED_LOG_LIKELIHOOD to pin it."
    )
else:
    _record = check_pinned(log_likelihood, _pinned_expected, label="numba_cpu_mge_mass", rtol=1e-6)
    if _record is not None:
        _pinned_drift.append(_record)

record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
