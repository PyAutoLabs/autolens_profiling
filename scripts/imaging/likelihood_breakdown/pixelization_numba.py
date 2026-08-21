"""
Numba CPU Profiling: Pixelization Imaging Likelihood (Per-Step Breakdown)
=========================================================================

Decomposes the **numba CPU sparse-operator** likelihood (``use_jax=False`` +
``apply_sparse_operator_cpu()`` — the workspace ``cpu_fast_modeling.py`` path)
into its per-evaluation steps, so the dominant costs are visible.

Unlike the JAX breakdown sibling (which re-implements each step as a separate
JIT block), the eager numpy/numba path is decomposed by **sequential
cached-property access**: each repeat builds a fresh ``FitImaging`` and then
touches each lazy property in dependency order, timing every access. A later
property reuses everything already cached, so each timing isolates that step's
incremental cost. The step list follows the traced call graph:

1.  ``FitImaging`` construction
2.  Blurred image (FFT convolution of non-linear light; ~zero for linear MGE)
3.  Profile-subtracted image
4.  Inversion build — ray trace, adaptive mesh + interpolation, mapper
5.  PSF-weighted data                       [numba]
6.  MGE operated mapping matrix — the 60 linear-Gaussian lens-light images
    + their PSF convolution (cached, reused by data vector + curvature)
7.  Mapper sparse triplets — the sparse [data -> source-pixel] mapping
    weights (cached, reused by data vector + curvature)
8.  Data vector D                           [numba]
9.  Curvature matrix F (sparse operator)    [numba]
10. Regularization matrix H
11. F + H
12. Reconstruction solve (BLAS)
13. Mapped reconstructed image              [numba + FFT]
14. Model data + chi-squared
15. log det (F + H)                         (Cholesky, LAPACK)
16. log det H                               (Cholesky, LAPACK)
17. Regularization term s'Hs
18. Log evidence (figure of merit)

A directly-timed full ``log_likelihood_function`` evaluation cross-checks that
the step total accounts for the whole pipeline.

The first evaluation is warm-up only (lazy numba compilation; on a COLD numba
cache the first call of ``psf_weighted_data_from`` is additionally known to
return garbage — see the runtime sibling's hazard note).

Output
------
``results/breakdown/imaging/pixelization_numba_breakdown_<instrument>_v<version>.{json,png}``
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
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    rect_mesh_classes,
    record_pinned_check,
    resolve_output_paths,
)

_cli = parse_profile_cli()

instrument = _cli.instrument or "euclid"  # default; override via --instrument

# ===================================================================
# Setup — identical fiducial to the runtime sibling
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

dataset = dataset.apply_sparse_operator_cpu()

print("\n--- Model construction ---")

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

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
    mesh=rect_mesh_classes(_cli)[0](shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

analysis = al.AnalysisImaging(
    dataset=dataset,
    settings=al.Settings(use_border_relocator=True),
    use_jax=False,
)

from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)

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
# Per-step decomposition via sequential cached-property access
# ===================================================================

STEP_ACCESSORS = [
    ("FitImaging construct", None),  # handled specially (constructor)
    ("Blurred image (FFT convolve)", lambda fit: fit.blurred_image),
    ("Profile subtracted image", lambda fit: fit.profile_subtracted_image),
    ("Inversion build (trace+mesh+mapper)", lambda fit: fit.inversion),
    ("PSF-weighted data [numba]", lambda fit: fit.inversion.psf_weighted_data),
    (
        "MGE operated mapping matrix (60 funcs)",
        lambda fit: fit.inversion.linear_func_operated_mapping_matrix_dict,
    ),
    (
        "Mapper sparse triplets",
        lambda fit: [
            obj.sparse_triplets_data
            for obj in fit.inversion.linear_obj_list
            if hasattr(obj, "sparse_triplets_data")
        ],
    ),
    ("Data vector D [numba]", lambda fit: fit.inversion.data_vector),
    ("Curvature matrix F [numba sparse-op]", lambda fit: fit.inversion.curvature_matrix),
    ("Regularization matrix H", lambda fit: fit.inversion.regularization_matrix),
    ("F + H", lambda fit: fit.inversion.curvature_reg_matrix),
    ("Reconstruction solve (BLAS)", lambda fit: fit.inversion.reconstruction),
    (
        "Mapped reconstruction [numba+FFT]",
        lambda fit: fit.inversion.mapped_reconstructed_operated_data,
    ),
    ("Model data + chi^2", lambda fit: (fit.model_data, fit.chi_squared)),
    (
        "log det (F+H) [Cholesky]",
        lambda fit: fit.inversion.log_det_curvature_reg_matrix_term,
    ),
    (
        "log det H [Cholesky]",
        lambda fit: fit.inversion.log_det_regularization_matrix_term,
    ),
    ("Regularization term s'Hs", lambda fit: fit.inversion.regularization_term),
    ("Log evidence (figure of merit)", lambda fit: fit.figure_of_merit),
]


def one_decomposed_evaluation() -> tuple[dict[str, float], float]:
    """Run one likelihood evaluation, timing each step's incremental cost."""
    step_times: dict[str, float] = {}

    start = time.perf_counter()
    fit = al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=list(instance.galaxies)),
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    step_times["FitImaging construct"] = time.perf_counter() - start

    for label, accessor in STEP_ACCESSORS[1:]:
        start = time.perf_counter()
        result = accessor(fit)
        step_times[label] = time.perf_counter() - start

    return step_times, float(result)  # final accessor is figure_of_merit


# Warm-up evaluation: triggers lazy numba materialization + compilation (and
# absorbs the cold-cache first-call hazard — see module docstring).
print("\n--- Warm-up evaluation (numba compile) ---")
_warm_start = time.perf_counter()
_warm_steps, _warm_fom = one_decomposed_evaluation()
warmup_s = time.perf_counter() - _warm_start
print(f"  warm-up total: {warmup_s:.4f} s (figure_of_merit = {_warm_fom})")

fit_check = analysis.fit_from(instance=instance)
assert isinstance(fit_check.inversion, InversionImagingSparseNumba), (
    f"Expected InversionImagingSparseNumba, got {type(fit_check.inversion).__name__}"
)
del fit_check

n_repeats = 10

print(f"\n--- Timed decomposition (x{n_repeats}) ---")

accumulated: dict[str, float] = {label: 0.0 for label, _ in STEP_ACCESSORS}
for _ in range(n_repeats):
    step_times, figure_of_merit = one_decomposed_evaluation()
    for label, elapsed in step_times.items():
        accumulated[label] += elapsed

likelihood_steps = [(label, accumulated[label] / n_repeats) for label, _ in STEP_ACCESSORS]

# Cross-check: a directly timed, undecomposed evaluation via the production
# entry point. The step total should account for (nearly) all of it.
start = time.perf_counter()
for _ in range(n_repeats):
    log_likelihood_direct = analysis.log_likelihood_function(instance=instance)
direct_per_call = (time.perf_counter() - start) / n_repeats

# ===================================================================
# Per-step breakdown summary + JSON + PNG
# ===================================================================

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"NUMBA CPU PER-STEP BREAKDOWN — {instrument.upper()} — v{al_version}")
print("=" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print(f"      {'Direct log_likelihood_function':<{max_label}}  {direct_per_call:>12.6f} s")
print(f"      {'Coverage (steps / direct)':<{max_label}}  {step_total / direct_per_call:>11.1%}")
print("=" * 70)

breakdown_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "mesh_shape": list(mesh_shape),
        "rect_mesh": _cli.rect_mesh,
        "source_pixels": int(n_source_pixels),
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "lens_light": "mge_60_linear",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "direct_log_likelihood_function_per_call": direct_per_call,
    "warmup_incl_numba_compile_s": warmup_s,
    "log_likelihood": float(log_likelihood_direct),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"pixelization_numba_breakdown_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color="#4C72B0", edgecolor="white", height=0.6)

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
    f"Numba CPU Pixelization Likelihood — Per-Step Breakdown — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_over_sampled_pixels} over-sampled  |  {mesh_shape[0]}x{mesh_shape[1]} mesh  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.margins(x=0.15)
fig.tight_layout()

fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

# ===================================================================
# Pinned-value drift record (soft — profiling records drift, never adjudicates
# correctness; boundary rule in results/notes/design_lock_in.md).
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
        f"log_likelihood = {float(log_likelihood_direct)!r}"
    )
else:
    _rec = check_pinned(
        float(log_likelihood_direct), _pinned_expected, label="numba_cpu", rtol=1e-6
    )
    if _rec is not None:
        _pinned_drift.append(_rec)

record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")
