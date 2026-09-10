"""
Numba CPU Profiling: Pixelization Interferometer Likelihood (Per-Step Breakdown)
================================================================================

Decomposes the **recovered numba w-tilde** interferometer likelihood for the
production rectangular fiducial (32x32 = 1024 adaptive rectangular source
pixels, ``reg.Constant`` regularization, ``Isothermal`` + ``ExternalShear`` lens
mass, no lens light) into its per-evaluation steps.

The likelihood itself is the pack under ``scripts/misc/numba_interferometer/``
— the numba kernels PyAutoArray deleted in Dec 2025 – Feb 2026, wired to today's
Mapper / Mask2D / inversion API. PyAutoArray is untouched by this script; the
JAX/FFT sparse path is the comparator (the same one
``likelihood_runtime/pixelization.py`` profiles), and its figure of merit is
recorded here as ``jax_reference_log_evidence``.

Decomposition method (as ``scripts/imaging/likelihood_breakdown/pixelization_numba.py``):
each repeat builds a fresh ``FitInterferometer`` and a fresh
``InversionInterferometerNumba``, then touches each lazy cached property in
dependency order, timing every access — each timing isolates that step's
incremental cost. The adapt image and the ``W~`` preload + dirty image are
one-off per dataset, so neither is a per-evaluation step and neither is timed
here.

A directly-timed full numba log-evidence evaluation cross-checks the step total.
The first evaluation is warm-up only (lazy numba compile).

Output
------
``results/breakdown/interferometer/pixelization_numba_breakdown_<instrument>_v<version>.{json,png}``
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

import argparse
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
from autoarray.fit import fit_util

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402
from numba_interferometer.inversion import (  # noqa: E402
    KERNELS,
    InversionInterferometerNumba,
)
from numba_interferometer.preload import (  # noqa: E402
    NumbaPreload,
    curvature_preload_from,
)

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)
from instruments.interferometer import INSTRUMENTS  # noqa: E402

_cli = parse_profile_cli()

# ``--kernel`` selects which curvature kernel assembles F. The shared profiling CLI
# uses ``parse_known_args``, so this second parser can claim its own flag without
# either seeing the other's. ``"jax"`` is not a numba kernel: it swaps the whole
# inversion for ``InversionInterferometerSparse``, the JAX/FFT path this pack exists
# to be compared against, so that both arms are measured by one harness on one fit.
_kernel_parser = argparse.ArgumentParser(add_help=False)
_kernel_parser.add_argument("--kernel", default="reference")
kernel = _kernel_parser.parse_known_args()[0].kernel

if kernel != "jax" and kernel not in KERNELS:
    raise SystemExit(f"--kernel {kernel!r} is not available; choose one of {KERNELS + ('jax',)}.")

is_jax_arm = kernel == "jax"

instrument = _cli.instrument or "sma"  # default; override via --instrument

CELL = "pixelization_numba"

mesh_pixels_yx = 32  # 32x32 = 1024 source pixels — 1000-tier production fiducial
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
regularization_coefficient = 1.0

# The reference scatter kernel is O(N_pix^2 * P^2) per curvature build, and the
# bilinear rectangular mapper has P = 4 against Delaunay's 3. Scale the repeat
# count so a run stays minutes, not hours, on the larger instruments.
N_REPEATS = {"sma": 10}.get(instrument, 3)

# ===================================================================
# Setup — identical fiducial to the likelihood_runtime/pixelization.py sibling
# ===================================================================

print(f"\n--- Dataset loading [{instrument}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
mask_radius = INSTRUMENTS[instrument]["mask_radius"]
transformer_chunk_size = INSTRUMENTS[instrument].get("transformer_chunk_size", None)

dataset_path = Path("dataset") / "interferometer" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="interferometer",
    instrument=instrument,
    workspace_root=_workspace_root,
)

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)


def _build_transformer(uv_wavelengths, real_space_mask):
    """Per-instrument NUFFT chunk_size, as ``delaunay.py`` does it (PyAutoArray#330)."""
    return al.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=transformer_chunk_size,
    )


dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=_build_transformer,
)

n_visibilities = dataset.uv_wavelengths.shape[0]
print(f"  Total visibilities: {n_visibilities}")

print("\n--- Sparse operator (one-off: W~ preload + dirty image) ---")

# use_jax=False keeps the whole numba path off JAX. The preload this builds is the
# same array the numba pack reads; `dirty_image` is taken straight off the operator.
#
# The preload is `O(N_pix * K)` — seconds at sma, 10-15 minutes at alma's million
# visibilities — and it is model-independent, geometry-keyed and identical across every
# kernel arm. It is therefore built once and cached beside the dataset, then handed to
# both `apply_sparse_operator` and `NumbaPreload`, so a paired B/A/B/A sweep over
# kernels pays it once rather than twice per arm. The cache key carries the grid shape
# and mask radius because those are what the array's shape and values depend on.
_preload_cache_path = (
    dataset_path
    / f"nufft_precision_operator_{real_space_shape[0]}x{real_space_shape[1]}_r{mask_radius}.npy"
)

_preload_start = time.perf_counter()

if _preload_cache_path.exists():
    curvature_preload = np.load(_preload_cache_path)
    preload_cache_hit = True
else:
    curvature_preload = curvature_preload_from(dataset)
    np.save(_preload_cache_path, curvature_preload)
    preload_cache_hit = False

dataset = dataset.apply_sparse_operator(use_jax=False, nufft_precision_operator=curvature_preload)
numba_preload = NumbaPreload.from_curvature_preload(dataset, curvature_preload)
preload_s = time.perf_counter() - _preload_start
print(
    f"  preload + dirty image: {preload_s:.4f} s "
    f"({'cache hit' if preload_cache_hit else 'built + cached'}: {_preload_cache_path})"
)

print("\n--- Adapt image (one-off; drives the adaptive rectangular mesh) ---")

adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

print("\n--- Model construction ---")

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

pixelization = al.Pixelization(
    mesh=rect_mesh_classes(_cli)[1](shape=mesh_shape, weight_power=1.0, weight_floor=0.0),
    regularization=al.reg.Constant(coefficient=regularization_coefficient),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

adapt_images = al.AdaptImages(
    galaxy_image_dict={instance.galaxies.source: adapt_image},
    galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
)

settings = al.Settings()

n_image_pixels = int(dataset.real_space_mask.pixels_in_mask)
n_source_pixels = mesh_shape[0] * mesh_shape[1]

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Visibilities:            {n_visibilities}")
print(f"  Mesh shape:              {mesh_shape[0]} x {mesh_shape[1]}")
print(f"  Source pixels:           {n_source_pixels}")
print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")

# ===================================================================
# Per-step decomposition via sequential cached-property access
# ===================================================================

# -------------------------------------------------------------------
# Curvature matrix F split
# -------------------------------------------------------------------
#
# F is built by one numba scatter over image-pixel pairs
# (``InversionInterferometerNumba.curvature_matrix_scatter``, the recovered
# ``curvature_matrix_via_w_tilde_curvature_preload_interferometer_from``) and then
# finished by ``curvature_matrix``, which applies only the no-regularization
# diagonal add — the scatter kernel loops the full N x N pair space, so unlike the
# imaging numba class and the modern interferometer sparse class there is no
# mirroring pass to pay for.
#
# Both are ``cached_property``, and the scatter row is touched first, so the F row
# that follows is a *direct measurement* of the remaining work rather than an
# arithmetic residual (the imaging sibling's helpers are uncached, which is why it
# has to subtract). The two rows therefore sum, by construction, to the whole cost
# of ``inversion.curvature_matrix``, recorded as ``curvature_matrix_f_total_s``.

INVERSION_BUILD_LABEL = "Inversion build (trace+mesh+mapper)"
H_LABEL = "Regularization matrix H (Constant)"
F_SCATTER_LABEL = "F: mapper×mapper [numba preload scatter]"
F_RESIDUAL_LABEL = "Curvature matrix F [residual: mirror + diag-add]"

_state: dict = {}


def _build_inversion(fit):
    """Ray-trace, place the mesh, build the mapper, then wrap it in the numba inversion.

    ``fit.inversion`` does the per-evaluation geometry (tracing the data grid to the
    source plane, placing the adaptive rectangular mesh, border relocation and the
    mapper). ``InversionInterferometerNumba`` then reuses that
    ``linear_obj_list`` verbatim, so the only thing this adds to the step is the
    inversion object itself plus the precondition checks — the preload is passed in,
    never rebuilt.
    """
    if is_jax_arm:
        _state["inversion"] = fit.inversion
        return _state["inversion"]

    _state["inversion"] = InversionInterferometerNumba(
        dataset=fit.inversion.dataset,
        linear_obj_list=fit.inversion.linear_obj_list,
        settings=fit.inversion.settings,
        xp=np,
        numba_preload=numba_preload,
        kernel=kernel,
    )
    return _state["inversion"]


def _inversion(_fit):
    return _state["inversion"]


def _term(name: str, accessor):
    """Time an evidence term and stash its value for the final assembly row.

    ``fast_chi_squared``, ``regularization_term`` and the two log-determinants are
    plain (uncached) properties on ``AbstractInversion``, so re-reading them in the
    'Log evidence' row would time them a second time and push the step total above
    the directly-measured likelihood. Each term row therefore records its own value
    and the final row assembles those, measuring only ``log_evidence_from`` plus
    ``noise_normalization``.
    """

    def step(fit):
        _state[name] = accessor(_inversion(fit))
        return _state[name]

    return step


def _log_evidence(fit):
    return fit_util.log_evidence_from(
        chi_squared=_state["chi_squared"],
        regularization_term=_state["regularization_term"],
        log_curvature_regularization_term=_state["log_det_curvature_reg"],
        log_regularization_term=_state["log_det_reg"],
        noise_normalization=fit.noise_normalization,
    )


STEP_ACCESSORS = [
    ("FitInterferometer construct", None),  # handled specially (constructor)
    (INVERSION_BUILD_LABEL, _build_inversion),
    ("Mapper index/weight arrays", lambda fit: _inversion(fit).mapper_index_arrays),
    ("Data vector D [numba]", lambda fit: _inversion(fit).data_vector),
    (F_SCATTER_LABEL, lambda fit: _inversion(fit).curvature_matrix_scatter),
    (F_RESIDUAL_LABEL, lambda fit: _inversion(fit).curvature_matrix),
    (
        H_LABEL,
        lambda fit: _inversion(fit).regularization_matrix,
    ),
    ("F + H", lambda fit: _inversion(fit).curvature_reg_matrix),
    ("Reconstruction solve", lambda fit: _inversion(fit).reconstruction),
    ("Fast chi^2", _term("chi_squared", lambda inv: inv.fast_chi_squared)),
    (
        "log det (F+H) [Cholesky]",
        _term("log_det_curvature_reg", lambda inv: inv.log_det_curvature_reg_matrix_term),
    ),
    (
        "log det H [Cholesky]",
        _term("log_det_reg", lambda inv: inv.log_det_regularization_matrix_term),
    ),
    (
        "Regularization term s'Hs",
        _term("regularization_term", lambda inv: inv.regularization_term),
    ),
    ("Log evidence (figure of merit)", _log_evidence),
]


# -------------------------------------------------------------------
# The JAX/FFT arm's step list
# -------------------------------------------------------------------
#
# ``InversionInterferometerSparse`` exposes ``data_vector`` and ``curvature_matrix`` as
# plain (uncached) properties, so walking the same sequential cached-property chain the
# numba arm walks would recompute F inside ``curvature_reg_matrix`` and charge it twice.
# The JAX arm therefore records the rows it can measure directly — the ones the
# comparison actually needs, F above all — and times the whole likelihood separately;
# ``steps_are_partial`` says so in the JSON rather than implying a total that is not one.

JAX_F_LABEL = "F: mapper×mapper [sparse-op FFT]"


def _jax_triplets(fit):
    inversion = _inversion(fit)
    mapper = inversion.cls_list_from(cls=Mapper)[0]
    return inversion._sparse_triplets_curvature_from(mapper=mapper)


JAX_STEP_ACCESSORS = [
    ("FitInterferometer construct", None),  # handled specially (constructor)
    (INVERSION_BUILD_LABEL, _build_inversion),
    ("Mapper sparse triplets [extent-flat]", _jax_triplets),
    ("Data vector D [sparse-op]", lambda fit: _inversion(fit).data_vector),
    (JAX_F_LABEL, lambda fit: np.asarray(_inversion(fit).curvature_matrix)),
    (H_LABEL, lambda fit: np.asarray(_inversion(fit).regularization_matrix)),
]


def jax_curvature_jit_callable(inversion):
    """A jitted, warm callable returning the JAX/FFT arm's ``F``.

    Read eagerly — which is what a step-by-step breakdown does — the sparse operator's
    ``lax.fori_loop`` over source-column blocks dispatches op by op, so the eager ``F``
    row is dominated by dispatch, not by the FFT. Production wraps the log likelihood in
    ``jax.jit``; this closure reproduces that regime for the one row the comparison turns
    on, with the mapper triplets as arguments and the operator (hence ``Khat``) closed
    over exactly as an inversion closes over it.
    """
    import jax
    import jax.numpy as jnp

    mapper = inversion.cls_list_from(cls=Mapper)[0]
    rows, cols, vals = inversion._sparse_triplets_curvature_from(mapper=mapper)

    rows = jnp.asarray(rows)
    cols = jnp.asarray(cols)
    vals = jnp.asarray(vals)

    operator = inversion.dataset.sparse_operator
    source_pixels = int(mapper.params)

    # `xp=jnp` keeps this arm on the operator's JAX branch: since PyAutoArray#544 the
    # sparse-operator methods default to `xp=np`, which under `jit` would call
    # `np.asarray` on a tracer and raise `TracerArrayConversionError`.
    jitted = jax.jit(
        lambda r, c, v: operator.curvature_matrix_diag_from(r, c, v, S=source_pixels, xp=jnp)
    )

    def call():
        result = jitted(rows, cols, vals)
        result.block_until_ready()
        return result

    return call


# ===================================================================
# Machine-speed controls
# ===================================================================


def dgemm_control_s(n: int = 1500, repeats: int = 3) -> float:
    """A fixed 1500x1500 ``dgemm``, timed identically in every arm.

    Run at the head and tail of an arm it is the machine-speed normaliser the paired
    B/A/B/A protocol needs: this laptop (i9-10885H) throttles, so a cross-arm ratio is
    only believable if the same BLAS call costs the same at both ends of both arms.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))

    a @ b  # warm the BLAS thread pool / first-touch the pages

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        samples.append(time.perf_counter() - start)

    return float(np.median(samples))


def cholesky_control_s(matrix, repeats: int = 3) -> float:
    """A Cholesky of the arm's own ``F + H``.

    Arm-invariant by construction (the two arms assemble the *same* matrix, pinned to
    ``rtol=1e-6`` on the log evidence) and magnitude-matched, so a drift here is the
    machine, not the kernel.
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    np.linalg.cholesky(matrix)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.linalg.cholesky(matrix)
        samples.append(time.perf_counter() - start)

    return float(np.median(samples))


def _fit_from() -> al.FitInterferometer:
    return al.FitInterferometer(
        dataset=dataset,
        tracer=al.Tracer(galaxies=list(instance.galaxies)),
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )


def one_decomposed_evaluation() -> tuple[dict[str, float], float]:
    """Run one likelihood evaluation, timing each step's incremental cost."""
    step_times: dict[str, float] = {}
    _state.clear()

    start = time.perf_counter()
    fit = _fit_from()
    step_times["FitInterferometer construct"] = time.perf_counter() - start

    for label, accessor in ARM_STEP_ACCESSORS[1:]:
        start = time.perf_counter()
        result = accessor(fit)
        step_times[label] = time.perf_counter() - start

    # The numba arm's last accessor is the log evidence; the JAX arm's step list stops
    # at H (see JAX_STEP_ACCESSORS), so there is no scalar to return there.
    return step_times, float("nan") if is_jax_arm else float(result)


def direct_log_evidence() -> float:
    """One whole likelihood, undecomposed — the step total's cross-check.

    On the JAX arm this is ``FitInterferometer.figure_of_merit`` itself, i.e. the
    production JAX/FFT likelihood, so the two arms' whole-evaluation numbers are
    like for like."""
    fit = _fit_from()

    if is_jax_arm:
        return float(fit.figure_of_merit)

    inversion = InversionInterferometerNumba(
        dataset=fit.inversion.dataset,
        linear_obj_list=fit.inversion.linear_obj_list,
        settings=fit.inversion.settings,
        xp=np,
        numba_preload=numba_preload,
        kernel=kernel,
    )

    return float(
        fit_util.log_evidence_from(
            chi_squared=inversion.fast_chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=fit.noise_normalization,
        )
    )


ARM_STEP_ACCESSORS = JAX_STEP_ACCESSORS if is_jax_arm else STEP_ACCESSORS

print("\n--- Machine-speed control (head of arm) ---")
control_dgemm_head_s = dgemm_control_s()
print(f"  1500x1500 dgemm: {control_dgemm_head_s:.4f} s")

print(f"\n--- Warm-up evaluation (kernel={kernel}; numba/JIT compile) ---")
_warm_start = time.perf_counter()
_warm_steps, _warm_fom = one_decomposed_evaluation()
warmup_s = time.perf_counter() - _warm_start
print(f"  warm-up total: {warmup_s:.4f} s (log evidence = {_warm_fom})")

n_repeats = N_REPEATS

print(f"\n--- Timed decomposition (x{n_repeats}) ---")

accumulated: dict[str, float] = {label: 0.0 for label, _ in ARM_STEP_ACCESSORS}
for _ in range(n_repeats):
    step_times, log_evidence_step = one_decomposed_evaluation()
    for label, elapsed in step_times.items():
        accumulated[label] += elapsed

likelihood_steps = [(label, accumulated[label] / n_repeats) for label, _ in ARM_STEP_ACCESSORS]

_step_dict = dict(likelihood_steps)
curvature_matrix_f_total = (
    _step_dict[JAX_F_LABEL]
    if is_jax_arm
    else _step_dict[F_SCATTER_LABEL] + _step_dict[F_RESIDUAL_LABEL]
)

start = time.perf_counter()
for _ in range(n_repeats):
    log_evidence_direct = direct_log_evidence()
direct_per_call = (time.perf_counter() - start) / n_repeats

# The kernel identity is read off the inversion the arm actually built, not from
# the CLI string, so a mis-wired dispatch cannot label itself correctly.
arm_kernel = "jax" if is_jax_arm else _state["inversion"].kernel

# The JAX arm's fair F row: jit-compiled and warm, i.e. the regime production runs in.
# The eager row in the step list above is kept because it is what an un-jitted
# evaluation actually costs, but it is dispatch-bound and is not the comparator.
curvature_matrix_f_jit_steady_s = None

if is_jax_arm:
    _jit_call = jax_curvature_jit_callable(_state["inversion"])
    _jit_call()  # compile

    _jit_start = time.perf_counter()
    for _ in range(n_repeats):
        _jit_call()
    curvature_matrix_f_jit_steady_s = (time.perf_counter() - _jit_start) / n_repeats

    print(
        f"\n--- JAX F (jit warm, steady x{n_repeats}) ---\n"
        f"  {curvature_matrix_f_jit_steady_s:.6f} s"
    )


control_cholesky_s = cholesky_control_s(_state["inversion"].curvature_reg_matrix)

print("\n--- Machine-speed control (tail of arm) ---")
control_dgemm_tail_s = dgemm_control_s()
print(
    f"  1500x1500 dgemm: {control_dgemm_tail_s:.4f} s "
    f"(head {control_dgemm_head_s:.4f} s, drift {control_dgemm_tail_s / control_dgemm_head_s:.3f}x)"
)

# ===================================================================
# JAX comparator — the same dataset/model through the FFT sparse path
# ===================================================================

print("\n--- JAX/FFT reference (InversionInterferometerSparse) ---")

_jax_fit = _fit_from()
jax_reference_log_evidence = float(_jax_fit.figure_of_merit)
print(f"  sparse-path figure_of_merit = {jax_reference_log_evidence}")
del _jax_fit

# ===================================================================
# Per-step breakdown summary + JSON + PNG
# ===================================================================

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 78)
print(
    f"NUMBA CPU PIXELIZATION INTERFEROMETER PER-STEP BREAKDOWN — "
    f"{instrument.upper()} — kernel={arm_kernel} — v{al_version}"
)
print("=" * 78)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 78)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print(f"      {'Direct numba log evidence':<{max_label}}  {direct_per_call:>12.6f} s")
print(f"      {'Coverage (steps / direct)':<{max_label}}  {step_total / direct_per_call:>11.1%}")
print("=" * 78)

breakdown_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "real_space_shape": list(real_space_shape),
        "image_pixels_masked": int(n_image_pixels),
        "visibilities": int(n_visibilities),
        "mesh_shape": list(mesh_shape),
        "rect_mesh": _cli.rect_mesh,
        "source_pixels": int(n_source_pixels),
        "regularization_coefficient": regularization_coefficient,
        # "sparse_numba" is the token scripts/misc/tooling/build_readme.py maps
        # to the "sparse (numba)" dashboard label; this cell is the numba w-tilde
        # (sparse) formalism, not the dense transformed-mapping-matrix one.
        "inversion_path": "sparse" if is_jax_arm else "sparse_numba",
        "formalism": "w_tilde_fft_jax" if is_jax_arm else "w_tilde_numba",
        "kernel": arm_kernel,
        "use_jax": is_jax_arm,
        "mesh": f"rectangular_adapt_image_{mesh_pixels_yx}x{mesh_pixels_yx}",
        "regularization": "constant",
        "lens_light": None,
        "n_repeats": n_repeats,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS", None),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", None),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", None),
        "xla_flags": os.environ.get("XLA_FLAGS", None),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", None),
    },
    "steps_are_partial": is_jax_arm,
    "steps_note": (
        "The JAX/FFT arm records only the rows it can measure directly: "
        "InversionInterferometerSparse exposes data_vector and curvature_matrix as "
        "plain (uncached) properties, so walking the numba arm's sequential "
        "cached-property chain would recompute and double-charge F. Its whole "
        "evaluation is measured separately as "
        "direct_log_likelihood_function_per_call (= FitInterferometer.figure_of_merit), "
        "which is the number to compare across arms."
    )
    if is_jax_arm
    else None,
    "control_dgemm_head_s": control_dgemm_head_s,
    "control_dgemm_tail_s": control_dgemm_tail_s,
    "control_cholesky_s": control_cholesky_s,
    "control_note": (
        "control_dgemm_* is a fixed 1500x1500 BLAS dgemm timed at the head and tail "
        "of the arm (this laptop throttles, so a cross-arm ratio is only believable "
        "if these agree); control_cholesky_s is a Cholesky of the arm's own F + H, "
        "which both arms assemble identically."
    ),
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "curvature_matrix_f_total_s": curvature_matrix_f_total,
    "curvature_matrix_f_jit_steady_s": curvature_matrix_f_jit_steady_s,
    "curvature_matrix_f_jit_note": (
        "JAX arm only: F assembled through a jax.jit-compiled, warm "
        "InterferometerSparseOperator.curvature_matrix_diag_from — the regime "
        "production runs in, and the row to compare against the numba arm's "
        "curvature_matrix_f_total_s. The eager step-list row is dispatch-bound "
        "(lax.fori_loop over source-column blocks, op by op) and is reported "
        "only as what an un-jitted evaluation costs."
    ),
    "curvature_matrix_f_split_note": (
        "The two 'F: ...' / 'Curvature matrix F' rows sum to "
        "curvature_matrix_f_total_s, the whole cost of "
        "inversion.curvature_matrix. Both are cached properties and the scatter "
        "row is touched first, so the second row is a direct measurement of the "
        "remaining work (the no-regularization diagonal add) rather than an "
        "arithmetic residual: the recovered kernel loops the full N x N "
        "image-pixel pair space and returns a complete symmetric F, so there is "
        "no mirroring pass here, unlike the imaging numba and modern "
        "interferometer sparse classes."
    ),
    "preload_build_s": preload_s,
    "preload_cache_hit": preload_cache_hit,
    "preload_note": (
        "The W~ preload and dirty image are one-off per dataset (built by "
        "dataset.apply_sparse_operator(use_jax=False) and read by NumbaPreload), "
        "so they are recorded here but are not a per-evaluation step. Phase 3 of "
        "the numba-interferometer-revisit epic profiles the preload build itself."
    ),
    "direct_log_likelihood_function_per_call": direct_per_call,
    "warmup_incl_numba_compile_s": warmup_s,
    "log_evidence": float(log_evidence_direct),
    "jax_reference_log_evidence": jax_reference_log_evidence,
    "jax_reference_note": (
        "figure_of_merit of the same dataset and model through "
        "InversionInterferometerSparse — the JAX/FFT path "
        "likelihood_runtime/pixelization.py profiles. The numba log evidence must "
        "match it; only the application of W~ differs between the two, not the "
        "algebra."
    ),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "interferometer",
    default_basename=f"{CELL}_{arm_kernel}_breakdown_{instrument}_v{al_version}",
    cell=f"{CELL}_{arm_kernel}",
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(10, 7.0))
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
    f"Numba CPU Pixelization Interferometer Likelihood — Per-Step Breakdown — {instrument.upper()} — kernel={arm_kernel}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} image pixels  |  '
    f"{n_visibilities} visibilities  |  {mesh_shape[0]}x{mesh_shape[1]} mesh  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.margins(x=0.15)
fig.tight_layout()

fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

# ===================================================================
# Pinned-value drift record (soft)
# ===================================================================

_pinned_drift: list = []

# Pinned 2026-09-07 (v2026.8.17.1, 32x32 adaptive rectangular fiducial). Repeats
# are bistable at the ~1e-8 relative level (summation-order nondeterminism) —
# rtol=1e-6 covers it.
EXPECTED_LOG_EVIDENCE: dict[str, float] = {
    "sma": -3168.280345651575,
}

_pinned_expected = EXPECTED_LOG_EVIDENCE.get(instrument)

if _pinned_expected is None:
    print(
        f"  Pinned check SKIPPED for {instrument} (no pinned value). "
        f"log_evidence = {float(log_evidence_direct)!r}"
    )
else:
    _rec = check_pinned(float(log_evidence_direct), _pinned_expected, label="numba_cpu", rtol=1e-6)
    if _rec is not None:
        _pinned_drift.append(_rec)

# The JAX/FFT comparator is pinned against the same value: the two paths share the
# algebra, so a divergence between them is a bug in one of the applications of W~.
_rec = check_pinned(
    jax_reference_log_evidence,
    float(log_evidence_direct),
    label="jax_sparse_vs_numba",
    rtol=1e-6,
)
if _rec is not None:
    _pinned_drift.append(_rec)

record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")
