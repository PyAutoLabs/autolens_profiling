"""
Numba CPU Profiling: Delaunay Imaging Likelihood (Per-Step Breakdown)
=====================================================================

Decomposes the **numba CPU sparse-operator** likelihood for the production
Delaunay fiducial (Hilbert image-mesh 1250 vertices from the adapt image,
``al.mesh.Delaunay`` + ``ConstantSplit`` regularization, MGE-60 linear lens
light, ``use_jax=False``) into its per-evaluation steps.

Decomposition method (as ``pixelization_numba.py``): each repeat builds a fresh
``FitImaging`` and touches each lazy cached property in dependency order,
timing every access — each timing isolates that step's incremental cost. The
Hilbert image-mesh placement is a one-off per analysis (vertices arrive via
``al.AdaptImages``), so it is *not* a per-evaluation step; the per-evaluation
"inversion build" step covers ray tracing, the scipy Delaunay triangulation,
barycentric interpolation and border relocation.

A directly-timed full ``log_likelihood_function`` evaluation cross-checks the
step total. First evaluation is warm-up only (lazy numba compile; cold-cache
first-call hazard — see the ``pixelization_numba.py`` runtime sibling).

Output
------
``results/breakdown/imaging/delaunay_numba_breakdown_<instrument>_v<version>.{json,png}``
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

# ===================================================================
# Setup — identical fiducial to the delaunay_numba runtime sibling
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

print("\n--- Adapt image + Hilbert image mesh (one-off) ---")

n_mesh_vertices = 1250

adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=adapt_image
)

print("\n--- Model construction ---")

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
    mesh=al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0),
    regularization=al.reg.ConstantSplit(coefficient=1.0),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={
        instance.galaxies.source: image_plane_mesh_grid,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid,
    },
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=al.Settings(use_border_relocator=True),
    use_jax=False,
)

from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402

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
# Per-step decomposition via sequential cached-property access
# ===================================================================

# -------------------------------------------------------------------
# Curvature matrix F sub-block instrumentation (PyAutoArray#505 step 0)
# -------------------------------------------------------------------
#
# F is assembled from three blocks (see
# ``autoarray/inversion/inversion/imaging_numba/sparse.py``):
#
#   1. mapper x mapper       ``_curvature_matrix_mapper_diag``          [sparse-op numba kernel]
#   2. mapper x linear-func  ``_curvature_matrix_mapper_func_blocks_from`` [batched FFT conv + numba scatter]
#   3. linear-func x l-func  ``_curvature_matrix_func_func_blocks_from``   [BLAS dot]
#
# and then, in the ``curvature_matrix`` cached property, a global mirror plus
# (when linear funcs are present) the no-regularization diagonal add.
#
# The three helpers are plain (uncached) methods, so touching them here does not
# prime the cached property: the F step that follows recomputes all three and
# then mirrors + adds the diagonal. The F row is therefore reported as a
# RESIDUAL,
#
#     F residual = t(curvature_matrix) - (t_block_1 + t_block_2 + t_block_3)
#
# using the three measured block costs as the estimate of the recompute inside
# the cached property. This is the design that stays truthful *and* keeps the
# artifact comparable with pre-instrumentation runs: the four F rows sum to
# exactly the un-instrumented ``curvature_matrix`` step, so "TOTAL
# (step-by-step)" and the coverage cross-check are unchanged in meaning. The raw
# (unsplit) F total is written to the result JSON as
# ``curvature_matrix_f_total_s``.
#
# Caveat: the residual is a difference of averaged timings, so at the 1e-4 s
# level it carries the noise of all four measurements and can in principle come
# out slightly negative. It is recorded as measured, never clipped.

F_MAPPER_MAPPER_LABEL = "F: mapper×mapper block [sparse-op]"
F_MAPPER_FUNC_LABEL = "F: mapper×linear-func block [FFT conv + scatter]"
F_FUNC_FUNC_LABEL = "F: linear-func×linear-func block [BLAS]"
F_RESIDUAL_LABEL = "Curvature matrix F [residual: mirror + diag-add]"

_f_scratch: dict[int, np.ndarray] = {}


def _f_block_scratch(inversion) -> np.ndarray:
    """Reusable (P, P) buffer the timed F sub-block helpers write into.

    Reused across repeats so each timing measures the block itself rather than a
    fresh multi-MB allocation. Both helpers *assign* their block (they never
    accumulate into it), so a dirty buffer cannot change what is measured.
    """
    total_params = inversion.total_params
    scratch = _f_scratch.get(total_params)
    if scratch is None:
        scratch = np.zeros((total_params, total_params))
        _f_scratch[total_params] = scratch
    return scratch


# -------------------------------------------------------------------
# MGE operated mapping matrix sub-block instrumentation (PyAutoArray#507 step 0)
# -------------------------------------------------------------------
#
# ``inversion.linear_func_operated_mapping_matrix_dict`` builds, for the linear
# MGE lens light, the PSF-convolved image of every one of its 60 Gaussians. On
# the numpy path that is
# ``LightProfileLinearObjFuncList.operated_mapping_matrix_override``
# (autogalaxy/profiles/light/linear/abstract.py), whose fast branch is three
# distinct pieces of work:
#
#   1. ``mapping_matrix``            -- 60x ``image_2d_from(grid)``, stacked
#   2. the blurring stack            -- 60x ``image_2d_from(blurring_grid)``, stacked
#   3. one batched real-space PSF convolution of the two stacks
#
# The override is a ``cached_property`` on the linear-func object and the three
# pieces below are recomputed from scratch (``mapping_matrix`` is a plain
# ``property``), so timing them does *not* prime the step that follows. As with
# the F sub-blocks, the pre-existing row is therefore reported as a RESIDUAL,
#
#     MGE residual = t(linear_func_operated_mapping_matrix_dict) - (t1 + t2 + t3)
#
# so the four MGE rows sum to exactly the un-instrumented step and the artifact
# stays comparable with pre-instrumentation runs. The raw (unsplit) MGE total is
# written to the result JSON as ``mge_operated_mapping_matrix_total_s``.
#
# ``mge_split_reproduces_step`` in the JSON records a one-off bit-identical
# check that the three timed pieces really do reconstruct the dict the step
# returns -- if a dataset ever takes a different branch of the override the
# split is flagged rather than silently mis-attributed.
#
# Caveat (as for F): the residual is a difference of averaged timings and can in
# principle come out slightly negative. It is recorded as measured, never clipped.

MGE_PROFILE_IMAGE_LABEL = "MGE: image_2d_from(grid) x60 [numpy]"
MGE_BLURRING_IMAGE_LABEL = "MGE: image_2d_from(blurring_grid) x60 [numpy]"
MGE_CONVOLVE_LABEL = "MGE: batched PSF convolution [real-space np]"
MGE_RESIDUAL_LABEL = "MGE operated mapping matrix [residual: dict assembly]"

_mge_scratch: dict[str, list] = {}


def _mge_linear_func_list(inversion) -> list:
    """The inversion's linear-func objects (the MGE lens light), never its mappers."""
    return [
        obj for obj in inversion.linear_obj_list if hasattr(obj, "light_profile_list")
    ]


def _mge_mapping_matrix(fit) -> list:
    """Piece 1: the 60 unblurred profile images on the (over-sampled) data grid."""
    matrices = [
        linear_func.mapping_matrix
        for linear_func in _mge_linear_func_list(fit.inversion)
    ]
    _mge_scratch["mapping_matrix"] = matrices
    return matrices


def _mge_blurring_stack(linear_func):
    """The blurring stack of one linear-func object, built the way the override builds it.

    ``operated_mapping_matrix_override`` stacked the 60 blurring-grid images with an
    inline per-profile loop until PyAutoArray#507 step 3, which moved both stacks behind
    ``LightProfileLinearObjFuncList._image_slim_list_from`` so that an MGE basis shares one
    reference-frame transform and one eccentric-radius grid. This harness is run against
    both libraries (the paired before/after of that step), so it dispatches on the
    attribute rather than hard-coding either shape -- a hand-rolled loop would keep timing
    the *old* code on the new library and mis-attribute the win to the residual row. Which
    branch was taken is recorded in the result JSON as ``mge_blurring_stack_path``.
    """
    if hasattr(linear_func, "_image_slim_list_from"):
        _mge_scratch["blurring_stack_path"] = "shared_geometry"

        return np.stack(
            linear_func._image_slim_list_from(grid=linear_func.blurring_grid, xp=np),
            axis=1,
        )

    _mge_scratch["blurring_stack_path"] = "per_profile_loop"

    return np.stack(
        [
            light_profile.image_2d_from(
                grid=linear_func.blurring_grid, xp=np
            ).slim.array
            for light_profile in linear_func.light_profile_list
        ],
        axis=1,
    )


def _mge_blurring_mapping_matrix(fit) -> list:
    """Piece 2: the same 60 profiles on the blurring grid (flux blurred in from
    outside the mask), stacked exactly as the override stacks them."""
    matrices = [
        _mge_blurring_stack(linear_func)
        for linear_func in _mge_linear_func_list(fit.inversion)
    ]
    _mge_scratch["blurring_mapping_matrix"] = matrices
    return matrices


def _mge_convolved(fit) -> list:
    """Piece 3: the single batched real-space convolution of the two stacks."""
    return [
        linear_func.psf.convolved_mapping_matrix_via_real_space_np_from(
            mapping_matrix=_mge_scratch["mapping_matrix"][index],
            mask=linear_func.grid.mask,
            blurring_mapping_matrix=_mge_scratch["blurring_mapping_matrix"][index],
            blurring_mask=linear_func.blurring_grid.mask,
        )
        for index, linear_func in enumerate(_mge_linear_func_list(fit.inversion))
    ]


def _mge_split_reproduces_step(fit) -> bool:
    """Bit-identical check that pieces 1-3 reconstruct the timed step's output."""
    _mge_mapping_matrix(fit)
    _mge_blurring_mapping_matrix(fit)
    reconstructed = _mge_convolved(fit)

    actual = list(fit.inversion.linear_func_operated_mapping_matrix_dict.values())

    if len(actual) != len(reconstructed) or len(actual) == 0:
        return False

    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(actual, reconstructed)
    )


def _geometry_constants(fit) -> dict:
    """Model-independent geometry of the sparse operator and of each mapper.

    Recorded once, outside the timed loop, so that the complexity of the
    mapper x mapper kernel (`curvature_matrix_via_sparse_operator_from`) is a
    measurement rather than an estimate. `mapper_mapper_inner_ops` is the exact
    count of that kernel's innermost accumulations,

        sum over stored (data_0, data_1) pairs of u0(data_0) * u1(data_1),

    and the `two_stage_*` counts are the same geometry costed for the phase-2
    reformulation (a dense n_source accumulator per data pixel, then one AXPY
    per mapping). PyAutoArray#507 step 0.
    """
    inversion = fit.inversion

    lengths = np.asarray(inversion.sparse_operator.lengths).astype("int64")
    indexes = np.asarray(inversion.sparse_operator.indexes).astype("int64")

    geometry = {
        "psf_shape_native": [int(s) for s in dataset.psf.kernel.shape_native],
        "image_pixels_masked": int(n_image_pixels),
        "psf_precision_pairs_stored": int(indexes.shape[0]),
        "psf_precision_lengths_sum": int(lengths.sum()),
        "psf_precision_lengths_mean": float(lengths.mean()),
        "psf_precision_lengths_max": int(lengths.max()),
        "mappers": [],
    }

    for mapper in inversion.cls_list_from(cls=Mapper):
        pix_lengths = np.asarray(mapper.unique_mappings.pix_lengths).astype("int64")

        pair_u0 = np.repeat(pix_lengths, lengths)
        pair_u1 = pix_lengths[indexes]

        n_source = int(mapper.params)
        data_pixels = int(lengths.shape[0])

        stage_1_ops = int(pair_u1.sum())
        stage_2_ops = int(pix_lengths.sum()) * n_source
        clear_ops = data_pixels * n_source

        geometry["mappers"].append(
            {
                "params": n_source,
                "data_to_pix_unique_shape": [
                    int(s)
                    for s in np.asarray(mapper.unique_mappings.data_to_pix_unique).shape
                ],
                "pix_lengths_mean": float(pix_lengths.mean()),
                "pix_lengths_max": int(pix_lengths.max()),
                "pix_lengths_sum": int(pix_lengths.sum()),
                "mapper_mapper_inner_ops": int((pair_u0 * pair_u1).sum()),
                "two_stage_stage_1_scatter_ops": stage_1_ops,
                "two_stage_stage_2_axpy_ops": stage_2_ops,
                "two_stage_clear_ops": clear_ops,
                "two_stage_total_ops": stage_1_ops + stage_2_ops + clear_ops,
            }
        )

    return geometry


STEP_ACCESSORS = [
    ("FitImaging construct", None),  # handled specially (constructor)
    ("Blurred image (FFT convolve)", lambda fit: fit.blurred_image),
    ("Profile subtracted image", lambda fit: fit.profile_subtracted_image),
    (
        "Inversion build (trace+Delaunay+mapper)",
        lambda fit: fit.inversion,
    ),
    ("PSF-weighted data [numba]", lambda fit: fit.inversion.psf_weighted_data),
    (MGE_PROFILE_IMAGE_LABEL, _mge_mapping_matrix),
    (MGE_BLURRING_IMAGE_LABEL, _mge_blurring_mapping_matrix),
    (MGE_CONVOLVE_LABEL, _mge_convolved),
    (
        MGE_RESIDUAL_LABEL,
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
    (
        F_MAPPER_MAPPER_LABEL,
        lambda fit: fit.inversion._curvature_matrix_mapper_diag,
    ),
    (
        F_MAPPER_FUNC_LABEL,
        lambda fit: fit.inversion._curvature_matrix_mapper_func_blocks_from(
            curvature_matrix=_f_block_scratch(fit.inversion)
        ),
    ),
    (
        F_FUNC_FUNC_LABEL,
        lambda fit: fit.inversion._curvature_matrix_func_func_blocks_from(
            curvature_matrix=_f_block_scratch(fit.inversion)
        ),
    ),
    (F_RESIDUAL_LABEL, lambda fit: fit.inversion.curvature_matrix),
    ("Regularization matrix H (ConstantSplit)", lambda fit: fit.inversion.regularization_matrix),
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
        adapt_images=adapt_images,
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    step_times["FitImaging construct"] = time.perf_counter() - start

    for label, accessor in STEP_ACCESSORS[1:]:
        start = time.perf_counter()
        result = accessor(fit)
        step_times[label] = time.perf_counter() - start

    return step_times, float(result)  # final accessor is figure_of_merit


print("\n--- Warm-up evaluation (numba compile) ---")
_warm_start = time.perf_counter()
_warm_steps, _warm_fom = one_decomposed_evaluation()
warmup_s = time.perf_counter() - _warm_start
print(f"  warm-up total: {warmup_s:.4f} s (figure_of_merit = {_warm_fom})")

fit_check = analysis.fit_from(instance=instance)
assert isinstance(fit_check.inversion, InversionImagingSparseNumba), (
    f"Expected InversionImagingSparseNumba, got {type(fit_check.inversion).__name__}"
)

# One-off, untimed (PyAutoArray#507 step 0).
mge_split_reproduces_step = _mge_split_reproduces_step(fit_check)
geometry_constants = _geometry_constants(fit_check)

print("\n--- Geometry constants (model-independent) ---")
print(json.dumps(geometry_constants, indent=2))
print(f"  MGE split reproduces step bit-identically: {mge_split_reproduces_step}")
print(f"  MGE blurring stack path: {_mge_scratch.get('blurring_stack_path')}")

del fit_check

n_repeats = 10

print(f"\n--- Timed decomposition (x{n_repeats}) ---")

accumulated: dict[str, float] = {label: 0.0 for label, _ in STEP_ACCESSORS}
for _ in range(n_repeats):
    step_times, figure_of_merit = one_decomposed_evaluation()
    for label, elapsed in step_times.items():
        accumulated[label] += elapsed

likelihood_steps = [(label, accumulated[label] / n_repeats) for label, _ in STEP_ACCESSORS]

# Convert the raw F step into the residual (mirror + diag-add): see the
# "Curvature matrix F sub-block instrumentation" note above. The four F rows
# then sum to the raw, un-split F cost recorded here.
_step_dict = dict(likelihood_steps)
curvature_matrix_f_total = _step_dict[F_RESIDUAL_LABEL]
_f_block_total = (
    _step_dict[F_MAPPER_MAPPER_LABEL]
    + _step_dict[F_MAPPER_FUNC_LABEL]
    + _step_dict[F_FUNC_FUNC_LABEL]
)

# The same scheme for the MGE row (PyAutoArray#507 step 0): the three timed
# pieces of the operated-mapping-matrix override are subtracted from the raw
# step, so the four MGE rows sum to the raw, un-split MGE cost recorded here.
mge_operated_mapping_matrix_total = _step_dict[MGE_RESIDUAL_LABEL]
_mge_piece_total = (
    _step_dict[MGE_PROFILE_IMAGE_LABEL]
    + _step_dict[MGE_BLURRING_IMAGE_LABEL]
    + _step_dict[MGE_CONVOLVE_LABEL]
)

_residual_of = {
    F_RESIDUAL_LABEL: curvature_matrix_f_total - _f_block_total,
    MGE_RESIDUAL_LABEL: mge_operated_mapping_matrix_total - _mge_piece_total,
}

likelihood_steps = [
    (label, _residual_of.get(label, per_call)) for label, per_call in likelihood_steps
]

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
print(f"NUMBA CPU DELAUNAY PER-STEP BREAKDOWN — {instrument.upper()} — v{al_version}")
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
        "mesh_shape": [n_mesh_vertices],
        "source_pixels": int(n_source_pixels),
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "mesh": "delaunay_hilbert_1250",
        "regularization": "constant_split",
        "lens_light": "mge_60_linear",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "curvature_matrix_f_total_s": curvature_matrix_f_total,
    "mge_operated_mapping_matrix_total_s": mge_operated_mapping_matrix_total,
    "mge_operated_mapping_matrix_split_note": (
        "The four 'MGE: ...' / 'MGE operated mapping matrix' rows sum to "
        "mge_operated_mapping_matrix_total_s, the raw un-split cost of the "
        "inversion.linear_func_operated_mapping_matrix_dict step. The three "
        "piece rows re-run, uncached, the three parts of "
        "LightProfileLinearObjFuncList.operated_mapping_matrix_override (60 "
        "profile images on the data grid, 60 on the blurring grid, one batched "
        "real-space PSF convolution); the residual row is the step minus those "
        "three. mge_split_reproduces_step records that the three pieces "
        "reconstruct the step's output bit-identically. PyAutoArray#507 step 0."
    ),
    "mge_split_reproduces_step": mge_split_reproduces_step,
    "mge_blurring_stack_path": _mge_scratch.get("blurring_stack_path"),
    "geometry": geometry_constants,
    "curvature_matrix_f_split_note": (
        "The four 'F: ...' / 'Curvature matrix F' rows sum to "
        "curvature_matrix_f_total_s, the raw un-split cost of the "
        "inversion.curvature_matrix step. The three block rows are direct "
        "measurements of the (uncached) per-block helpers in "
        "imaging_numba/sparse.py; the 'Curvature matrix F [residual: ...]' row "
        "is t(curvature_matrix) minus those three, i.e. the global mirror, the "
        "no-regularization diagonal add and assembly overhead, and it carries "
        "the combined noise of the four timings. PyAutoArray#505 step 0."
    ),
    "direct_log_likelihood_function_per_call": direct_per_call,
    "warmup_incl_numba_compile_s": warmup_s,
    "log_likelihood": float(log_likelihood_direct),
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"delaunay_numba_breakdown_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(10, 7.8))
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
    f"Numba CPU Delaunay Likelihood — Per-Step Breakdown — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_over_sampled_pixels} over-sampled  |  {n_mesh_vertices} vertices  |  "
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

# Pinned 2026-08-20 (v2026.8.17.1, 1250-vertex fiducial). Delaunay repeats are bistable at the
# ~1e-8 relative level (summation-order nondeterminism) — rtol=1e-6 covers it.
EXPECTED_LOG_LIKELIHOOD: dict[str, float] = {
    "euclid": 7215.3687893658935,
    "hst": 29090.527192092646,
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
