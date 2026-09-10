"""
JAX Profiling: Pixelization Imaging Likelihood — Per-Step Breakdown
====================================================================

Decomposes the JAX likelihood function for an imaging dataset (rectangular
pixelization source model) into its individual pipeline steps and
JIT-profiles each one separately. This script is the **breakdown** counterpart
to ``likelihood_runtime/imaging/pixelization.py``, which measures only the
full-pipeline single-JIT cost.

Steps profiled:

1. Ray-trace grids through the lens
2. Blurred image of lens light (non-linear profiles)
3. Profile-subtracted image (lens light subtraction)
4. Border relocation of traced grid
5. Overlay grid (source pixel centres)
6. Interpolation weights and mapper construction
7. Mapping matrix
8. Blurred mapping matrix / Inversion setup (steps 4-8 combined)
9. Data vector (D)
10. Curvature matrix (F)
11. Regularization matrix (H)
12. Regularized reconstruction: s = (F + H)^{-1} D
13. Map reconstruction to image + log evidence

Per-step timing is approximate: XLA may fuse operations differently when
compiled as one program vs separate pieces, but the breakdown is still useful
for identifying which step dominates the runtime budget.

All JAX timings block over every leaf of the returned pytree
(``likelihood_breakdown.timing.block``) to force synchronous measurement, so
tuple-returning prefixes are timed on the same terms as single-array steps.

Staged prefixes (``--split-setup``)
-----------------------------------

The combined inversion-setup row (steps 4-8) is attributed to its stages by
nested prefix JITs of the same staged computation — ``params -> step-5 output``,
``-> 6``, ``-> 7``, ``-> 8`` — with successive differences giving the per-stage
rows, exactly as ``likelihood_breakdown/delaunay.py`` does. The differences
inherit the fusion caveat (XLA may move work across a prefix boundary, so a
small negative row is noise) and every prefix pays the ray-trace preamble,
which lands in the first difference.

Regularization matrix (H) attribution
-------------------------------------

Since 2026-09-10 the H row is the prefix difference
``t(params -> H) - t(params -> interpolator outputs)``, matching the Delaunay
cell. ``_setup_prefix_fn(11)`` returns the step-6 outputs **alongside** H so it
is a strict superset of ``_setup_prefix_fn(6)``; returning H alone would let
XLA dead-code-eliminate the query side of the mapper and the difference would
go negative.

For this cell the difference is expected to read **~0**: the rectangular mesh's
``neighbors`` are fixed by ``mesh_shape`` and ``al.reg.Constant`` reads nothing
else, so H does not depend on the model parameters at all and XLA hoists it out
of the prefix. That is a true statement about the rectangular + ``Constant``
pairing rather than a measurement artifact — the Delaunay family's split
schemes are where H has a per-call cost. The assembly cost of the matrix in
isolation is still timed and recorded as
``regularization_matrix_assembly_s`` (the ``constant_regularization_matrix_from``
call this row used to report), and the absolute prefix is
``regularization_matrix_prefix_s``.

Batched re-timing (``--vmap-batch N``)
--------------------------------------

With ``--vmap-batch N`` the combined inversion-setup block, each
``--split-setup`` prefix and the params->H prefix are re-timed under
``jax.jit(jax.vmap(fn))`` on a params pytree broadcast to batch ``N``, and
reported as amortized per-call time (batch time / N) beside the unbatched
column. Batch 16 matches the ``n_batch=16`` of the Nautilus reference runs. The
whole block is wrapped in ``try/except`` so an OOM keeps the unbatched rows.

Sparse (w-tilde) step map
-------------------------

``--sparse`` attaches the w-tilde sparse operator
(``dataset.apply_sparse_operator(batch_size=--sparse-batch-size)``) and — since
2026-09-10 — actually **times the w-tilde steps**. Before that the ``_sparse``
rows were dense tables with a sparse dataset attached: the cell called
``apply_sparse_operator()`` and then timed ``operated_mapping_matrix`` and
``curvature_matrix_via_mapping_matrix_from``, neither of which
``InversionImagingSparse`` ever calls, which is why dense and ``_sparse`` rows
agreed to <1 %.

The dense rows and their sparse replacements:

===================================== =========================================
dense                                 sparse
===================================== =========================================
Inversion setup (steps 4-8 combined)  Inversion setup (sparse, steps 4-8
                                      combined) — triplets + MGE operated
                                      basis + PSF-weighted data
Data vector (D)                       Data vector (D, w-tilde)
Curvature matrix (F)                  Curvature matrix (F, w-tilde)
Mapped recon + log evidence           Mapped recon + log evidence (sparse)
===================================== =========================================

Rows 1-5 (ray-trace, lens light, PSF, profile subtraction, overlay grid) are
path-independent and are timed identically in both legs. H, ``F + λH``, the
NNLS reconstruction and both Choleskys are **the same code in both legs** —
that is the point of the comparison: the w-tilde path replaces the mapping
matrix, not the log-det.

Two extra JSON-only tables come out of the sparse leg:

- ``steps_sparse_setup_rows`` — the setup pieces timed separately: ``Sparse
  triplets (data + curvature)``, ``MGE operated basis (params prefix)`` (60
  PSF-convolved columns, still dense, and a matrix-free line has to beat it too)
  and ``PSF-weighted data``. The MGE row is a ``params -> basis`` prefix rather
  than an array-level convolution because a linear light profile overrides the
  inversion's convolution (``operated_mapping_matrix_override``) to include flux
  that blurs in from outside the mask.
- ``steps_sparse_sub_rows`` — the three blocks inside the single F row: ``F diag
  (FFT blocks)`` (``ImagingSparseOperator.curvature_matrix_diag_from``, the
  ``ceil(S / batch_size)`` rFFT2 column sweep), ``F off-diag (mapper x MGE)``
  and ``F MGE x MGE GEMM``.

Neither table is added to ``total_step_by_step`` — they overlap rows that are.

The standalone step functions live in
``scripts/misc/likelihood_breakdown/sparse_steps.py``; see that module and
``scripts/misc/likelihood_breakdown/README.md`` for the library provenance of
each one.

Output
------

Results JSON and PNG are written to ``results/breakdown/imaging/`` using
the basename ``pixelization_breakdown_{instrument}_v{al_version}``.
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


import sys
from pathlib import Path

import autoarray as aa
import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autofit.jax import register_model as _register_model_pytrees

# Shared adapt-image loader: load or compute+cache `lensed_source.fits`
# next to the dataset, then return the masked ``aa.Array2D``.
sys.path.insert(0, str(_profiling_root()))
# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------
# AUTOLENS_PROFILING_SMOKE=1 short-circuit (Phase 5 / CI lint smoke).
# Verifies the import graph + module-level setup succeeded without running
# the full profiling pipeline. Skipped entirely when the env var is unset.
import os as _smoke_os
import sys as _smoke_sys

# Shared breakdown helpers. Imported *before* the smoke short-circuit so the CI
# import smoke covers the package too.
from likelihood_breakdown import sparse_steps, timing  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

# Sweep-driver CLI args (--config-name / --output-dir / --use-mixed-precision).
# Tolerates extra/unknown args via parse_known_args inside the helper.
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _production_config import observe_thread_env as _observe_thread_env  # noqa: E402
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

_vmap_batch = timing.parse_vmap_batch(sys.argv)
if _vmap_batch is not None and _vmap_batch < 1:
    raise ValueError(f"--vmap-batch must be >= 1 (got {_vmap_batch})")

_split_setup = "--split-setup" in sys.argv

instrument = "hst"  # <-- change this to profile a different instrument


# ---------------------------------------------------------------------------
# Profiling helpers — the shared implementations, bound to this cell's Timer
# ---------------------------------------------------------------------------

Timer = timing.Timer
block = timing.block

timer = Timer()
likelihood_steps = []  # (label, per_call_seconds) for the final summary
jit_records: dict[str, dict] = {}  # {label: {lower_s, compile_s, first_call_s, ...}}


def jit_profile(func, label, *args, n_repeats=10):
    """Cell-local binding of ``timing.jit_profile`` (this cell's timer/records)."""
    return timing.jit_profile(
        func,
        label,
        *args,
        n_repeats=n_repeats,
        timer=timer,
        jit_records=jit_records,
    )


# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
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
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=1,
    )

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )

    if _cli.use_sparse_operator:
        # Engage the w-tilde sparse-operator path. See the runtime sibling
        # script for the rationale (autolens_profiling#44). The operator build
        # itself is two rFFT2s of the padded PSF and is timed here, eagerly, as
        # a one-off — it is not a per-call cost.
        with timer.section("sparse_operator_build"):
            dataset = dataset.apply_sparse_operator(batch_size=_cli.sparse_batch_size)
        sparse_operator_build_s = timer.records[-1][1]
    else:
        sparse_operator_build_s = None

# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

mesh_pixels_yx = 39  # 39x39 = 1521 source pixels — 1500-tier production fiducial
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

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

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

    # ``RectangularBilinearAdaptImage`` weights mesh pixels by the lensed-source
    # adapt image — the production-grade alternative to the coordinate-
    # density-only ``RectangularBilinearAdaptDensity``. Adapt image is loaded /
    # cached below; the same shape and regularization are kept.
    reg_coefficient_model = 1.0
    pixelization = al.Pixelization(
        mesh=rect_mesh_classes(_cli)[1](shape=mesh_shape, weight_power=1.0, weight_floor=0.0),
        regularization=al.reg.Constant(coefficient=reg_coefficient_model),
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

# Regularization provenance, recorded in the JSON on the same terms as the
# Delaunay family's ``regularization`` block (which carries the resolved
# ``--regularization`` scheme). This cell's mesh is paired with plain
# ``Constant`` and takes no ``--regularization`` flag.
reg_provenance = {"scheme": "constant", "coefficient": reg_coefficient_model}

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Mesh shape: {mesh_shape}")
print(f"  Source pixels: {mesh_pixels_yx * mesh_pixels_yx}")
print(f"  Regularization: constant (coefficient={reg_coefficient_model})")

# ---------------------------------------------------------------------------
# 3. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)
tracer = al.Tracer(galaxies=list(instance.galaxies))

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# Key configuration that dictates run time
# ---------------------------------------------------------------------------

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
print(f"  Inversion path:          {'sparse (w-tilde)' if _cli.use_sparse_operator else 'dense'}")
if _cli.use_sparse_operator:
    print(f"  Sparse batch size:       {_cli.sparse_batch_size}")

# ---------------------------------------------------------------------------
# 4. Adapt image — PSF-convolved lensed-source image used by
#    ``RectangularBilinearAdaptImage`` to weight mesh pixels.
# ---------------------------------------------------------------------------

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
    adapt_images = al.AdaptImages(
        galaxy_image_dict={instance.galaxies.source: adapt_image},
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
    )

print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

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
# PART B — Per-step JIT profiling
# ===================================================================

print("\n" + "=" * 70)
print("PER-STEP JIT PROFILING")
print("=" * 70)

# Extract raw arrays from autoarray types via .array so they can cross
# JIT boundaries.  See CLAUDE.md for rationale.

grid_lp_raw = jnp.array(dataset.grids.lp.array)
grid_pix_raw = jnp.array(dataset.grids.pixelization.array)
grid_blurring_raw = jnp.array(dataset.grids.blurring.array)
data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)

# Keep autoarray objects for eager calls that need them.
grid_lp = dataset.grids.lp
grid_blurring = dataset.grids.blurring

# ---------------------------------------------------------------------------
# Step 1: Ray-trace grids
# ---------------------------------------------------------------------------

print("\n--- Step 1: Ray-trace grids ---")

with timer.section("ray_trace_eager"):
    traced_grids = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)
    for tg in traced_grids:
        block(tg)

print(f"  Number of planes traced: {len(traced_grids)}")


def ray_trace_raw(grid_raw):
    """Wraps ray-tracing so inputs/outputs are raw arrays."""
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, traced_grids_raw = jit_profile(ray_trace_raw, "ray_trace_jit", grid_pix_raw)
likelihood_steps.append(("Ray-trace grids", timer.records[-1][1] / 10))

print(f"  traced_grids shape: {traced_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 2: Blurred image of non-linear light profiles (lens light)
# ---------------------------------------------------------------------------

print("\n--- Step 2: Blurred image (lens light profiles) ---")

# Sub-step 2a: Compute raw lens light images (JIT-profiled)
#
# We rebuild ``Grid2D`` (uniform, masked) instead of ``Grid2DIrregular``
# inside the function so the lens-light Basis can produce its
# ``Array2D`` zero-vector for ``LightProfileLinear`` components — see
# ``PyAutoGalaxy:autogalaxy/profiles/basis.py:151`` which does
# ``mask=grid.mask`` and would AttributeError on an irregular grid.
# The masks are captured from module-level scope; they're static
# (Python-level constants from the JIT's perspective).
_grid_lp_mask = dataset.grids.lp.mask
_grid_blurring_mask = dataset.grids.blurring.mask


def lens_image_raw(grid_raw, blurring_grid_raw):
    """Compute lens light images on masked + blurring grids (no PSF)."""
    grid = aa.Grid2D(values=grid_raw, mask=_grid_lp_mask, xp=jnp)
    blurring_grid = aa.Grid2D(values=blurring_grid_raw, mask=_grid_blurring_mask, xp=jnp)
    image = tracer.image_2d_from(grid=grid, xp=jnp)
    blurring_image = tracer.image_2d_from(grid=blurring_grid, xp=jnp)
    return image.array, blurring_image.array


with timer.section("lens_image_eager"):
    img_eager, blur_img_eager = lens_image_raw(grid_lp_raw, grid_blurring_raw)
    block(img_eager)
    block(blur_img_eager)

_, (img_jit, blur_img_jit) = jit_profile(
    lens_image_raw, "lens_image_jit", grid_lp_raw, grid_blurring_raw
)
likelihood_steps.append(("Lens light images (pre-PSF)", timer.records[-1][1] / 10))

# Sub-step 2b: PSF convolution (eager — requires autoarray mask objects)
with timer.section("blurred_image_eager"):
    blurred_image = tracer.blurred_image_2d_from(
        grid=grid_lp,
        psf=dataset.psf,
        blurring_grid=grid_blurring,
        xp=jnp,
    )
    block(blurred_image)

print(f"  blurred_image shape: {blurred_image.array.shape}")


def blurred_image_from_params(params_tree):
    """Compute blurred image directly from a pytree ModelInstance — fully JIT-traceable."""
    t = al.Tracer(galaxies=list(params_tree.galaxies))
    result = t.blurred_image_2d_from(
        grid=grid_lp,
        psf=dataset.psf,
        blurring_grid=grid_blurring,
        xp=jnp,
    )
    return result.array


_, blurred_img_jit = jit_profile(blurred_image_from_params, "blurred_image_jit", params_tree)
likelihood_steps.append(("Blurred image (PSF convolution)", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Step 3: Profile-subtracted image (lens light subtraction)
# ---------------------------------------------------------------------------

print("\n--- Step 3: Profile-subtracted image ---")


def profile_subtract(data, blurred_image):
    return data - blurred_image


with timer.section("profile_subtract_eager"):
    blurred_img_jnp = jnp.array(blurred_image.array)
    profile_subtracted = profile_subtract(data_array, blurred_img_jnp)
    block(profile_subtracted)

_, profile_subtracted = jit_profile(
    profile_subtract, "profile_subtract_jit", data_array, blurred_img_jnp
)
likelihood_steps.append(("Profile-subtracted image", timer.records[-1][1] / 10))

print(f"  profile_subtracted shape: {profile_subtracted.shape}")

# ---------------------------------------------------------------------------
# Step 4: Border relocation of traced grid
# ---------------------------------------------------------------------------

print("\n--- Step 4: Border relocation ---")

from autoarray.inversion.mesh.border_relocator import BorderRelocator

with timer.section("border_relocator_setup"):
    border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)

# The source plane grid is the last entry (index -1) of the traced grids list.
traced_source_grid = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)[-1]

with timer.section("border_relocation_eager"):
    relocated_grid = border_relocator.relocated_grid_from(grid=traced_source_grid)
    block(relocated_grid)

print(f"  relocated_grid shape: {relocated_grid.array.shape}")

# For JIT profiling, extract the relocation logic as a raw-array function
relocated_grid_raw = jnp.array(relocated_grid.array)

# ---------------------------------------------------------------------------
# Step 5: Overlay grid (source pixel centres)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Overlay grid (source pixel centres) ---")

# `overlay_grid_from` lives in the RTU module after PyAutoArray split the old
# `rectangular_adapt_density` into `rectangular_bilinear_adapt_density` /
# `rectangular_rtu_adapt_density` (f9aceea3): `RectangularBilinearAdaptDensity`
# subclasses `RectangularRTUAdaptDensity` and inherits this overlay, so the one
# function serves both `--rect-mesh` families.
from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import overlay_grid_from

with timer.section("overlay_grid_eager"):
    mesh_grid = overlay_grid_from(
        shape_native=mesh_shape,
        grid=al.Grid2DIrregular(relocated_grid),
        xp=jnp,
    )
    block(mesh_grid)


def overlay_grid_raw_fn(relocated_grid_raw):
    grid = al.Grid2DIrregular(values=relocated_grid_raw, xp=jnp)
    return overlay_grid_from(shape_native=mesh_shape, grid=grid, xp=jnp)


_, mesh_grid_raw = jit_profile(overlay_grid_raw_fn, "overlay_grid_jit", relocated_grid_raw)
likelihood_steps.append(("Overlay grid (source pixel centres)", timer.records[-1][1] / 10))

print(f"  mesh_grid shape: {mesh_grid_raw.shape}")

# ---------------------------------------------------------------------------
# Step 6: Interpolation + Mapper construction
# ---------------------------------------------------------------------------

print("\n--- Step 6: Interpolation + Mapper ---")

pixelization_obj = instance.galaxies.source.pixelization

with timer.section("interpolation_and_mapper"):
    # ``RectangularBilinearAdaptImage.interpolator_from`` consumes ``adapt_data`` to
    # build the per-pixel weight map; pass the lensed-source image through.
    interpolator = pixelization_obj.mesh.interpolator_from(
        source_plane_data_grid=relocated_grid,
        source_plane_mesh_grid=al.Grid2DIrregular(mesh_grid),
        adapt_data=adapt_image,
    )
    mapper = al.Mapper(interpolator=interpolator, xp=jnp)

print(f"  mapper.pixels (source): {mapper.pixels}")
print(f"  pix_indexes shape: {mapper.pix_indexes_for_sub_slim_index.shape}")

# ---------------------------------------------------------------------------
# Steps 7-13: Extract matrices from FitImaging inversion for consistency
# ---------------------------------------------------------------------------
# The FitImaging pipeline handles edge pixel zeroing, curvature diagonal
# adjustments, and settings that are difficult to replicate manually.
# We extract the correct matrices from fit.inversion so the step-by-step
# matches the reference, then JIT-profile the linear algebra operations.

print("\n--- Extracting inversion matrices from FitImaging ---")

inversion = fit.inversion
print(f"  inversion class: {type(inversion).__name__}")

with timer.section("extract_inversion_matrices"):
    # The operated_mapping_matrix is the blurred mapping matrix (post-PSF).
    # The sparse leg never builds it; only the dense leg reads it.
    if not _cli.use_sparse_operator:
        bmm_ref = jnp.array(inversion.operated_mapping_matrix)
        mapping_matrix_ref = jnp.array(inversion.mapping_matrix)
    else:
        bmm_ref = None
        mapping_matrix_ref = None

    # Extract mapper from the inversion's linear object list
    inv_mapper = inversion.cls_list_from(cls=al.Mapper)[0]
    neighbors = inv_mapper.neighbors
    neighbors_array = jnp.array(np.asarray(neighbors))
    neighbors_sizes = jnp.array(neighbors.sizes)

    reg_coefficient = pixelization_obj.regularization.coefficient

if not _cli.use_sparse_operator:
    print(f"  operated_mapping_matrix shape: {bmm_ref.shape}")
    print(f"  mapping_matrix shape: {mapping_matrix_ref.shape}")

profile_sub_jnp = jnp.array(fit.profile_subtracted_image.array)
noise_jnp = jnp.array(dataset.noise_map.array)

# ---------------------------------------------------------------------------
# Steps 7-8 (dense): Mapping matrix, blurred mapping matrix
# ---------------------------------------------------------------------------

if not _cli.use_sparse_operator:
    print("\n--- Step 7: Mapping matrix ---")

    with timer.section("mapping_matrix"):
        mapping_matrix = inv_mapper.mapping_matrix

    print(f"  mapping_matrix shape: {mapping_matrix.shape}")

    print("\n--- Step 8: Blurred mapping matrix ---")

    with timer.section("blurred_mapping_matrix"):
        blurred_mapping_matrix = dataset.psf.convolved_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            mask=dataset.mask,
            xp=jnp,
        )
        block(blurred_mapping_matrix)

    print(f"  blurred_mapping_matrix shape: {blurred_mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Staged prefix JITs of the inversion-setup block (steps 4-8)
# ---------------------------------------------------------------------------
# Nested prefix-JITs of the same staged computation: params -> step-5 output,
# -> 6, -> 7, -> 8, and (``upto=11``) -> the regularization matrix H. Successive
# differences attribute the combined block's cost to border relocation / overlay
# grid + interpolation / mapping matrix / PSF convolution.
#
# ``--split-setup`` selects whether the whole walk is timed. The interpolator
# prefix (``upto=6``) is timed **either way**, because the H row is attributed as
# t(params -> H) - t(params -> interpolator outputs).
#
# Under ``--sparse`` stages 7 and 8 become the w-tilde setup: 7 the mapper's
# sparse triplets, 8 the whole sparse setup (triplets + MGE operated basis +
# PSF-weighted data) read off a JIT-built ``InversionImagingSparse``. Stage 8 is
# therefore also the sparse leg's combined-setup row, exactly as
# ``blurred_mm_from_params`` is the dense leg's.


def _adapt_images_from(pt):
    """Rebuild ``AdaptImages`` against JIT-rebuilt galaxies.

    The ``galaxy_image_dict`` lookup is by object identity, so the dict has to
    be keyed on the traced instance; the path-keyed dict is kept as a redundant
    fallback.
    """
    return al.AdaptImages(
        galaxy_image_dict={pt.galaxies.source: adapt_image},
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
    )


def _fit_from(pt, xp=jnp):
    """A ``FitImaging`` on this cell's dataset/settings from a params pytree."""
    return al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=list(pt.galaxies)),
        adapt_images=_adapt_images_from(pt),
        settings=al.Settings(
            use_border_relocator=True,
            use_mixed_precision=_cli.use_mixed_precision,
        ),
        xp=xp,
    )


def _sparse_setup_outputs(inv):
    """Everything the sparse steps consume, off an ``InversionImagingSparse``."""
    m = inv.cls_list_from(cls=al.Mapper)[0]
    rows_d, cols_d, vals_d = m.sparse_triplets_data
    rows_c, _, _ = m.sparse_triplets_curvature
    operated = next(iter(inv.linear_func_operated_mapping_matrix_dict.values()))
    return rows_d, cols_d, vals_d, rows_c, operated, inv.psf_weighted_data


def blurred_mm_from_params(params_tree):
    """The dense combined-setup row: params -> operated (blurred) mapping matrix."""
    return jnp.array(_fit_from(params_tree).inversion.operated_mapping_matrix)


def sparse_setup_from_params(params_tree):
    """The sparse combined-setup row: params -> triplets + MGE basis + weighted data."""
    return _sparse_setup_outputs(_fit_from(params_tree).inversion)


def _setup_prefix_fn(upto):
    """Return a ``params_tree -> <stage output>`` function for the given stage.

    ``upto`` is the pipeline step the prefix stops at: 4/5 border relocation +
    overlay grid, 6 interpolator + mapper, 7 mapping matrix (dense) or sparse
    triplets (``--sparse``), 8 blurred mapping matrix (dense) or the whole
    sparse setup, and 11 the regularization matrix H.

    Prefix 11 returns the step-6 outputs **as well as** H. That is load-bearing
    rather than cosmetic: a prefix returning H alone lets XLA dead-code-eliminate
    the query side of the mapper, the prefixes stop nesting and their difference
    goes negative (measured on the Delaunay family: a 215 ms H prefix against a
    391 ms interpolator prefix). Keeping the step-6 outputs live makes prefix 11
    a strict superset of prefix 6.
    """

    def fn(pt):
        if upto in (7, 8) and _cli.use_sparse_operator:
            # The sparse setup rides on the inversion the library builds, so
            # the triplets and the MGE basis are the same objects the w-tilde
            # steps below consume.
            outputs = _sparse_setup_outputs(_fit_from(pt).inversion)
            if upto == 7:
                return outputs[:4]
            return outputs

        t = al.Tracer(galaxies=list(pt.galaxies))
        traced_source = t.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)[-1]
        relocated = border_relocator.relocated_grid_from(grid=traced_source, xp=jnp)
        if upto == 5:
            return relocated.array
        overlay = overlay_grid_from(
            shape_native=mesh_shape,
            grid=al.Grid2DIrregular(values=relocated.array, xp=jnp),
            xp=jnp,
        )
        interp = pixelization_obj.mesh.interpolator_from(
            source_plane_data_grid=relocated,
            source_plane_mesh_grid=al.Grid2DIrregular(values=overlay, xp=jnp),
            adapt_data=adapt_image,
            xp=jnp,
        )
        m = al.Mapper(interpolator=interp, xp=jnp)
        if upto == 6:
            return (
                m.pix_indexes_for_sub_slim_index,
                m.pix_weights_for_sub_slim_index,
            )
        if upto == 11:
            # H is the mapper's own regularization block: the same call the
            # inversion makes. For this cell's rectangular mesh + ``Constant``
            # the neighbours are fixed by ``mesh_shape``, so H is
            # param-independent and the prefix difference reads ~0 (see the
            # module docstring). The step-6 outputs ride along so the prefix
            # strictly contains prefix 6 either way.
            return (
                m.pix_indexes_for_sub_slim_index,
                m.pix_weights_for_sub_slim_index,
                pixelization_obj.regularization.regularization_matrix_from(linear_obj=m, xp=jnp),
            )
        mm = m.mapping_matrix
        if upto == 7:
            return mm
        return dataset.psf.convolved_mapping_matrix_from(
            mapping_matrix=mm, mask=dataset.mask, xp=jnp
        )

    return fn


if _cli.use_sparse_operator:
    _prefix_labels = {
        5: "Border relocation",
        6: "Overlay grid + interpolation",
        7: "Sparse triplets",
        8: "MGE operated basis + PSF-weighted data",
    }
    _combined_label = "Inversion setup (sparse, steps 4-8 combined)"
    _combined_fn = sparse_setup_from_params
    _combined_jit_label = "inversion_setup_sparse_jit"
else:
    _prefix_labels = {
        5: "Border relocation",
        6: "Overlay grid + interpolation",
        7: "Mapping matrix",
        8: "Blurred mapping matrix (PSF)",
    }
    _combined_label = "Inversion setup (steps 4-8 combined)"
    _combined_fn = blurred_mm_from_params
    _combined_jit_label = "inversion_setup_jit"

print(f"\n--- {_combined_label} ---")

_, _combined_result = jit_profile(_combined_fn, _combined_jit_label, params_tree)
_combined_per_call = timer.records[-1][1] / 10
likelihood_steps.append((_combined_label, _combined_per_call))

_prefix_per_call: dict[int, float] = {}
_setup_split: dict | None = None

# Stage 8's prefix. Dense: its own direct ``params -> blurred mapping matrix``
# prefix, exactly as ``delaunay.py`` times it, so the four-way split is
# comparable between the cells. Sparse: the combined FitImaging row *is* stage 8
# (it is what builds the triplets, the MGE basis and the PSF-weighted data), so
# it is reused rather than compiled twice.
if _cli.use_sparse_operator:
    _prefix_per_call[8] = _combined_per_call
    _stage_8_needs_own_prefix = False
else:
    _stage_8_needs_own_prefix = True

if _split_setup:
    print("\n--- Inversion setup four-way split (--split-setup) ---")
    _prefix_stages = (5, 6, 7, 8) if _stage_8_needs_own_prefix else (5, 6, 7)
else:
    print("\n--- Interpolator prefix (needed for the H attribution) ---")
    _prefix_stages = (6,)

for _upto in _prefix_stages:
    jit_profile(_setup_prefix_fn(_upto), f"setup_prefix_{_upto}", params_tree)
    _prefix_per_call[_upto] = timer.records[-1][1] / 10

if _split_setup:
    _setup_split = timing.split_by_successive_differences(_prefix_per_call, _prefix_labels)

    print(
        "  prefix per-call: "
        + ", ".join(f"..{u}={_prefix_per_call[u] * 1000:.2f} ms" for u in (5, 6, 7, 8))
    )
    for _label, _dt in _setup_split.items():
        print(f"    {_label}: {_dt * 1000:8.2f} ms")
    print(f"  (combined single-JIT reference: {_combined_per_call * 1000:.2f} ms)")
else:
    print(f"  interpolator prefix (..6) per-call: {_prefix_per_call[6] * 1000:.2f} ms")

# ---------------------------------------------------------------------------
# Sparse setup rows, timed standalone
# ---------------------------------------------------------------------------
# The three pieces the sparse setup is made of, each on explicit arrays so it is
# its own compiled program. They overlap the combined row above and are reported
# in the JSON as ``steps_sparse_setup_rows`` rather than added to the step total.

sparse_setup_rows: dict[str, float] = {}
sparse_ctx = None
sparse_nnz = None

if _cli.use_sparse_operator:
    print("\n--- Sparse setup rows (standalone) ---")

    sparse_ctx = sparse_steps.sparse_context_from(inversion=inversion, dataset=dataset)

    _pix_indexes = jnp.asarray(inv_mapper.pix_indexes_for_sub_slim_index)
    _pix_weights = jnp.asarray(inv_mapper.pix_weights_for_sub_slim_index)
    _slim_index_for_sub = jnp.asarray(inv_mapper.slim_index_for_sub_slim_index)
    _fft_index = jnp.asarray(dataset.mask.fft_index_for_masked_pixel)
    _sub_fraction = jnp.asarray(inv_mapper.over_sampler.sub_fraction.array)

    _, _triplets = jit_profile(
        sparse_steps.sparse_triplets,
        "sparse_triplets_jit",
        _pix_indexes,
        _pix_weights,
        _slim_index_for_sub,
        _fft_index,
        _sub_fraction,
    )
    sparse_setup_rows["Sparse triplets (data + curvature)"] = timer.records[-1][1] / 10

    rows_data_jnp, cols_jnp, vals_jnp, rows_curv_jnp = _triplets
    sparse_nnz = int(cols_jnp.shape[0])
    print(f"  sparse nnz: {sparse_nnz}")

    # MGE operated basis — 60 PSF-convolved columns. Still dense; kept visible
    # because a matrix-free line has to beat this too.
    #
    # Timed as a **params prefix**, not as an array-level PSF convolution, and
    # that is not a stylistic choice: a linear light profile overrides the
    # inversion's convolution entirely
    # (``LightProfileLinearObjFuncList.operated_mapping_matrix_override``,
    # PyAutoGalaxy) because its flux outside the mask blurs *into* the mask, and
    # the mapping matrix has no columns for that region. Convolving
    # ``linear_func_mapping_matrix_dict`` at image resolution therefore does not
    # reproduce the basis the inversion uses — measured 2.1e-2 relative on F when
    # this row was first written that way. The basis is a function of the model
    # parameters, so the honest program is params -> basis.
    _mge_func = inversion.cls_list_from(cls=aa.AbstractLinearObjFuncList)[0]

    def _mge_operated_basis_from_params(pt):
        inv = _fit_from(pt).inversion
        return next(iter(inv.linear_func_operated_mapping_matrix_dict.values()))

    jit_profile(_mge_operated_basis_from_params, "mge_operated_basis_jit", params_tree)
    sparse_setup_rows["MGE operated basis (params prefix)"] = timer.records[-1][1] / 10

    # The array the w-tilde steps consume is the inversion's own, so F and D are
    # assembled from exactly the object ``InversionImagingSparse`` assembles them
    # from.
    operated_mge_jnp = jnp.asarray(inversion.linear_func_operated_mapping_matrix_dict[_mge_func])
    print(f"  MGE operated basis shape: {operated_mge_jnp.shape}")

    _weight_map_native = jnp.asarray(dataset.sparse_operator.weight_map.array)
    _psf_kernel_native = jnp.asarray(dataset.psf.kernel.native)
    _native_for_slim = jnp.asarray(dataset.mask.derive_indexes.native_for_slim)

    _, psf_weighted_data_jnp = jit_profile(
        sparse_steps.psf_weighted_data,
        "psf_weighted_data_jit",
        _weight_map_native,
        _psf_kernel_native,
        _native_for_slim,
    )
    sparse_setup_rows["PSF-weighted data"] = timer.records[-1][1] / 10

# ---------------------------------------------------------------------------
# Step 9: Data vector (D)
# ---------------------------------------------------------------------------

print("\n--- Step 9: Data vector ---")

sparse_sub_rows: dict[str, float] = {}


def compute_data_vector(blurred_mapping_matrix, image, noise_map):
    return al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=image,
        noise_map=noise_map,
    )


if _cli.use_sparse_operator:
    from functools import partial

    _data_vector_sparse = partial(sparse_steps.data_vector, sparse_ctx)

    with timer.section("data_vector_eager"):
        data_vector = _data_vector_sparse(
            psf_weighted_data_jnp,
            rows_data_jnp,
            cols_jnp,
            vals_jnp,
            operated_mge_jnp,
            profile_sub_jnp,
            noise_jnp,
        )
        block(data_vector)

    _, data_vector = jit_profile(
        _data_vector_sparse,
        "data_vector_sparse_jit",
        psf_weighted_data_jnp,
        rows_data_jnp,
        cols_jnp,
        vals_jnp,
        operated_mge_jnp,
        profile_sub_jnp,
        noise_jnp,
    )
    likelihood_steps.append(("Data vector (D, w-tilde)", timer.records[-1][1] / 10))
else:
    with timer.section("data_vector_eager"):
        data_vector = compute_data_vector(bmm_ref, profile_sub_jnp, noise_jnp)
        block(data_vector)

    _, data_vector = jit_profile(
        compute_data_vector, "data_vector_jit", bmm_ref, profile_sub_jnp, noise_jnp
    )
    likelihood_steps.append(("Data vector (D)", timer.records[-1][1] / 10))

print(f"  data_vector shape: {data_vector.shape}")

# ---------------------------------------------------------------------------
# Step 10: Curvature matrix (F)
# ---------------------------------------------------------------------------

print("\n--- Step 10: Curvature matrix ---")

# Match the FitImaging inversion: add_to_curvature_diag=True, with settings
no_reg_list = list(inversion.no_regularization_index_list)


def compute_curvature_matrix(blurred_mapping_matrix, noise_map):
    return al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        noise_map=noise_map,
        settings=fit.settings,
        add_to_curvature_diag=True,
        no_regularization_index_list=no_reg_list,
        xp=jnp,
    )


if _cli.use_sparse_operator:
    _curvature_sparse = partial(sparse_steps.curvature_matrix, sparse_ctx)

    with timer.section("curvature_matrix_eager"):
        curvature_matrix = _curvature_sparse(
            rows_curv_jnp, cols_jnp, vals_jnp, operated_mge_jnp, noise_jnp
        )
        block(curvature_matrix)

    _, curvature_matrix = jit_profile(
        _curvature_sparse,
        "curvature_matrix_sparse_jit",
        rows_curv_jnp,
        cols_jnp,
        vals_jnp,
        operated_mge_jnp,
        noise_jnp,
    )
    likelihood_steps.append(("Curvature matrix (F, w-tilde)", timer.records[-1][1] / 10))

    # The three blocks inside that one row, each its own compiled program.
    print("\n--- F sub-rows (w-tilde blocks) ---")

    jit_profile(
        partial(sparse_steps.curvature_diag, sparse_ctx),
        "curvature_F_diag_jit",
        rows_curv_jnp,
        cols_jnp,
        vals_jnp,
    )
    sparse_sub_rows["F diag (FFT blocks)"] = timer.records[-1][1] / 10

    jit_profile(
        partial(sparse_steps.curvature_off_diag_func_list, sparse_ctx),
        "curvature_F_off_diag_jit",
        rows_curv_jnp,
        cols_jnp,
        vals_jnp,
        operated_mge_jnp,
        noise_jnp,
    )
    sparse_sub_rows["F off-diag (mapper x MGE)"] = timer.records[-1][1] / 10

    jit_profile(
        sparse_steps.curvature_func_func,
        "curvature_F_func_func_jit",
        operated_mge_jnp,
        noise_jnp,
    )
    sparse_sub_rows["F MGE x MGE GEMM"] = timer.records[-1][1] / 10
else:
    with timer.section("curvature_matrix_eager"):
        curvature_matrix = compute_curvature_matrix(bmm_ref, noise_jnp)
        block(curvature_matrix)

    _, curvature_matrix = jit_profile(
        compute_curvature_matrix, "curvature_matrix_jit", bmm_ref, noise_jnp
    )
    likelihood_steps.append(("Curvature matrix (F)", timer.records[-1][1] / 10))

print(f"  curvature_matrix shape: {curvature_matrix.shape}")

# ---------------------------------------------------------------------------
# Sparse F / D equivalence assertion
# ---------------------------------------------------------------------------
# The standalone sparse steps must reproduce ``InversionImagingSparse`` exactly,
# not approximately: that is the whole claim of the sparse leg. Compared against
# the eager inversion's own F and D, scaled by the matrix/vector magnitude.

sparse_equivalence: dict | None = None

if _cli.use_sparse_operator:
    print("\n--- Sparse F / D equivalence vs fit.inversion ---")

    _F_ref = np.asarray(inversion.curvature_matrix, dtype=float)
    _D_ref = np.asarray(inversion.data_vector, dtype=float)
    _F_got = np.asarray(curvature_matrix, dtype=float)
    _D_got = np.asarray(data_vector, dtype=float)

    _F_scale = max(float(np.max(np.abs(_F_ref))), 1e-300)
    _D_scale = max(float(np.max(np.abs(_D_ref))), 1e-300)
    _F_diff = float(np.max(np.abs(_F_got - _F_ref))) / _F_scale
    _D_diff = float(np.max(np.abs(_D_got - _D_ref))) / _D_scale

    print(f"  F: max |diff| / max|F| = {_F_diff:.3e}  (shape {_F_got.shape})")
    print(f"  D: max |diff| / max|D| = {_D_diff:.3e}  (shape {_D_got.shape})")

    sparse_equivalence = {
        "curvature_matrix_max_rel_diff": _F_diff,
        "data_vector_max_rel_diff": _D_diff,
        "tolerance": 1e-8,
    }

    assert _F_diff <= 1e-8, (
        f"sparse_steps.curvature_matrix does not reproduce "
        f"InversionImagingSparse.curvature_matrix (max rel diff {_F_diff:.3e} > 1e-8)"
    )
    assert _D_diff <= 1e-8, (
        f"sparse_steps.data_vector does not reproduce "
        f"InversionImagingSparse.data_vector (max rel diff {_D_diff:.3e} > 1e-8)"
    )
    print("  Assertion PASSED: sparse F and D match fit.inversion to 1e-8")

# ---------------------------------------------------------------------------
# Step 11: Regularization matrix (H)
# ---------------------------------------------------------------------------

print("\n--- Step 11: Regularization matrix ---")

# TIMING: the reported row is the JIT-timed ``params -> H`` prefix minus the
# ``params -> interpolator outputs`` prefix (module docstring, "Regularization
# matrix (H) attribution"). For this cell's rectangular mesh + ``Constant`` the
# neighbours are param-independent, so that difference is expected to read ~0;
# the assembly cost in isolation is kept as ``regularization_matrix_assembly_s``.
#
# VALUE: steps 12 and 13 consume the inversion's own matrix, so the correctness
# assertions compare like with like against eager FitImaging.

jit_profile(_setup_prefix_fn(11), "regularization_matrix_jit", params_tree)
reg_matrix_prefix_per_call = timer.records[-1][1] / 10
reg_matrix_attributed = reg_matrix_prefix_per_call - _prefix_per_call[6]
likelihood_steps.append(("Regularization matrix (H)", reg_matrix_attributed))

print(f"  params->H prefix per-call:       {reg_matrix_prefix_per_call * 1000:9.3f} ms")
print(f"  params->interpolator prefix:     {_prefix_per_call[6] * 1000:9.3f} ms")
print(f"  attributed H row (difference):   {reg_matrix_attributed * 1000:9.3f} ms")
if reg_matrix_attributed < 0.0:
    print(
        "  NOTE: negative attribution — H is param-independent for this mesh / "
        "regularization pairing (or XLA fused across the prefix boundary); read "
        "the row as ~0 and the absolute prefix as the bound."
    )


def compute_regularization_matrix(neighbors_array, neighbors_sizes):
    return al.util.regularization.constant_regularization_matrix_from(
        coefficient=reg_coefficient,
        neighbors=neighbors_array,
        neighbors_sizes=neighbors_sizes,
        xp=jnp,
    )


# The isolated assembly cost — what this row reported before 2026-09-10. Kept as
# a JSON field so the historic number stays recoverable, not as a step row.
jit_profile(
    compute_regularization_matrix,
    "regularization_matrix_assembly_jit",
    neighbors_array,
    neighbors_sizes,
)
reg_matrix_assembly_per_call = timer.records[-1][1] / 10
print(f"  isolated H assembly per-call:    {reg_matrix_assembly_per_call * 1000:9.3f} ms")

# Full block-diagonal regularization matrix from the inversion: (1581, 1581)
# with a zero (60, 60) linear-bulge block at indices [0..59]. Step 12 (F+H
# solve) and Step 13 (log evidence) both run on this full-size matrix; the
# log_det terms index it down to the mapper rows via ``inversion.mapper_indices``
# to avoid the singular zero diagonal.
regularization_matrix_full = jnp.array(inversion.regularization_matrix)
print(f"  regularization_matrix shape: {regularization_matrix_full.shape}")

# ---------------------------------------------------------------------------
# Batched re-timing of the setup block and H (--vmap-batch N)
# ---------------------------------------------------------------------------
# Broadcast every leaf of the params pytree to a leading batch axis, wrap in
# ``jax.jit(jax.vmap(fn))`` and report ``batch_time / N``. Placed after step 11
# so the H prefix is available; the compiles are the expensive part of this
# block, and the whole thing is guarded so an OOM keeps the unbatched rows.

_vmap_steps: dict[str, float] | None = None
_vmap_split: dict[str, float] | None = None
_vmap_error: str | None = None
_vmap_h_prefix = None
_vmap_interp_prefix = None

if _vmap_batch is not None:
    print(f"\n--- Batched re-timing (--vmap-batch {_vmap_batch}) ---")

    import traceback as _traceback

    _params_batched = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (_vmap_batch, *leaf.shape)),
        params_tree,
    )

    def _vmap_profile(func, label):
        return timing.vmap_profile(
            func,
            label,
            _params_batched,
            _vmap_batch,
            timer=timer,
            jit_records=jit_records,
        )

    try:
        _vmap_prefix_per_call: dict[int, float] = {}
        _vmap_stages = (5, 6, 7, 8, 11) if _stage_8_needs_own_prefix else (5, 6, 7, 11)
        for _upto in _vmap_stages:
            _vmap_prefix_per_call[_upto] = _vmap_profile(
                _setup_prefix_fn(_upto), f"setup_prefix_{_upto}"
            )
        _vmap_combined = _vmap_profile(_combined_fn, "inversion_setup")
        if not _stage_8_needs_own_prefix:
            _vmap_prefix_per_call[8] = _vmap_combined

        _vmap_split = timing.split_by_successive_differences(_vmap_prefix_per_call, _prefix_labels)

        _vmap_steps = {
            _combined_label: _vmap_combined,
            "Regularization matrix (H)": (_vmap_prefix_per_call[11] - _vmap_prefix_per_call[6]),
        }
        _vmap_h_prefix = _vmap_prefix_per_call[11]
        _vmap_interp_prefix = _vmap_prefix_per_call[6]
    except Exception:  # noqa: BLE001 — a vmap failure must not lose the unbatched run
        _vmap_error = _traceback.format_exc()
        _vmap_split = None
        _vmap_steps = None
        print("  VMAP FAILED — unbatched results are unaffected. Traceback:")
        print(_vmap_error)

# ---------------------------------------------------------------------------
# Step 12: Regularized reconstruction: s = NNLS(F + H, D)
# ---------------------------------------------------------------------------
# Identical code in both legs — the w-tilde path replaces the mapping matrix,
# not the solve.

print("\n--- Step 12: Regularized reconstruction ---")


def compute_reconstruction(data_vector, curvature_matrix, regularization_matrix):
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    return al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        xp=jnp,
    )


with timer.section("reconstruction_eager"):
    reconstruction = compute_reconstruction(
        jnp.array(data_vector),
        jnp.array(curvature_matrix),
        regularization_matrix_full,
    )
    block(reconstruction)

_, reconstruction = jit_profile(
    compute_reconstruction,
    "reconstruction_jit",
    jnp.array(data_vector),
    jnp.array(curvature_matrix),
    regularization_matrix_full,
)
likelihood_steps.append(("Regularized reconstruction", timer.records[-1][1] / 10))

print(f"  reconstruction shape: {reconstruction.shape}")

# ---------------------------------------------------------------------------
# Step 13: Map reconstruction to image + log evidence
# ---------------------------------------------------------------------------

print("\n--- Step 13: Mapped reconstruction + log evidence ---")


def compute_log_evidence(
    data,
    noise_map,
    blurred_image,
    blurred_mapping_matrix,
    reconstruction,
    curvature_matrix,
    regularization_matrix,
    mapper_indices,
):
    """Compute the full log evidence including all five terms:

    -2 ln e = chi^2 + s^T H s + ln[det(F+H)] - ln[det(H)] + noise_norm

    Matches the production formula in
    ``autoarray/inversion/inversion/abstract.py:log_det_*`` — reduces both the
    curvature_reg_matrix and the regularization_matrix to the rows/cols indexed
    by ``mapper_indices`` before the log_det. This drops the no-regularization
    rows (e.g. MGE Basis linear components) which otherwise make
    ``det(H) = 0`` and the slogdet return -inf, then uses Cholesky for a
    numerically stable log_det.
    """
    # Map reconstruction to image
    mapped_recon = al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        reconstruction=reconstruction,
        xp=jnp,
    )

    # model_data = lens light + pixelized source
    model_data = blurred_image + mapped_recon

    # Chi-squared
    residual = data - model_data
    chi_squared = jnp.sum((residual / noise_map) ** 2)

    # Regularization term: s^T H s
    regularization_term = jnp.dot(reconstruction, jnp.dot(regularization_matrix, reconstruction))

    # Curvature + regularization matrix
    curvature_reg_matrix = curvature_matrix + regularization_matrix

    # Reduce to pixelization rows/cols only (matches production
    # ``*_matrix_reduced``): required for models with non-regularised
    # linear components (e.g. MGE lens light).
    creg_reduced = curvature_reg_matrix[mapper_indices][:, mapper_indices]
    reg_reduced = regularization_matrix[mapper_indices][:, mapper_indices]

    log_det_curvature_reg = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(creg_reduced))))
    log_det_regularization = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(reg_reduced))))

    # Noise normalization
    noise_normalization = jnp.sum(jnp.log(2 * jnp.pi * noise_map**2))

    return -0.5 * (
        chi_squared
        + regularization_term
        + log_det_curvature_reg
        - log_det_regularization
        + noise_normalization
    )


# For the JIT profiling we use the step-by-step matrices for timing.
# For the correctness assertion we use the inversion's own matrices, because
# cumulative floating-point differences between JIT-compiled and eager paths
# (especially through ill-conditioned solves) can compound significantly.

blurred_img_jnp = jnp.array(blurred_image.array)
recon_jnp = jnp.array(reconstruction)
curv_jnp = jnp.array(curvature_matrix)
reg_jnp = regularization_matrix_full
mapper_indices_jnp = jnp.array(np.asarray(inversion.mapper_indices))
inv_recon_jnp = jnp.array(inversion.reconstruction)
inv_curv_jnp = jnp.array(inversion.curvature_matrix)

if _cli.use_sparse_operator:
    # Sparse model image: the mapper's contribution via the sparse operator +
    # PSF, the MGE's via the already-operated basis. Reduced blocks are taken
    # off the inversion, matching the Delaunay cell's signature.
    _reg_reduced_jnp = jnp.array(inversion.regularization_matrix_reduced)
    _curv_reg_reduced_jnp = jnp.array(inversion.curvature_reg_matrix_reduced)

    _log_evidence_sparse = partial(sparse_steps.log_evidence, sparse_ctx)
    _log_evidence_args = (
        data_array,
        noise_jnp,
        blurred_img_jnp,
        recon_jnp,
        rows_curv_jnp,
        cols_jnp,
        vals_jnp,
        operated_mge_jnp,
        mapper_indices_jnp,
        _reg_reduced_jnp,
        _curv_reg_reduced_jnp,
    )

    with timer.section("log_evidence_eager"):
        log_evidence = _log_evidence_sparse(*_log_evidence_args)
        block(log_evidence)

    _, log_evidence = jit_profile(
        _log_evidence_sparse, "log_evidence_sparse_jit", *_log_evidence_args
    )
    likelihood_steps.append(("Mapped recon + log evidence (sparse)", timer.records[-1][1] / 10))

    log_evidence_check = _log_evidence_sparse(
        data_array,
        noise_jnp,
        blurred_img_jnp,
        inv_recon_jnp,
        rows_curv_jnp,
        cols_jnp,
        vals_jnp,
        operated_mge_jnp,
        mapper_indices_jnp,
        _reg_reduced_jnp,
        _curv_reg_reduced_jnp,
    )
else:
    with timer.section("log_evidence_eager"):
        log_evidence = compute_log_evidence(
            data_array,
            noise_jnp,
            blurred_img_jnp,
            bmm_ref,
            recon_jnp,
            curv_jnp,
            reg_jnp,
            mapper_indices_jnp,
        )
        block(log_evidence)

    _, log_evidence = jit_profile(
        compute_log_evidence,
        "log_evidence_jit",
        data_array,
        noise_jnp,
        blurred_img_jnp,
        bmm_ref,
        recon_jnp,
        curv_jnp,
        reg_jnp,
        mapper_indices_jnp,
    )
    likelihood_steps.append(("Mapped recon + log evidence", timer.records[-1][1] / 10))

    log_evidence_check = compute_log_evidence(
        data_array,
        noise_jnp,
        blurred_img_jnp,
        bmm_ref,
        inv_recon_jnp,
        inv_curv_jnp,
        reg_jnp,
        mapper_indices_jnp,
    )

print(f"  log_evidence (step-by-step) = {log_evidence}")
print(f"  log_evidence (inv matrices) = {log_evidence_check}")
print(f"  log_evidence (reference)    = {log_evidence_ref}")

_step_by_step_drift = check_pinned(
    float(log_evidence_check),
    float(log_evidence_ref),
    label=f"imaging/pixelization[{instrument}] step-by-step vs eager FitImaging",
    rtol=1e-4,
)
if _step_by_step_drift is None:
    print("  Check PASSED: inversion-matrix log_evidence matches FitImaging.log_evidence")

# ===================================================================
# Per-step breakdown summary + JSON + PNG
# ===================================================================

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"PER-STEP BREAKDOWN SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Instrument:            {instrument}")
print(f"  Pixel scale:           {pixel_scale} arcsec/pixel")
print(f"  Mask radius:           {mask_radius} arcsec")
print(f"  Image pixels (masked): {n_image_pixels}")
print(f"  Over-sampled pixels:   {n_over_sampled_pixels}")
print(f"  Mesh shape:            {mesh_shape}")
print(f"  Source pixels:         {n_source_pixels}")
print(f"  Inversion path:        {'sparse (w-tilde)' if _cli.use_sparse_operator else 'dense'}")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
_have_vmap = _vmap_steps is not None
step_total = 0.0

if _have_vmap:
    print(f"      {'':<{max_label}}  {'unbatched':>14}  {f'vmap/{_vmap_batch} per call':>22}")
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    if _have_vmap and label in _vmap_steps:
        print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s  {_vmap_steps[label]:>20.6f} s")
    else:
        print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print("=" * 70)

print(f"  Regularization matrix (H) — params->H prefix:  {reg_matrix_prefix_per_call:.6f} s")
print(f"  Regularization matrix (H) — interp. prefix:    {_prefix_per_call[6]:.6f} s")
print(f"  Regularization matrix (H) — isolated assembly: {reg_matrix_assembly_per_call:.6f} s")

if _setup_split is not None or _vmap_split is not None:
    print("-" * 70)
    _split_label_width = max(len(k) for k in _prefix_labels.values())
    print(
        f"  inversion-setup split{'':<{max(_split_label_width - 21, 0)}}  "
        f"{'unbatched':>14}"
        + (f"  {f'vmap/{_vmap_batch} per call':>22}" if _vmap_split is not None else "")
    )
    for _upto in (5, 6, 7, 8):
        _lab = _prefix_labels[_upto]
        _unb = f"{_setup_split[_lab]:12.6f} s" if _setup_split is not None else f"{'—':>14}"
        _bat = f"  {_vmap_split[_lab]:20.6f} s" if _vmap_split is not None else ""
        print(f"    {_lab:<{_split_label_width}}  {_unb}{_bat}")

if sparse_setup_rows:
    print("-" * 70)
    print("  sparse setup rows (standalone; overlap the combined row, not summed)")
    _w = max(len(k) for k in sparse_setup_rows)
    for _lab, _dt in sparse_setup_rows.items():
        print(f"    {_lab:<{_w}}  {_dt:12.6f} s")

if sparse_sub_rows:
    print("-" * 70)
    print("  F sub-rows (w-tilde blocks; sum to the single F row, not summed here)")
    _w = max(len(k) for k in sparse_sub_rows)
    for _lab, _dt in sparse_sub_rows.items():
        print(f"    {_lab:<{_w}}  {_dt:12.6f} s")

if _vmap_error is not None:
    print("-" * 70)
    print(f"  vmap batch {_vmap_batch}: FAILED (traceback in the result JSON).")

# --- Save results dictionary ---

_configuration = {
    "pixel_scale_arcsec": pixel_scale,
    "mask_radius_arcsec": mask_radius,
    "image_pixels_masked": int(n_image_pixels),
    "over_sampled_pixels": int(n_over_sampled_pixels),
    "mesh_shape": list(mesh_shape),
    "rect_mesh": _cli.rect_mesh,
    "source_pixels": int(n_source_pixels),
    "inversion_path": "sparse" if _cli.use_sparse_operator else "dense",
    "total_params": int(inversion.total_params),
    # Provenance only (autolens_profiling#235 decision 3): this cell's
    # over-sampling and mesh are the A100-pinned JAX configuration and are
    # deliberately NOT production-matched — GPU representativeness is a
    # separate task. What it does record is the thread environment as found
    # (never pinned here) and the NNLS cross-evaluation warm-start memo,
    # which on the JAX path is inert (it seeds the numba fnnls loop only).
    "thread_env": _observe_thread_env(),
    "memo": "library_default (inert on the JAX path)",
    "over_sample_size_lp_rule": {
        "sub_size_list": [4, 2, 2],
        "radial_list": [0.3, 0.6],
        "centre": [0.0, 0.0],
        "note": (
            "Outer sub-size 1 retired repo-wide on 2026-09-08 "
            "(autolens_profiling#235): it leaves the outermost annulus "
            "un-over-sampled and causes gradient issues."
        ),
    },
}

if _cli.use_sparse_operator:
    _configuration["sparse_batch_size"] = int(_cli.sparse_batch_size)
    _configuration["sparse_nnz"] = int(sparse_nnz)
    _configuration["sparse_operator_build_s"] = float(sparse_operator_build_s)
    if _cli.use_mixed_precision:
        _configuration["mixed_precision_note"] = (
            "--use-mixed-precision does not reach the w-tilde blocks: "
            "ImagingSparseOperator casts triplets, vectors and FFT state to "
            "float64 unconditionally."
        )

breakdown_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": _configuration,
    # Regularization scheme + coefficients, on the same terms as the Delaunay
    # family's block. A result JSON without this key is a pre-2026-09-10 row.
    "regularization": reg_provenance,
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    # Absolute prefix times behind the attributed "Regularization matrix (H)"
    # row: the row is ``regularization_matrix_prefix_s - interpolator_prefix_s``.
    "regularization_matrix_prefix_s": float(reg_matrix_prefix_per_call),
    "interpolator_prefix_s": float(_prefix_per_call[6]),
    "regularization_matrix_assembly_s": float(reg_matrix_assembly_per_call),
    # lower / compile / first-call / steady per label, for every timed JIT.
    "jit_phases": jit_records,
}

if _setup_split is not None:
    breakdown_summary["setup_split"] = {k: float(v) for k, v in _setup_split.items()}

if sparse_setup_rows:
    breakdown_summary["steps_sparse_setup_rows"] = {
        k: float(v) for k, v in sparse_setup_rows.items()
    }

if sparse_sub_rows:
    breakdown_summary["steps_sparse_sub_rows"] = {k: float(v) for k, v in sparse_sub_rows.items()}

if sparse_equivalence is not None:
    breakdown_summary["sparse_equivalence"] = sparse_equivalence

if _vmap_batch is not None:
    breakdown_summary["vmap_batch"] = int(_vmap_batch)
    if _vmap_error is not None:
        breakdown_summary["vmap_error"] = _vmap_error
    else:
        breakdown_summary["steps_vmap_per_call"] = {k: float(v) for k, v in _vmap_steps.items()}
        breakdown_summary["setup_split_vmap"] = {k: float(v) for k, v in _vmap_split.items()}
        breakdown_summary["regularization_matrix_prefix_vmap_per_call_s"] = float(_vmap_h_prefix)
        breakdown_summary["interpolator_prefix_vmap_per_call_s"] = float(_vmap_interp_prefix)

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"pixelization_breakdown_{instrument}_v{al_version}",
    cell="pixelization",
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
    f"Pixelization Imaging Likelihood — Per-Step Breakdown — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_over_sampled_pixels} over-sampled  |  {mesh_shape[0]}x{mesh_shape[1]} mesh  |  "
    f"{'sparse (w-tilde)' if _cli.use_sparse_operator else 'dense'}  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.margins(x=0.15)
fig.tight_layout()

fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Regression check — eager log_evidence only
# ===================================================================

# Keyed by --rect-mesh since 2026-08-26 — the Bilinear (rank-CDF) and RTU
# (kernel-CDF) families reconstruct differently, so one number cannot pin both.
# Same fiducial, and the same paired measurement, as
# likelihood_runtime/pixelization.py: PyAutoArray 72fb01d1 (#490) fixed the
# mirrored bilinear ROW weights and the round-off-dependent cell assignment in
# the shared adaptive rectangular mapper, and bilinear gives 28370.240585918986
# at 72fb01d1^ (the 2026-05-18 pin, to 1.3e-6) against the value below at
# 72fb01d1.
# RE-MEASURED 2026-09-08 (autolens_profiling#237), one eager run per
# `--rect-mesh` on the local WSL host. Two changes are folded into these values:
# the light-profile radial bins moved from [4, 2, 1] to [4, 2, 2] repo-wide
# (autolens_profiling#235, sub-size 1 causes gradient issues), and the step-5
# import was repaired — the cell used to raise on `main` importing
# `autoarray.inversion.mesh.mesh.rectangular_adapt_density`, a module
# PyAutoArray split into `rectangular_bilinear_adapt_density` /
# `rectangular_rtu_adapt_density`, so #235 could take no eager value here and
# left the pins as previously measured. Both old pins in fact still PASSED at
# `rtol=1e-4` (bilinear 28622.397322591198 -> 28621.128714095972, 4.4e-5; rtu
# 28506.318157467784 -> 28505.343980143432, 3.4e-5); they are replaced by the
# measured values so the pin describes what the cell computes today.
# Libraries: PyAutoArray 35aa681f, PyAutoFit 74884c5e, PyAutoGalaxy f1225037,
# PyAutoLens 08a05858, PyAutoNerves 0e7163bc; autolens 2026.8.17.1.
#
# The dense and sparse legs share one pin: they fit the same data with the same
# mesh and the same regularization, so a sparse row that does not reproduce the
# dense evidence is a bug in the w-tilde path, not a different measurement.
#
# Since 2026-09-10 this is ``check_pinned`` rather than a hard
# ``assert_allclose``: a profiling run records and flags drift, it does not
# adjudicate library correctness (results/notes/design_lock_in.md), and the
# timings from a drifted run are still data.
EXPECTED_LOG_EVIDENCE_HST = {
    # 39x39 = 1521 source pixels, MGE-60 lens light, adapt_image=lensed_source
    "bilinear": 28621.128714095972,
    "rtu": 28505.343980143432,
}[_cli.rect_mesh]

_pin_drift = check_pinned(
    log_evidence_ref,
    EXPECTED_LOG_EVIDENCE_HST,
    label=f"imaging/pixelization[{instrument}, {_cli.rect_mesh}] eager log_evidence",
    rtol=1e-4,
)
if _pin_drift is None:
    print(f"  Eager regression check PASSED: log_evidence matches {EXPECTED_LOG_EVIDENCE_HST:.6f}")
_rel_to_pin = abs(log_evidence_ref - EXPECTED_LOG_EVIDENCE_HST) / abs(EXPECTED_LOG_EVIDENCE_HST)
print(f"  relative difference vs pin: {_rel_to_pin:.3e}")

_drift_records = [r for r in (_pin_drift, _step_by_step_drift) if r is not None]
record_pinned_check(dict_path, EXPECTED_LOG_EVIDENCE_HST, _drift_records)
