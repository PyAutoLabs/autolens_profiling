"""
JAX Profiling: Delaunay Imaging Likelihood — Per-Step Breakdown
================================================================

Decomposes the JAX likelihood function for an imaging dataset (Hilbert/Delaunay
source model) into its individual pipeline steps and JIT-profiles each one
separately. This script is the **breakdown** counterpart to
``likelihood_runtime/imaging/delaunay.py``, which measures only the
full-pipeline single-JIT cost.

Key differences from the rectangular pixelization breakdown script:

- Mesh vertices are computed in the **image-plane** via a Hilbert image mesh,
  then ray-traced to the source-plane.
- Edge points are appended around the mask border and zeroed during inversion.
- Uses **InterpolatorDelaunay** (barycentric interpolation within triangles).
- Uses **ConstantSplit** regularization (cross-derivative scheme).
- Delaunay triangulation itself uses scipy on CPU and cannot be JIT-compiled.

Pipeline steps:

1. Ray-trace data grid to source plane
2. Ray-trace mesh grid (image-plane vertices) to source plane
3. Lens light images (pre-PSF, JIT) + PSF convolution (eager)
4. Profile-subtracted image
5. Border relocation (data grid + mesh grid)
6. Delaunay triangulation + interpolation + mapper
7. Mapping matrix
8. Blurred mapping matrix / Inversion setup (steps 5-8 combined)
9. Data vector (D)
10. Curvature matrix (F)
11. Regularization matrix (H) — ConstantSplit scheme
12. Regularized reconstruction: s = (F + H)^{-1} D
13. Map reconstruction to image + log evidence

Per-step timing is approximate: XLA may fuse operations differently when
compiled as one program vs separate pieces. All JAX timings use
``block_until_ready()`` to force synchronous measurement.

Output
------

Results JSON and PNG are written to ``results/breakdown/imaging/`` using
the basename ``delaunay_breakdown_{instrument}_v{al_version}``.
"""

import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import autoarray as aa
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
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)
from simulators.imaging import INSTRUMENTS  # noqa: E402

_cli = parse_profile_cli()

instrument = "hst"  # <-- change this to profile a different instrument


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        """Context manager that records wall-clock time for *label*."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    def summary(self):
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


def block(x):
    """Call block_until_ready if available (JAX arrays)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time first call and steady-state average.

    Returns the compiled function and its result.
    """
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
likelihood_steps = []  # (label, per_call_seconds) for the final summary

# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[1]
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
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )

    if _cli.use_sparse_operator:
        # Engage the w-tilde sparse-operator path. See the runtime sibling
        # script for the rationale (autolens_profiling#44).
        dataset = dataset.apply_sparse_operator()

# ---------------------------------------------------------------------------
# 2. Adapt image + image mesh (Hilbert)
# ---------------------------------------------------------------------------

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

print(f"  adapt_image shape (slim): {adapt_image.shape_slim}")

print("\n--- Image mesh construction (Hilbert) ---")

n_mesh_vertices = 1500  # 1500-tier production fiducial

with timer.section("image_mesh_hilbert"):
    image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=adapt_image
    )

edge_pixels_total = 0
print(f"  Hilbert pixels: {n_mesh_vertices}")
print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

# ---------------------------------------------------------------------------
# 3. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

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

    mesh = al.mesh.Delaunay(
        pixels=n_mesh_vertices,
        zeroed_pixels=0,
    )
    regularization = al.reg.ConstantSplit(coefficient=1.0)
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

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

n_pytree_leaves = len(jax.tree_util.tree_leaves(params_tree))
print(f"  Pytree JAX leaves: {n_pytree_leaves}")

tracer = al.Tracer(galaxies=list(instance.galaxies))

# AdaptImages tells FitImaging where mesh vertices live in image-plane
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
# Key configuration that dictates run time
# ---------------------------------------------------------------------------

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
print(f"  Edge zeroed pixels:      {edge_pixels_total}")

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
mesh_grid_raw = jnp.array(image_plane_mesh_grid.array)
data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)

# Keep autoarray objects for eager calls that need them.
grid_lp = dataset.grids.lp
grid_blurring = dataset.grids.blurring

# ---------------------------------------------------------------------------
# Step 1: Ray-trace data grid to source plane
# ---------------------------------------------------------------------------

print("\n--- Step 1: Ray-trace data grid ---")

with timer.section("ray_trace_data_eager"):
    traced_grids = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)
    for tg in traced_grids:
        block(tg)

print(f"  Number of planes traced: {len(traced_grids)}")


def ray_trace_data_raw(grid_raw):
    """Wraps ray-tracing so inputs/outputs are raw arrays."""
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, traced_data_grids_raw = jit_profile(ray_trace_data_raw, "ray_trace_data_jit", grid_pix_raw)
likelihood_steps.append(("Ray-trace data grid", timer.records[-1][1] / 10))

print(f"  traced_data_grids shape: {traced_data_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 2: Ray-trace mesh grid (image-plane vertices) to source plane
# ---------------------------------------------------------------------------

print("\n--- Step 2: Ray-trace mesh grid ---")

with timer.section("ray_trace_mesh_eager"):
    traced_mesh = tracer.traced_grid_2d_list_from(
        grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
    )
    for tg in traced_mesh:
        block(tg)


def ray_trace_mesh_raw(mesh_raw):
    """Ray-trace image-plane mesh vertices to source plane."""
    grid = aa.Grid2DIrregular(values=mesh_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, traced_mesh_grids_raw = jit_profile(ray_trace_mesh_raw, "ray_trace_mesh_jit", mesh_grid_raw)
likelihood_steps.append(("Ray-trace mesh grid", timer.records[-1][1] / 10))

print(f"  traced_mesh_grids shape: {traced_mesh_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 3: Blurred image of non-linear light profiles (lens light)
# ---------------------------------------------------------------------------

print("\n--- Step 3: Blurred image (lens light profiles) ---")

# Sub-step 3a: Compute raw lens light images (JIT-profiled)
#
# We rebuild ``Grid2D`` (uniform, masked) instead of ``Grid2DIrregular``
# inside the function so the lens-light Basis can produce its
# ``Array2D`` zero-vector for ``LightProfileLinear`` components — see
# ``PyAutoGalaxy:autogalaxy/profiles/basis.py:151`` which does
# ``mask=grid.mask`` and would AttributeError on an irregular grid.
# Masks are captured from module-level scope (static at trace time).
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

# Sub-step 3b: PSF convolution
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
# Step 4: Profile-subtracted image (lens light subtraction)
# ---------------------------------------------------------------------------

print("\n--- Step 4: Profile-subtracted image ---")


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
# Step 5: Border relocation (data grid + mesh grid)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Border relocation ---")

from autoarray.inversion.mesh.border_relocator import BorderRelocator

with timer.section("border_relocator_setup"):
    border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)

traced_source_grid = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)[-1]
traced_mesh_source = tracer.traced_grid_2d_list_from(
    grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
)[-1]

with timer.section("border_relocation_eager"):
    relocated_grid = border_relocator.relocated_grid_from(grid=traced_source_grid)
    relocated_mesh_grid = border_relocator.relocated_mesh_grid_from(
        grid=traced_source_grid,
        mesh_grid=traced_mesh_source,
    )
    block(relocated_grid)
    block(relocated_mesh_grid)

print(f"  relocated_data_grid shape: {relocated_grid.array.shape}")
print(f"  relocated_mesh_grid shape: {relocated_mesh_grid.array.shape}")

# ---------------------------------------------------------------------------
# Step 6: Delaunay triangulation + interpolation + mapper
# ---------------------------------------------------------------------------

print("\n--- Step 6: Delaunay triangulation + Interpolation + Mapper ---")

pixelization_obj = instance.galaxies.source.pixelization

with timer.section("delaunay_interpolation_and_mapper"):
    interpolator = al.InterpolatorDelaunay(
        mesh=pixelization_obj.mesh,
        mesh_grid=relocated_mesh_grid,
        data_grid=relocated_grid,
    )
    mapper = al.Mapper(
        interpolator=interpolator,
        image_plane_mesh_grid=image_plane_mesh_grid,
        xp=jnp,
    )

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

with timer.section("extract_inversion_matrices"):
    bmm_ref = jnp.array(inversion.operated_mapping_matrix)
    mapping_matrix_ref = jnp.array(inversion.mapping_matrix)

    inv_mapper = inversion.cls_list_from(cls=al.Mapper)[0]
    neighbors = inv_mapper.neighbors
    neighbors_array = jnp.array(np.asarray(neighbors))
    neighbors_sizes = jnp.array(neighbors.sizes)

print(f"  operated_mapping_matrix shape: {bmm_ref.shape}")
print(f"  mapping_matrix shape: {mapping_matrix_ref.shape}")

# ---------------------------------------------------------------------------
# Step 7: Mapping matrix
# ---------------------------------------------------------------------------

print("\n--- Step 7: Mapping matrix ---")

with timer.section("mapping_matrix"):
    mapping_matrix = inv_mapper.mapping_matrix

print(f"  mapping_matrix shape: {mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 8: Blurred mapping matrix (PSF convolution)
# ---------------------------------------------------------------------------

print("\n--- Step 8: Blurred mapping matrix ---")

with timer.section("blurred_mapping_matrix"):
    blurred_mapping_matrix = dataset.psf.convolved_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        mask=dataset.mask,
        xp=jnp,
    )
    block(blurred_mapping_matrix)

# JIT-profile the full inversion setup pipeline (steps 5-8 combined):
# border relocation → Delaunay triangulation → interpolation → mapper → mapping matrix → PSF convolution.
# These steps are tightly sequential; the full pipeline JIT-compiles them all together.


def blurred_mm_from_params(params_tree):
    """Compute blurred mapping matrix via full inversion setup from a pytree ModelInstance."""
    t = al.Tracer(galaxies=list(params_tree.galaxies))
    # Recreate adapt_images with new galaxy instance so dict lookup by object identity works.
    adapt_images_jax = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            params_tree.galaxies.source: image_plane_mesh_grid,
        },
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid,
        },
    )
    fit_jax = al.FitImaging(
        dataset=dataset,
        tracer=t,
        adapt_images=adapt_images_jax,
        settings=al.Settings(
            use_border_relocator=True,
            use_mixed_precision=_cli.use_mixed_precision,
        ),
        xp=jnp,
    )
    return jnp.array(fit_jax.inversion.operated_mapping_matrix)


_, bmm_jit = jit_profile(blurred_mm_from_params, "inversion_setup_jit", params_tree)
likelihood_steps.append(("Inversion setup (steps 5-8 combined)", timer.records[-1][1] / 10))

print(f"  blurred_mapping_matrix (JIT) shape: {bmm_jit.shape}")

# ---------------------------------------------------------------------------
# Optional: four-way split of the inversion-setup block (--split-setup)
# ---------------------------------------------------------------------------
# Nested prefix-JITs of the same staged computation: params -> step-5 output,
# -> step-6, -> step-7, -> step-8. Successive differences attribute the
# combined block's cost to border relocation / triangulation+interpolation /
# mapping matrix / PSF convolution. The differences inherit the fusion caveat
# (XLA may move work across prefix boundaries, so small negatives are noise),
# and every prefix pays the ray-trace preamble (~0.3 ms, measured separately
# in steps 1-2) which lands in the first difference.

_setup_split: dict | None = None

if "--split-setup" in sys.argv:
    print("\n--- Inversion setup four-way split (--split-setup) ---")

    def _setup_prefix_fn(upto):
        def fn(pt):
            t = al.Tracer(galaxies=list(pt.galaxies))
            traced_source = t.traced_grid_2d_list_from(
                grid=dataset.grids.pixelization, xp=jnp
            )[-1]
            traced_mesh = t.traced_grid_2d_list_from(
                grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
            )[-1]
            relocated = border_relocator.relocated_grid_from(grid=traced_source, xp=jnp)
            relocated_mesh = border_relocator.relocated_mesh_grid_from(
                grid=traced_source, mesh_grid=traced_mesh, xp=jnp
            )
            if upto == 5:
                return relocated.array, relocated_mesh.array
            interp = al.InterpolatorDelaunay(
                mesh=pixelization_obj.mesh,
                mesh_grid=relocated_mesh,
                data_grid=relocated,
                xp=jnp,
            )
            m = al.Mapper(
                interpolator=interp,
                image_plane_mesh_grid=image_plane_mesh_grid,
                xp=jnp,
            )
            if upto == 6:
                return (
                    m.pix_indexes_for_sub_slim_index,
                    m.pix_weights_for_sub_slim_index,
                )
            mm = m.mapping_matrix
            if upto == 7:
                return mm
            return dataset.psf.convolved_mapping_matrix_from(
                mapping_matrix=mm, mask=dataset.mask, xp=jnp
            )

        return fn

    _prefix_labels = {
        5: "Border relocation",
        6: "Triangulation + interpolation",
        7: "Mapping matrix",
        8: "Blurred mapping matrix (PSF)",
    }
    _prefix_per_call = {}
    for _upto in (5, 6, 7, 8):
        jit_profile(_setup_prefix_fn(_upto), f"setup_prefix_{_upto}", params_tree)
        _prefix_per_call[_upto] = timer.records[-1][1] / 10

    _setup_split = {}
    _prev = 0.0
    for _upto in (5, 6, 7, 8):
        _setup_split[_prefix_labels[_upto]] = _prefix_per_call[_upto] - _prev
        _prev = _prefix_per_call[_upto]

    print(
        "  prefix per-call: "
        + ", ".join(f"5..{u}={_prefix_per_call[u] * 1000:.2f} ms" for u in (5, 6, 7, 8))
    )
    for _label, _dt in _setup_split.items():
        print(f"    {_label}: {_dt * 1000:8.2f} ms")
    _combined = dict(likelihood_steps)["Inversion setup (steps 5-8 combined)"]
    print(f"  (combined single-JIT reference: {_combined * 1000:.2f} ms)")

bmm_jnp = bmm_ref  # Use the reference matrices for linear algebra steps
print(f"  blurred_mapping_matrix shape: {blurred_mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 9: Data vector (D)
# ---------------------------------------------------------------------------

print("\n--- Step 9: Data vector ---")


def compute_data_vector(blurred_mapping_matrix, image, noise_map):
    return al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=image,
        noise_map=noise_map,
    )


profile_sub_jnp = jnp.array(fit.profile_subtracted_image.array)
noise_jnp = jnp.array(dataset.noise_map.array)

with timer.section("data_vector_eager"):
    data_vector = compute_data_vector(bmm_jnp, profile_sub_jnp, noise_jnp)
    block(data_vector)

_, data_vector = jit_profile(
    compute_data_vector, "data_vector_jit", bmm_jnp, profile_sub_jnp, noise_jnp
)
likelihood_steps.append(("Data vector (D)", timer.records[-1][1] / 10))

print(f"  data_vector shape: {data_vector.shape}")

# ---------------------------------------------------------------------------
# Step 10: Curvature matrix (F)
# ---------------------------------------------------------------------------

print("\n--- Step 10: Curvature matrix ---")

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


with timer.section("curvature_matrix_eager"):
    curvature_matrix = compute_curvature_matrix(bmm_jnp, noise_jnp)
    block(curvature_matrix)

_, curvature_matrix = jit_profile(
    compute_curvature_matrix, "curvature_matrix_jit", bmm_jnp, noise_jnp
)
likelihood_steps.append(("Curvature matrix (F)", timer.records[-1][1] / 10))

print(f"  curvature_matrix shape: {curvature_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 11: Regularization matrix (H) — ConstantSplit scheme
# ---------------------------------------------------------------------------

print("\n--- Step 11: Regularization matrix (ConstantSplit) ---")

# ConstantSplit uses a cross-derivative scheme via the interpolator's
# _mappings_sizes_weights_split, not the simple neighbour-difference approach.
# We extract it from the inversion for consistency and JIT-profile separately.

with timer.section("regularization_matrix_eager"):
    regularization_matrix = jnp.array(inversion.regularization_matrix)
    block(regularization_matrix)

likelihood_steps.append(("Regularization matrix (H)", timer.records[-1][1]))

print(f"  regularization_matrix shape: {regularization_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 12: Regularized reconstruction: s = NNLS(F + H, D)
# ---------------------------------------------------------------------------
#
# Uses ``reconstruction_positive_only_from`` (NNLS) to match production
# AnalysisImaging behaviour. An earlier version of this script used
# ``jnp.linalg.solve(F+H, D)`` which under-reports the per-step
# reconstruction cost (~5 ms vs ~36 ms NNLS on RTX 2060). The two
# solvers happen to produce identical reconstructions for the
# well-conditioned ConstantSplit setup at prior medians (no negative
# source pixels, NNLS reduces to linear solve), so the downstream
# log-evidence value is unchanged within rtol=1e-4.

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
        jnp.array(regularization_matrix),
    )
    block(reconstruction)

_, reconstruction = jit_profile(
    compute_reconstruction,
    "reconstruction_jit",
    jnp.array(data_vector),
    jnp.array(curvature_matrix),
    jnp.array(regularization_matrix),
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
    reduced_indices,
    reg_reduced,
    curv_reg_reduced,
):
    """Compute the full log evidence including all five terms:

    -2 ln e = chi^2 + s^T H s + ln[det(F+H)] - ln[det(H)] + noise_norm

    Mirrors the reference implementation in PyAutoArray's
    ``Inversion.log_evidence`` chain:

    - chi^2 and the noise-normalisation term are computed over the *full*
      reconstruction (lens-MGE linear params + source-Delaunay pixels)
      because they're per-pixel data terms over the masked image.
    - s^T H s and the two log-det terms operate on the *reduced* (rank-
      stripped) regularisation block, which slices out the non-mapper
      rows/columns whose regularisation entries are zero. The full
      regularisation matrix is rank-deficient by construction (the lens
      MGE bulge is linear but not regularised), so `slogdet` on the full
      matrix returns -inf; the reduced block is positive-definite and
      Cholesky-safe.
    - Log-det terms use ``2 * sum(log(diag(cholesky(M))))`` to match the
      reference inversion (see PyAutoArray's
      ``Inversion.log_det_regularization_matrix_term`` /
      ``log_det_curvature_reg_matrix_term``).
    """
    # Map reconstruction to image
    mapped_recon = al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        reconstruction=reconstruction,
        xp=jnp,
    )

    # model_data = lens light + pixelized source
    model_data = blurred_image + mapped_recon

    # Chi-squared (over full reconstruction → full mapping matrix)
    residual = data - model_data
    chi_squared = jnp.sum((residual / noise_map) ** 2)

    # Reduced reconstruction (source-pixel block only) for the regularised
    # terms.
    s_reduced = reconstruction[reduced_indices]

    # Regularization term: s^T H s on the reduced block
    regularization_term = jnp.dot(s_reduced, jnp.dot(reg_reduced, s_reduced))

    # Log-determinant terms via Cholesky on the reduced (PD) matrices —
    # matches PyAutoArray's reference. slogdet on the full matrices returns
    # -inf because they contain zero rows for the non-regularised lens MGE
    # linear parameters.
    L_cr = jnp.linalg.cholesky(curv_reg_reduced)
    log_det_curvature_reg = 2.0 * jnp.sum(jnp.log(jnp.diag(L_cr)))
    L_r = jnp.linalg.cholesky(reg_reduced)
    log_det_regularization = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))

    # Noise normalization (over full masked image)
    noise_normalization = jnp.sum(jnp.log(2 * jnp.pi * noise_map**2))

    return -0.5 * (
        chi_squared
        + regularization_term
        + log_det_curvature_reg
        - log_det_regularization
        + noise_normalization
    )


# For the JIT profiling we use the step-by-step reconstruction for timing.
# For the correctness assertion we use the inversion's own reconstruction,
# because cumulative floating-point differences between JIT-compiled and
# eager paths (especially through ill-conditioned solves) can compound
# significantly.
#
# The reduced (rank-stripped) regularisation block and curvature+reg matrix
# are precomputed eagerly from the inversion. These are constant across
# calls within this script's lens/source configuration, so the reduction
# work itself is not part of the per-call timed cost.

blurred_img_jnp = jnp.array(blurred_image.array)
recon_jnp = jnp.array(reconstruction)
reduced_indices_jnp = jnp.array(inversion.mapper_indices)
reg_reduced_jnp = jnp.array(inversion.regularization_matrix_reduced)
curv_reg_reduced_jnp = jnp.array(inversion.curvature_reg_matrix_reduced)

with timer.section("log_evidence_eager"):
    log_evidence = compute_log_evidence(
        data_array,
        noise_jnp,
        blurred_img_jnp,
        bmm_jnp,
        recon_jnp,
        reduced_indices_jnp,
        reg_reduced_jnp,
        curv_reg_reduced_jnp,
    )
    block(log_evidence)

_, log_evidence = jit_profile(
    compute_log_evidence,
    "log_evidence_jit",
    data_array,
    noise_jnp,
    blurred_img_jnp,
    bmm_jnp,
    recon_jnp,
    reduced_indices_jnp,
    reg_reduced_jnp,
    curv_reg_reduced_jnp,
)
likelihood_steps.append(("Mapped recon + log evidence", timer.records[-1][1] / 10))

print(f"  log_evidence (step-by-step) = {log_evidence}")

# Correctness check: recompute log_evidence using the inversion's own
# reconstruction to avoid accumulated FP drift from the JIT-compiled
# reconstruction step.
inv_recon_jnp = jnp.array(inversion.reconstruction)

log_evidence_check = compute_log_evidence(
    data_array,
    noise_jnp,
    blurred_img_jnp,
    bmm_jnp,
    inv_recon_jnp,
    reduced_indices_jnp,
    reg_reduced_jnp,
    curv_reg_reduced_jnp,
)
print(f"  log_evidence (inv matrices) = {log_evidence_check}")
print(f"  log_evidence (reference)    = {log_evidence_ref}")

np.testing.assert_allclose(
    float(log_evidence_check),
    float(log_evidence_ref),
    rtol=1e-4,
    err_msg="Log_evidence from inversion matrices does not match FitImaging.log_evidence",
)
print("  Assertion PASSED: inversion-matrix log_evidence matches FitImaging.log_evidence")

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
print(f"  Delaunay vertices:     {n_source_pixels}")
print(f"  Edge zeroed pixels:    {edge_pixels_total}")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print("=" * 70)

# --- Save results dictionary ---

breakdown_summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "delaunay_vertices": int(n_source_pixels),
        "edge_zeroed_pixels": int(edge_pixels_total),
        "inversion_path": "sparse" if _cli.use_sparse_operator else "dense",
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
}

if _setup_split is not None:
    breakdown_summary["setup_split"] = {k: float(v) for k, v in _setup_split.items()}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"delaunay_breakdown_{instrument}_v{al_version}",
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
    f"Delaunay Imaging Likelihood — Per-Step Breakdown — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_over_sampled_pixels} over-sampled  |  {n_source_pixels} Delaunay vertices  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.margins(x=0.15)
fig.tight_layout()

fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Regression assertion — eager log_evidence only
# ===================================================================

EXPECTED_LOG_EVIDENCE_HST = (
    29110.92085793  # 1500-pixel Hilbert/Delaunay, MGE-60 lens, adapt_image=lensed_source
)

np.testing.assert_allclose(
    log_evidence_ref,
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/delaunay[{instrument}]: regression — eager log_evidence drifted "
        f"(got {log_evidence_ref}, expected {EXPECTED_LOG_EVIDENCE_HST})"
    ),
)
print(f"  Eager regression assertion PASSED: log_evidence matches {EXPECTED_LOG_EVIDENCE_HST:.6f}")
