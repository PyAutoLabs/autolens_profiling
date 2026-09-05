"""
JAX Profiling: DelaunayNN Imaging Likelihood — Per-Step Breakdown
==================================================================

The **Sibson natural-neighbour sibling** of
``likelihood_breakdown/imaging/delaunay.py``. Every knob is deliberately
identical — HST, 3.5" mask, radial-bin over-sampling, 1500-vertex Hilbert image
mesh, MGE-60 lens light, Isothermal + ExternalShear mass, ConstantSplit
regularization, border relocator on, dense inversion path — and the **only**
difference is the source mesh's interpolation scheme:

- ``delaunay.py``    — ``al.mesh.Delaunay`` + ``al.InterpolatorDelaunay``:
  C0 barycentric interpolation inside the containing triangle, 3 weights per
  query coordinate.
- ``delaunay_nn.py`` — ``al.mesh.DelaunayNN`` + ``al.InterpolatorDelaunayNN``:
  Sibson natural-neighbour interpolation over the whole circumcircle cavity,
  continuous through Delaunay diagonal flips, and linearly precise.

So a per-step diff between the two JSONs isolates the interpolation scheme with
nothing else varying, and the four-way ``--split-setup`` labels mean exactly the
same thing in both.

Static caps (this is where the cost difference lives)
-----------------------------------------------------

Sibson weights need fixed-shape arrays for ``jax.jit``, so PyAutoArray pins
production caps in ``autoarray/inversion/mesh/interpolator/sibson.py``:

    SIBSON_MAX_CAVITY_TRIANGLES = 32
    SIBSON_MAX_NEIGHBORS        = 32
    SIBSON_QUERY_CHUNK          = 256

``autoarray/inversion/mesh/mesh/delaunay_nn.py`` binds them onto the mesh class
as ``max_cavity_triangles`` / ``max_neighbors`` / ``query_chunk``. The workspace
mass-model audit (101 traced Hilbert meshes) observed maxima of 25 cavity
triangles and 27 natural neighbours; caps 16 and 24 overflowed, so 32 is the
smallest tested-safe shape. Every mapper row is therefore 32 wide here against
Delaunay's 3, which is the structural reason the mapper geometry benchmark
reads ~157 ms (cap 32) vs ~37 ms (Delaunay) unbatched on the A100 — this
breakdown is what says how much of that survives into the whole likelihood.
If a cap is exceeded the affected weights become NaN (the sample is rejected)
rather than being silently truncated.

Pipeline steps:

1. Ray-trace data grid to source plane
2. Ray-trace mesh grid (image-plane vertices) to source plane
3. Lens light images (pre-PSF, JIT) + PSF convolution (eager)
4. Profile-subtracted image
5. Border relocation (data grid + mesh grid)
6. Delaunay triangulation + Sibson interpolation + mapper
7. Mapping matrix
8. Blurred mapping matrix / Inversion setup (steps 5-8 combined)
9. Data vector (D)
10. Curvature matrix (F)
11. Regularization matrix (H) — ConstantSplit scheme (JIT-timed from params)
12. Regularized reconstruction: s = (F + H)^{-1} D
13. Map reconstruction to image + log evidence

Per-step timing is approximate: XLA may fuse operations differently when
compiled as one program vs separate pieces. All JAX timings use
``block_until_ready()`` to force synchronous measurement (over every leaf of
the returned pytree, so multi-output prefixes are synchronised too).

Regularization matrix (H) attribution
-------------------------------------

The H row is a **JIT-timed step**, not a host-to-device copy. A prefix
function ``params_tree -> regularization matrix`` is compiled — Tracer from
the params pytree, traced grids, border relocation, ``InterpolatorDelaunayNN``,
``Mapper``, then ``ConstantSplit.regularization_matrix_from(linear_obj=mapper,
xp=jnp)`` — and the row is reported as the difference::

    t(params -> H) - t(params -> interpolator outputs)

where the interpolator prefix is exactly ``_setup_prefix_fn(6)``. The shared
tracer / relocation / triangulation work is already charged to "Triangulation +
interpolation" in the four-way ``--split-setup`` table, so subtracting the
interpolator prefix leaves the incremental cost of the split-point walk plus
the regularization assembly and avoids double-counting the mesh build. The
absolute prefix time is kept in the JSON as ``regularization_matrix_prefix_s``
beside the attributed row.

For that subtraction to mean anything the two prefixes must nest, and they do
not by default: ConstantSplit reads the interpolator's split-point walk, never
the per-query interpolation, so a prefix returning H alone lets XLA
dead-code-eliminate the whole query side of the mapper. Measured on local CPU
(DelaunayNN/HST, 2026-09-05) that gave a 215 ms H prefix against a 391 ms
interpolator prefix — a nonsensical -175 ms row. ``_setup_prefix_fn(11)``
therefore returns the step-6 outputs alongside H, making it a strict superset
of ``_setup_prefix_fn(6)``.

Before 2026-09 this row timed ``jnp.array(inversion.regularization_matrix)`` —
a host-to-device copy of the 19.5 MB matrix the *eager NumPy* ``FitImaging``
had already computed. On the A100 that read as ~14.4 ms of PCIe traffic and
was not a JIT step at all (autolens_profiling#219). The downstream
reconstruction and log-evidence steps still consume the inversion's own
``regularization_matrix``, so every correctness assertion is unchanged.

Batched re-timing (``--vmap-batch N``)
--------------------------------------

With ``--vmap-batch N`` the combined inversion-setup block, each
``--split-setup`` prefix and the params->H prefix are re-timed under
``jax.jit(jax.vmap(fn))`` on a params pytree broadcast to batch ``N``, and
reported as amortized per-call time (batch time / N) in a column beside the
unbatched one. Batch 16 matches the ``n_batch=16`` of the Nautilus reference
runs, so the breakdown can be reconciled against their per-evaluation cost.
The Delaunay triangulation's qhull ``pure_callback`` is
``vmap_method="sequential"`` (one host call per lane), so the triangulation
row is expected to stay roughly linear in ``N`` while the dense linear algebra
amortizes.

Output
------

Results JSON and PNG are written to ``results/breakdown/imaging/`` using
the basename ``delaunay_nn_breakdown_{instrument}_v{al_version}``. Under
``--config-name`` the cell name is passed explicitly as ``delaunay_nn``,
because ``resolve_output_paths``'s default first-token derivation would give
``delaunay`` and clobber the Delaunay cell's row.
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

sys.path.insert(0, str(_profiling_root()))
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
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

_cli = parse_profile_cli()


def _parse_vmap_batch(argv) -> int | None:
    """Parse ``--vmap-batch N`` / ``--vmap-batch=N`` out of *argv*.

    Read straight from ``sys.argv`` rather than added to
    ``_profile_cli.parse_profile_cli`` because it is a breakdown-only flag —
    the runtime cells resolve their batch from the VRAM table / probe JSON
    instead, and the shared parser stays the sweep-driver contract.
    """
    for i, arg in enumerate(argv):
        if arg == "--vmap-batch" and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith("--vmap-batch="):
            return int(arg.split("=", 1)[1])
    return None


_vmap_batch = _parse_vmap_batch(sys.argv)
if _vmap_batch is not None and _vmap_batch < 1:
    raise ValueError(f"--vmap-batch must be >= 1 (got {_vmap_batch})")

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
    """Force synchronisation on every JAX array in *x* (array or pytree).

    Tuple-returning prefixes (steps 5 and 6 of the ``--split-setup`` walk)
    used to slip through an ``hasattr(x, "block_until_ready")`` test and were
    therefore timed asynchronously, which is exactly the artifact the H-row
    fix removes elsewhere. Blocking over ``tree_leaves`` makes every timed
    step synchronous on the same terms.
    """
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
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

    # Matches ``_delaunay_nn_model`` in scripts/misc/searches/_setup.py — the
    # registered DelaunayNN target — so the breakdown and the sampler runs
    # profile the same mesh. ``DelaunayNN`` is a ``Delaunay`` subclass with the
    # identical (pixels, zeroed_pixels, areas_factor) constructor.
    mesh = al.mesh.DelaunayNN(
        pixels=n_mesh_vertices,
        areas_factor=0.5,
        zeroed_pixels=0,
    )
    regularization = al.reg.ConstantSplit(coefficient=1.0)
    pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  DelaunayNN pixels: {n_mesh_vertices}")
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
print(f"  DelaunayNN vertices:     {n_source_pixels}")
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

print("\n--- Step 6: Delaunay triangulation + Sibson interpolation + Mapper ---")

pixelization_obj = instance.galaxies.source.pixelization

# Single symbol for the mesh family's interpolator, so this script differs from
# its barycentric sibling (``delaunay.py``) only in the mesh class and this
# line. Every direct interpolator construction below goes through it.
_INTERPOLATOR_CLS = al.InterpolatorDelaunayNN

with timer.section("delaunay_nn_interpolation_and_mapper"):
    interpolator = _INTERPOLATOR_CLS(
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
# Staged prefix JITs of the inversion-setup block
# ---------------------------------------------------------------------------
# Nested prefix-JITs of the same staged computation: params -> step-5 output,
# -> step-6, -> step-7, -> step-8, and (``upto=11``) -> the ConstantSplit
# regularization matrix. Successive differences attribute the combined block's
# cost to border relocation / triangulation+interpolation / mapping matrix /
# PSF convolution. The differences inherit the fusion caveat (XLA may move work
# across prefix boundaries, so small negatives are noise), and every prefix pays
# the ray-trace preamble (~0.3 ms, measured separately in steps 1-2) which lands
# in the first difference.
#
# ``--split-setup`` selects whether the whole four-way walk is timed. The
# interpolator prefix (``upto=6``) is timed **either way**, because step 11
# attributes the H row as t(params -> H) - t(params -> interpolator outputs);
# when ``--split-setup`` is on its timing is reused rather than compiled twice.


def _setup_prefix_fn(upto):
    """Return a ``params_tree -> <stage output>`` function for the given stage.

    ``upto`` is the pipeline step the prefix stops at: 5 border relocation,
    6 interpolator+mapper, 7 mapping matrix, 8 blurred mapping matrix, and
    11 the ConstantSplit regularization matrix H (the step-11 row's prefix;
    it branches off after the mapper, before the mapping matrix, because
    that is exactly what the inversion's H depends on).

    Prefix 11 returns the step-6 outputs **as well as** H. That is load-bearing,
    not cosmetic: H is built from the interpolator's split-point walk
    (``_mappings_sizes_weights_split``) and does not consume the per-query
    interpolation at all, so returning H alone lets XLA dead-code-eliminate the
    whole query side of the mapper — the prefixes stop nesting and their
    difference goes *negative* (measured: 215 ms H prefix vs 391 ms interpolator
    prefix on local CPU, DelaunayNN/HST). Keeping the step-6 outputs live makes
    prefix 11 a strict superset of prefix 6, so the difference is exactly the
    incremental cost of the split walk plus the regularization assembly, with
    the shared triangulation charged once to "Triangulation + interpolation".
    """

    def fn(pt):
        t = al.Tracer(galaxies=list(pt.galaxies))
        traced_source = t.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)[-1]
        traced_mesh = t.traced_grid_2d_list_from(
            grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
        )[-1]
        relocated = border_relocator.relocated_grid_from(grid=traced_source, xp=jnp)
        relocated_mesh = border_relocator.relocated_mesh_grid_from(
            grid=traced_source, mesh_grid=traced_mesh, xp=jnp
        )
        if upto == 5:
            return relocated.array, relocated_mesh.array
        interp = _INTERPOLATOR_CLS(
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
        if upto == 11:
            # H is the mapper's own regularization block: the same call the
            # inversion makes (``AbstractRegularization.regularization_matrix_from``
            # on each linear object, block-diagonalised). The lens MGE's block
            # is all-zero and is not built here — this prefix times the source
            # mesh's ConstantSplit matrix, which is the whole non-trivial cost.
            #
            # The step-6 outputs ride along so this prefix strictly contains
            # prefix 6 (see the docstring) — without them XLA prunes the query
            # interpolation and the attribution difference goes negative.
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


_prefix_labels = {
    5: "Border relocation",
    6: "Triangulation + interpolation",
    7: "Mapping matrix",
    8: "Blurred mapping matrix (PSF)",
}
_split_setup = "--split-setup" in sys.argv
_prefix_per_call: dict[int, float] = {}
_setup_split: dict | None = None

if _split_setup:
    print("\n--- Inversion setup four-way split (--split-setup) ---")
    _prefix_stages = (5, 6, 7, 8)
else:
    print("\n--- Interpolator prefix (needed for the step-11 H attribution) ---")
    _prefix_stages = (6,)

for _upto in _prefix_stages:
    jit_profile(_setup_prefix_fn(_upto), f"setup_prefix_{_upto}", params_tree)
    _prefix_per_call[_upto] = timer.records[-1][1] / 10

if _split_setup:
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
else:
    print(f"  interpolator prefix (5..6) per-call: {_prefix_per_call[6] * 1000:.2f} ms")

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
#
# TIMING: the reported row is the JIT-timed ``params -> H`` prefix minus the
# ``params -> interpolator outputs`` prefix (module docstring, "Regularization
# matrix (H) attribution"). Timing ``jnp.array(inversion.regularization_matrix)``
# instead — as this script did before 2026-09 — measures a host-to-device copy
# of an eagerly-computed NumPy matrix, not a JIT step.
#
# VALUE: steps 12 and 13 still consume the inversion's own matrix, so the
# correctness assertions compare like with like against eager FitImaging.

jit_profile(_setup_prefix_fn(11), "regularization_matrix_jit", params_tree)
reg_matrix_prefix_per_call = timer.records[-1][1] / 10
reg_matrix_attributed = reg_matrix_prefix_per_call - _prefix_per_call[6]
likelihood_steps.append(("Regularization matrix (H)", reg_matrix_attributed))

print(f"  params->H prefix per-call:       {reg_matrix_prefix_per_call * 1000:9.3f} ms")
print(f"  params->interpolator prefix:     {_prefix_per_call[6] * 1000:9.3f} ms")
print(f"  attributed H row (difference):   {reg_matrix_attributed * 1000:9.3f} ms")
if reg_matrix_attributed < 0.0:
    print(
        "  NOTE: negative attribution — XLA fused work across the prefix "
        "boundary; read the row as ~0 and the absolute prefix as the bound."
    )

# The eager copy is still made (steps 12-13 need the value); it is no longer
# reported as the H row's cost.
with timer.section("regularization_matrix_eager_copy"):
    regularization_matrix = jnp.array(inversion.regularization_matrix)
    block(regularization_matrix)

print(f"  regularization_matrix shape: {regularization_matrix.shape}")

# ---------------------------------------------------------------------------
# Batched re-timing of the setup block and H (--vmap-batch N)
# ---------------------------------------------------------------------------
# Mirrors PART D of ``likelihood_runtime/delaunay.py``: broadcast every leaf of
# the params pytree to a leading batch axis, wrap in ``jax.jit(jax.vmap(fn))``
# and report ``batch_time / N``. Placed after step 11 so the H prefix is
# available; the compiles are the expensive part of this block.

_vmap_steps: dict[str, float] | None = None
_vmap_split: dict[str, float] | None = None
_vmap_error: str | None = None

if _vmap_batch is not None:
    print(f"\n--- Batched re-timing (--vmap-batch {_vmap_batch}) ---")

    import traceback as _traceback

    _params_batched = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (_vmap_batch, *leaf.shape)),
        params_tree,
    )

    def _vmap_profile(func, label, n_repeats=10):
        """Time ``jax.jit(jax.vmap(func))``; return amortized per-call seconds."""
        fn = jax.jit(jax.vmap(func))
        with timer.section(f"{label}_vmap{_vmap_batch}_first_call"):
            block(fn(_params_batched))
        with timer.section(f"{label}_vmap{_vmap_batch}_steady_x{n_repeats}"):
            for _ in range(n_repeats):
                block(fn(_params_batched))
        batch_time = timer.records[-1][1] / n_repeats
        per_call = batch_time / _vmap_batch
        print(
            f"    -> batch {_vmap_batch}: {batch_time * 1000:9.3f} ms; "
            f"per call: {per_call * 1000:9.3f} ms"
        )
        return per_call

    try:
        _vmap_prefix_per_call: dict[int, float] = {}
        for _upto in (5, 6, 7, 8, 11):
            _vmap_prefix_per_call[_upto] = _vmap_profile(
                _setup_prefix_fn(_upto), f"setup_prefix_{_upto}"
            )
        _vmap_combined = _vmap_profile(blurred_mm_from_params, "inversion_setup")

        _vmap_split = {}
        _prev = 0.0
        for _upto in (5, 6, 7, 8):
            _vmap_split[_prefix_labels[_upto]] = _vmap_prefix_per_call[_upto] - _prev
            _prev = _vmap_prefix_per_call[_upto]

        _vmap_steps = {
            "Inversion setup (steps 5-8 combined)": _vmap_combined,
            "Regularization matrix (H)": (_vmap_prefix_per_call[11] - _vmap_prefix_per_call[6]),
        }
        _vmap_h_prefix = _vmap_prefix_per_call[11]
        _vmap_interp_prefix = _vmap_prefix_per_call[6]
    except Exception:  # noqa: BLE001 — a vmap failure must not lose the unbatched run
        _vmap_error = _traceback.format_exc()
        _vmap_h_prefix = None
        _vmap_interp_prefix = None
        print("  VMAP FAILED — unbatched results are unaffected. Traceback:")
        print(_vmap_error)
else:
    _vmap_h_prefix = None
    _vmap_interp_prefix = None

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
      reconstruction (lens-MGE linear params + source-DelaunayNN pixels)
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
print(f"  DelaunayNN vertices:   {n_source_pixels}")
print(f"  Edge zeroed pixels:    {edge_pixels_total}")
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

if _setup_split is not None or _vmap_split is not None:
    print("-" * 70)
    _split_label_width = max(len(k) for k in _prefix_labels.values())
    print(
        f"  inversion-setup split{'':<{_split_label_width - 21}}  "
        f"{'unbatched':>14}"
        + (f"  {f'vmap/{_vmap_batch} per call':>22}" if _vmap_split is not None else "")
    )
    for _upto in (5, 6, 7, 8):
        _lab = _prefix_labels[_upto]
        _unb = f"{_setup_split[_lab]:12.6f} s" if _setup_split is not None else f"{'—':>14}"
        _bat = f"  {_vmap_split[_lab]:20.6f} s" if _vmap_split is not None else ""
        print(f"    {_lab:<{_split_label_width}}  {_unb}{_bat}")
if _vmap_error is not None:
    print("-" * 70)
    print(f"  vmap batch {_vmap_batch}: FAILED (traceback in the result JSON).")

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
    # Absolute prefix times behind the attributed "Regularization matrix (H)"
    # row: the row is ``regularization_matrix_prefix_s - interpolator_prefix_s``.
    "regularization_matrix_prefix_s": float(reg_matrix_prefix_per_call),
    "interpolator_prefix_s": float(_prefix_per_call[6]),
}

if _setup_split is not None:
    breakdown_summary["setup_split"] = {k: float(v) for k, v in _setup_split.items()}

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
    default_basename=f"delaunay_nn_breakdown_{instrument}_v{al_version}",
    # Explicit: the default first-token rule resolves "delaunay_nn_..." to the
    # cell "delaunay" and would overwrite the Delaunay cell's config-tagged
    # JSON/PNG (autolens_profiling#219).
    cell="delaunay_nn",
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
    f"DelaunayNN Imaging Likelihood — Per-Step Breakdown — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_over_sampled_pixels} over-sampled  |  {n_source_pixels} DelaunayNN vertices  |  "
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

# Pinned from the first eager CPU run of this script: 2026-09-05, local CPU
# (WSL, JAX fp64), PyAutoLens v2026.8.17.1 / PyAutoNerves 8f6a0b25 /
# PyAutoFit 12b3e6b6 / PyAutoArray a1e4c0ef / PyAutoGalaxy 6b8b18b6.
# 1500-pixel Hilbert/DelaunayNN, MGE-60 lens, adapt_image=lensed_source.
# Compare: the barycentric Delaunay sibling pins 29110.92085793 — the meshes
# have the same vertices, so the ~34 nat gap is the interpolation scheme.
EXPECTED_LOG_EVIDENCE_HST = 29144.581943885652

np.testing.assert_allclose(
    log_evidence_ref,
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/delaunay_nn[{instrument}]: regression — eager log_evidence drifted "
        f"(got {log_evidence_ref}, expected {EXPECTED_LOG_EVIDENCE_HST})"
    ),
)
print(f"  Eager regression assertion PASSED: log_evidence matches {EXPECTED_LOG_EVIDENCE_HST:.6f}")
