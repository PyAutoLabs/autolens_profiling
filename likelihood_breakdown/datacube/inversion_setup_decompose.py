"""
JAX Profiling: Decompose the datacube "inversion setup" step
============================================================

The cube breakdown (``likelihood_breakdown/datacube/delaunay.py``) reports a
single dominant "Inversion setup, incl. NUFFT (per channel)" step that is ~78%
of the cube cost. That step is one combined JIT block running the whole chain:

    params -> Tracer -> AdaptImages -> FitInterferometer -> inversion
           -> mapper (Delaunay triangulation + neighbours + pix weights)
           -> mapping_matrix  L
           -> data_vector  D = Lᵀ · dirty_image

To scope the cross-`Analysis` shared-state optimisation (PyAutoFit feature,
prompt ``autofit/analysis_shared_state_cross_factor.md``) we need to know how
that ~78% splits between work that is **channel-invariant** (shareable once
across all channels when the lens model + source mesh are shared) and work that
is **per-channel** (the data vector, which folds in each channel's distinct
visibilities via ``dirty_image``).

Method: JIT four cutpoints of the same chain and read the deltas. NOTE the
cutpoints are NOT a linear superset chain — XLA dead-code-eliminates per leaf,
so a cutpoint only pays for the branch its returned leaf actually needs. The
true dependency tree is::

    A. trace        -> stacked traced grids                 (ray-trace only)
       └─ B. mapper -> mapping_matrix.sum()                 (A + Delaunay mapper + L)
            ├─ C. curvature   -> curvature_matrix.sum()     (B + triplets + Lᵀ W̃ L)
            └─ D. data_vector -> data_vector  = Lᵀ·dirty_image  (B + the per-channel matmul)

C (curvature) and D (data vector) are PARALLEL CHILDREN of the shared mapper+L
block, not a sequence — ``data_vector`` does not depend on the curvature matrix,
so XLA prunes the curvature FFT out of cutpoint D. Therefore the only valid
subtractions are against the common parent B:

    ray-trace          = A
    mapper + L         = B - A
    curvature (Lᵀ W̃ L) = C - B      (curvature-specific work over the shared mapper)
    data vector D      = D - B      (the Lᵀ·dirty_image matmul; the ONLY per-channel sliver)

(An earlier version of this script wrongly used ``D - C``, treating the cutpoints
as a linear chain; that produced a negative data-vector delta and a >100%
"shareable" fraction — the tell-tale signature of the per-leaf DCE pruning.)

Everything except the data-vector matmul is channel-invariant under the shared
lens+mesh assumption (``dirty_image`` is the only per-channel input, entering
only at the final ``Lᵀ·dirty_image``) — i.e. the headline block the shared-state
optimisation collapses from ``N ×`` to ``1 ×``.

Runs on CPU at SMA scale by default (190 visibilities — the sparse precompute is
trivial, no GPU needed). Single-channel timings; the cube cost is ``N × per-call``
for the channel-variant rows and ``1 × per-call`` once the optimisation lands.

Usage::

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
    PYAUTO_SKIP_API_GATE=1 python3 \
    likelihood_breakdown/datacube/inversion_setup_decompose.py --instrument sma

pyauto-api-gate: skip
  (false positive on the real ``from autofit.jax import register_model`` import,
  identical to the shipped ``likelihood_breakdown/datacube/delaunay.py``; the
  gate's static validator can't resolve the ``autofit.jax`` submodule even though
  Python imports it fine.)
"""

import numpy as np
import jax
import jax.numpy as jnp
import os
import time
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al
import autoarray as aa
from autofit.jax import register_model as _register_model_pytrees

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

import os as _smoke_os
import sys as _smoke_sys
if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from _profile_cli import (  # noqa: E402
    parse_profile_cli,
    device_info_dict,
    resolve_output_paths,
    auto_simulate_if_missing,
)
from simulators.interferometer import INSTRUMENTS  # noqa: E402
_cli = parse_profile_cli()

instrument = _cli.instrument or "sma"
hilbert_pixels = 1500
regularization_coefficient = 1.0


class Timer:
    def __init__(self):
        self.records = []

    @contextmanager
    def section(self, label):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


timer = Timer()


def jit_profile(func, label, *args, n_repeats=10):
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
    return per_call


# ---------------------------------------------------------------------------
# Setup (mirrors likelihood_breakdown/datacube/delaunay.py)
# ---------------------------------------------------------------------------

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[1]
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
dataset_path = Path("dataset") / "interferometer" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="interferometer",
    instrument=instrument,
    workspace_root=_workspace_root,
)

mask_radius = INSTRUMENTS[instrument]["mask_radius"]
real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)
transformer_chunk_size = INSTRUMENTS[instrument].get("transformer_chunk_size", None)


def _build_transformer(uv_wavelengths, real_space_mask):
    return al.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=transformer_chunk_size,
    )


print(f"\n--- Dataset load [{instrument}] ---")
dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=_build_transformer,
).apply_sparse_operator(use_jax=True, show_progress=False)

n_visibilities = dataset.uv_wavelengths.shape[0]
print(f"  Visibilities: {n_visibilities}")

print("\n--- Adapt image + Hilbert mesh ---")
adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
image_mesh = al.image_mesh.Hilbert(pixels=hilbert_pixels, weight_power=1.0, weight_floor=0.0)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.real_space_mask, adapt_data=adapt_image
)
n_mesh_vertices = image_plane_mesh_grid.shape[0]
print(f"  Delaunay vertices: {n_mesh_vertices}")

print("\n--- Model ---")
mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_ell[0], sigma=0.01)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_ell[1], sigma=0.01)
shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
mesh = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
regularization = al.reg.ConstantSplit(coefficient=regularization_coefficient)
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

param_vector = model.physical_values_from_prior_medians
instance = model.instance_from_vector(vector=param_vector)
_register_model_pytrees(model)
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

grid_pix_raw = jnp.array(dataset.grids.pixelization.array)
mesh_grid_raw = jnp.array(image_plane_mesh_grid.array)


def _adapt_images_from(params_tree):
    return al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            params_tree.galaxies.source: image_plane_mesh_grid,
        },
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid,
        },
    )


def _fit_from(params_tree):
    tracer = al.Tracer(galaxies=list(params_tree.galaxies))
    return al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        adapt_images=_adapt_images_from(params_tree),
        settings=al.Settings(use_mixed_precision=_cli.use_mixed_precision),
        xp=jnp,
    )


# ---------------------------------------------------------------------------
# Cutpoint A — ray-trace only
# ---------------------------------------------------------------------------

print("\n=== Cutpoint A: ray-trace only ===")


def cut_trace(params_tree):
    tracer = al.Tracer(galaxies=list(params_tree.galaxies))
    grid = aa.Grid2DIrregular(values=mesh_grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


cost_A = jit_profile(cut_trace, "A_trace", params_tree)

# ---------------------------------------------------------------------------
# Cutpoint B — + Delaunay mapper + mapping matrix L
# ---------------------------------------------------------------------------

print("\n=== Cutpoint B: + mapper + mapping matrix L ===")


def cut_mapper(params_tree):
    fit = _fit_from(params_tree)
    return jnp.asarray(fit.inversion.mapping_matrix).sum()


cost_B = jit_profile(cut_mapper, "B_mapper", params_tree)

# ---------------------------------------------------------------------------
# Cutpoint C — + sparse triplets + curvature F = Lᵀ W̃ L
# ---------------------------------------------------------------------------

print("\n=== Cutpoint C: + curvature F = Lᵀ W̃ L ===")


def cut_curvature(params_tree):
    fit = _fit_from(params_tree)
    return jnp.asarray(fit.inversion.curvature_matrix).sum()


cost_C = jit_profile(cut_curvature, "C_curvature", params_tree)

# ---------------------------------------------------------------------------
# Cutpoint D — + data vector D = Lᵀ · dirty_image (full current step)
# ---------------------------------------------------------------------------

print("\n=== Cutpoint D: + data vector D (full inversion-setup step) ===")


def cut_data_vector(params_tree):
    fit = _fit_from(params_tree)
    return jnp.asarray(fit.inversion.data_vector)


cost_D = jit_profile(cut_data_vector, "D_data_vector", params_tree)

# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------

# C and D are parallel children of the shared mapper+L parent B (XLA prunes the
# curvature branch out of D), so both subtract B — never each other.
ray_trace = cost_A
mapper_L = cost_B - cost_A
curvature = cost_C - cost_B
data_vector = cost_D - cost_B

sub_steps = {
    "ray_trace": ray_trace,
    "mapper_plus_mapping_matrix_L": mapper_L,
    "curvature_F_LtWtL": curvature,
    "data_vector_D": data_vector,
}
raw_cutpoints = {
    "A_trace": cost_A,
    "B_mapper": cost_B,
    "C_curvature": cost_C,
    "D_data_vector": cost_D,
}

# Channel-invariant = ray-trace + mapper + L + curvature (all pure functions of
# the shared lens model + source mesh + channel-invariant Khat). Channel-variant
# = only the Lᵀ·dirty_image matmul, since dirty_image is the per-channel input.
channel_invariant = ray_trace + mapper_L + curvature
channel_variant = data_vector

# Full per-channel inversion cost (what the optimisation would collapse): the
# shared block computed once + the per-channel matmul kept N times.
per_channel_total = channel_invariant + channel_variant

print("\n" + "=" * 64)
print(f"INVERSION-SETUP DECOMPOSITION — {instrument.upper()} — {n_visibilities} vis")
print("=" * 64)
print(f"  {'ray-trace':<34}{ray_trace:>12.6f} s")
print(f"  {'mapper + mapping matrix L':<34}{mapper_L:>12.6f} s")
print(f"  {'curvature F = Lᵀ W̃ L':<34}{curvature:>12.6f} s")
print(f"  {'data vector D (per-channel)':<34}{data_vector:>12.6f} s")
print("-" * 64)
print(f"  {'per-channel inversion total':<34}{per_channel_total:>12.6f} s")
print("-" * 64)
print(f"  {'channel-INVARIANT (shareable 1x)':<34}{channel_invariant:>12.6f} s")
print(f"  {'channel-VARIANT (stays Nx)':<34}{channel_variant:>12.6f} s")
if per_channel_total > 0:
    print(f"  shareable fraction of inversion total: {100*channel_invariant/per_channel_total:.1f}%")
print("-" * 64)
print("  raw cutpoint leaves (NOT a linear chain — C,D are parallel children of B):")
print(f"    A trace={cost_A:.6f}  B mapper+L={cost_B:.6f}  C +curv={cost_C:.6f}  D +datavec={cost_D:.6f}")
print("=" * 64)

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

import json

al_version = al.__version__
summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "model": "delaunay",
    "purpose": "decompose the datacube inversion-setup step into channel-invariant vs channel-variant sub-steps (sparse w-tilde route)",
    "configuration": {
        "visibilities": int(n_visibilities),
        "delaunay_vertices": int(n_mesh_vertices),
        "use_mixed_precision": bool(_cli.use_mixed_precision),
    },
    "note": "cutpoints are a dependency TREE not a linear chain — XLA DCE prunes per leaf, so C(curvature) and D(data_vector) are parallel children of B(mapper+L); deltas subtract the common parent B, never C from D",
    "raw_cutpoints_per_call_s": raw_cutpoints,
    "sub_steps_per_call_s": sub_steps,
    "per_channel_inversion_total_per_call_s": per_channel_total,
    "channel_invariant_per_call_s": channel_invariant,
    "channel_variant_per_call_s": channel_variant,
    "shareable_fraction_of_inversion_total": (
        channel_invariant / per_channel_total if per_channel_total > 0 else None
    ),
}

dict_path, _ = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "datacube",
    default_basename=f"inversion_setup_decompose_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(summary, indent=2))
print(f"\n  Saved: {dict_path}")
