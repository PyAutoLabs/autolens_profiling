"""
Numba CPU Witness: split-regularization lever 1 is bit-identical
================================================================

Witness for **lever 1** of the numba CPU levers task (autolens_profiling#267).

Lever 1 replaced the two pure-Python loops in PyAutoArray's split-regularization
assembly — ``regularization_util.reg_split_np_from`` (the split-point stencil
completion) and ``regularization_util.pixel_splitted_regularization_matrix_np_from``
(the dense ``(P, P)`` scatter) — with numba kernels, and **retained the original
loop bodies verbatim** as ``_reg_split_reference`` and
``_pixel_splitted_regularization_matrix_reference``. The library's unit tests
assert the kernels against those references on synthetic tables. This cell
asserts the same thing on the **production system the CPU timing legs measure**:
the HST Delaunay N=1500 source-only (S3) ``AdaptSplit`` system that
``fixed_light_numba.py`` route ``b`` builds.

A speedup row without this witness beside it is a number for a *possibly
different* matrix. Three things are checked, in the order a reader should want
them:

``W1`` **the matrix itself**. The interpolator's
``_mappings_sizes_weights_split`` tables are run through
``_reg_split_reference`` (on copies — it mutates mappings and sizes in place)
and then through ``_pixel_splitted_regularization_matrix_reference``, and the
result is asserted ``np.array_equal`` with the library's own
``inversion.regularization_matrix`` (the kernel path). ``array_equal``, not
``allclose``: the claim is bit-identity, and the max absolute difference is
printed and recorded either way.

  One caveat is recorded rather than assumed away. ``_reg_split_reference``
  carries a ``size == 0`` index leak that ``_reg_split_kernel`` deliberately
  fixes (``j`` keeps its value from the previous row when a row's stencil is
  empty, so the flag-zero branch writes at a stale ``j + 1``). The two are
  therefore bit-identical only on tables with no empty rows. The count of
  zero-size rows is asserted to be 0 and recorded as
  ``n_zero_size_stencil_rows``; a non-zero count would make W1's verdict a
  statement about the leak, not about the lever.

``W2`` **the caller's tables are not consumed**. ``reg_split_np_from`` copies
mappings and sizes on entry (``np.array(...)``) and rebinds weights
(``-1.0 * w``), so the interpolator's ``cached_property`` stencil tables must be
byte-for-byte unchanged after a full library inversion. They are snapshotted
before the library builds ``H`` and compared after. An in-place kernel that ate
its input would still produce the right ``H`` on the first evaluation and the
wrong one on every evaluation after it — which a single-call witness on ``H``
alone cannot see.

``W3`` **the likelihood**. The fit's ``figure_of_merit`` (the log evidence) is
computed twice on **fresh fit objects**: once with the library as it ships, once
with both module-level entry points monkeypatched to the ``_reference`` bodies
(each given copies). Bit-identity of ``H`` should make the relative difference
exactly ``0.0``; the gate passes at ``<= 1e-9``. The monkeypatch is at module
scope on ``regularization_util``, which is where ``reg_split_from`` and
``pixel_splitted_regularization_matrix_from`` look their numpy branches up, so
the library's own dispatch is what routes into the references.

What this cell is NOT
--------------------

It takes **no timings** and asserts no pin. The A/B milliseconds are
``hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_ral_hst_fp64``'s
two arms of ``fixed_light_numba.py``; this cell is the correctness leg that runs
beside the feature arm and says the two arms computed the same matrix.

It imports **no JAX** of its own and sets no ``JAX_*``/``XLA_*`` variable. Lever
1 did not touch the JAX branch of either function (``reg_split_from`` and
``pixel_splitted_regularization_matrix_from`` both dispatch on ``if xp == np``,
and only that branch changed), so there is nothing for a JAX witness to say
here. Note that ``jax`` still lands in ``sys.modules`` via the first
``FitImaging`` regardless.

Construction
------------

The dataset, mask, over-sampling, Hilbert mesh, MGE lens light, mass/shear
priors, instance stream and ``AdaptSplit(0.1, 10.0, signal_scale=0.1)``
regularization are the same block ``fixed_light_numba.py`` runs, and the S3
system is built through the same
``likelihood_breakdown.fixed_light_system.fixed_light_system_from`` with
``sparse_operator="rebake_cpu"``, i.e. route ``b``'s
``InversionImagingSparseNumba`` on the subtracted dataset. The construction is
**replicated rather than imported**: ``fixed_light_numba.py`` is a top-to-bottom
script with no ``__main__`` guard, so importing it would run all four of its
legs. Every shared helper it uses is imported from the same module, so the two
cells cannot drift in the mesh, the model or the regularization — only in this
prologue, which is why the instance seed, the mask radius, the over-sample bins
and the mesh vertex count are literals here exactly as they are there.

The dispatch is a hard ``isinstance`` assert, and ``fit._xp is np`` is asserted
too: a witness that silently ran the dense formalism, or the JAX branch, would
be asserting bit-identity of something the timing legs do not run.

Output
------
``results/breakdown/imaging/fixed_light_numba_levers_witness_<config-name>.json``
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

# ---------------------------------------------------------------------------
# HEADER ORDER IS LOAD-BEARING — the same constraint fixed_light_numba.py has
# ---------------------------------------------------------------------------
# OpenBLAS / MKL / OpenMP read their thread-count variables once, when the
# shared library loads, and numba reads NUMBA_NUM_THREADS once, when it
# initialises its threading layer. Both therefore have to be set before
# `import numpy` and before anything imports numba. `_production_config` and
# `_profile_cli` are stdlib-only at import time, which is what makes this
# possible. NUMBA_NUM_THREADS is set cell-locally and deliberately NOT added to
# `_production_config.THREAD_ENV_VARS`, for the reason that cell gives.
import argparse as _argparse  # noqa: E402
import os as _os  # noqa: E402

from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli as _parse_profile_cli  # noqa: E402

_cli = _parse_profile_cli()

_cell_parser = _argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--mesh", choices=("delaunay",), default="delaunay")
_cell_parser.add_argument("--dataset", choices=("hst", "euclid"), default="hst")
_cell_parser.add_argument("--threads", type=int, default=1)
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh
DATASET = _cell_args.dataset
N_THREADS = max(1, int(_cell_args.threads))

#: W3's gate. Bit-identity of H should give exactly 0.0; the tolerance exists so
#: a last-bit difference in the log-evidence sum is reported as a pass with its
#: value printed rather than as a crash with no number.
W3_RTOL = 1e-9

thread_env = _pin_thread_env(N_THREADS)

_numba_threads_before = _os.environ.get("NUMBA_NUM_THREADS")
_os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)
numba_thread_env = {
    "NUMBA_NUM_THREADS": _os.environ["NUMBA_NUM_THREADS"],
    "preexisting": _numba_threads_before,
}

# The cross-evaluation NNLS warm-start memo is OFF here. This cell takes no
# timing, and every figure_of_merit it compares must be a cold solve of its own
# system: a memo shared between the library fit and the monkeypatched fit would
# let the second one start from the first's passive set, which is exactly the
# channel W3 is trying to look through.
_os.environ["AUTOARRAY_NNLS_WARM_START"] = "0"

import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

import autoarray  # noqa: E402
from autoarray.inversion.inversion.imaging.mapping import (  # noqa: E402
    InversionImagingMapping,
)
from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402
from autoarray.inversion.regularization import regularization_util  # noqa: E402
from likelihood_breakdown import fixed_light_system  # noqa: E402
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    delaunay_regularization,
    machine_info_dict,
)

_t_start = time.time()
_workspace_root = _profiling_root()

print("=" * 70)
print("FIXED-LIGHT NUMBA LEVERS — LEVER 1 BIT-IDENTITY WITNESS (#267)")
print("=" * 70)
print(f"  autoarray.__file__:  {autoarray.__file__}")
print(f"  autolens.__version__: {al.__version__}")
print(f"  hostname:            {socket.gethostname()}")
print(f"  thread env:          {thread_env}")
print(f"  numba thread env:    {numba_thread_env}")


# ===================================================================
# PART A — the S3 system route `b` of fixed_light_numba.py builds
# ===================================================================

instrument = DATASET

print(f"\n--- Dataset loading & masking [{instrument}, mesh={MESH}] ---")

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

n_mesh_vertices = 1500 if _cli.source_pixels is None else int(_cli.source_pixels)

print("\n--- Adapt image (lensed source) ---")
adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

print("\n--- Image mesh construction (Hilbert) ---")
image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=adapt_image
)
print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

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

mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization:        {reg_scheme} ({reg_provenance})")

# Instance 0 of `fixed_light_numba.py`'s `--instances iid` stream: the same seed
# and the same central-20 % draw, so this witness stands on the very first
# instance every timed row of the A/B legs starts from.
_IID_SEED = 263
_rng = np.random.default_rng(_IID_SEED)
instance = model.instance_from_unit_vector(
    unit_vector=list(_rng.uniform(0.4, 0.6, size=model.prior_count))
)


def adapt_images_of(one_instance):
    """``AdaptImages`` for one instance — the dicts are keyed on its own galaxy."""
    return al.AdaptImages(
        galaxy_image_dict={one_instance.galaxies.source: adapt_image},
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
        galaxy_image_plane_mesh_grid_dict={one_instance.galaxies.source: image_plane_mesh_grid},
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": image_plane_mesh_grid},
    )


adapt_images = adapt_images_of(instance)

_settings = al.Settings(use_border_relocator=True)

print("\n--- S0: full FitImaging with linear MGE lens light (eager, dense) ---")
fit_s0 = al.FitImaging(
    dataset=dataset,
    tracer=al.Tracer(galaxies=list(instance.galaxies)),
    adapt_images=adapt_images,
    settings=_settings,
    xp=np,
)
assert isinstance(fit_s0.inversion, InversionImagingMapping), (
    f"S0 dispatched {type(fit_s0.inversion).__name__}, not InversionImagingMapping"
)
print(f"  inversion class: {type(fit_s0.inversion).__name__}")

print("\n--- S3: lens light converted to regular profiles + subtracted (sparse numba) ---")
system_s3 = fixed_light_system.fixed_light_system_from(
    fit_s0,
    dataset,
    adapt_images=adapt_images,
    settings=_settings,
    name="S3_sparse_numba",
    scaling="numpy",
    sparse_operator="rebake_cpu",
)
dataset_s3 = system_s3.dataset
source_only_tracer = system_s3.source_only_tracer
print(
    f"  S3: n={system_s3.n_params} (mapper {system_s3.n_mapper} + funcs {system_s3.n_funcs}), "
    f"edge-zeroed {int(system_s3.edge_zero_mask.sum())}"
)
print(f"  S3 operator:     {type(dataset_s3.sparse_operator).__name__}")
print(f"  subtracted light flux: {system_s3.subtracted_light_flux}")


def _s3_instance(one_instance):
    """``one_instance`` with the lens light stripped, for the S3 fit."""
    import copy

    stripped = copy.deepcopy(one_instance)
    galaxies = list(
        fixed_light_system._light_stripped_tracer(
            al.Tracer(galaxies=list(one_instance.galaxies))
        ).galaxies
    )
    stripped.galaxies.lens = galaxies[0]
    stripped.galaxies.source = galaxies[1]
    return stripped


adapt_images_s3 = adapt_images_of(_s3_instance(instance))


def _eager_s3_fit():
    """A fresh route-``b`` fit: the subtracted dataset, the source-only tracer, numpy."""
    return al.FitImaging(
        dataset=dataset_s3,
        tracer=source_only_tracer,
        adapt_images=adapt_images_s3,
        settings=_settings,
        xp=np,
    )


# ===================================================================
# PART B — the three witness checks
# ===================================================================

results: dict[str, dict] = {}


def _record(name, status, detail):
    results[name] = {"status": status, **detail}
    print(f"  [{status:>4}] {name}: {detail.get('summary', '')}")
    return results[name]


print("\n" + "=" * 70)
print("WITNESS")
print("=" * 70)

fit = _eager_s3_fit()
inversion = fit.inversion

assert isinstance(inversion, InversionImagingSparseNumba), (
    f"the S3 witness fit dispatched {type(inversion).__name__}, not "
    f"InversionImagingSparseNumba — route b's formalism is not what is being witnessed"
)
assert fit._xp is np, f"fit._xp is {fit._xp!r}, not numpy — this witness is the numpy path"

mapper = inversion.cls_list_from(cls=Mapper)[0]
interpolator = mapper.interpolator
print(f"  inversion class:  {type(inversion).__name__}")
print(f"  mapper params:    {mapper.params}")
print(f"  interpolator:     {type(interpolator).__name__}")
print(f"  regularization:   {type(pixelization.regularization).__name__}")

# The interpolator's stencil tables. `_mappings_sizes_weights_split` is a
# cached_property, so these are the very arrays the library hands
# `reg_split_from` — which is what makes W2 meaningful.
_mappings, _sizes, _weights = interpolator._mappings_sizes_weights_split
_mappings = np.asarray(_mappings)
_sizes = np.asarray(_sizes)
_weights = np.asarray(_weights)

print(
    f"  stencil tables:   mappings {_mappings.shape} {_mappings.dtype}, "
    f"sizes {_sizes.shape} {_sizes.dtype}, weights {_weights.shape} {_weights.dtype}"
)

# Snapshots for W2, taken BEFORE the library builds H.
_mappings_before = _mappings.copy()
_sizes_before = _sizes.copy()
_weights_before = _weights.copy()

n_zero_size_rows = int((_sizes == 0).sum())

regularization_weights = np.asarray(
    pixelization.regularization.regularization_weights_from(linear_obj=mapper, xp=np),
    dtype=float,
)

# --- the library's own H (the kernel path) ---------------------------------
_t0 = time.time()
_h_library_full = np.asarray(inversion.regularization_matrix, dtype=float)
_t_library = time.time() - _t0

# S3 is source-only, so the inversion's regularization matrix should already be
# the mapper's own (P, P) block. It is sliced rather than assumed: a leg that
# ever grows a linear-func block would otherwise compare a padded matrix to an
# unpadded one and report a shape error instead of a verdict.
_P = int(mapper.params)
if _h_library_full.shape == (_P, _P):
    _h_library = _h_library_full
    _sliced = False
else:
    _idx = np.asarray(inversion.mapper_indices)
    _h_library = _h_library_full[np.ix_(_idx, _idx)]
    _sliced = True

print(
    f"  library H:        shape {_h_library_full.shape}"
    f"{' -> mapper block ' + str(_h_library.shape) if _sliced else ''}, "
    f"built in {_t_library * 1e3:.1f} ms"
)

# --- W1: the reference chain reproduces it, bit for bit --------------------
_t0 = time.time()
(
    _ref_mappings,
    _ref_sizes,
    _ref_weights,
) = regularization_util._reg_split_reference(
    np.array(_mappings),
    np.array(_sizes),
    np.array(_weights),
)
_h_reference = regularization_util._pixel_splitted_regularization_matrix_reference(
    regularization_weights,
    _ref_mappings,
    _ref_sizes,
    _ref_weights,
)
_t_reference = time.time() - _t0

_equal = bool(np.array_equal(_h_reference, _h_library))
_max_abs_diff = float(np.max(np.abs(_h_reference - _h_library))) if not _equal else 0.0

_record(
    "W1_regularization_matrix_bit_identical",
    "PASS" if (_equal and n_zero_size_rows == 0) else "FAIL",
    {
        "array_equal": _equal,
        "max_abs_diff": _max_abs_diff,
        "shape": list(_h_library.shape),
        "n_zero_size_stencil_rows": n_zero_size_rows,
        "reference_wall_s": _t_reference,
        "library_wall_s": _t_library,
        "summary": (
            f"array_equal={_equal}, max|diff|={_max_abs_diff:.3e}, "
            f"zero-size rows={n_zero_size_rows}, reference {_t_reference * 1e3:.1f} ms "
            f"vs library {_t_library * 1e3:.1f} ms"
        ),
    },
)

# --- W2: the caller's stencil tables survived -----------------------------
_w2_mappings_ok = bool(np.array_equal(_mappings, _mappings_before))
_w2_sizes_ok = bool(np.array_equal(_sizes, _sizes_before))
_w2_weights_ok = bool(np.array_equal(_weights, _weights_before))
_w2_ok = _w2_mappings_ok and _w2_sizes_ok and _w2_weights_ok

_record(
    "W2_interpolator_tables_unmutated",
    "PASS" if _w2_ok else "FAIL",
    {
        "mappings_unchanged": _w2_mappings_ok,
        "sizes_unchanged": _w2_sizes_ok,
        "weights_unchanged": _w2_weights_ok,
        "summary": (f"mappings={_w2_mappings_ok}, sizes={_w2_sizes_ok}, weights={_w2_weights_ok}"),
    },
)

# --- W3: the log evidence is the same -------------------------------------
# Fresh fit objects on both sides. `fit` above already has its cached inversion
# warm, so it is dropped rather than reused: a figure_of_merit read off it would
# be read off an inversion built before the monkeypatch existed.
del fit, inversion, mapper

_fit_kernel = _eager_s3_fit()
_fom_kernel = float(_fit_kernel.figure_of_merit)
print(f"  figure_of_merit (kernel path):    {_fom_kernel!r}")

_kernel_reg_split = regularization_util.reg_split_np_from
_kernel_pixel_split = regularization_util.pixel_splitted_regularization_matrix_np_from

_patch_calls = {"reg_split": 0, "pixel_splitted": 0}


def _reference_reg_split_np_from(splitted_mappings, splitted_sizes, splitted_weights):
    """``reg_split_np_from``'s contract, served by the pure-Python reference.

    The reference mutates mappings and sizes in place, so both are copied here —
    the same copies ``reg_split_np_from`` itself makes.
    """
    _patch_calls["reg_split"] += 1
    return regularization_util._reg_split_reference(
        np.array(splitted_mappings),
        np.array(splitted_sizes),
        np.array(splitted_weights),
    )


def _reference_pixel_splitted_np_from(
    regularization_weights, splitted_mappings, splitted_sizes, splitted_weights
):
    """``pixel_splitted_regularization_matrix_np_from``'s contract, via the reference.

    The weights are passed **unsquared**: the kernel entry point squares them
    before the call, the reference squares them itself.
    """
    _patch_calls["pixel_splitted"] += 1
    return regularization_util._pixel_splitted_regularization_matrix_reference(
        regularization_weights,
        splitted_mappings,
        splitted_sizes,
        splitted_weights,
    )


regularization_util.reg_split_np_from = _reference_reg_split_np_from
regularization_util.pixel_splitted_regularization_matrix_np_from = _reference_pixel_splitted_np_from
try:
    _fit_reference = _eager_s3_fit()
    _fom_reference = float(_fit_reference.figure_of_merit)
finally:
    regularization_util.reg_split_np_from = _kernel_reg_split
    regularization_util.pixel_splitted_regularization_matrix_np_from = _kernel_pixel_split

print(f"  figure_of_merit (reference path): {_fom_reference!r}")
print(f"  monkeypatch call counts:          {_patch_calls}")

_w3_rel = abs(_fom_reference - _fom_kernel) / max(abs(_fom_kernel), 1e-300)
_w3_fired = _patch_calls["reg_split"] > 0 and _patch_calls["pixel_splitted"] > 0

_record(
    "W3_log_evidence_identical",
    "PASS" if (_w3_rel <= W3_RTOL and _w3_fired) else "FAIL",
    {
        "figure_of_merit_kernel": _fom_kernel,
        "figure_of_merit_reference": _fom_reference,
        "d_log_evidence_nats": _fom_reference - _fom_kernel,
        "relative_difference": _w3_rel,
        "rtol": W3_RTOL,
        "monkeypatch_calls": dict(_patch_calls),
        "monkeypatch_fired": _w3_fired,
        "summary": (
            f"kernel {_fom_kernel:.12f} vs reference {_fom_reference:.12f} "
            f"(rel {_w3_rel:.3e} <= {W3_RTOL:g}), patch calls {_patch_calls}"
        ),
    },
)

# ===================================================================
# PART C — the record
# ===================================================================

_all_pass = all(entry["status"] == "PASS" for entry in results.values())

_config_name = _cli.config_name or "local"

witness = {
    "cell": "fixed_light_numba_levers_witness",
    "issue": "autolens_profiling#267 (lever 1: numba split-regularization assembly)",
    "verdict": "PASS" if _all_pass else "FAIL",
    "results": results,
    "configuration": {
        "config_name": _config_name,
        "mesh": MESH,
        "dataset": DATASET,
        "source_pixels": int(n_mesh_vertices),
        "mesh_vertices_placed": int(image_plane_mesh_grid.shape[0]),
        "regularization": reg_provenance,
        "threads": N_THREADS,
        "thread_env": thread_env,
        "numba_thread_env": numba_thread_env,
        "nnls_warm_start_env": os.environ.get("AUTOARRAY_NNLS_WARM_START"),
        "use_jax": False,
        "instance_seed": _IID_SEED,
        "w3_rtol": W3_RTOL,
    },
    "system": {
        "n_params": int(system_s3.n_params),
        "n_mapper": int(system_s3.n_mapper),
        "n_funcs": int(system_s3.n_funcs),
        "edge_zeroed_pixels": int(system_s3.edge_zero_mask.sum()),
        "subtracted_light_flux": float(system_s3.subtracted_light_flux),
        "mapper_params": _P,
        "h_shape_full": list(_h_library_full.shape),
        "h_sliced_to_mapper_block": _sliced,
        "stencil_mappings_shape": list(_mappings.shape),
        "stencil_sizes_max": int(_sizes.max()),
        "n_zero_size_stencil_rows": n_zero_size_rows,
    },
    "provenance": {
        "autoarray_file": autoarray.__file__,
        "autoarray_version": autoarray.__version__,
        "autolens_version": al.__version__,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "jax_in_sys_modules": "jax" in sys.modules,
        "wall_s": time.time() - _t_start,
    },
    "machine": machine_info_dict(),
    "notes": (
        "W1 asserts np.array_equal between the retained pure-Python reference chain "
        "(_reg_split_reference -> _pixel_splitted_regularization_matrix_reference) and the "
        "library's own inversion.regularization_matrix on the HST Delaunay N=1500 source-only "
        "(S3) AdaptSplit system fixed_light_numba.py route b builds. W2 asserts the "
        "interpolator's cached stencil tables are byte-identical after the library inversion. "
        "W3 asserts the fit's figure_of_merit is unchanged when both numpy entry points are "
        "monkeypatched to the references. No timing is taken here and no pin is asserted; the "
        "A/B milliseconds are the two arms of "
        "hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_ral_hst_fp64."
    ),
}

_out_dir = _workspace_root / "results" / "breakdown" / "imaging"
_out_dir.mkdir(parents=True, exist_ok=True)
_out_path = _out_dir / f"fixed_light_numba_levers_witness_{_config_name}.json"
_out_path.write_text(json.dumps(witness, indent=2, default=str))

print("\n" + "=" * 70)
print(f"VERDICT: {witness['verdict']}")
for _name, _entry in results.items():
    print(f"  {_entry['status']:>4}  {_name}")
print(f"  Witness JSON saved to: {_out_path}")
print("=" * 70)

if not _all_pass:
    raise SystemExit(
        "lever 1 witness FAILED — see the per-check summaries above and the witness JSON. "
        "The A/B timing rows in this leg describe a matrix the reference does not reproduce."
    )
