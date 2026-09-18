"""
Numba CPU Witness: which curvature-matrix kernel is right on Delaunay
=====================================================================

Witness for **lever 4a** of the fixed-lens-light numba CPU campaign
(autolens_profiling#274, phase 4 of ``fixed-lens-light-numba-cpu``).

What lever 4a is
----------------

``inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from`` is a
**dispatcher**, not a kernel. It picks between two implementations with
identical inputs, outputs and contracts on one pixel-count cap::

    if pix_pixels <= two_stage_max_pix_pixels:      # CURVATURE_TWO_STAGE_MAX_PIX_PIXELS = 4096
        return curvature_matrix_via_sparse_operator_two_stage_from(...)
    return curvature_matrix_via_sparse_operator_direct_from(...)

At the 1500 source pixels this campaign measures, production therefore always
takes the **two-stage** branch. That cap was calibrated on the HST *rectangular*
fiducial — bilinear mappings, ``u0 = 4``, where two-stage wins 2.9x — and the
two-stage kernel's own docstring says it can lose on barycentric Delaunay
(``u0 ~ 1.55``), because stage 2 does a dense ``pix_pixels``-long AXPY per
mapping of every data pixel and then re-zeroes the whole accumulator regardless
of how few indices stage 1 actually touched. It has never been measured on this
cell. Lever 4a is that measurement, and a third kernel — the two-stage body with
stage 2 and the re-zero restricted to the touched indices — is measured beside
the two the library already carries.

This cell is the **correctness** leg of that measurement. It says the three
kernels compute the same matrix and the same evidence, on the production system,
in one process. It is **not** the A/B row: the milliseconds that decide the lever
are rows ``b``, ``b_direct`` and ``b_touched`` of
``hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_s4_delaunay_ral_hst_fp64``,
under the cell's own ABBA harness at ``--n-repeats 64``.

The four checks
---------------

``W1`` **each kernel fires, exactly once per evaluation**. Under each injection
the seam's counter must read exactly 1 after one ``fit.figure_of_merit``, and the
row records which dotted name actually ran. The inversion resolves the dispatcher
by module attribute (``imaging_numba/sparse.py:385``), so an injection that
silently stopped firing would leave a perfectly plausible number wearing the
wrong kernel's label — which is the only way this witness can be wrong about
what it measured.

``W2`` **the matrix**. The mapper block of ``curvature_matrix``, computed by all
three kernels on the **same seven arrays** read off the fit's own mapper and
sparse operator, so nothing but the kernel differs:

  * ``direct`` vs ``two_stage`` — maximum relative difference, gate ``1e-9``. The
    two sum the same products in a different order, so they agree to
    floating-point reassociation and nothing tighter; the value is recorded.
  * ``two_stage_touched`` vs ``two_stage`` — ``np.array_equal``, **bit-identical**,
    and recorded explicitly rather than only as "the gate cleared". The touched
    kernel's whole promotion case is that it removes additions of exact ``+0.0``
    and reorders nothing, so exact equality is the claim and a merely-tolerant
    pass would refute it.

``W3`` **the evidence**. ``fit.figure_of_merit`` under each injection, against
the un-injected library fit, in nats and relative, gate ``1e-9`` — the phase's
Witness pin. The two two-stage forms should be bit-identical here as well, and
W3 records whether they were.

``W4`` **the site cost, in isolation**. Medians of 20 single-thread calls of each
kernel on the identical inputs. **RECORDED, never gated**: no calibrated ratio
exists for an arbitrary host, and the row that decides the lever is the
whole-call A/B, not this. It is here so that a site-level ratio can be read
against the whole-call movement the A/B reports, and so that a kernel that wins
the site but not the call is visible as such.

What this cell is NOT
---------------------

It asserts **no pin** on a log-evidence value. Nothing here compares against a
calibrated literal: every gate is a difference between kernels measured in the
same process, on the same system, from the same arrays.

It imports **no JAX** of its own and sets no ``JAX_*``/``XLA_*`` variable. The
JAX path never reaches this dispatcher at all — the sparse operator route is
CPU-only — so there is nothing here for a GPU arm to witness. Note that ``jax``
still lands in ``sys.modules`` via the first ``FitImaging`` regardless.

Construction
------------

Part A below is **lever 3's witness verbatim**, which is in turn lever 2's and
lever 1's: the dataset, mask, over-sampling, Hilbert mesh, MGE lens light,
mass/shear priors, instance stream and ``AdaptSplit(0.1, 10.0, signal_scale=0.1)``
regularization are the same block ``fixed_light_numba.py`` runs, and the S3
system is built through the same
``likelihood_breakdown.fixed_light_system.fixed_light_system_from`` with
``sparse_operator="rebake_cpu"``, i.e. route ``b``'s
``InversionImagingSparseNumba`` on the subtracted dataset. The construction is
**replicated rather than imported**: ``fixed_light_numba.py`` is a top-to-bottom
script with no ``__main__`` guard, so importing it would run all of its legs.
Every shared helper it uses is imported from the same module, so the two cells
cannot drift in the mesh, the model or the regularization — only in this
prologue, which is why the instance seed, the mask radius, the over-sample bins
and the mesh vertex count are literals here exactly as they are there.

The dispatch is a hard ``isinstance`` assert, and ``fit._xp is np`` is asserted
too: a witness that silently ran the dense formalism, or the JAX branch, would be
witnessing something the timing legs do not run. The two library kernels and the
dispatcher's ``two_stage_max_pix_pixels`` parameter are asserted at import time,
so running this cell against a PyAutoArray without them fails with the sentence
that names what is missing rather than with an ``AttributeError`` three hundred
lines later.

Output
------
``results/breakdown/imaging/fixed_light_numba_s4_witness_<config-name>.json``
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

#: W2's gate, on the mapper block of `curvature_matrix` itself.
#:
#: The two LIBRARY kernels sum the same products in a different order, so they
#: agree to floating-point reassociation and nothing tighter. The library's own
#: pin on the pair is `rtol=1e-6`; the measured maximum relative difference on
#: the production HST geometries is ~4e-13. 1e-9 is this campaign's tolerance —
#: three decades inside the library's pin, three decades outside the measured
#: disagreement — and it is the same number the cell's P4 gate uses.
#:
#: The TOUCHED kernel is held to a different standard entirely, and W2 records
#: both: against `two_stage` it must be **bit-identical**, because its whole
#: promotion case is that it removes additions of exact +0.0 and reorders
#: nothing. A pass at 1e-15 and a pass at exact equality are different facts
#: about that kernel, and only the second licenses dropping it into
#: `curvature_matrix_via_sparse_operator_two_stage_from` without touching the
#: library's existing reference test.
W2_RTOL = 1e-9

#: W3's gate, on the fit's `figure_of_merit` — the phase's Witness pin, and the
#: number the issue's Witness line names ("returns the library's log evidence to
#: <= 1e-9 relative"). Recorded in nats as well as relative, and W3 records
#: whether the two two-stage forms come out BIT-IDENTICAL rather than only
#: whether they cleared the gate: they should, and a merely-tolerant pass there
#: is a finding about the touched kernel.
W3_RTOL = 1e-9

#: W4's sample count. Medians of this many calls per kernel, so one slow
#: iteration on a shared node cannot become the row.
W4_N_REPEATS = 20

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

import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

import autoarray  # noqa: E402
from autoarray.inversion.inversion.imaging.mapping import (  # noqa: E402
    InversionImagingMapping,
)
from autoarray.inversion.inversion.imaging_numba import (  # noqa: E402
    inversion_imaging_numba_util,
)
from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402
from likelihood_breakdown import (
    fixed_light_numpy_kernels,  # noqa: E402
    fixed_light_system,  # noqa: E402
)
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    delaunay_regularization,
    machine_info_dict,
)

_t_start = time.time()
_workspace_root = _profiling_root()

# Fail here, with these sentences, rather than with an AttributeError three
# hundred lines down: this cell witnesses a choice between two kernels that must
# both exist, reached through a dispatcher that must still take the argument the
# seam supplies. One assert per thing that could be missing.
for _branch in (
    "curvature_matrix_via_sparse_operator_from",
    "curvature_matrix_via_sparse_operator_two_stage_from",
    "curvature_matrix_via_sparse_operator_direct_from",
):
    assert hasattr(inversion_imaging_numba_util, _branch), (
        f"inversion_imaging_numba_util has no {_branch} — this PyAutoArray "
        f"({autoarray.__file__}) does not carry the two-kernel dispatch this cell "
        f"witnesses a choice between."
    )

# `curvature_matrix_via_sparse_operator_from` is `numba_util.jit`-decorated, so
# what the attribute holds is either the lazy placeholder (a `functools.wraps`
# wrapper, whose signature follows `__wrapped__`) or, once numba has materialized
# it, a Dispatcher carrying `py_func`. Read the signature off whichever it is
# rather than assuming one, so this assert says something true in both states.
_dispatcher = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from
_dispatcher_parameters = inspect.signature(getattr(_dispatcher, "py_func", _dispatcher)).parameters
assert "two_stage_max_pix_pixels" in _dispatcher_parameters, (
    "curvature_matrix_via_sparse_operator_from has no `two_stage_max_pix_pixels` "
    f"parameter ({autoarray.__file__}). The `two_stage` and `direct` kernels of this "
    "witness ARE that keyword; without it there is no way to reach either branch on "
    "purpose and the A/B has nothing to compare."
)

CURVATURE_TWO_STAGE_MAX_PIX_PIXELS = int(
    inversion_imaging_numba_util.CURVATURE_TWO_STAGE_MAX_PIX_PIXELS
)

_scipy_version = scipy.__version__

print("=" * 70)
print("FIXED-LIGHT NUMBA S4 — CURVATURE-MATRIX KERNEL WITNESS (#274, lever 4a)")
print("=" * 70)
print(f"  autoarray.__file__:  {autoarray.__file__}")
print(f"  autolens.__version__: {al.__version__}")
print(f"  hostname:            {socket.gethostname()}")
print(f"  thread env:          {thread_env}")
print(f"  numba thread env:    {numba_thread_env}")
print(f"  scipy:               {_scipy_version}")
print("  two-stage / direct kernels:         present")
print("  dispatcher takes two_stage_max_pix_pixels: True")
print(f"  CURVATURE_TWO_STAGE_MAX_PIX_PIXELS: {CURVATURE_TWO_STAGE_MAX_PIX_PIXELS}")


# ===================================================================
# PART A — the S3 system route `b` of fixed_light_numba.py builds
# (lever 2's witness verbatim: same dataset, mask, over-sampling, mesh, model,
#  instance stream, regularization and S3 construction — the same system)
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
# PART B — the four witness checks
# ===================================================================

results: dict[str, dict] = {}


def _record(name, status, detail):
    results[name] = {"status": status, **detail}
    print(f"  [{status:>4}] {name}: {detail.get('summary', '')}")
    return results[name]


print("\n" + "=" * 70)
print("WITNESS")
print("=" * 70)

#: The three kernels, in the order every table below prints them. `two_stage` is
#: first because it is the control: it is what the un-injected dispatcher runs at
#: this source size, and both other rows are read against it.
KERNEL_NAMES = ("two_stage", "direct", "two_stage_touched")

# --- the reference fit, un-injected ---------------------------------------
# Built BEFORE any injection, so nothing in its construction can have run
# through a patched dispatcher. Its figure_of_merit is route b's, and the seven
# arrays W2 and W4 use are read off its own mapper and sparse operator.
fit_library = _eager_s3_fit()
inversion = fit_library.inversion

assert isinstance(inversion, InversionImagingSparseNumba), (
    f"the S3 witness fit dispatched {type(inversion).__name__}, not "
    f"InversionImagingSparseNumba — route b's formalism is not what is being witnessed"
)
assert fit_library._xp is np, (
    f"fit._xp is {fit_library._xp!r}, not numpy — this witness is the numpy path"
)

mapper = inversion.cls_list_from(cls=Mapper)[0]
_sparse_operator = dataset_s3.sparse_operator

print(f"  inversion class:  {type(inversion).__name__}")
print(f"  mapper params:    {mapper.params}")
print(f"  interpolator:     {type(mapper.interpolator).__name__}")
print(f"  regularization:   {type(pixelization.regularization).__name__}")
print(f"  sparse operator:  {type(_sparse_operator).__name__}")

_t0 = time.time()
_fom_library = float(fit_library.figure_of_merit)
_t_fom_library = time.time() - _t0
print(
    f"  figure_of_merit (library, un-injected): {_fom_library!r}  [{_t_fom_library * 1e3:.1f} ms]"
)

# The seven arguments `imaging_numba/sparse.py:385` passes, read off this fit
# rather than rebuilt: every kernel below sees the SAME arrays, so a difference
# between two rows is a difference between two kernels and nothing else.
KERNEL_ARGUMENTS = {
    "psf_precision_operator": _sparse_operator.psf_precision_operator_sparse,
    "psf_precision_indexes": _sparse_operator.indexes,
    "psf_precision_lengths": _sparse_operator.lengths,
    "data_to_pix_unique": mapper.unique_mappings.data_to_pix_unique,
    "data_weights": mapper.unique_mappings.data_weights,
    "pix_lengths": mapper.unique_mappings.pix_lengths,
    "pix_pixels": mapper.params,
}

_pix_pixels = int(KERNEL_ARGUMENTS["pix_pixels"])
_data_pixels = int(np.asarray(KERNEL_ARGUMENTS["psf_precision_lengths"]).shape[0])
_stored_pairs = int(np.sum(np.asarray(KERNEL_ARGUMENTS["psf_precision_lengths"])))
_mean_mappings_u0 = float(np.mean(np.asarray(KERNEL_ARGUMENTS["pix_lengths"])))
_production_branch = "two_stage" if _pix_pixels <= CURVATURE_TWO_STAGE_MAX_PIX_PIXELS else "direct"

print(
    f"  geometry: pix_pixels {_pix_pixels}, data_pixels {_data_pixels}, "
    f"stored pairs {_stored_pairs}, mean mappings per data pixel (u0) "
    f"{_mean_mappings_u0:.3f}"
)
print(
    f"  production branch at this size: {_production_branch} "
    f"(cap {CURVATURE_TWO_STAGE_MAX_PIX_PIXELS})"
)


# --- W1: each kernel fires, exactly once per figure_of_merit --------------
_w1_calls: dict[str, dict] = {}
_fom_by_kernel: dict[str, float] = {}

for _name in KERNEL_NAMES:
    with fixed_light_numpy_kernels.curvature_kernel_injected(
        _name, label=f"s4_witness_{_name}"
    ) as _counts:
        _fit = _eager_s3_fit()
        _t0 = time.time()
        _fom_by_kernel[_name] = float(_fit.figure_of_merit)
        _t_fom = time.time() - _t0
        _w1_calls[_name] = {
            "n_calls": int(_counts["n_calls"]),
            "kernel_dotted_name": _counts["dotted_name"],
            "last_pix_pixels": _counts["last_pix_pixels"],
            "figure_of_merit_wall_s": _t_fom,
        }
    print(
        f"  {_name:<18} n_calls {_w1_calls[_name]['n_calls']}, "
        f"figure_of_merit {_fom_by_kernel[_name]!r}  [{_t_fom * 1e3:.1f} ms]"
    )

_w1_all_once = all(entry["n_calls"] == 1 for entry in _w1_calls.values())
_w1_dotted_ok = all(
    _w1_calls[name]["kernel_dotted_name"] == fixed_light_numpy_kernels.KERNEL_DOTTED_NAMES[name]
    for name in KERNEL_NAMES
)
_w1_sizes_ok = all(entry["last_pix_pixels"] == _pix_pixels for entry in _w1_calls.values())

_record(
    "W1_each_kernel_fires_once",
    "PASS" if (_w1_all_once and _w1_dotted_ok and _w1_sizes_ok) else "FAIL",
    {
        "calls": _w1_calls,
        "all_called_exactly_once": _w1_all_once,
        "dotted_names_agree": _w1_dotted_ok,
        "pix_pixels_agree": _w1_sizes_ok,
        "patched_dotted_name": (fixed_light_numpy_kernels.LIBRARY_CURVATURE_DISPATCHER_DOTTED),
        "production_branch_at_this_size": _production_branch,
        "curvature_two_stage_max_pix_pixels": CURVATURE_TWO_STAGE_MAX_PIX_PIXELS,
        "summary": (
            "n_calls "
            + ", ".join(f"{name} {_w1_calls[name]['n_calls']}" for name in KERNEL_NAMES)
            + f" per figure_of_merit at pix_pixels {_pix_pixels}"
        ),
    },
)


# --- W2: the matrix, from the identical seven arrays ----------------------
_matrices: dict[str, np.ndarray] = {}
for _name in KERNEL_NAMES:
    with fixed_light_numpy_kernels.curvature_kernel_injected(
        _name, label=f"s4_witness_w2_{_name}"
    ) as _counts:
        _matrices[_name] = np.asarray(
            inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from(
                **KERNEL_ARGUMENTS
            ),
            dtype=float,
        )
        assert _counts["n_calls"] == 1, (
            f"W2: the {_name} injection fired {_counts['n_calls']} time(s) on one direct "
            f"call of the dispatcher; the matrix below would not be that kernel's"
        )

_reference = _matrices["two_stage"]
_reference_scale = max(float(np.max(np.abs(_reference))), 1e-300)

_w2_rows: dict[str, dict] = {}
for _name in KERNEL_NAMES:
    _difference = _matrices[_name] - _reference
    _max_abs = float(np.max(np.abs(_difference)))
    _w2_rows[_name] = {
        "max_abs_diff_vs_two_stage": _max_abs,
        "max_rel_diff_vs_two_stage": _max_abs / _reference_scale,
        "bit_identical_to_two_stage": bool(np.array_equal(_matrices[_name], _reference)),
        "frobenius_norm": float(np.linalg.norm(_matrices[_name])),
        "shape": list(_matrices[_name].shape),
        "symmetric": bool(np.array_equal(_matrices[_name], _matrices[_name].T)),
    }
    print(
        f"  {_name:<18} max|Δ| {_max_abs:.3e}, rel "
        f"{_w2_rows[_name]['max_rel_diff_vs_two_stage']:.3e}, bit-identical "
        f"{_w2_rows[_name]['bit_identical_to_two_stage']}, symmetric "
        f"{_w2_rows[_name]['symmetric']}"
    )

# The un-injected dispatcher must reproduce the `two_stage` row exactly, or the
# control is not the production path and every ratio in the A/B is against the
# wrong baseline.
_matrix_library = np.asarray(
    inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from(**KERNEL_ARGUMENTS),
    dtype=float,
)
_w2_control_is_production = bool(np.array_equal(_matrix_library, _reference))

_w2_direct_rel = _w2_rows["direct"]["max_rel_diff_vs_two_stage"]
_w2_touched_identical = _w2_rows["two_stage_touched"]["bit_identical_to_two_stage"]
_w2_all_symmetric = all(row["symmetric"] for row in _w2_rows.values())

_record(
    "W2_curvature_matrix_agrees",
    "PASS"
    if (
        _w2_direct_rel <= W2_RTOL
        and _w2_touched_identical
        and _w2_all_symmetric
        and _w2_control_is_production
    )
    else "FAIL",
    {
        "kernels": _w2_rows,
        "rtol": W2_RTOL,
        "direct_vs_two_stage_max_rel_diff": _w2_direct_rel,
        "direct_within_rtol": bool(_w2_direct_rel <= W2_RTOL),
        "touched_bit_identical_to_two_stage": _w2_touched_identical,
        "touched_max_abs_diff": _w2_rows["two_stage_touched"]["max_abs_diff_vs_two_stage"],
        "all_symmetric": _w2_all_symmetric,
        "uninjected_dispatcher_matches_two_stage": _w2_control_is_production,
        "note": (
            "The two library kernels sum the same products in a different order and "
            "agree to floating-point reassociation, so `direct` is gated at a tolerance "
            "and its value is recorded. `two_stage_touched` removes additions of exact "
            "+0.0 and reorders nothing, so it is gated at EXACT equality — a pass at "
            "1e-15 there would refute its promotion case rather than support it."
        ),
        "summary": (
            f"direct vs two-stage rel {_w2_direct_rel:.3e} (<= {W2_RTOL:g}: "
            f"{_w2_direct_rel <= W2_RTOL}), touched bit-identical "
            f"{_w2_touched_identical}, control == un-injected dispatcher "
            f"{_w2_control_is_production}"
        ),
    },
)


# --- W3: the evidence -----------------------------------------------------
_w3_rows: dict[str, dict] = {}
for _name in KERNEL_NAMES:
    _delta = _fom_by_kernel[_name] - _fom_library
    _w3_rows[_name] = {
        "figure_of_merit": _fom_by_kernel[_name],
        "delta_nats_vs_library": _delta,
        "relative_difference": abs(_delta) / max(abs(_fom_library), 1e-300),
        "bit_identical_to_library": _fom_by_kernel[_name] == _fom_library,
        "within_rtol": bool(abs(_delta) / max(abs(_fom_library), 1e-300) <= W3_RTOL),
    }
    print(
        f"  {_name:<18} log evidence {_fom_by_kernel[_name]!r}, Δ {_delta:+.3e} nats, rel "
        f"{_w3_rows[_name]['relative_difference']:.3e}, bit-identical "
        f"{_w3_rows[_name]['bit_identical_to_library']}"
    )

_w3_all_within = all(row["within_rtol"] for row in _w3_rows.values())
_w3_two_stage_forms_identical = (
    _w3_rows["two_stage"]["bit_identical_to_library"]
    and _w3_rows["two_stage_touched"]["bit_identical_to_library"]
)

_record(
    "W3_log_evidence_agrees",
    "PASS" if _w3_all_within else "FAIL",
    {
        "kernels": _w3_rows,
        "rtol": W3_RTOL,
        "figure_of_merit_library": _fom_library,
        "all_within_rtol": _w3_all_within,
        "two_stage_forms_bit_identical_to_library": _w3_two_stage_forms_identical,
        "note": (
            "The phase's Witness pin: every kernel returns the library's log evidence to "
            "<= 1e-9 relative. The two two-stage forms are expected BIT-IDENTICAL because "
            "they produce a bit-identical matrix; recorded either way, because a merely "
            "tolerant pass there is a finding about the touched kernel and not a pass."
        ),
        "summary": (
            "max rel "
            f"{max(row['relative_difference'] for row in _w3_rows.values()):.3e} "
            f"(<= {W3_RTOL:g}: {_w3_all_within}); two-stage forms bit-identical "
            f"{_w3_two_stage_forms_identical}"
        ),
    },
)


# --- W4: the site cost, in isolation. RECORDED, not gated ----------------
def _time_kernel(name, n_repeats):
    """Medians of ``n_repeats`` calls of one kernel on ``KERNEL_ARGUMENTS``."""
    builder = fixed_light_numpy_kernels.KERNELS[name]
    kernel = builder(inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from)

    # One untimed call: numba compiles the touched kernel here, and the two
    # library branches warm their own caches.
    kernel(**KERNEL_ARGUMENTS)

    samples = []
    for _repeat in range(n_repeats):
        _started = time.perf_counter()
        kernel(**KERNEL_ARGUMENTS)
        samples.append((time.perf_counter() - _started) * 1e3)
    return samples


_w4_rows: dict[str, dict] = {}
for _name in KERNEL_NAMES:
    _samples = _time_kernel(_name, W4_N_REPEATS)
    _w4_rows[_name] = {
        "ms_median": float(np.median(_samples)),
        "ms_min": float(np.min(_samples)),
        "ms_max": float(np.max(_samples)),
        "ms_samples": _samples,
    }
    print(
        f"  {_name:<18} {_w4_rows[_name]['ms_median']:.3f} ms median "
        f"({_w4_rows[_name]['ms_min']:.3f} - {_w4_rows[_name]['ms_max']:.3f}) "
        f"over {W4_N_REPEATS}"
    )

_w4_control_ms = _w4_rows["two_stage"]["ms_median"]
_w4_speedups = {
    name: _w4_control_ms / max(_w4_rows[name]["ms_median"], 1e-300) for name in KERNEL_NAMES
}

_record(
    "W4_kernel_site_cost",
    "RECORDED",
    {
        "kernels": _w4_rows,
        "n_repeats": int(W4_N_REPEATS),
        "threads": N_THREADS,
        "speedup_vs_two_stage": _w4_speedups,
        "geometry": {
            "pix_pixels": _pix_pixels,
            "data_pixels": _data_pixels,
            "stored_pairs": _stored_pairs,
            "mean_mappings_per_data_pixel_u0": _mean_mappings_u0,
        },
        "note": (
            "RECORDED, never gated: no calibrated ratio exists for an arbitrary host, and "
            "the row that decides lever 4a is the WHOLE-CALL A/B — rows b, b_direct and "
            "b_touched of the s4 submit under the cell's ABBA harness at --n-repeats 64. "
            "A kernel that wins here and not there has not won. Compare these ratios with "
            "the A/B's `sparse_numba.curvature_matrix` site rows, not with a literal."
        ),
        "summary": (
            ", ".join(
                f"{name} {_w4_rows[name]['ms_median']:.3f} ms ({_w4_speedups[name]:.2f}x)"
                for name in KERNEL_NAMES
            )
        ),
    },
)


# ===================================================================
# PART C — the record
# ===================================================================

_all_pass = all(entry["status"] in ("PASS", "RECORDED") for entry in results.values())

_config_name = _cli.config_name or "local"

witness = {
    "cell": "fixed_light_numba_s4_witness",
    "issue": "autolens_profiling#274 (lever 4a: the curvature-matrix kernel A/B on Delaunay)",
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
        "kernels": list(KERNEL_NAMES),
        "kernel_dotted_names": dict(fixed_light_numpy_kernels.KERNEL_DOTTED_NAMES),
        "patched_dotted_name": fixed_light_numpy_kernels.LIBRARY_CURVATURE_DISPATCHER_DOTTED,
        "curvature_two_stage_max_pix_pixels": CURVATURE_TWO_STAGE_MAX_PIX_PIXELS,
        "production_branch_at_this_size": _production_branch,
        "w2_rtol": W2_RTOL,
        "w3_rtol": W3_RTOL,
        "w4_n_repeats": int(W4_N_REPEATS),
        "log_det_method": str(_settings.log_det_method),
    },
    "system": {
        "n_params": int(system_s3.n_params),
        "n_mapper": int(system_s3.n_mapper),
        "n_funcs": int(system_s3.n_funcs),
        "edge_zeroed_pixels": int(system_s3.edge_zero_mask.sum()),
        "subtracted_light_flux": float(system_s3.subtracted_light_flux),
        "inversion_class": type(inversion).__name__,
        "interpolator_class": type(mapper.interpolator).__name__,
        "regularization_class": type(pixelization.regularization).__name__,
        "sparse_operator_class": type(_sparse_operator).__name__,
        "mapper_params": int(mapper.params),
        "curvature_matrix_mapper_block_shape": list(_reference.shape),
        "pix_pixels": _pix_pixels,
        "data_pixels": _data_pixels,
        "stored_pairs": _stored_pairs,
        "mean_mappings_per_data_pixel_u0": _mean_mappings_u0,
    },
    "provenance": {
        "autoarray_file": autoarray.__file__,
        "autoarray_version": autoarray.__version__,
        "autolens_version": al.__version__,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "jax_in_sys_modules": "jax" in sys.modules,
        "scipy_version": _scipy_version,
        "wall_s": time.time() - _t_start,
    },
    "machine": machine_info_dict(),
    "notes": (
        "W1 wraps the library's curvature-matrix dispatcher in a counting injection and "
        "asserts each of the three kernels is called exactly ONCE during one "
        "fit.figure_of_merit on the HST Delaunay N=1500 source-only (S3) AdaptSplit system "
        "fixed_light_numba.py route b builds, recording the dotted name that actually ran "
        "and the pix_pixels it saw. W2 computes the mapper block of curvature_matrix with "
        "all three kernels from the SAME seven arrays read off that fit's mapper and sparse "
        "operator: `direct` vs `two_stage` is gated at 1e-9 relative with the value "
        "recorded (they sum the same products in a different order), `two_stage_touched` vs "
        "`two_stage` is gated at np.array_equal because it removes additions of exact +0.0 "
        "and reorders nothing, and the un-injected dispatcher is checked to reproduce the "
        "`two_stage` row exactly so the control really is the production path. W3 is the "
        "same comparison on figure_of_merit, gate 1e-9 relative (the phase's Witness pin), "
        "in nats and recording bit-identity. W4 times each kernel on those identical arrays, "
        "median of 20 at one thread, and RECORDS the ratios. The milliseconds that decide "
        "lever 4a are rows b, b_direct and b_touched of "
        "hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_s4_delaunay_ral_hst_fp64, "
        "not this cell: a kernel is a lever only if it beats route b by >= 5 % on the WHOLE "
        "CALL with the ABBA gate PASS."
    ),
}

_out_dir = _workspace_root / "results" / "breakdown" / "imaging"
_out_dir.mkdir(parents=True, exist_ok=True)
_out_path = _out_dir / f"fixed_light_numba_s4_witness_{_config_name}.json"
_out_path.write_text(json.dumps(witness, indent=2, default=str))

print("\n" + "=" * 70)
print(f"VERDICT: {witness['verdict']}")
for _name, _entry in results.items():
    print(f"  {_entry['status']:>4}  {_name}")
print(f"  Witness JSON saved to: {_out_path}")
print("=" * 70)

if not _all_pass:
    raise SystemExit(
        "fixed-light s4 witness FAILED — see the per-check summaries above and the witness "
        "JSON. The A/B timing rows in this leg describe a curvature matrix this cell cannot "
        "vouch for."
    )
