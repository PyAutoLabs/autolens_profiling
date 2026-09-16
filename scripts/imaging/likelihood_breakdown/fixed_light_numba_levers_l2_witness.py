"""
Numba CPU Witness: split-regularization lever 2 factorizes H sparsely
=====================================================================

Witness for **lever 2** of the numba CPU levers task (autolens_profiling#267).

Lever 2 changed how ``AbstractInversion.log_det_regularization_matrix_term``
factorizes ``H``. That term historically ran a **dense** Cholesky of
``regularization_matrix_reduced``, a ``(pixels, pixels)`` matrix which, under the
split regularization family, is a mesh's adjacency and so carries ``O(1)``
non-zeros per row — 37.2 ms of the 299.7 ms lever-1 HST Delaunay ``pixels=1500``
numba CPU call, almost all of it spent on structural zeros. On the **NumPy path
only** the matrix is now offered first to
``inversion_util.log_det_sparse_spd_from``: a SuperLU factorization in symmetric
mode (``permc_spec="MMD_AT_PLUS_A"``, ``diag_pivot_thresh=0.0``,
``SymmetricMode=True``), whose ``diag(U)`` are the pivots, so
``log det H = sum(log(diag(U)))`` with ``L`` unit-diagonal. That function returns
``None`` — leaving the unchanged dense Cholesky to run — outside two named
regimes, ``SPARSE_LOG_DET_MIN_PIXELS = 256`` and
``SPARSE_LOG_DET_MAX_NNZ_PER_ROW = 32``. The JAX branch is untouched
(``sparse=self._xp is np`` at the call site, and the routine is SciPy).

**Lever 2 is not bit-identical, and this cell is written around that.** Lever 1's
witness asserted ``np.array_equal``; here the two routes are two different fp64
factorizations of the same cond ~9e12 matrix, so they differ in their last bits by
construction. The claim lever 2 actually makes is that the difference is the
*matrix's* round-off rather than an error introduced on top of it, and that is a
statement about tolerances, recorded with the conditioning beside it so a reader
can judge the tolerance rather than take it on trust.

A speedup row without this witness beside it is a number for a route that might
never have fired. Five things are checked, in the order a reader should want them:

``W1`` **the sparse route fires at all**. ``log_det_sparse_spd_from`` is wrapped in
a counting wrapper before the fit is built, and during one ``fit.figure_of_merit``
it must be called **exactly once** and must return a ``float``, not ``None``. A
``None`` return would mean one of the two gates rejected this matrix and the fit
took the dense Cholesky — in which case every other check in this cell would
compare the dense route against itself and pass vacuously, and the A/B row beside
it would be measuring nothing. The count is also the check that the sibling term
``log_det_curvature_reg_matrix_term`` did **not** reach the sparse route:
``F + λH`` is dense and its call site passes no ``sparse=True``, so a count of two
would mean the route leaked into the wrong term.

``W2`` **the log determinant itself**. ``log_det_regularization_matrix_term`` is
read off the sparse fit, then off a **fresh fit** with
``log_det_sparse_spd_from`` monkeypatched to return ``None``, which forces the
same dense Cholesky the library shipped before lever 2 through the library's own
fallback rather than through a hand-rolled copy of it. The absolute difference in
nats and the relative difference are both recorded; the gate is ``1e-8``
relative. The two fits' ``regularization_matrix_reduced`` are compared with
``np.array_equal`` and recorded as ``h_identical_between_fits`` so the reader
knows W2 compares two factorizations of one matrix and not two matrices.

  The conditioning is recorded rather than assumed: the extreme eigenvalues of
  ``regularization_matrix_reduced`` via ``eigvalsh`` and their ratio. ``1e-8`` is
  not a tolerance anyone should accept without it — at cond ~9e12 a fp64
  factorization carries ~1e-4 of relative error in the *matrix*, and NumPy's own
  ``slogdet`` differs from the dense Cholesky by up to 2.8e-8 on matrices of this
  conditioning, so a difference at 1e-10 is well inside the round-off the dense
  route already had.

``W3`` **the likelihood**. The same two fits' ``figure_of_merit`` (the log
evidence), sparse against dense-forced, absolute nats and relative difference,
gate ``1e-9`` relative. This is the phase's Witness pin: the log det is one of six
evidence terms and a relative move in it is diluted by the sum, so the pin on the
likelihood is tighter than the pin on the term.

``W4`` **both gates admit this matrix**. ``pixels``, the total non-zeros and
``nnz/row`` of ``regularization_matrix_reduced``, checked against
``SPARSE_LOG_DET_MIN_PIXELS`` and ``SPARSE_LOG_DET_MAX_NNZ_PER_ROW`` as the
library's own function reads them (``nnz`` mean over rows, not a per-row maximum).
W1 says the route fired; W4 says *why* it fired, and how much margin the
production system has against either regime switch.

``W5`` **the site cost, in isolation**. The sparse helper against the dense
``2*sum(log(diag(cholesky)))`` on the **same matrix**, median of 20 each, at one
thread — the factorization's own ratio, with the A/B legs' process-level noise
taken out. It also gives a same-matrix log-det difference, which is W2's
comparison with the matrix rebuild removed. This is the only timing this cell
takes and it is a **site** measurement: the row that matters for the lever is the
whole-call A/B, not this.

What this cell is NOT
--------------------

It is **not** the A/B row. The milliseconds that decide the lever are the two arms
of
``hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_l2_delaunay_ral_hst_fp64``
(control = lever 1 ``bae9296e``, feature = lever 2 ``0b17c292``, so the difference
isolates lever 2); this cell is the correctness leg that runs beside the feature
arm and says the two arms computed the same log determinant to a stated tolerance.

It asserts **no pin** on a log-evidence value. Nothing here compares against a
calibrated literal: every gate is a difference between two routes measured in the
same process on the same system.

It imports **no JAX** of its own and sets no ``JAX_*``/``XLA_*`` variable. Lever 2
did not touch the JAX branch, so there is nothing for a JAX witness to say here;
the A100 pair is
``hpc/batch_gpu/submit_breakdown_imaging_fixed_light_numba_levers_l2_delaunay_a100_hst_fp64``,
whose question is whether the sparse route is worth carrying to the GPU at all.
Note that ``jax`` still lands in ``sys.modules`` via the first ``FitImaging``
regardless.

Construction
------------

Part A below is **lever 1's witness verbatim**: the dataset, mask, over-sampling,
Hilbert mesh, MGE lens light, mass/shear priors, instance stream and
``AdaptSplit(0.1, 10.0, signal_scale=0.1)`` regularization are the same block
``fixed_light_numba.py`` runs, and the S3 system is built through the same
``likelihood_breakdown.fixed_light_system.fixed_light_system_from`` with
``sparse_operator="rebake_cpu"``, i.e. route ``b``'s
``InversionImagingSparseNumba`` on the subtracted dataset. The construction is
**replicated rather than imported**: ``fixed_light_numba.py`` is a top-to-bottom
script with no ``__main__`` guard, so importing it would run all four of its legs.
Every shared helper it uses is imported from the same module, so the two cells
cannot drift in the mesh, the model or the regularization — only in this prologue,
which is why the instance seed, the mask radius, the over-sample bins and the mesh
vertex count are literals here exactly as they are there.

The dispatch is a hard ``isinstance`` assert, and ``fit._xp is np`` is asserted
too: a witness that silently ran the dense formalism, or the JAX branch, would be
witnessing something the timing legs do not run. The presence of
``log_det_sparse_spd_from`` and of both constants is asserted at import time, so
running this cell against a pre-lever-2 checkout fails with that sentence rather
than with an ``AttributeError`` three hundred lines later.

Output
------
``results/breakdown/imaging/fixed_light_numba_levers_l2_witness_<config-name>.json``
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
_cell_args, _ = _cell_parser.parse_known_args()

MESH = _cell_args.mesh
DATASET = _cell_args.dataset
N_THREADS = max(1, int(_cell_args.threads))

#: W2's gate, on `log_det_regularization_matrix_term` itself. The sparse SuperLU
#: factorization and the dense Cholesky are two fp64 factorizations of the SAME
#: cond ~9e12 matrix, so they differ in their last bits by construction — 7.4e-10
#: relative when this was measured locally. 1e-8 is two decades above that and
#: still an order of magnitude BELOW the 2.8e-8 by which numpy's own `slogdet`
#: already differed from the dense Cholesky on matrices of this conditioning, so
#: a pass here says the sparse route sits inside the round-off the dense route
#: had rather than adding error on top of it. The conditioning is recorded with
#: the verdict so the tolerance can be judged rather than taken on trust.
W2_RTOL = 1e-8

#: W3's gate, on the fit's `figure_of_merit` — the phase's Witness pin. Tighter
#: than W2 because the log det is one of six evidence terms and its relative move
#: is diluted by the sum: 7.4e-10 on the term was 9.8e-11 on the evidence locally.
W3_RTOL = 1e-9

#: W5's sample count. Medians of this many calls per route, so one slow iteration
#: on a shared node cannot become the row.
W5_N_REPEATS = 20

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
import scipy  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

import autoarray  # noqa: E402
from autoarray.inversion.inversion import inversion_util  # noqa: E402
from autoarray.inversion.inversion.imaging.mapping import (  # noqa: E402
    InversionImagingMapping,
)
from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)
from autoarray.inversion.inversion.inversion_util import (  # noqa: E402
    SPARSE_LOG_DET_MAX_NNZ_PER_ROW,
    SPARSE_LOG_DET_MIN_PIXELS,
)
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402
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

# Fail here, with this sentence, rather than with an AttributeError three hundred
# lines down: this cell witnesses lever 2 and has nothing to say about a checkout
# that does not carry it.
assert hasattr(inversion_util, "log_det_sparse_spd_from"), (
    "inversion_util has no log_det_sparse_spd_from — this is a pre-lever-2 PyAutoArray "
    f"({autoarray.__file__}). There is no sparse route here to witness."
)

_scipy_version = scipy.__version__

print("=" * 70)
print("FIXED-LIGHT NUMBA LEVERS — LEVER 2 SPARSE LOG DET WITNESS (#267)")
print("=" * 70)
print(f"  autoarray.__file__:  {autoarray.__file__}")
print(f"  autolens.__version__: {al.__version__}")
print(f"  hostname:            {socket.gethostname()}")
print(f"  thread env:          {thread_env}")
print(f"  numba thread env:    {numba_thread_env}")
print(f"  scipy:               {_scipy_version}")
print(f"  SPARSE_LOG_DET_MIN_PIXELS:      {SPARSE_LOG_DET_MIN_PIXELS}")
print(f"  SPARSE_LOG_DET_MAX_NNZ_PER_ROW: {SPARSE_LOG_DET_MAX_NNZ_PER_ROW}")


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
# PART B — the five witness checks
# ===================================================================

results: dict[str, dict] = {}


def _record(name, status, detail):
    results[name] = {"status": status, **detail}
    print(f"  [{status:>4}] {name}: {detail.get('summary', '')}")
    return results[name]


print("\n" + "=" * 70)
print("WITNESS")
print("=" * 70)

# --- W1: the sparse route fires, exactly once, and returns a float ---------
# The wrapper goes on BEFORE the fit is built, so nothing in the construction can
# slip a call past the counter. It wraps the module attribute
# `inversion_util.log_det_sparse_spd_from`, which is where `_log_det_symmetric_from`
# looks the function up (`abstract.py` does `from ... import inversion_util`), so
# the library's own call site is what routes into the wrapper.
_sparse_calls: list[dict] = []
_real_sparse = inversion_util.log_det_sparse_spd_from


def _counting_sparse_log_det(matrix):
    """``log_det_sparse_spd_from`` with its calls, shapes and returns recorded."""
    _t0 = time.perf_counter()
    _value = _real_sparse(matrix=matrix)
    _sparse_calls.append(
        {
            "shape": list(np.asarray(matrix).shape),
            "returned_none": _value is None,
            "value": None if _value is None else float(_value),
            "wall_s": time.perf_counter() - _t0,
        }
    )
    return _value


inversion_util.log_det_sparse_spd_from = _counting_sparse_log_det
try:
    fit_sparse = _eager_s3_fit()
    inversion = fit_sparse.inversion

    assert isinstance(inversion, InversionImagingSparseNumba), (
        f"the S3 witness fit dispatched {type(inversion).__name__}, not "
        f"InversionImagingSparseNumba — route b's formalism is not what is being witnessed"
    )
    assert fit_sparse._xp is np, (
        f"fit._xp is {fit_sparse._xp!r}, not numpy — this witness is the numpy path"
    )

    mapper = inversion.cls_list_from(cls=Mapper)[0]
    print(f"  inversion class:  {type(inversion).__name__}")
    print(f"  mapper params:    {mapper.params}")
    print(f"  interpolator:     {type(mapper.interpolator).__name__}")
    print(f"  regularization:   {type(pixelization.regularization).__name__}")

    _t0 = time.time()
    _fom_sparse = float(fit_sparse.figure_of_merit)
    _t_fom_sparse = time.time() - _t0

    # Snapshot the counter HERE: W1's claim is about one `figure_of_merit`, and the
    # term is read again below, which would add a call.
    _calls_during_fom = list(_sparse_calls)

    _term_sparse = float(inversion.log_det_regularization_matrix_term)
finally:
    inversion_util.log_det_sparse_spd_from = _real_sparse

print(f"  figure_of_merit (sparse):              {_fom_sparse!r}  [{_t_fom_sparse * 1e3:.1f} ms]")
print(f"  log_det_regularization_matrix_term:    {_term_sparse!r}")
print(f"  log_det_sparse_spd_from calls in fom:  {len(_calls_during_fom)}")
for _i, _call in enumerate(_calls_during_fom):
    print(
        f"    call {_i}: shape {_call['shape']}, returned_none={_call['returned_none']}, "
        f"value={_call['value']}, {_call['wall_s'] * 1e3:.2f} ms"
    )

_w1_once = len(_calls_during_fom) == 1
_w1_is_float = _w1_once and isinstance(_calls_during_fom[0]["value"], float)
_w1_not_none = _w1_once and not _calls_during_fom[0]["returned_none"]

_record(
    "W1_sparse_route_fires",
    "PASS" if (_w1_once and _w1_is_float and _w1_not_none) else "FAIL",
    {
        "n_calls_during_figure_of_merit": len(_calls_during_fom),
        "called_exactly_once": _w1_once,
        "returned_float": _w1_is_float,
        "returned_none": not _w1_not_none,
        "calls": _calls_during_fom,
        "figure_of_merit_sparse": _fom_sparse,
        "log_det_regularization_matrix_term_sparse": _term_sparse,
        "summary": (
            f"{len(_calls_during_fom)} call(s) during figure_of_merit, "
            f"returned {'float' if _w1_is_float else 'None/non-float'}"
            + (
                f" = {_calls_during_fom[0]['value']!r} in "
                f"{_calls_during_fom[0]['wall_s'] * 1e3:.2f} ms"
                if _w1_once
                else ""
            )
        ),
    },
)

# --- the matrix, its sparsity and its conditioning ------------------------
# `regularization_matrix_reduced` is a cached_property, so this is the very array
# the library handed the factorization above — W2/W4/W5 all describe it.
_h = np.asarray(inversion.regularization_matrix_reduced, dtype=float)
_pixels = int(_h.shape[0])
_non_zeros_per_row = np.count_nonzero(_h != 0.0, axis=1)
_nnz_total = int(_non_zeros_per_row.sum())
_nnz_per_row_mean = _nnz_total / _pixels
_nnz_per_row_max = int(_non_zeros_per_row.max())

print(f"\n  H shape:          {_h.shape}")
print(
    f"  H non-zeros:      {_nnz_total} total, {_nnz_per_row_mean:.3f} per row (mean), "
    f"{_nnz_per_row_max} (max row)"
)

# Conditioning via the extreme eigenvalues of the symmetric matrix rather than
# `np.linalg.cond`'s SVD: same number on an SPD matrix, and the two eigenvalues
# themselves are what make the 1e-8 gate judgeable rather than arbitrary.
_t0 = time.time()
_eigenvalues = np.linalg.eigvalsh(_h)
_t_eig = time.time() - _t0
_eig_min = float(_eigenvalues[0])
_eig_max = float(_eigenvalues[-1])
_cond = float(abs(_eig_max) / abs(_eig_min)) if _eig_min != 0.0 else float("inf")
print(
    f"  H eigenvalues:    min {_eig_min:.6e}, max {_eig_max:.6e}, "
    f"ratio (cond) {_cond:.3e}  [eigvalsh {_t_eig:.2f} s]"
)

# --- W2 / W3: sparse vs a dense Cholesky forced through the library ------
# `log_det_sparse_spd_from` patched to return None is exactly the pre-lever-2
# library: the same dense `2*sum(log(diag(cholesky)))` runs, reached through the
# library's own fallback rather than through a copy of it written here.
_forced_dense_calls = {"n": 0}


def _dense_forcing_sparse_log_det(matrix):
    """``log_det_sparse_spd_from`` forced to decline, so the dense Cholesky runs."""
    _forced_dense_calls["n"] += 1
    return None


inversion_util.log_det_sparse_spd_from = _dense_forcing_sparse_log_det
try:
    fit_dense = _eager_s3_fit()
    _inversion_dense = fit_dense.inversion
    _h_dense = np.asarray(_inversion_dense.regularization_matrix_reduced, dtype=float)
    _term_dense = float(_inversion_dense.log_det_regularization_matrix_term)
    _fom_dense = float(fit_dense.figure_of_merit)
finally:
    inversion_util.log_det_sparse_spd_from = _real_sparse

print(f"\n  log_det term (dense-forced):   {_term_dense!r}")
print(f"  figure_of_merit (dense-forced): {_fom_dense!r}")
print(f"  dense-forcing patch calls:      {_forced_dense_calls['n']}")

_h_identical = bool(np.array_equal(_h, _h_dense))
print(f"  H identical between fits:       {_h_identical}")

_w2_abs = _term_sparse - _term_dense
_w2_rel = abs(_w2_abs) / max(abs(_term_dense), 1e-300)
_w2_patch_fired = _forced_dense_calls["n"] > 0

_record(
    "W2_log_det_sparse_vs_dense",
    "PASS" if (_w2_rel <= W2_RTOL and _w2_patch_fired and _h_identical) else "FAIL",
    {
        "log_det_sparse": _term_sparse,
        "log_det_dense": _term_dense,
        "d_log_det_nats": _w2_abs,
        "relative_difference": _w2_rel,
        "rtol": W2_RTOL,
        "dense_forcing_patch_calls": _forced_dense_calls["n"],
        "dense_forcing_patch_fired": _w2_patch_fired,
        "h_identical_between_fits": _h_identical,
        "h_eigenvalue_min": _eig_min,
        "h_eigenvalue_max": _eig_max,
        "h_condition_number": _cond,
        "h_eigvalsh_wall_s": _t_eig,
        "summary": (
            f"sparse {_term_sparse:.9f} vs dense {_term_dense:.9f}, "
            f"Δ {_w2_abs:+.3e} nats (rel {_w2_rel:.3e} <= {W2_RTOL:g}) on a matrix of "
            f"cond {_cond:.2e}; same H both fits={_h_identical}"
        ),
    },
)

_w3_abs = _fom_sparse - _fom_dense
_w3_rel = abs(_w3_abs) / max(abs(_fom_dense), 1e-300)

_record(
    "W3_figure_of_merit_sparse_vs_dense",
    "PASS" if (_w3_rel <= W3_RTOL and _w2_patch_fired) else "FAIL",
    {
        "figure_of_merit_sparse": _fom_sparse,
        "figure_of_merit_dense": _fom_dense,
        "d_log_evidence_nats": _w3_abs,
        "relative_difference": _w3_rel,
        "rtol": W3_RTOL,
        "dense_forcing_patch_fired": _w2_patch_fired,
        "summary": (
            f"sparse {_fom_sparse:.9f} vs dense {_fom_dense:.9f}, "
            f"Δ {_w3_abs:+.3e} nats (rel {_w3_rel:.3e} <= {W3_RTOL:g})"
        ),
    },
)

# --- W4: both regime gates admit this matrix ------------------------------
# Read exactly as `log_det_sparse_spd_from` reads them: `pixels` against
# SPARSE_LOG_DET_MIN_PIXELS, and the TOTAL non-zeros against
# SPARSE_LOG_DET_MAX_NNZ_PER_ROW * pixels — i.e. the MEAN per row, not the max.
_gate_pixels_ok = _pixels >= SPARSE_LOG_DET_MIN_PIXELS
_gate_nnz_ok = _nnz_total <= SPARSE_LOG_DET_MAX_NNZ_PER_ROW * _pixels

_record(
    "W4_both_regime_gates_admit",
    "PASS" if (_gate_pixels_ok and _gate_nnz_ok) else "FAIL",
    {
        "pixels": _pixels,
        "min_pixels_constant": int(SPARSE_LOG_DET_MIN_PIXELS),
        "pixels_gate_admits": _gate_pixels_ok,
        "pixels_margin": _pixels - int(SPARSE_LOG_DET_MIN_PIXELS),
        "nnz_total": _nnz_total,
        "nnz_per_row_mean": _nnz_per_row_mean,
        "nnz_per_row_max": _nnz_per_row_max,
        "max_nnz_per_row_constant": int(SPARSE_LOG_DET_MAX_NNZ_PER_ROW),
        "nnz_gate_admits": _gate_nnz_ok,
        "density": _nnz_total / float(_pixels * _pixels),
        "summary": (
            f"pixels {_pixels} >= {SPARSE_LOG_DET_MIN_PIXELS} ({_gate_pixels_ok}), "
            f"nnz/row {_nnz_per_row_mean:.3f} <= {SPARSE_LOG_DET_MAX_NNZ_PER_ROW} "
            f"({_gate_nnz_ok}), density {_nnz_total / float(_pixels * _pixels):.3e}"
        ),
    },
)

# --- W5: the site cost, both routes, on the same matrix -------------------
# The dense expression is `_log_det_symmetric_from`'s Cholesky branch written out
# (`2 * sum(log(diag(cholesky(H))))`), which is what the library ran before lever
# 2 and what it still runs when the helper declines. Medians, not means: a single
# slow iteration on a shared node should not become the row.


def _median_ms(fn, n):
    """Median wall time of *n* calls to *fn*, in ms, with the last value returned."""
    _samples = []
    _value = None
    for _ in range(n):
        _t0 = time.perf_counter()
        _value = fn()
        _samples.append((time.perf_counter() - _t0) * 1e3)
    return float(np.median(_samples)), _samples, _value


def _sparse_route():
    return _real_sparse(matrix=_h)


def _dense_route():
    return 2.0 * float(np.sum(np.log(np.diag(np.linalg.cholesky(_h)))))


print(
    f"\n  W5: timing both routes on the same H, median of {W5_N_REPEATS} at {N_THREADS} thread(s)"
)
_sparse_ms, _sparse_samples, _sparse_value = _median_ms(_sparse_route, W5_N_REPEATS)
_dense_ms, _dense_samples, _dense_value = _median_ms(_dense_route, W5_N_REPEATS)

_same_matrix_abs = float(_sparse_value) - _dense_value
_same_matrix_rel = abs(_same_matrix_abs) / max(abs(_dense_value), 1e-300)
_w5_speedup = _dense_ms / _sparse_ms if _sparse_ms > 0 else float("inf")

print(f"    sparse (SuperLU):  {_sparse_ms:.3f} ms  -> {_sparse_value!r}")
print(f"    dense (Cholesky):  {_dense_ms:.3f} ms  -> {_dense_value!r}")
print(f"    speedup:           {_w5_speedup:.2f}x")
print(f"    same-matrix Δ:     {_same_matrix_abs:+.3e} nats (rel {_same_matrix_rel:.3e})")

# W5 is a measurement, not a threshold: there is no calibrated ratio to assert on
# an arbitrary host, so it PASSes as long as both routes ran and agreed to W2's
# tolerance on the one matrix. The ratio is RECORDED.
_w5_ok = (_sparse_value is not None) and (_same_matrix_rel <= W2_RTOL)

_record(
    "W5_site_timing_sparse_vs_dense",
    "PASS" if _w5_ok else "FAIL",
    {
        "n_repeats": int(W5_N_REPEATS),
        "threads": N_THREADS,
        "sparse_ms_median": _sparse_ms,
        "dense_ms_median": _dense_ms,
        "sparse_ms_samples": _sparse_samples,
        "dense_ms_samples": _dense_samples,
        "speedup_dense_over_sparse": _w5_speedup,
        "sparse_returned_none": _sparse_value is None,
        "log_det_sparse_same_matrix": None if _sparse_value is None else float(_sparse_value),
        "log_det_dense_same_matrix": _dense_value,
        "d_log_det_same_matrix_nats": _same_matrix_abs,
        "relative_difference_same_matrix": _same_matrix_rel,
        "rtol": W2_RTOL,
        "note": (
            "RECORDED, not pinned: no calibrated ratio exists for an arbitrary host. The gate is "
            "that the sparse route ran (did not return None) and agreed with the dense Cholesky on "
            "the one matrix to W2's tolerance. Compare the ratio with the A/B legs' whole-call rows, "
            "not with a literal."
        ),
        "summary": (
            f"sparse {_sparse_ms:.3f} ms vs dense {_dense_ms:.3f} ms ({_w5_speedup:.2f}x), "
            f"same-matrix Δ {_same_matrix_abs:+.3e} nats (rel {_same_matrix_rel:.3e})"
        ),
    },
)

# ===================================================================
# PART C — the record
# ===================================================================

_all_pass = all(entry["status"] == "PASS" for entry in results.values())

_config_name = _cli.config_name or "local"

witness = {
    "cell": "fixed_light_numba_levers_l2_witness",
    "issue": "autolens_profiling#267 (lever 2: sparse log det of the split-regularization H)",
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
        "w2_rtol": W2_RTOL,
        "w3_rtol": W3_RTOL,
        "w5_n_repeats": int(W5_N_REPEATS),
        "sparse_log_det_min_pixels": int(SPARSE_LOG_DET_MIN_PIXELS),
        "sparse_log_det_max_nnz_per_row": int(SPARSE_LOG_DET_MAX_NNZ_PER_ROW),
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
        "mapper_params": int(mapper.params),
        "h_shape": list(_h.shape),
        "h_pixels": _pixels,
        "h_nnz_total": _nnz_total,
        "h_nnz_per_row_mean": _nnz_per_row_mean,
        "h_nnz_per_row_max": _nnz_per_row_max,
        "h_density": _nnz_total / float(_pixels * _pixels),
        "h_eigenvalue_min": _eig_min,
        "h_eigenvalue_max": _eig_max,
        "h_condition_number": _cond,
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
        "W1 wraps inversion_util.log_det_sparse_spd_from in a counter and asserts it is called "
        "exactly once during one fit.figure_of_merit and returns a float, not None — i.e. the "
        "sparse route fired on the HST Delaunay N=1500 source-only (S3) AdaptSplit system "
        "fixed_light_numba.py route b builds, and did not leak into "
        "log_det_curvature_reg_matrix_term. W2 compares log_det_regularization_matrix_term sparse "
        "against a fresh fit with that helper monkeypatched to return None (the library's own dense "
        "Cholesky fallback), gate 1e-8 relative, with the matrix's extreme eigenvalues and their "
        "ratio recorded beside it. W3 is the same comparison on figure_of_merit, gate 1e-9 "
        "relative. W4 records pixels and nnz/row against SPARSE_LOG_DET_MIN_PIXELS and "
        "SPARSE_LOG_DET_MAX_NNZ_PER_ROW. W5 times both routes on the same matrix, median of 20 at "
        "one thread, and RECORDS the ratio. Lever 2 is NOT bit-identical and no check here asserts "
        "that it is: the two routes are two fp64 factorizations of a cond ~9e12 matrix. The A/B "
        "milliseconds are the two arms of "
        "hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_l2_delaunay_ral_hst_fp64 "
        "(control = lever 1 bae9296e, feature = lever 2 0b17c292)."
    ),
}

_out_dir = _workspace_root / "results" / "breakdown" / "imaging"
_out_dir.mkdir(parents=True, exist_ok=True)
_out_path = _out_dir / f"fixed_light_numba_levers_l2_witness_{_config_name}.json"
_out_path.write_text(json.dumps(witness, indent=2, default=str))

print("\n" + "=" * 70)
print(f"VERDICT: {witness['verdict']}")
for _name, _entry in results.items():
    print(f"  {_entry['status']:>4}  {_name}")
print(f"  Witness JSON saved to: {_out_path}")
print("=" * 70)

if not _all_pass:
    raise SystemExit(
        "lever 2 witness FAILED — see the per-check summaries above and the witness JSON. "
        "The A/B timing rows in this leg describe a log determinant this cell cannot vouch for."
    )
