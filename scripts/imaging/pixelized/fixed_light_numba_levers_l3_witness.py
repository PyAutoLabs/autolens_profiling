"""
Numba CPU Witness: lever 3 reads log det(F + λH) off the NNLS Cholesky factor
=============================================================================

Witness for **lever 3** of the numba CPU levers task (autolens_profiling#267).

The positive-only reconstruction is a non-negative least squares solve, and
``fnnls_cholesky`` finishes it holding a Cholesky factor ``U`` of ``M[P][:, P]``,
where ``M`` is ``curvature_reg_matrix_reduced`` and ``P`` is the solve's final
**passive** set — 1485 of the 1500 columns on the production HST Delaunay
``pixels=1500`` ``AdaptSplit`` system. The Bayesian evidence then needs
``log det M`` of the full system, and until lever 3
``AbstractInversion.log_det_curvature_reg_matrix_term`` factorized that whole
``(1500, 1500)`` matrix a **second** time from scratch: 39.596 ms of lever 2's
267.448 ms call, redoing 99 % of the work the solver had just done.

Lever 3 is three changes in one commit (PyAutoArray ``b4322c3e``):

1. ``fnnls_cholesky`` gains an optional ``factor`` out-dict, filled on return with
   ``U_buffer`` (by reference — it is local to the call and untouched afterwards,
   so there is no 18 MB copy), ``k_active``, ``passive_set`` **in the order the
   factor's rows and columns are in**, and ``matrix_shape``. Like ``stats`` it is
   purely observational and the solve is byte-identical whether or not it is
   passed; ``reconstruction_positive_only_from`` clears the dict on entry so a
   memo-seeded attempt that raised cannot leave a discarded solve's factor behind.
2. ``inversion_util.log_det_from_passive_cholesky_from`` turns that factor into
   ``log det M`` by the block-determinant identity
   ``log det M = log det M_PP + log det(M_AA - M_AP M_PP^-1 M_PA)`` — the first
   term free from ``diag(U)``, the second a Schur complement over the ~15
   **active** columns.
3. ``AbstractInversion.curvature_reg_matrix`` becomes a ``cached_property``. It
   was a plain ``property`` reached twice per evaluation (4.081 ms across the two
   in lever 2's row), and that rebuild is part of what made the second
   factorization necessary at all.

On the numpy path the term takes the fast route only under every one of its
guards — numpy backend, ``log_det_method == "cholesky"``,
``use_positive_only_solver``, all linear objects regularized, a factor present
with a non-empty passive block, a ``matrix_shape`` match, and the solve covering
the whole reduced system — and falls back to the unchanged dense route otherwise
or on ``np.linalg.LinAlgError``.

**Lever 3's tolerance is three decades tighter than lever 2's, and that is the
whole point of this cell.** Lever 2 swapped one factorization for a genuinely
different one and had to accept a round-off-level value change (7.4e-10 relative
on its term). Lever 3 re-uses **the same** factor, so the only arithmetic that
differs between its two routes is a Schur complement over ~15 columns against the
corresponding tail of a dense ``dpotrf``: the design measured ``<= 1.8e-12`` nats
across six instances, with the fiducial fit's log evidence **bit-identical**. W2
therefore gates at ``1e-10``, not at lever 2's ``1e-8``. An arm that only reaches
lever 2's tolerance is a **finding** — it means the factor is being rebuilt
somewhere rather than shared — and a gate inherited from lever 2 would have waved
it through.

A speedup row without this witness beside it is a number for a route that might
never have fired. Five things are checked, in the order a reader should want them:

``W1`` **the fast path fires at all**. ``log_det_from_passive_cholesky_from`` is
wrapped in a counting wrapper before the fit is built, and during one
``fit.figure_of_merit`` it must be called **exactly once** and return a **finite
float**. A count of zero would mean one of the seven guards rejected this system
and the fit took the dense route — in which case every other check in this cell
would compare the dense route against itself and pass vacuously, and the A/B row
beside it would be measuring nothing. Recorded beside the count, from
``fit.inversion._nnls_factor`` itself: ``k_active`` (~1485), ``len(passive_set)``
and that the two agree, and ``matrix_shape`` against the reduced matrix's own
shape — the published factor has to belong to the system being asked about, and
``(1500, 1500)`` is what that is here.

``W2`` **the log determinant itself**. ``log_det_curvature_reg_matrix_term`` read
off the fast fit, then off a **fresh fit** with
``log_det_from_passive_cholesky_from`` monkeypatched to raise
``np.linalg.LinAlgError``, which drives the library's own ``except`` clause into
the dense route it shipped before lever 3 — the pre-lever-3 code path reached
through the library's own fallback rather than through a hand-rolled copy of it.
Absolute nats and relative difference are both recorded; the gate is ``1e-10``
relative. The two fits' ``curvature_reg_matrix_reduced`` are compared with
``np.array_equal`` and recorded as ``matrix_identical_between_fits`` so the reader
knows W2 compares two log determinants of one matrix and not of two matrices.

  The conditioning is recorded rather than assumed: the extreme eigenvalues of
  ``curvature_reg_matrix_reduced`` via ``eigvalsh`` and their ratio. Unlike lever
  2 this is context rather than justification — the two routes here share their
  factor, so the conditioning bounds what *either* of them knows about the matrix,
  not the gap between them.

``W3`` **the likelihood**. The same two fits' ``figure_of_merit`` (the log
evidence), fast against dense-forced, absolute nats and relative difference, gate
``1e-9`` relative. This is the phase's Witness pin. The design saw this come out
**bit-identical**, so W3 records ``bit_identical`` explicitly rather than only
whether the gate cleared: a pass at 1e-12 and a pass at exact equality are
different facts about the lever.

``W4`` **the cache, and that the solve did not change**. Two independent claims
lever 3 makes that are not about the log determinant:

  * ``curvature_reg_matrix`` is the **same object** on two accesses (``is``, not
    ``array_equal``), and the class attribute is a ``functools.cached_property``.
    That is part 3 of the lever, and it is what turns the cell's own
    ``n_calls`` accounting from 2 into 1.
  * the ``fnnls`` solution is **byte-identical** with and without the fast path:
    ``fit.inversion.reconstruction`` from the fast fit against the same array from
    the dense-forced fit, ``np.array_equal``. The ``factor`` out-dict is
    documented as purely observational, and this is the check on that sentence —
    if publishing the factor perturbed the solve, ``solver.fnnls_cholesky``'s
    milliseconds in the A/B row would be measuring a different solve.

``W5`` **the site cost, in isolation**. Three medians of 20 at one thread, on the
same matrix and the same published factor:
``log_det_from_passive_cholesky_from`` on the factor, the dense
``2*sum(log(diag(cholesky(M))))`` the library ran before lever 3, and one
``_xp.add(F, H)`` build — the third because part 3 of the lever removes one of
those per evaluation and the A/B row will show it as movement in
``inversion.curvature_reg_matrix``. It also gives a same-matrix log-det
difference, which is W2's comparison with the fit rebuild removed. This is the
only timing this cell takes and it is a **site** measurement: the row that matters
for the lever is the whole-call A/B, not this.

What this cell is NOT
--------------------

It is **not** the A/B row. The milliseconds that decide the lever are the two arms
of
``hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_l3_delaunay_ral_hst_fp64``
(control = lever 2 ``0b17c292``, feature = lever 3 ``b4322c3e``, so the difference
isolates lever 3); this cell is the correctness leg that runs beside the feature
arm and says the two arms computed the same log determinant to a stated tolerance.

It asserts **no pin** on a log-evidence value. Nothing here compares against a
calibrated literal: every gate is a difference between two routes measured in the
same process on the same system.

It imports **no JAX** of its own and sets no ``JAX_*``/``XLA_*`` variable. Lever 3
did not touch the JAX branch — and could not have, because the JAX path never
runs ``fnnls`` at all, so no factor exists there to share. The A100 pair is
``hpc/batch_gpu/submit_breakdown_imaging_fixed_light_numba_levers_l3_delaunay_a100_hst_fp64``,
whose question is whether the shared-factor route is worth carrying to the GPU
(the A100's dense Cholesky of the same matrix is ~1.14 ms, launch-bound). Note
that ``jax`` still lands in ``sys.modules`` via the first ``FitImaging``
regardless.

Construction
------------

Part A below is **lever 2's witness verbatim**, which is in turn lever 1's: the
dataset, mask, over-sampling, Hilbert mesh, MGE lens light, mass/shear priors,
instance stream and ``AdaptSplit(0.1, 10.0, signal_scale=0.1)`` regularization are
the same block ``fixed_light_numba.py`` runs, and the S3 system is built through
the same ``likelihood_breakdown.fixed_light_system.fixed_light_system_from`` with
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
witnessing something the timing legs do not run. All three parts of lever 3 are
asserted at import time — the helper's presence, ``fnnls_cholesky``'s ``factor``
parameter and ``curvature_reg_matrix`` being a ``cached_property`` — so running
this cell against a pre-lever-3 checkout fails with the sentence that names the
missing part rather than with an ``AttributeError`` three hundred lines later.

Output
------
``results/breakdown/imaging/fixed_light_numba_levers_l3_witness_<config-name>.json``
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

#: W2's gate, on `log_det_curvature_reg_matrix_term` itself — and it is
#: deliberately THREE DECADES TIGHTER than lever 2's 1e-8.
#:
#: Lever 2 swapped one factorization for a genuinely different one (SuperLU
#: against a dense Cholesky of a cond ~9e12 matrix) and had to accept a
#: round-off-level value change: 7.4e-10 relative. Lever 3 re-uses **the same**
#: factor the NNLS solve already computed, so the only arithmetic that differs
#: between the two routes is the Schur complement over the ~15 active columns
#: against the corresponding tail of a dense `dpotrf`. The design measured that at
#: **<= 1.8e-12 nats** across six instances of this system, with the fiducial
#: fit's log evidence bit-identical, so 1e-10 is two decades above the measured
#: disagreement and still three decades below lever 2's tolerance.
#:
#: **A lever-3 arm that only reaches lever 2's tolerance is a finding, not a
#: pass.** It would mean the factor is being rebuilt somewhere rather than
#: shared — which is precisely the failure mode this gate exists to catch, and
#: which a 1e-8 gate inherited from lever 2 would have waved through.
W2_RTOL = 1e-10

#: W3's gate, on the fit's `figure_of_merit` — the phase's Witness pin. The log det
#: is one of six evidence terms, so a relative move in it is diluted by the sum;
#: the design saw the evidence come out BIT-IDENTICAL, and W3 records whether it
#: did here too rather than only whether it cleared the gate.
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

import functools  # noqa: E402
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
from autoarray.inversion.inversion import inversion_util  # noqa: E402
from autoarray.inversion.inversion.abstract import AbstractInversion  # noqa: E402
from autoarray.inversion.inversion.imaging.mapping import (  # noqa: E402
    InversionImagingMapping,
)
from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402
from autoarray.util import fnnls  # noqa: E402
from autonerves import cached_property as autonerves_cached_property  # noqa: E402
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

# Fail here, with these sentences, rather than with an AttributeError three
# hundred lines down: this cell witnesses lever 3 and has nothing to say about a
# checkout that does not carry it. One assert per part of the lever, so the
# message names which part is missing.
assert hasattr(inversion_util, "log_det_from_passive_cholesky_from"), (
    "inversion_util has no log_det_from_passive_cholesky_from — this is a pre-lever-3 "
    f"PyAutoArray ({autoarray.__file__}). There is no shared-factor route here to witness."
)
assert "factor" in inspect.signature(fnnls.fnnls_cholesky).parameters, (
    "fnnls_cholesky has no `factor` parameter — this PyAutoArray carries the log-det helper "
    f"but not the out-dict that publishes the factor to it ({autoarray.__file__}). Nothing "
    "would ever fill it, so the fast path could not fire."
)
# Both descriptor types, because PyAutoArray decorates with
# `autonerves.cached_property` — the `CachedProperty` class, which caches into
# `obj.__dict__` exactly as the stdlib one does — and NOT with
# `functools.cached_property`. A check against the stdlib class alone reports
# `False` on a library that does cache it, and this assert would then reject a
# perfectly good lever-3 checkout.
CURVATURE_REG_MATRIX_IS_CACHED = isinstance(
    inspect.getattr_static(AbstractInversion, "curvature_reg_matrix"),
    (functools.cached_property, autonerves_cached_property),
)
CURVATURE_REG_MATRIX_DESCRIPTOR = type(
    inspect.getattr_static(AbstractInversion, "curvature_reg_matrix")
).__name__
assert CURVATURE_REG_MATRIX_IS_CACHED, (
    "AbstractInversion.curvature_reg_matrix is a "
    f"{CURVATURE_REG_MATRIX_DESCRIPTOR}, not a cached property — this PyAutoArray is "
    f"missing the third part of lever 3 ({autoarray.__file__}). W4 pins it."
)

_scipy_version = scipy.__version__

print("=" * 70)
print("FIXED-LIGHT NUMBA LEVERS — LEVER 3 SHARED CHOLESKY FACTOR WITNESS (#267)")
print("=" * 70)
print(f"  autoarray.__file__:  {autoarray.__file__}")
print(f"  autolens.__version__: {al.__version__}")
print(f"  hostname:            {socket.gethostname()}")
print(f"  thread env:          {thread_env}")
print(f"  numba thread env:    {numba_thread_env}")
print(f"  scipy:               {_scipy_version}")
print("  log_det_from_passive_cholesky_from: present")
print("  fnnls_cholesky has `factor`:        True")
print(f"  curvature_reg_matrix cached:        {CURVATURE_REG_MATRIX_IS_CACHED}")


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

# --- W1: the fast path fires, exactly once, and returns a finite float -----
# The wrapper goes on BEFORE the fit is built, so nothing in the construction can
# slip a call past the counter. It wraps the module attribute
# `inversion_util.log_det_from_passive_cholesky_from`, which is where
# `log_det_curvature_reg_matrix_term` looks the function up (`abstract.py` does
# `from ... import inversion_util`), so the library's own call site is what routes
# into the wrapper.
_fast_calls: list[dict] = []
_real_fast = inversion_util.log_det_from_passive_cholesky_from


def _counting_fast_log_det(matrix, U_buffer, k_active, passive_set):
    """``log_det_from_passive_cholesky_from`` with its calls and returns recorded."""
    _t0 = time.perf_counter()
    _value = _real_fast(
        matrix=matrix,
        U_buffer=U_buffer,
        k_active=k_active,
        passive_set=passive_set,
    )
    _fast_calls.append(
        {
            "matrix_shape": list(np.asarray(matrix).shape),
            "u_buffer_shape": list(np.asarray(U_buffer).shape),
            "k_active": int(k_active),
            "n_passive": int(np.asarray(passive_set).size),
            "value": float(_value),
            "finite": bool(np.isfinite(_value)),
            "wall_s": time.perf_counter() - _t0,
        }
    )
    return _value


inversion_util.log_det_from_passive_cholesky_from = _counting_fast_log_det
try:
    fit_fast = _eager_s3_fit()
    inversion = fit_fast.inversion

    assert isinstance(inversion, InversionImagingSparseNumba), (
        f"the S3 witness fit dispatched {type(inversion).__name__}, not "
        f"InversionImagingSparseNumba — route b's formalism is not what is being witnessed"
    )
    assert fit_fast._xp is np, (
        f"fit._xp is {fit_fast._xp!r}, not numpy — this witness is the numpy path"
    )

    mapper = inversion.cls_list_from(cls=Mapper)[0]
    print(f"  inversion class:  {type(inversion).__name__}")
    print(f"  mapper params:    {mapper.params}")
    print(f"  interpolator:     {type(mapper.interpolator).__name__}")
    print(f"  regularization:   {type(pixelization.regularization).__name__}")

    # The seven guards, read off the fit rather than assumed, so a zero count in
    # W1 can be attributed to a named guard instead of to a mystery.
    _guards = {
        "xp_is_numpy": fit_fast._xp is np,
        "log_det_method": str(inversion.settings.log_det_method),
        "log_det_method_is_cholesky": inversion.settings.log_det_method == "cholesky",
        "use_positive_only_solver": bool(inversion.settings.use_positive_only_solver),
        "all_linear_obj_have_regularization": bool(inversion.all_linear_obj_have_regularization),
    }
    print(f"  fast-path guards: {_guards}")

    _t0 = time.time()
    _fom_fast = float(fit_fast.figure_of_merit)
    _t_fom_fast = time.time() - _t0

    # Snapshot the counter HERE: W1's claim is about one `figure_of_merit`, and the
    # term is read again below, which would add a call.
    _calls_during_fom = list(_fast_calls)

    _term_fast = float(inversion.log_det_curvature_reg_matrix_term)
finally:
    inversion_util.log_det_from_passive_cholesky_from = _real_fast

# The factor the solve published, read off the instance. `U_buffer` is held by
# reference and nothing re-solves this inversion (the cross-evaluation memo is off
# and `reconstruction` is a `cached_property`), so it is still the factor the call
# above read — which is what makes W5 able to re-time the same work.
_factor = inversion._nnls_factor
_factor_ids = inversion._nnls_factor_ids

_m = np.asarray(inversion.curvature_reg_matrix_reduced, dtype=float)

_factor_k_active = int(_factor["k_active"])
_factor_n_passive = int(np.asarray(_factor["passive_set"]).size)
_factor_matrix_shape = tuple(int(v) for v in _factor["matrix_shape"])

print(f"  figure_of_merit (fast):                {_fom_fast!r}  [{_t_fom_fast * 1e3:.1f} ms]")
print(f"  log_det_curvature_reg_matrix_term:     {_term_fast!r}")
print(f"  log_det_from_passive_cholesky calls:   {len(_calls_during_fom)}")
for _i, _call in enumerate(_calls_during_fom):
    print(
        f"    call {_i}: matrix {_call['matrix_shape']}, k_active={_call['k_active']}, "
        f"n_passive={_call['n_passive']}, value={_call['value']!r}, "
        f"{_call['wall_s'] * 1e3:.2f} ms"
    )
print(
    f"  factor: k_active={_factor_k_active}, len(passive_set)={_factor_n_passive}, "
    f"matrix_shape={_factor_matrix_shape}, reduced matrix shape={_m.shape}"
)
print(
    f"  _nnls_factor_ids: {'None' if _factor_ids is None else f'array({np.asarray(_factor_ids).shape})'}"
)

_w1_once = len(_calls_during_fom) == 1
_w1_finite = _w1_once and _calls_during_fom[0]["finite"]
_w1_k_agrees = _factor_k_active == _factor_n_passive
_w1_shape_agrees = _factor_matrix_shape == tuple(_m.shape)
_w1_shape_is_1500 = _factor_matrix_shape == (1500, 1500)

_record(
    "W1_fast_path_fires",
    "PASS" if (_w1_once and _w1_finite and _w1_k_agrees and _w1_shape_agrees) else "FAIL",
    {
        "n_calls_during_figure_of_merit": len(_calls_during_fom),
        "called_exactly_once": _w1_once,
        "returned_finite_float": _w1_finite,
        "calls": _calls_during_fom,
        "factor_k_active": _factor_k_active,
        "factor_n_passive": _factor_n_passive,
        "k_active_agrees_with_passive_set": _w1_k_agrees,
        "factor_matrix_shape": list(_factor_matrix_shape),
        "reduced_matrix_shape": list(_m.shape),
        "matrix_shape_agrees": _w1_shape_agrees,
        "matrix_shape_is_1500": _w1_shape_is_1500,
        "n_active_columns": int(_m.shape[0]) - _factor_n_passive,
        "nnls_factor_ids_is_none": _factor_ids is None,
        "nnls_factor_ids_is_identity": (
            None
            if _factor_ids is None
            else bool(np.array_equal(np.asarray(_factor_ids), np.arange(_m.shape[0])))
        ),
        "fast_path_guards": _guards,
        "figure_of_merit_fast": _fom_fast,
        "log_det_curvature_reg_matrix_term_fast": _term_fast,
        "summary": (
            f"{len(_calls_during_fom)} call(s) during figure_of_merit, "
            f"k_active {_factor_k_active} == len(passive_set) {_factor_n_passive} "
            f"({_w1_k_agrees}), matrix_shape {_factor_matrix_shape} == reduced "
            f"{tuple(_m.shape)} ({_w1_shape_agrees}), "
            f"{int(_m.shape[0]) - _factor_n_passive} active column(s)"
            + (
                f", value {_calls_during_fom[0]['value']!r} in "
                f"{_calls_during_fom[0]['wall_s'] * 1e3:.2f} ms"
                if _w1_once
                else ""
            )
        ),
    },
)

# --- the matrix and its conditioning --------------------------------------
# Context rather than justification: unlike lever 2 the two routes here SHARE
# their factor, so the conditioning bounds what either of them knows about the
# matrix, not the gap between them.
_pixels = int(_m.shape[0])
_t0 = time.time()
_eigenvalues = np.linalg.eigvalsh(_m)
_t_eig = time.time() - _t0
_eig_min = float(_eigenvalues[0])
_eig_max = float(_eigenvalues[-1])
_cond = float(abs(_eig_max) / abs(_eig_min)) if _eig_min != 0.0 else float("inf")
print(f"\n  F+λH shape:       {_m.shape}")
print(
    f"  F+λH eigenvalues: min {_eig_min:.6e}, max {_eig_max:.6e}, "
    f"ratio (cond) {_cond:.3e}  [eigvalsh {_t_eig:.2f} s]"
)

# --- W2 / W3: fast vs a dense Cholesky forced through the library ---------
# Raising `np.linalg.LinAlgError` from the helper drives the library's own
# `except np.linalg.LinAlgError: pass` clause, so the pre-lever-3 dense
# `_log_det_symmetric_from(self.curvature_reg_matrix_reduced)` runs — reached
# through the library's fallback rather than through a copy of it written here.
_forced_dense_calls = {"n": 0}


def _linalg_error_fast_log_det(matrix, U_buffer, k_active, passive_set):
    """``log_det_from_passive_cholesky_from`` forced to fail, so the dense route runs."""
    _forced_dense_calls["n"] += 1
    raise np.linalg.LinAlgError("lever 3 witness W2: forcing the dense fallback")


inversion_util.log_det_from_passive_cholesky_from = _linalg_error_fast_log_det
try:
    fit_dense = _eager_s3_fit()
    _inversion_dense = fit_dense.inversion
    _m_dense = np.asarray(_inversion_dense.curvature_reg_matrix_reduced, dtype=float)
    _term_dense = float(_inversion_dense.log_det_curvature_reg_matrix_term)
    _fom_dense = float(fit_dense.figure_of_merit)
    _reconstruction_dense = np.asarray(_inversion_dense.reconstruction)
finally:
    inversion_util.log_det_from_passive_cholesky_from = _real_fast

print(f"\n  log_det term (dense-forced):    {_term_dense!r}")
print(f"  figure_of_merit (dense-forced): {_fom_dense!r}")
print(f"  LinAlgError patch calls:        {_forced_dense_calls['n']}")

_matrix_identical = bool(np.array_equal(_m, _m_dense))
print(f"  F+λH identical between fits:    {_matrix_identical}")

_w2_abs = _term_fast - _term_dense
_w2_rel = abs(_w2_abs) / max(abs(_term_dense), 1e-300)
_w2_patch_fired = _forced_dense_calls["n"] > 0
_w2_bit_identical = _term_fast == _term_dense

_record(
    "W2_log_det_fast_vs_dense",
    "PASS" if (_w2_rel <= W2_RTOL and _w2_patch_fired and _matrix_identical) else "FAIL",
    {
        "log_det_fast": _term_fast,
        "log_det_dense": _term_dense,
        "d_log_det_nats": _w2_abs,
        "relative_difference": _w2_rel,
        "rtol": W2_RTOL,
        "bit_identical": _w2_bit_identical,
        "linalg_error_patch_calls": _forced_dense_calls["n"],
        "linalg_error_patch_fired": _w2_patch_fired,
        "matrix_identical_between_fits": _matrix_identical,
        "matrix_eigenvalue_min": _eig_min,
        "matrix_eigenvalue_max": _eig_max,
        "matrix_condition_number": _cond,
        "matrix_eigvalsh_wall_s": _t_eig,
        "lever2_rtol_for_reference": 1e-8,
        "summary": (
            f"fast {_term_fast:.9f} vs dense {_term_dense:.9f}, "
            f"Δ {_w2_abs:+.3e} nats (rel {_w2_rel:.3e} <= {W2_RTOL:g}) on a matrix of "
            f"cond {_cond:.2e}; bit-identical={_w2_bit_identical}; "
            f"same F+λH both fits={_matrix_identical}"
        ),
    },
)

_w3_abs = _fom_fast - _fom_dense
_w3_rel = abs(_w3_abs) / max(abs(_fom_dense), 1e-300)
_w3_bit_identical = _fom_fast == _fom_dense

_record(
    "W3_figure_of_merit_fast_vs_dense",
    "PASS" if (_w3_rel <= W3_RTOL and _w2_patch_fired) else "FAIL",
    {
        "figure_of_merit_fast": _fom_fast,
        "figure_of_merit_dense": _fom_dense,
        "d_log_evidence_nats": _w3_abs,
        "relative_difference": _w3_rel,
        "rtol": W3_RTOL,
        "bit_identical": _w3_bit_identical,
        "linalg_error_patch_fired": _w2_patch_fired,
        "summary": (
            f"fast {_fom_fast:.9f} vs dense {_fom_dense:.9f}, "
            f"Δ {_w3_abs:+.3e} nats (rel {_w3_rel:.3e} <= {W3_RTOL:g}); "
            f"bit-identical={_w3_bit_identical}"
        ),
    },
)

# --- W4: the cache, and that the solve did not change ---------------------
# Two claims lever 3 makes that are not about the log determinant. The `is`
# comparison is the cache itself — `array_equal` would pass on a rebuilt array and
# say nothing. The reconstruction comparison is the check on the library's own
# sentence that the `factor` out-dict is purely observational.
_first_access = inversion.curvature_reg_matrix
_second_access = inversion.curvature_reg_matrix
_w4_same_object = _first_access is _second_access

_reconstruction_fast = np.asarray(inversion.reconstruction)
_w4_reconstruction_identical = bool(np.array_equal(_reconstruction_fast, _reconstruction_dense))
_recon_max_abs_diff = float(
    np.max(np.abs(_reconstruction_fast - _reconstruction_dense))
    if _reconstruction_fast.shape == _reconstruction_dense.shape
    else np.inf
)

print(f"\n  curvature_reg_matrix same object on two accesses: {_w4_same_object}")
print(f"  curvature_reg_matrix is cached_property:          {CURVATURE_REG_MATRIX_IS_CACHED}")
print(f"  reconstruction byte-identical across the two fits: {_w4_reconstruction_identical}")
print(f"  reconstruction max |Δ|:                           {_recon_max_abs_diff:.3e}")

_record(
    "W4_cache_and_byte_identical_solve",
    "PASS"
    if (_w4_same_object and CURVATURE_REG_MATRIX_IS_CACHED and _w4_reconstruction_identical)
    else "FAIL",
    {
        "curvature_reg_matrix_same_object": _w4_same_object,
        "curvature_reg_matrix_is_cached_property": CURVATURE_REG_MATRIX_IS_CACHED,
        "curvature_reg_matrix_descriptor": CURVATURE_REG_MATRIX_DESCRIPTOR,
        "reconstruction_byte_identical": _w4_reconstruction_identical,
        "reconstruction_max_abs_difference": _recon_max_abs_diff,
        "reconstruction_shape": list(_reconstruction_fast.shape),
        "reconstruction_n_nonzero_fast": int(np.count_nonzero(_reconstruction_fast)),
        "reconstruction_n_nonzero_dense": int(np.count_nonzero(_reconstruction_dense)),
        "summary": (
            f"curvature_reg_matrix same object={_w4_same_object} "
            f"(cached_property={CURVATURE_REG_MATRIX_IS_CACHED}); reconstruction "
            f"byte-identical={_w4_reconstruction_identical} (max |Δ| "
            f"{_recon_max_abs_diff:.3e}, {int(np.count_nonzero(_reconstruction_fast))} "
            f"non-zero of {_reconstruction_fast.shape[0]})"
        ),
    },
)

# --- W5: the site cost, all three routes, on the same matrix --------------
# The dense expression is `_log_det_symmetric_from`'s Cholesky branch written out
# (`2 * sum(log(diag(cholesky(M))))`), which is what the library ran before lever
# 3 and what it still runs when the factor is unusable. The `_xp.add(F, H)` build
# is part 3 of the lever: `curvature_reg_matrix` was a plain `property` that paid
# for one of these on every access. Medians, not means: a single slow iteration on
# a shared node should not become the row.


def _median_ms(fn, n):
    """Median wall time of *n* calls to *fn*, in ms, with the last value returned."""
    _samples = []
    _value = None
    for _ in range(n):
        _t0 = time.perf_counter()
        _value = fn()
        _samples.append((time.perf_counter() - _t0) * 1e3)
    return float(np.median(_samples)), _samples, _value


_f = np.asarray(inversion.curvature_matrix, dtype=float)
_h = np.asarray(inversion.regularization_matrix, dtype=float)


def _fast_route():
    return _real_fast(
        matrix=_m,
        U_buffer=_factor["U_buffer"],
        k_active=_factor["k_active"],
        passive_set=_factor["passive_set"],
    )


def _dense_route():
    return 2.0 * float(np.sum(np.log(np.diag(np.linalg.cholesky(_m)))))


def _add_route():
    return np.add(_f, _h)


print(
    f"\n  W5: timing on the same F+λH and the same published factor, median of "
    f"{W5_N_REPEATS} at {N_THREADS} thread(s)"
)
_fast_ms, _fast_samples, _fast_value = _median_ms(_fast_route, W5_N_REPEATS)
_dense_ms, _dense_samples, _dense_value = _median_ms(_dense_route, W5_N_REPEATS)
_add_ms, _add_samples, _add_value = _median_ms(_add_route, W5_N_REPEATS)

_same_matrix_abs = float(_fast_value) - _dense_value
_same_matrix_rel = abs(_same_matrix_abs) / max(abs(_dense_value), 1e-300)
_w5_speedup = _dense_ms / _fast_ms if _fast_ms > 0 else float("inf")

print(f"    fast (shared factor + Schur): {_fast_ms:.3f} ms  -> {_fast_value!r}")
print(f"    dense (Cholesky of F+λH):     {_dense_ms:.3f} ms  -> {_dense_value!r}")
print(f"    one _xp.add(F, H) build:      {_add_ms:.3f} ms  -> shape {_add_value.shape}")
print(f"    speedup:                      {_w5_speedup:.2f}x")
print(
    f"    same-matrix Δ:                {_same_matrix_abs:+.3e} nats (rel {_same_matrix_rel:.3e})"
)

# W5 is a measurement, not a threshold: there is no calibrated ratio to assert on
# an arbitrary host, so it PASSes as long as the fast route ran and agreed with the
# dense one to W2's tolerance on the one matrix. The ratios are RECORDED.
_w5_ok = np.isfinite(_fast_value) and (_same_matrix_rel <= W2_RTOL)

_record(
    "W5_site_timing_fast_vs_dense",
    "PASS" if _w5_ok else "FAIL",
    {
        "n_repeats": int(W5_N_REPEATS),
        "threads": N_THREADS,
        "fast_ms_median": _fast_ms,
        "dense_ms_median": _dense_ms,
        "add_f_plus_h_ms_median": _add_ms,
        "fast_ms_samples": _fast_samples,
        "dense_ms_samples": _dense_samples,
        "add_f_plus_h_ms_samples": _add_samples,
        "speedup_dense_over_fast": _w5_speedup,
        "log_det_fast_same_matrix": float(_fast_value),
        "log_det_dense_same_matrix": _dense_value,
        "d_log_det_same_matrix_nats": _same_matrix_abs,
        "relative_difference_same_matrix": _same_matrix_rel,
        "bit_identical_same_matrix": float(_fast_value) == _dense_value,
        "rtol": W2_RTOL,
        "note": (
            "RECORDED, not pinned: no calibrated ratio exists for an arbitrary host. The gate is "
            "that the fast route ran and agreed with the dense Cholesky on the one matrix to W2's "
            "tolerance. The add row is part 3 of the lever: curvature_reg_matrix was a plain "
            "property paying for one of these on every access, and the A/B row shows it as "
            "movement in inversion.curvature_reg_matrix with n_calls 2 -> 1. Compare these ratios "
            "with the A/B legs' whole-call rows, not with a literal."
        ),
        "summary": (
            f"fast {_fast_ms:.3f} ms vs dense {_dense_ms:.3f} ms ({_w5_speedup:.2f}x), "
            f"one _xp.add(F, H) {_add_ms:.3f} ms, same-matrix Δ {_same_matrix_abs:+.3e} nats "
            f"(rel {_same_matrix_rel:.3e})"
        ),
    },
)

# ===================================================================
# PART C — the record
# ===================================================================

_all_pass = all(entry["status"] == "PASS" for entry in results.values())

_config_name = _cli.config_name or "local"

witness = {
    "cell": "fixed_light_numba_levers_l3_witness",
    "issue": "autolens_profiling#267 (lever 3: log det(F + λH) off the NNLS Cholesky factor)",
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
        "w2_rtol_lever2_for_reference": 1e-8,
        "w3_rtol": W3_RTOL,
        "w5_n_repeats": int(W5_N_REPEATS),
        "log_det_method": str(_settings.log_det_method),
        "curvature_reg_matrix_is_cached_property": CURVATURE_REG_MATRIX_IS_CACHED,
        "curvature_reg_matrix_descriptor": CURVATURE_REG_MATRIX_DESCRIPTOR,
        "fast_path_guards": _guards,
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
        "curvature_reg_matrix_reduced_shape": list(_m.shape),
        "pixels": _pixels,
        "factor_k_active": _factor_k_active,
        "factor_n_passive": _factor_n_passive,
        "n_active_columns": _pixels - _factor_n_passive,
        "factor_matrix_shape": list(_factor_matrix_shape),
        "matrix_eigenvalue_min": _eig_min,
        "matrix_eigenvalue_max": _eig_max,
        "matrix_condition_number": _cond,
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
        "W1 wraps inversion_util.log_det_from_passive_cholesky_from in a counter and asserts it is "
        "called exactly once during one fit.figure_of_merit and returns a finite float — i.e. the "
        "shared-factor fast path fired on the HST Delaunay N=1500 source-only (S3) AdaptSplit "
        "system fixed_light_numba.py route b builds — recording the published factor's k_active, "
        "len(passive_set) and matrix_shape against the reduced matrix's own shape, plus the "
        "fast-path guards read off the fit. W2 compares log_det_curvature_reg_matrix_term fast "
        "against a fresh fit with that helper monkeypatched to raise np.linalg.LinAlgError (the "
        "library's own dense fallback), gate 1e-10 relative — THREE DECADES tighter than lever 2's "
        "1e-8, because lever 3 re-uses the same factor rather than computing a different "
        "factorization, and an arm that only reaches lever 2's tolerance is a finding. W3 is the "
        "same comparison on figure_of_merit, gate 1e-9 relative, recording whether it is "
        "bit-identical. W4 pins part 3 of the lever (curvature_reg_matrix is the same object on two "
        "accesses and a functools.cached_property) and checks the library's claim that the factor "
        "out-dict is purely observational, by comparing the two fits' reconstruction arrays with "
        "np.array_equal. W5 times the fast route, the dense Cholesky and one _xp.add(F, H) build on "
        "the same matrix, median of 20 at one thread, and RECORDS the ratios. The A/B milliseconds "
        "are the two arms of "
        "hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_l3_delaunay_ral_hst_fp64 "
        "(control = lever 2 0b17c292, feature = lever 3 b4322c3e)."
    ),
}

_out_dir = _workspace_root / "results" / "breakdown" / "imaging"
_out_dir.mkdir(parents=True, exist_ok=True)
_out_path = _out_dir / f"fixed_light_numba_levers_l3_witness_{_config_name}.json"
_out_path.write_text(json.dumps(witness, indent=2, default=str))

print("\n" + "=" * 70)
print(f"VERDICT: {witness['verdict']}")
for _name, _entry in results.items():
    print(f"  {_entry['status']:>4}  {_name}")
print(f"  Witness JSON saved to: {_out_path}")
print("=" * 70)

if not _all_pass:
    raise SystemExit(
        "lever 3 witness FAILED — see the per-check summaries above and the witness JSON. "
        "The A/B timing rows in this leg describe a log determinant this cell cannot vouch for."
    )
