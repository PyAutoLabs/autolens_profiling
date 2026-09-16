"""
Numba CPU Profiling: Fixed Lens Light — where the CPU likelihood actually goes
==============================================================================

Phase 1 of the numba CPU profiling campaign (autolens_profiling#263).

Every fixed-lens-light number the epic has produced so far is a **JAX** number:
phase 0's kernels and phases 1-5's whole-likelihood rows were measured under
``jax.jit`` on an A100 or on a JAX CPU backend. Production CPU users run none of
that. Their likelihood is ``AnalysisImaging(..., use_jax=False)`` over a
``SparseLinAlgImagingNumba`` operator, their positivity solver is the library's
``fnnls`` Cholesky-updating NNLS on the NumPy path, and their thread count is
one. This cell measures **that** call, and decomposes it.

How the decomposition is taken
------------------------------

Not by touching cached properties in a chosen order — that measures a
reconstruction of the call, and two of ``delaunay_numba.py``'s rows are already
residuals because of it. Instead ``likelihood_breakdown.call_accounting``
replaces ~40 named descriptors and two module-level solver functions with timing
wrappers, and the production ``log_likelihood_function`` is then run
**untouched**. Each site reports inclusive time, exclusive (self) time and the
number of times the library actually reached it. Exclusive times are additive by
construction, so

    ``unattributed = call - sum(exclusive)``

is an honest remainder rather than an estimate, and it is written as its own row.
It is never folded into a neighbouring site.

Three protocol facts follow from instrumenting the real call:

- Every row **warms to a steady state** rather than to a fixed call count. The
  first call is where numba's lazy JIT compilation lands; calls then repeat until
  the median of the last three agrees with the median of the previous three to
  10 %, or twelve calls are reached. The entire per-call sequence is recorded
  under ``rows[*].warmup`` — a warm-up that never settles is evidence about the
  host, not something to discard.
- The quoted ``call_ms`` is a **clean, uninstrumented** call, and the clean and
  instrumented calls are **counterbalanced**: each block runs **A B B A** (clean,
  instrumented, instrumented, clean) and its overhead ratio is
  ``mean(B) / mean(A)``, which cancels any drift linear in call index exactly.
  Every decomposition row is then rescaled by ``clean / instrumented`` so the
  rows sum to the clean call, with the factor visible in the JSON.

  This matters more than it sounds. The sequential estimator this cell used
  first — one pass of clean calls, then one pass of instrumented calls —
  returned 0.71, 0.98, 1.10, 1.11, 1.24 and 1.37 across six rows of a single
  leg, *including ratios below 1, which no real instrumentation can produce*.
  The counterbalanced estimator on the same configuration returns 1.0085
  (median 1.0071, blocks 0.956-1.059): the instrumentation costs under 1 %, and
  every one of those six numbers was the host's own scatter. The threshold was
  never the problem; the estimator was. The ratio is therefore **asserted only
  from three counterbalanced blocks or more** (``--n-repeats 6``); below that it
  is RECORDED beside the observed clean-call spread, because one block cannot
  resolve a 3 % effect against a +-15 % noise floor.
- The cross-evaluation NNLS warm-start memo is **off by default** and the memo
  dict is cleared between rows *and between every call of a block*. Dense and
  sparse-numba S3 key identically (``_nnls_warm_start_fingerprint`` is built
  from shapes), so without this a sparse row would inherit the dense row's
  passive set and report a solve production never gets. ``--nnls-warm-start on``
  reverses the first half of that — the env goes to ``"1"`` and the
  **within-block** clears are skipped, so a block's calls warm-start from each
  other as a sampler's successive evaluations do. The **between-row** clears stay
  unconditional either way: they exist to stop one formalism seeding the other,
  which is not a production channel at all.

The four routes
---------------

============  ======================================================  ======================
Route         What runs                                               Kernel replaced
============  ======================================================  ======================
``a``         S0 — the joint system the library runs today (linear     none
              MGE lens light + source mapper)
``b``         S3 — source-only, lens light converted to regular         none
              profiles and subtracted. **The reference.**
``c``         S3 with ``use_positive_only_solver=False``                none
``d_np``      S3 with the factor-reuse NNLS injected                    ``nnls_factor_reuse``
``b_direct``  S3 with the library's DIRECT curvature kernel forced      ``..._direct_from``
``b_touched`` S3 with the touched-index two-stage curvature kernel      ``..._touched_from``
============  ======================================================  ======================

``--routes`` defaults to ``a,b,c``, so ``d_np``, ``b_direct`` and ``b_touched``
run only when they are named and every pre-existing invocation of this cell
produces the row set it always did.

There is no ``d``/``d0``/``e``: the certified active set is a JAX kernel and
this cell imports no JAX.

Route ``d_np``
--------------

Route ``b``'s likelihood with one function replaced: the library's positive-only
entry point is rebound, for the duration of the row, to
``fixed_light_numpy_solvers.nnls_factor_reuse`` — the Bro & de Jong active set
started from a single Cholesky factorisation that is then *kept*, instead of the
library's LU-for-the-sign-seed plus Cholesky-of-the-passive-block pair. Every
other part of the call is the library's own code on the library's own path; the
harness calls the library, not a copy of it.

The injection is entered **outside** ``call_accounting.install``, so the
accounting wrapper closes over the injected function and the decomposition's
``solver.reconstruction_positive_only_from`` row attributes the injected solve.
``solver.nnls_factor_reuse`` is instrumented as a site of its own, so the kernel
also appears as its own exclusive row. The injection's call counters are
**asserted non-zero** after the block: a patch that never fired would report
route ``b``'s timing wearing route ``d_np``'s label.

Routes ``b_direct`` and ``b_touched``
-------------------------------------

Route ``b``'s likelihood with one function replaced, one level below ``d_np``:
the library's curvature-matrix dispatcher
``inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from`` is
rebound, for the duration of the row, to one of
``fixed_light_numpy_kernels.KERNELS``. ``b_direct`` forces the library's own
direct quadruple loop (which production only reaches above 4096 source pixels);
``b_touched`` runs the candidate two-stage kernel whose stage 2 and accumulator
re-zero are restricted to the indices stage 1 touched.

The dispatcher picks the two-stage kernel below
``CURVATURE_TWO_STAGE_MAX_PIX_PIXELS = 4096``, so **route ``b`` is the two-stage
branch** at every source size this cell measures, and ``b`` / ``b_direct`` /
``b_touched`` in one process under the ABBA harness is the A/B. Row ``b`` is
deliberately left un-patched — it is the production path byte for byte, and the
row the verdict is taken against.

Both are entered **outside** ``call_accounting.install``, exactly as ``d_np`` is,
so the decomposition's ``sparse_numba.curvature_matrix`` site attributes the
injected kernel. Their call counters are **asserted non-zero** after the block.

**Route ``c`` is never quoted as a bare millisecond.** Dropping positivity also
silently drops edge zeroing — ``Inversion.solve_ids_to_keep`` returns ``None``
the moment positivity is off (``abstract.py:540-546``) — and the unconstrained
solution sits at a *higher* evidence than the constrained optimum because it is a
different, infeasible minimiser. Every route-``c`` row carries its Δlog-evidence
against ``b``, its negative-entry count, its negative-flux fraction and
``edge_zeroing_disabled: true``.

The two formalisms
------------------

Each route is run on both CPU formalisms, giving up to six rows keyed
``<route>_<formalism>``:

- ``dense`` — no operator on the dataset, so ``inversion_from`` dispatches
  ``InversionImagingMapping`` (``factory.py:154``);
- ``sparse_numba`` — ``dataset.apply_sparse_operator_cpu()``
  (``dataset.py:652``), so it dispatches ``InversionImagingSparseNumba``
  (``factory.py:136-145``).

The dispatch is a **hard ``isinstance`` assert** per row, not an inference from
a flag, and the class name is recorded as ``inversion_class``.

Re-baking the operator on the subtracted dataset
------------------------------------------------

The sparse S3 rows need an operator baked on the *subtracted* dataset. That is
safe here and it is asserted, not assumed (gate **P1**): the operator is a
function of the noise map, PSF and mask only (``dataset.py:682-706``), which the
subtraction does not touch, and ``InversionImagingSparseNumba.psf_weighted_data``
recomputes the weighted data from ``self.data`` on every inversion
(``imaging_numba/sparse.py:94-101``) rather than reading a baked weight map. The
JAX ``InversionImagingSparse`` does read a baked map, which is why the JAX legs
of this epic could only ever run dense — a different class, not a different
setting.

Gates
-----

``P1`` the re-baked operator equals the original's, on
``psf_precision_operator_sparse`` / ``indexes`` / ``lengths`` — **asserted
unconditionally**, including under ``--pins none``.
``P2`` dense and sparse-numba S3 agree on ``data_vector`` and the mapper block of
``curvature_matrix`` to ``rtol=1e-9`` — also unconditional.
``P3`` dense and sparse-numba S3 agree on ``figure_of_merit`` to ``rtol=1e-6``,
recording the Δ in nats and the number of passive-set differences between the two
solves. Under ``--pins none`` P3 becomes ``RECORDED``.
``P4`` (route ``d_np`` only, once per row) the injected factor-reuse solver and
the library's own solver reach the same ``figure_of_merit`` on the same instance
and formalism to ``rtol=1e-9``, recording the Δ in nats and the maximum absolute
reconstruction difference. Under ``--pins none`` P4 becomes ``RECORDED``. A
``d_np`` millisecond without P4 beside it is a number for a different problem.

Plus, per row: the structural dispatch assert, ``fit._xp is np``,
``"jax" not in sys.modules``, ``n_calls == 1`` on every site declared cached, the
ABBA ``instrumentation_overhead_ms <= 12.0`` — the ABBA ratio converted into the
absolute milliseconds it stands for, because the instrument's cost is fixed and
a ratio gate tightens every time the campaign makes the call shorter (asserted at
three blocks or more, RECORDED below that) — and coverage
``unattributed / call <= 5 %``. And once per leg: S3's mapper-block log
determinants equal S0's to ``rtol=1e-6`` with the same ``n_edge_zeroed``.

Everything else **records**. There is no pin for any numba fixed-light
configuration — none was ever calibrated — so the JSON carries
``pinned_expected: null`` and says so in as many words rather than leaving a
reader to assume a silent pass.

Output
------
``results/breakdown/imaging/fixed_light_numba_<mesh>[_<dataset>][_n<N>]_..{json,png}``
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
# HEADER ORDER IS LOAD-BEARING — read before moving anything below this line
# ---------------------------------------------------------------------------
# OpenBLAS / MKL / OpenMP read their thread-count variables ONCE, when the shared
# library loads, and numba reads NUMBA_NUM_THREADS once, when it initialises its
# threading layer. Both therefore have to be set before `import numpy` (which
# pulls in the BLAS) and before anything imports numba. `_production_config` and
# `_profile_cli` are stdlib-only at import time, which is what makes this
# possible.
#
# NUMBA_NUM_THREADS is set HERE, cell-locally, and deliberately NOT added to
# `_production_config.THREAD_ENV_VARS`: that tuple is recorded verbatim into
# every existing numba cell's `configuration.thread_env`, and widening it would
# change those recorded blocks and invalidate the pins that describe them. The
# same reasoning `preload_numba.py:207-219` gives for keeping its own
# `THREAD_ENV_KEYS`.
import argparse as _argparse  # noqa: E402
import os as _os  # noqa: E402

from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli as _parse_profile_cli  # noqa: E402

_cli = _parse_profile_cli()

#: Every route this cell can run, in the order the summary prints them. There is
#: no d/d0/e: those are the JAX certified-active-set rows of
#: ``fixed_light_library.py``, and this cell imports no JAX. ``d_np`` is the
#: numpy factor-reuse kernel injected into the library's own positive-only entry
#: point — a numpy row, not a JAX one. ``b_direct`` and ``b_touched`` sit next to
#: ``b`` because they are route ``b`` with one curvature kernel swapped, and the
#: three of them read as one A/B table.
ALL_ROUTE_KEYS = ("a", "b", "b_direct", "b_touched", "c", "d_np")

#: What ``--routes`` selects when it is not given. Deliberately NOT
#: ``ALL_ROUTE_KEYS``: ``d_np``, ``b_direct`` and ``b_touched`` each patch the
#: library for the duration of their row, and a cell invocation written before
#: those routes existed must produce the row set it always did. They run only
#: when they are named.
DEFAULT_ROUTE_KEYS = ("a", "b", "c")

#: The two CPU formalisms, in the order the summary prints them.
ALL_FORMALISM_KEYS = ("dense", "sparse_numba")

_cell_parser = _argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--mesh", choices=("delaunay", "rectangular"), default="delaunay")
_cell_parser.add_argument("--dataset", choices=("hst", "euclid"), default="hst")
_cell_parser.add_argument("--routes", default=None)
_cell_parser.add_argument("--formalism", choices=("dense", "sparse_numba", "both"), default="both")
_cell_parser.add_argument("--threads", type=int, default=1)
_cell_parser.add_argument("--n-repeats", type=int, default=8)
_cell_parser.add_argument("--instances", choices=("median", "iid"), default="median")
_cell_parser.add_argument("--decompose", dest="decompose", action="store_true", default=True)
_cell_parser.add_argument("--no-decompose", dest="decompose", action="store_false")
_cell_parser.add_argument("--row-order", choices=("grouped", "interleaved"), default="grouped")
_cell_parser.add_argument("--pins", choices=("fp64", "none"), default="fp64")
_cell_parser.add_argument("--nnls-warm-start", choices=("off", "on"), default="off")
_cell_args, _ = _cell_parser.parse_known_args()

MESH = _cell_args.mesh
DATASET = _cell_args.dataset
N_THREADS = max(1, int(_cell_args.threads))
N_REPEATS = max(1, int(_cell_args.n_repeats))
INSTANCE_MODE = _cell_args.instances
DECOMPOSE = bool(_cell_args.decompose)
ROW_ORDER = _cell_args.row_order
PINS_MODE = _cell_args.pins
PINS_ASSERT = PINS_MODE == "fp64"

#: ``on`` leaves the cross-evaluation NNLS memo enabled AND skips the
#: within-block memo clears, so a block's calls warm-start from each other the
#: way a sampler's successive evaluations do. The between-row clears are not
#: affected — see :func:`_clear_memos`.
NNLS_WARM_START = _cell_args.nnls_warm_start
NNLS_WARM_START_ON = NNLS_WARM_START == "on"


def _parse_routes(raw):
    """``--routes a,b`` -> the selected keys in canonical order; unknown is an error."""
    if raw is None:
        return tuple(DEFAULT_ROUTE_KEYS)
    wanted = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not wanted:
        raise ValueError("--routes was given but names no routes")
    unknown = [token for token in wanted if token not in ALL_ROUTE_KEYS]
    if unknown:
        raise ValueError(
            f"--routes: unknown route(s) {unknown}; choose from {list(ALL_ROUTE_KEYS)}"
        )
    return tuple(key for key in ALL_ROUTE_KEYS if key in set(wanted))


def _parse_formalisms(raw):
    """``--formalism`` -> the selected formalism keys, in canonical order."""
    if raw == "both":
        return tuple(ALL_FORMALISM_KEYS)
    return (raw,)


ROUTE_SELECTION = _parse_routes(_cell_args.routes)
FORMALISM_SELECTION = _parse_formalisms(_cell_args.formalism)

# The BLAS/OpenMP family, through the shared helper so the recorded block is the
# same shape every other production cell records...
thread_env = _pin_thread_env(N_THREADS)

# ...and numba's own knob, cell-locally (see the note above).
_numba_threads_before = _os.environ.get("NUMBA_NUM_THREADS")
_os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)
numba_thread_env = {
    "NUMBA_NUM_THREADS": _os.environ["NUMBA_NUM_THREADS"],
    "preexisting": _numba_threads_before,
    "overridden": (
        _numba_threads_before
        if _numba_threads_before is not None and _numba_threads_before != str(N_THREADS)
        else None
    ),
    "note": (
        "Set cell-locally before numpy/numba import. Deliberately NOT added to "
        "_production_config.THREAD_ENV_VARS: that tuple is recorded verbatim in "
        "every existing numba cell's configuration.thread_env and widening it "
        "would change those blocks and the pins that describe them."
    ),
}

# The cross-evaluation NNLS warm-start memo is OFF for every row by default.
# Dense and sparse-numba S3 produce the SAME memo key (the fingerprint is built
# from shapes, not from the matrix — abstract.py:551+), so a memo left on would
# let the second formalism start from the first's passive set and report a solve
# production never gets. The dict is also cleared between rows; both facts are
# recorded per row.
#
# `--nnls-warm-start on` turns it on deliberately, to measure the production
# sampler's own regime (successive nearby evaluations seeding each other). It
# also skips the WITHIN-BLOCK clears, because a memo that is cleared before every
# call is a memo that is never used. It does NOT skip the between-row clears:
# those stop one formalism seeding the other, which is not a channel production
# has.
_nnls_warm_start_before = _os.environ.get("AUTOARRAY_NNLS_WARM_START")
_os.environ["AUTOARRAY_NNLS_WARM_START"] = "1" if NNLS_WARM_START_ON else "0"

import functools  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from likelihood_breakdown import (  # noqa: E402
    call_accounting,
    fixed_light_numpy_kernels,
    fixed_light_numpy_solvers,
    fixed_light_system,
)
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    delaunay_regularization,
    machine_info_dict,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)

# NO `device_info_dict()` ANYWHERE IN THIS CELL. It imports jax unconditionally
# (`_profile_cli.py:392`), which would put a JAX runtime in a process whose whole
# purpose is to measure the likelihood without one — and would break the
# `"jax" not in sys.modules` gate below. `machine_info_dict()` is jax-free and is
# what this cell records instead, beside a cell-local device block.
assert "jax" not in sys.modules, (
    "this cell must not import JAX at any level: a JAX runtime in the process "
    "changes the thread pools and the BLAS the numba rows are measured through. "
    f"Imported by: {sorted(m for m in sys.modules if m.startswith('jax'))}"
)

from autoarray.inversion.inversion import (
    inversion_util,  # noqa: E402
    nnls_memo,  # noqa: E402
)
from autoarray.inversion.inversion.abstract import AbstractInversion  # noqa: E402
from autoarray.inversion.inversion.imaging.abstract import (  # noqa: E402
    AbstractInversionImaging,
)
from autoarray.inversion.inversion.imaging.mapping import (  # noqa: E402
    InversionImagingMapping,
)
from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402
from autoarray.inversion.mesh.interpolator.delaunay import (  # noqa: E402
    InterpolatorDelaunay,
)
from autoarray.util import fnnls as fnnls_module  # noqa: E402
from autolens.imaging.fit_imaging import FitImaging  # noqa: E402
from autolens.lens.to_inversion import TracerToInversion  # noqa: E402
from autonerves import cached_property as autonerves_cached_property  # noqa: E402

#: Relative tolerances of the three hard gates.
P2_RTOL = 1.0e-9
P3_RTOL = 1.0e-6
MAPPER_LOGDET_RTOL = 1.0e-6

#: The instrumentation budget, in MILLISECONDS of the call it measures.
#:
#: ``call_accounting`` costs a FIXED number of wrapper invocations per call —
#: ~40 descriptors and three module-level functions, each a closure entry, a
#: ``perf_counter`` pair and a dict update. That cost does not scale with the
#: call, so expressing it as a ratio makes the gate tighten every time the
#: campaign makes the call shorter, and eventually kills the very rows it is
#: measuring. It did: on RAL job 343356 the ratio was **1.0147 at a 413 ms
#: call** (6.1 ms of instrument) and on job 343355's feature arm **1.0366 at a
#: 224 ms call** (8.2 ms of instrument) — the shorter row cost 2 ms more in
#: absolute terms and was the one the 1.03 ratio killed, taking its result JSON
#: with it.
#:
#: 12.0 ms is the calibration the ratio actually encoded: 1.03 x the ~400 ms call
#: it was set on. The measured fixed cost is 6-8 ms across every recorded row of
#: this cell, so the budget sits ~1.5x above the instrument and still fails a
#: harness that has genuinely started changing the number it reports.
#:
#: The gated quantity is
#: ``overhead_ms = (ABBA ratio - 1) * mean clean call ms`` — the ABBA ratio
#: (below), converted into the absolute cost it stands for using the row's own
#: measured clean mean. Never a ratio of two sequential passes.
MAX_INSTRUMENTATION_OVERHEAD_MS = 12.0

#: The ratio this gate used to be. RECORDED, never asserted.
#:
#: It is kept because every row of levers 1-3 carries it and the note chains
#: them: 1.0147 at 413.301 ms, 1.0167 at 302.709, 1.0163 at 299.709, 1.0221 at
#: 267.448, 1.0182 at 268.681, 1.0366 at 224.330. Reading those rows next to
#: this cell's needs the number they were judged against.
#:
#: Measured 2026-09-15 on the HST / Delaunay / N=484 configuration, the estimator
#: behind it was also wrong: the sequential estimator returned 0.71, 0.98, 1.10,
#: 1.11, 1.24 and 1.37 on six rows of the same leg — a spread of a factor 1.9
#: around a quantity whose threshold was 1.03 — while the counterbalanced
#: estimator on the same row returned 1.0085 (median 1.0071, blocks
#: 0.956-1.059). That fix is kept; only the units of the threshold change here.
REFERENCE_OVERHEAD_RATIO = 1.03

#: Blocks needed before the overhead ratio is ASSERTED rather than RECORDED.
#: One block resolves nothing against a call-to-call scatter of +-15 %; three
#: blocks (six clean calls, six instrumented) put the standard error of the mean
#: block ratio near 1 %, which is the scale the threshold discriminates at. Below
#: this the comparison is still computed and written, with its observed spread,
#: and its status is ``RECORDED``.
MIN_BLOCKS_FOR_OVERHEAD_ASSERT = 3

#: At most 5 % of the call may be unattributed.
MAX_UNATTRIBUTED_FRACTION = 0.05

#: Warm-up runs to a STEADY STATE rather than a fixed count: calls are repeated
#: until the median of the last ``WARMUP_WINDOW`` agrees with the median of the
#: previous ``WARMUP_WINDOW`` to ``WARMUP_TOLERANCE``, or ``WARMUP_MAX_CALLS`` is
#: reached. The whole sequence is recorded either way — a warm-up that never
#: settles is evidence about the host, not something to hide.
WARMUP_WINDOW = 3
WARMUP_TOLERANCE = 0.10
WARMUP_MAX_CALLS = 12

#: The explicit remainder row. Never folded into a neighbouring site.
UNATTRIBUTED_LABEL = "unattributed (call - sum of exclusive times)"

#: Does the installed PyAutoArray cache `AbstractInversion.curvature_reg_matrix`?
#:
#: `F + lambda*H` became a `cached_property` in **lever 3 of #267** (PyAutoArray
#: `b4322c3e`); before it, it was a plain `property` that the likelihood rebuilt on
#: every access — twice per evaluation, 4.081 ms across the two in lever 2's row.
#:
#: This is read off the installed class rather than written as a literal `True`
#: because lever 3's own A/B runs ONE checkout of this cell against TWO PyAutoArray
#: revisions in one job: the control arm is the pre-lever-3 library, where the site
#: is genuinely reached twice, and the `cached_site_call_count_violations` gate
#: below raises `AssertionError` on any site declared cached and reached a
#: different number of times. A literal would therefore kill the control arm of the
#: measurement it exists to serve, while saying nothing the library does not already
#: say about itself. Declared this way, the gate fires as a hard `n_calls == 1` in
#: exactly the arm where the cache is claimed.
#:
#: The PIN is unconditional and lives elsewhere:
#: `scripts/misc/test/test_fixed_light_numba.py::test_curvature_reg_matrix_is_cached`
#: asserts `n_calls == 1` against whatever PyAutoArray it is run on, so a library
#: that stops caching it fails loudly rather than silently moving milliseconds
#: between the decomposition's rows.
#: Both descriptor types are checked because PyAutoArray decorates with
#: `autonerves.cached_property` — the `autonerves.tools.decorators.CachedProperty`
#: class, which caches into `obj.__dict__` exactly as the stdlib one does — and NOT
#: with `functools.cached_property`. A check against the stdlib class alone reports
#: `False` on a library that does cache it, which would silently un-declare the site.
CURVATURE_REG_MATRIX_IS_CACHED = isinstance(
    inspect.getattr_static(AbstractInversion, "curvature_reg_matrix"),
    (functools.cached_property, autonerves_cached_property),
)


# ===================================================================
# The instrumented site spec
# ===================================================================
# Roughly forty access points, grouped into the decomposition rows the summary
# prints. Every one was read in the library before it was wrapped; `cached=True`
# means the library caches it per object, which the gate below turns into a hard
# `n_calls == 1`. `missing_ok=True` marks a site that only exists on one
# formalism or one mesh — it is reported with `n_calls: 0` rather than vanishing
# from the row set, so the two legs always print the same labels.


def _site_spec():
    """The full spec, rebuilt per row so a previous row's counters cannot leak."""
    descriptor = call_accounting.descriptor_site
    function = call_accounting.function_site

    return [
        # --- image and blurring -------------------------------------------
        descriptor("fit.blurred_image", FitImaging, "blurred_image", cached=True, group="image"),
        descriptor(
            "fit.profile_subtracted_image",
            FitImaging,
            "profile_subtracted_image",
            cached=True,
            group="image",
        ),
        # --- mesh ----------------------------------------------------------
        descriptor("fit.tracer_to_inversion", FitImaging, "tracer_to_inversion", group="mesh"),
        descriptor(
            "to_inversion.traced_grid_2d_list_of_inversion",
            TracerToInversion,
            "traced_grid_2d_list_of_inversion",
            cached=True,
            group="mesh",
        ),
        descriptor(
            "to_inversion.image_plane_mesh_grid_pg_list",
            TracerToInversion,
            "image_plane_mesh_grid_pg_list",
            cached=True,
            group="mesh",
        ),
        descriptor(
            "to_inversion.traced_mesh_grid_pg_list",
            TracerToInversion,
            "traced_mesh_grid_pg_list",
            cached=True,
            group="mesh",
        ),
        descriptor(
            "delaunay.mesh_geometry",
            InterpolatorDelaunay,
            "mesh_geometry",
            cached=True,
            missing_ok=True,
            group="mesh",
        ),
        descriptor(
            "delaunay.triangulation",
            InterpolatorDelaunay,
            "delaunay",
            cached=True,
            missing_ok=True,
            group="mesh",
        ),
        # --- mapper and mapping matrix -------------------------------------
        descriptor(
            "to_inversion.lp_linear_func_list_galaxy_dict",
            TracerToInversion,
            "lp_linear_func_list_galaxy_dict",
            cached=True,
            group="mapper",
        ),
        descriptor(
            "to_inversion.mapper_galaxy_dict",
            TracerToInversion,
            "mapper_galaxy_dict",
            cached=True,
            group="mapper",
        ),
        descriptor(
            "mapper.unique_mappings", Mapper, "unique_mappings", cached=True, group="mapper"
        ),
        descriptor("mapper.mapping_matrix", Mapper, "mapping_matrix", cached=True, group="mapper"),
        descriptor(
            "mapper.sparse_triplets_data",
            Mapper,
            "sparse_triplets_data",
            cached=True,
            group="mapper",
        ),
        descriptor(
            "mapper.sparse_triplets_curvature",
            Mapper,
            "sparse_triplets_curvature",
            cached=True,
            group="mapper",
        ),
        # --- mapper weights (Delaunay family only) --------------------------
        descriptor(
            "delaunay.mappings_sizes_weights",
            InterpolatorDelaunay,
            "_mappings_sizes_weights",
            cached=True,
            missing_ok=True,
            group="mapper_weights",
        ),
        descriptor(
            "delaunay.mappings_sizes_weights_split",
            InterpolatorDelaunay,
            "_mappings_sizes_weights_split",
            cached=True,
            missing_ok=True,
            group="mapper_weights",
        ),
        # --- D and F --------------------------------------------------------
        descriptor(
            "inversion.to_inversion", TracerToInversion, "inversion", cached=True, group="DF"
        ),
        descriptor("fit.inversion", FitImaging, "inversion", cached=True, group="DF"),
        descriptor(
            "inversion.operated_mapping_matrix",
            AbstractInversion,
            "operated_mapping_matrix",
            cached=True,
            group="DF",
        ),
        descriptor(
            "inversion.linear_func_operated_mapping_matrix_dict",
            AbstractInversionImaging,
            "linear_func_operated_mapping_matrix_dict",
            cached=True,
            group="DF",
        ),
        descriptor(
            "sparse_numba.linear_func_operated_mapping_matrix_dict",
            InversionImagingSparseNumba,
            "linear_func_operated_mapping_matrix_dict",
            cached=True,
            missing_ok=True,
            group="DF",
        ),
        descriptor(
            "sparse_numba.psf_weighted_data",
            InversionImagingSparseNumba,
            "psf_weighted_data",
            cached=True,
            missing_ok=True,
            group="DF",
        ),
        descriptor(
            "sparse_numba.data_vector",
            InversionImagingSparseNumba,
            "data_vector",
            cached=True,
            missing_ok=True,
            group="DF",
        ),
        descriptor(
            "sparse_numba.curvature_matrix",
            InversionImagingSparseNumba,
            "curvature_matrix",
            cached=True,
            missing_ok=True,
            group="DF",
        ),
        descriptor(
            "dense.data_vector",
            InversionImagingMapping,
            "data_vector",
            cached=True,
            missing_ok=True,
            group="DF",
        ),
        descriptor(
            "dense.curvature_matrix",
            InversionImagingMapping,
            "curvature_matrix",
            cached=True,
            missing_ok=True,
            group="DF",
        ),
        # --- F + lambda H ---------------------------------------------------
        descriptor(
            "inversion.regularization_matrix",
            AbstractInversion,
            "regularization_matrix",
            cached=True,
            group="F_plus_H",
        ),
        descriptor(
            "inversion.regularization_matrix_reduced",
            AbstractInversion,
            "regularization_matrix_reduced",
            cached=True,
            group="F_plus_H",
        ),
        # Cached: it became a `cached_property` in lever 3 of #267, having been a
        # plain `property` reached twice per evaluation before it.
        descriptor(
            "inversion.curvature_reg_matrix",
            AbstractInversion,
            "curvature_reg_matrix",
            cached=CURVATURE_REG_MATRIX_IS_CACHED,
            group="F_plus_H",
        ),
        descriptor(
            "inversion.curvature_reg_matrix_reduced",
            AbstractInversion,
            "curvature_reg_matrix_reduced",
            group="F_plus_H",
        ),
        descriptor(
            "inversion.zeroed_ids_to_keep",
            AbstractInversion,
            "zeroed_ids_to_keep",
            cached=True,
            group="F_plus_H",
        ),
        # --- the solve ------------------------------------------------------
        descriptor(
            "inversion.reconstruction",
            AbstractInversion,
            "reconstruction",
            cached=True,
            group="solve",
        ),
        function(
            "solver.reconstruction_positive_only_from",
            inversion_util,
            "reconstruction_positive_only_from",
            group="solve",
        ),
        function(
            "solver.reconstruction_positive_negative_from",
            inversion_util,
            "reconstruction_positive_negative_from",
            group="solve",
        ),
        function("solver.fnnls_cholesky", fnnls_module, "fnnls_cholesky", group="solve"),
        # Route d_np's kernel, as its own exclusive row. `missing_ok` is not
        # needed (the module is always importable) but the site reports
        # `n_calls: 0` on every route that does not inject it, which is what
        # keeps the two legs printing the same labels.
        function(
            "solver.nnls_factor_reuse",
            fixed_light_numpy_solvers,
            "nnls_factor_reuse",
            group="solve",
        ),
        descriptor(
            "inversion.reconstruction_reduced",
            AbstractInversion,
            "reconstruction_reduced",
            cached=True,
            group="solve",
        ),
        # --- log determinants and the evidence -------------------------------
        descriptor(
            "inversion.regularization_term",
            AbstractInversion,
            "regularization_term",
            group="evidence",
        ),
        descriptor(
            "inversion.log_det_curvature_reg_matrix_term",
            AbstractInversion,
            "log_det_curvature_reg_matrix_term",
            group="evidence",
        ),
        descriptor(
            "inversion.log_det_regularization_matrix_term",
            AbstractInversion,
            "log_det_regularization_matrix_term",
            group="evidence",
        ),
        descriptor(
            "inversion.mapped_reconstructed_operated_data",
            AbstractInversion,
            "mapped_reconstructed_operated_data",
            group="evidence",
        ),
        descriptor("fit.model_data", FitImaging, "model_data", cached=True, group="evidence"),
        descriptor("fit.residual_map", FitImaging, "residual_map", cached=True, group="evidence"),
        descriptor(
            "fit.chi_squared_map", FitImaging, "chi_squared_map", cached=True, group="evidence"
        ),
        descriptor("fit.chi_squared", FitImaging, "chi_squared", cached=True, group="evidence"),
        descriptor(
            "fit.noise_normalization",
            FitImaging,
            "noise_normalization",
            cached=True,
            group="evidence",
        ),
        descriptor("fit.log_evidence", FitImaging, "log_evidence", cached=True, group="evidence"),
        descriptor(
            "fit.figure_of_merit", FitImaging, "figure_of_merit", cached=True, group="evidence"
        ),
    ]


#: Human-readable names for the decomposition groups, in print order.
GROUP_LABELS = {
    "image": "Image + blurring",
    "mesh": "Mesh (ray trace, placement, triangulation)",
    "mapper": "Mapper + mapping matrix",
    "mapper_weights": "Mapper weights (Delaunay)",
    "DF": "D and F assembly",
    "F_plus_H": "F + lambda H",
    "solve": "Positivity solve",
    "evidence": "Log determinants + evidence",
}


# ===================================================================
# PART A — Setup
# ===================================================================
# Deliberately the same construction as ``fixed_light_library.py`` PART A (which
# is itself the pixelized breakdown cells'), so a numba row and a JAX row of the
# same mesh and dataset describe the same model rather than two experiments.

instrument = DATASET

print(f"\n--- Dataset loading & masking [{instrument}, mesh={MESH}] ---")

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

SOURCE_PIXELS_REQUESTED = _cli.source_pixels
_default_source_pixels = 39 * 39 if MESH == "rectangular" else 1500

if MESH == "rectangular":
    _requested = (
        _default_source_pixels if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    )
    mesh_pixels_yx = int(round(math.sqrt(_requested)))
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
    n_source_pixels = mesh_pixels_yx * mesh_pixels_yx
    n_mesh_vertices = None
else:
    mesh_pixels_yx = None
    mesh_shape = None
    n_mesh_vertices = (
        _default_source_pixels if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    )
    n_source_pixels = n_mesh_vertices

print("\n--- Adapt image (lensed source) ---")

adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

image_plane_mesh_grid = None
if MESH != "rectangular":
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

if MESH == "rectangular":
    reg_scheme = "constant"
    mesh_obj = rect_mesh_classes(_cli)[1](shape=mesh_shape, weight_power=1.0, weight_floor=0.0)
    regularization = al.reg.Constant(coefficient=1.0)
    reg_provenance = {"scheme": reg_scheme, "coefficient": 1.0}
else:
    mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
    reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization: {reg_scheme} ({reg_provenance})")

# The instance stream. `median` repeats one prior-median draw (the default, and
# what the smoke leg runs); `iid` draws independently from the central 20 % of
# every prior, which is the regime a Nautilus worker actually sees. The lens
# LIGHT is fixed at instance 0 by construction — that is the whole premise of S3
# — so the S3 dataset is built once and only the mass/source parameters vary.
_IID_SEED = 263

if INSTANCE_MODE == "iid":
    _rng = np.random.default_rng(_IID_SEED)
    instances = [
        model.instance_from_unit_vector(
            unit_vector=list(_rng.uniform(0.4, 0.6, size=model.prior_count))
        )
        for _ in range(N_REPEATS + 1)
    ]
else:
    _median = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    instances = [_median] * (N_REPEATS + 1)

instance = instances[0]


def adapt_images_of(one_instance):
    """``AdaptImages`` for one instance — the dicts are keyed on its own galaxy."""
    kwargs = {
        "galaxy_image_dict": {one_instance.galaxies.source: adapt_image},
        "galaxy_name_image_dict": {"('galaxies', 'source')": adapt_image},
    }
    if image_plane_mesh_grid is not None:
        kwargs["galaxy_image_plane_mesh_grid_dict"] = {
            one_instance.galaxies.source: image_plane_mesh_grid
        }
        kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
            "('galaxies', 'source')": image_plane_mesh_grid
        }
    return al.AdaptImages(**kwargs)


adapt_images = adapt_images_of(instance)

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Mesh:                    {MESH}")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Source pixels:           {n_source_pixels}")
print(f"  Routes:                  {','.join(ROUTE_SELECTION)}")
print(f"  Formalisms:              {','.join(FORMALISM_SELECTION)}")
print(f"  Threads:                 {N_THREADS}")
print(f"  n-repeats:               {N_REPEATS}")
print(f"  Instances:               {INSTANCE_MODE}")
print(f"  Row order:               {ROW_ORDER}")
print(f"  Pins mode:               {PINS_MODE}")
print(f"  NNLS warm start:         {NNLS_WARM_START}")
print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
print(f"  NUMBA_NUM_THREADS:       {os.environ.get('NUMBA_NUM_THREADS', '(unset)')}")

_settings = al.Settings(use_border_relocator=True)

#: Positivity off. ``solve_ids_to_keep`` returns ``None`` the moment this is
#: false (abstract.py:540-546), so edge zeroing goes with it — which is why no
#: route-(c) number is ever quoted without its Δevidence and its negatives.
_settings_positive_negative = al.Settings(
    use_border_relocator=True,
    use_positive_only_solver=False,
)

tracer = al.Tracer(galaxies=list(instance.galaxies))

print("\n--- S0: full FitImaging with linear MGE lens light (eager, dense) ---")

fit_s0 = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    adapt_images=adapt_images,
    settings=_settings,
    xp=np,
)
system_s0 = fixed_light_system.linear_system_from(
    fit_s0, dataset, name="S0_current_mge60", scaling="numpy"
)
log_evidence_s0 = float(fit_s0.figure_of_merit)

print(f"  inversion class: {type(fit_s0.inversion).__name__}")
print(
    f"  S0: n={system_s0.n_params} (mapper {system_s0.n_mapper} + funcs {system_s0.n_funcs}), "
    f"edge-zeroed {int(system_s0.edge_zero_mask.sum())}"
)
print(f"  figure_of_merit (log_evidence) = {log_evidence_s0}")

print("\n--- S3: lens light converted to regular profiles + subtracted (eager) ---")

system_s3_dense = fixed_light_system.fixed_light_system_from(
    fit_s0,
    dataset,
    adapt_images=adapt_images,
    settings=_settings,
    name="S3_dense",
    scaling="numpy",
    sparse_operator="drop",
)
print(
    f"  S3: n={system_s3_dense.n_params} (mapper {system_s3_dense.n_mapper} + "
    f"funcs {system_s3_dense.n_funcs}), "
    f"edge-zeroed {int(system_s3_dense.edge_zero_mask.sum())}"
)
print(f"  subtracted light flux: {system_s3_dense.subtracted_light_flux}")

# The S3 instance is the S0 instance with the lens's light removed — the same
# galaxies `_light_stripped_tracer` put in the source-only tracer. The adapt
# images are keyed on the SOURCE galaxy, which is passed through by identity, so
# the lookups still hit.
_source_only_galaxies = list(system_s3_dense.source_only_tracer.galaxies)


def instance_s3_of(one_instance):
    """``one_instance`` with the lens light stripped, for the S3 routes."""
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


instances_s3 = [instance_s3_of(one) for one in instances]
adapt_images_s3 = adapt_images_of(instances_s3[0])

# The datasets each (route, formalism) pair runs on. S0 is the original dataset;
# S3 is the subtracted one. The sparse legs get a CPU operator baked on their own
# dataset, which is what makes the dispatch assert below possible.
dataset_s0_dense = dataset
dataset_s3_dense = system_s3_dense.dataset

print("\n--- Baking the numba CPU sparse operators ---")
dataset_s0_sparse = dataset.apply_sparse_operator_cpu()
system_s3_sparse = fixed_light_system.fixed_light_system_from(
    fit_s0,
    dataset,
    adapt_images=adapt_images,
    settings=_settings,
    name="S3_sparse_numba",
    scaling="numpy",
    sparse_operator="rebake_cpu",
)
dataset_s3_sparse = system_s3_sparse.dataset
print(f"  S0 operator: {type(dataset_s0_sparse.sparse_operator).__name__}")
print(f"  S3 operator: {type(dataset_s3_sparse.sparse_operator).__name__}")


# ===================================================================
# PART B — The gates that do not depend on a timing
# ===================================================================

gates: dict[str, dict] = {}


def _record_gate(name, status, detail):
    gates[name] = {"status": status, **detail}
    print(f"  [{status:>8}] {name}: {detail.get('summary', '')}")
    return gates[name]


print("\n" + "=" * 70)
print("GATES")
print("=" * 70)

# --- P1: the re-baked operator is the original's ---------------------------
# Asserted UNCONDITIONALLY, including under --pins none: this is not a
# calibrated numerical pin but a structural identity (the operator is a function
# of the noise map, PSF and mask, and the subtraction touches none of them). If
# it ever fails, every sparse S3 row in the JSON is measuring a different
# operator to the dense one it is compared against.

_op_original = dataset_s0_sparse.sparse_operator
_op_rebaked = dataset_s3_sparse.sparse_operator

_p1_fields = {}
for _field in ("psf_precision_operator_sparse", "indexes", "lengths"):
    _p1_fields[_field] = bool(
        np.array_equal(
            np.asarray(getattr(_op_original, _field)),
            np.asarray(getattr(_op_rebaked, _field)),
        )
    )

_p1_ok = all(_p1_fields.values())
_record_gate(
    "P1_rebaked_operator_equals_original",
    "PASS" if _p1_ok else "FAIL",
    {
        "fields": _p1_fields,
        "asserted_under_pins_none": True,
        "basis": (
            "apply_sparse_operator_cpu builds the operator from the noise map, PSF and "
            "mask alone (dataset.py:682-706); subtracting the fixed lens light changes "
            "only the data, so the re-baked operator must be identical element for "
            "element."
        ),
        "summary": f"{sum(_p1_fields.values())}/3 arrays identical",
    },
)
if not _p1_ok:
    raise AssertionError(
        f"P1 FAILED: the operator re-baked on the subtracted dataset differs from the "
        f"original's ({_p1_fields}). Every sparse S3 row would be measuring a different "
        f"operator to the dense row it is compared against."
    )

# --- S3 == S0 on the mapper block ------------------------------------------
# Fixing the lens light removes 60 unregularised columns from the linear system
# and must leave the mapper block of F + lambda H untouched. The log determinants
# are read off the library's own evidence terms, which are computed on exactly
# the rank-stripped blocks the evidence uses.

_mapper_log_dets = {}
for _key in ("log_det_curvature_reg_matrix_term", "log_det_regularization_matrix_term"):
    _s0_value = float(getattr(system_s0.inversion, _key))
    _s3_value = float(getattr(system_s3_dense.inversion, _key))
    _rel = abs(_s3_value - _s0_value) / max(abs(_s0_value), 1e-300)
    _mapper_log_dets[_key] = {
        "s0": _s0_value,
        "s3": _s3_value,
        "rel_diff": _rel,
        "status": "PASS" if _rel <= MAPPER_LOGDET_RTOL else "FAIL",
    }

_edge_s0 = int(system_s0.edge_zero_mask.sum())
_edge_s3 = int(system_s3_dense.edge_zero_mask.sum())
_mapper_ok = all(e["status"] == "PASS" for e in _mapper_log_dets.values()) and _edge_s0 == _edge_s3

_record_gate(
    "S3_mapper_block_equals_S0",
    "PASS" if _mapper_ok else "FAIL",
    {
        "log_dets": _mapper_log_dets,
        "n_edge_zeroed_s0": _edge_s0,
        "n_edge_zeroed_s3": _edge_s3,
        "rtol": MAPPER_LOGDET_RTOL,
        "summary": (
            f"log-dets rel "
            f"{max(e['rel_diff'] for e in _mapper_log_dets.values()):.3e}, "
            f"edge-zeroed {_edge_s0} vs {_edge_s3}"
        ),
    },
)
if not _mapper_ok:
    raise AssertionError(
        f"S3's mapper block is not S0's: log-dets {_mapper_log_dets}, edge-zeroed "
        f"{_edge_s0} vs {_edge_s3}. Every S0-vs-S3 delta in this JSON would be "
        f"comparing two different problems."
    )


def _eager_s3_fit(dataset_for_fit, settings_for_fit):
    """One eager source-only ``FitImaging`` on the given dataset."""
    return al.FitImaging(
        dataset=dataset_for_fit,
        tracer=system_s3_dense.source_only_tracer,
        adapt_images=adapt_images_s3,
        settings=settings_for_fit,
        xp=np,
    )


def _solve_with_stats(dataset_for_fit, settings_for_fit):
    """An eager S3 fit whose solver ``stats`` dict is captured as it is filled.

    ``fnnls_cholesky`` fills a ``stats`` dict its caller owns and
    ``reconstruction_positive_only_from`` adds two keys to that same dict after
    the solver returns (``inversion_util.py:305-312``), so the dict is only
    complete once the call is over — which is exactly what the wrapper's
    reference-not-copy capture gives back.
    """
    nnls_memo._nnls_passive_set_memo.clear()
    call_accounting.install(
        [
            call_accounting.function_site("fnnls", fnnls_module, "fnnls_cholesky", group="solve"),
        ]
    )
    try:
        fit = _eager_s3_fit(dataset_for_fit, settings_for_fit)
        figure_of_merit = float(fit.figure_of_merit)
        stats = call_accounting.captured_stats().get("fnnls", {})
    finally:
        call_accounting.uninstall()
    return fit, figure_of_merit, stats


print("\n  building the two eager S3 fits the P2/P3 gates compare...")
_fit_s3_dense, _fom_s3_dense, _stats_s3_dense = _solve_with_stats(dataset_s3_dense, _settings)
_fit_s3_sparse, _fom_s3_sparse, _stats_s3_sparse = _solve_with_stats(dataset_s3_sparse, _settings)

assert isinstance(_fit_s3_dense.inversion, InversionImagingMapping), (
    f"the dense S3 gate fit dispatched {type(_fit_s3_dense.inversion).__name__}, "
    f"not InversionImagingMapping"
)
assert isinstance(_fit_s3_sparse.inversion, InversionImagingSparseNumba), (
    f"the sparse S3 gate fit dispatched {type(_fit_s3_sparse.inversion).__name__}, "
    f"not InversionImagingSparseNumba"
)

# --- P2: the two formalisms assemble the same linear system ------------------

_dv_dense = np.asarray(_fit_s3_dense.inversion.data_vector, dtype=float)
_dv_sparse = np.asarray(_fit_s3_sparse.inversion.data_vector, dtype=float)
_n_mapper = int(system_s3_dense.n_mapper)
_cm_dense = np.asarray(_fit_s3_dense.inversion.curvature_matrix, dtype=float)[
    :_n_mapper, :_n_mapper
]
_cm_sparse = np.asarray(_fit_s3_sparse.inversion.curvature_matrix, dtype=float)[
    :_n_mapper, :_n_mapper
]


def _max_rel(got, reference):
    got_a = np.asarray(got, dtype=float)
    ref_a = np.asarray(reference, dtype=float)
    scale = max(float(np.max(np.abs(ref_a))), 1e-300)
    return float(np.max(np.abs(got_a - ref_a))) / scale


_p2_dv_rel = _max_rel(_dv_sparse, _dv_dense)
_p2_cm_rel = _max_rel(_cm_sparse, _cm_dense)
_p2_ok = _p2_dv_rel <= P2_RTOL and _p2_cm_rel <= P2_RTOL

_record_gate(
    "P2_dense_equals_sparse_numba_system",
    "PASS" if _p2_ok else "FAIL",
    {
        "data_vector_max_rel_diff": _p2_dv_rel,
        "curvature_matrix_mapper_block_max_rel_diff": _p2_cm_rel,
        "rtol": P2_RTOL,
        "n_mapper": _n_mapper,
        "asserted_under_pins_none": True,
        "summary": f"D {_p2_dv_rel:.3e}, F(mapper) {_p2_cm_rel:.3e} (rtol {P2_RTOL:g})",
    },
)
if not _p2_ok:
    raise AssertionError(
        f"P2 FAILED: the sparse-numba S3 system is not the dense one — data_vector "
        f"rel {_p2_dv_rel:.3e}, mapper-block curvature rel {_p2_cm_rel:.3e} "
        f"(rtol {P2_RTOL:g}). The subtraction has not reached the sparse data vector."
    )

# --- P3: ...and therefore solve to the same evidence -------------------------


def _passive_set(stats):
    value = stats.get("passive_set")
    if value is None:
        return None
    return np.sort(np.asarray(value, dtype=int))


_ps_dense = _passive_set(_stats_s3_dense)
_ps_sparse = _passive_set(_stats_s3_sparse)
if _ps_dense is None or _ps_sparse is None:
    _n_passive_differences = None
else:
    _n_passive_differences = int(len(np.setxor1d(_ps_dense, _ps_sparse, assume_unique=False)))

_p3_rel = abs(_fom_s3_sparse - _fom_s3_dense) / max(abs(_fom_s3_dense), 1e-300)
_p3_pass = _p3_rel <= P3_RTOL
_p3_status = ("PASS" if _p3_pass else "FAIL") if PINS_ASSERT else "RECORDED"

_record_gate(
    "P3_dense_equals_sparse_numba_evidence",
    _p3_status,
    {
        "figure_of_merit_dense": _fom_s3_dense,
        "figure_of_merit_sparse_numba": _fom_s3_sparse,
        "d_log_evidence_nats": _fom_s3_sparse - _fom_s3_dense,
        "rel_diff": _p3_rel,
        "rtol": P3_RTOL,
        "n_passive_set_differences": _n_passive_differences,
        "n_passive_dense": None if _ps_dense is None else int(_ps_dense.size),
        "n_passive_sparse_numba": None if _ps_sparse is None else int(_ps_sparse.size),
        "solver_stats_dense": {k: v for k, v in _stats_s3_dense.items() if k != "passive_set"},
        "solver_stats_sparse_numba": {
            k: v for k, v in _stats_s3_sparse.items() if k != "passive_set"
        },
        "summary": (
            f"Δ {_fom_s3_sparse - _fom_s3_dense:+.6e} nats (rel {_p3_rel:.3e}), "
            f"{_n_passive_differences} passive-set differences"
        ),
    },
)
if PINS_ASSERT and not _p3_pass:
    raise AssertionError(
        f"P3 FAILED: dense S3 figure_of_merit {_fom_s3_dense!r} vs sparse-numba "
        f"{_fom_s3_sparse!r} (rel {_p3_rel:.3e} > rtol {P3_RTOL:g}); "
        f"{_n_passive_differences} passive-set differences."
    )

# --- Route (c) diagnostics: what dropping positivity buys and costs ----------
# Built eagerly so the negatives and the Δevidence exist beside the timing, and
# so the route the library actually takes is verified rather than asserted.

print("\n--- Route (c) diagnostics: positive-negative, eager ---")

_fit_c = _eager_s3_fit(dataset_s3_dense, _settings_positive_negative)
_x_c = np.asarray(_fit_c.inversion.reconstruction, dtype=float)
_fom_c = float(_fit_c.figure_of_merit)
_c_negative = _x_c < 0.0

route_c_block = {
    "solve_ids_to_keep_is_none": _fit_c.inversion.solve_ids_to_keep is None,
    "edge_zeroing_disabled": True,
    "edge_zeroing_note": (
        "Inversion.solve_ids_to_keep returns None the moment use_positive_only_solver "
        "is False (abstract.py:540-546), so turning positivity off silently turns edge "
        "zeroing off with it. Route c solves the FULL system; routes a and b solve the "
        "edge-zeroed one."
    ),
    "log_evidence_eager": _fom_c,
    "d_log_evidence_vs_route_b": _fom_c - _fom_s3_dense,
    "n_negative_entries": int(np.sum(_c_negative)),
    "negative_flux_fraction": float(
        np.sum(np.abs(_x_c[_c_negative])) / max(float(np.sum(np.abs(_x_c))), 1e-300)
    ),
    "note": (
        "Never quote route c's milliseconds without these numbers. The unconstrained "
        "solution sits at a HIGHER evidence than the constrained optimum because it is "
        "a different, infeasible minimiser."
    ),
}
print(
    f"  Δlog-evidence vs b: {route_c_block['d_log_evidence_vs_route_b']:+.6e} nats; "
    f"{route_c_block['n_negative_entries']} negative entries "
    f"({route_c_block['negative_flux_fraction'] * 100:.4f} % of |flux|)"
)

del _fit_c, _fit_s3_dense, _fit_s3_sparse


# ===================================================================
# PART C — The rows
# ===================================================================

#: ``(dataset, instances, adapt images, settings, expected inversion class)`` per
#: (route, formalism). The expected class is the HARD dispatch assert: the
#: factory selects on ``dataset.sparse_operator`` and the linear-object mix
#: (``factory.py:128-160``), and a row that silently fell back to the other class
#: would be a correct number wearing the wrong label.
_EXPECTED_CLASS = {
    "dense": InversionImagingMapping,
    "sparse_numba": InversionImagingSparseNumba,
}

_ROUTE_DATASETS = {
    ("a", "dense"): dataset_s0_dense,
    ("a", "sparse_numba"): dataset_s0_sparse,
    ("b", "dense"): dataset_s3_dense,
    ("b", "sparse_numba"): dataset_s3_sparse,
    ("c", "dense"): dataset_s3_dense,
    ("c", "sparse_numba"): dataset_s3_sparse,
    # d_np is route b's problem with route b's datasets and route b's settings.
    # Only the solver differs, which is the whole point: any other difference
    # would make the b -> d_np delta a comparison of two problems.
    ("d_np", "dense"): dataset_s3_dense,
    ("d_np", "sparse_numba"): dataset_s3_sparse,
    # b_direct and b_touched are the same statement one level down: route b's
    # problem with one curvature kernel swapped.
    ("b_direct", "dense"): dataset_s3_dense,
    ("b_direct", "sparse_numba"): dataset_s3_sparse,
    ("b_touched", "dense"): dataset_s3_dense,
    ("b_touched", "sparse_numba"): dataset_s3_sparse,
}

_ROUTE_LABELS = {
    "a": "S0 fnnls (the joint system the library runs today)",
    "b": "S3 fnnls source-only (the reference)",
    "c": "S3 positive-negative (xp.linalg.solve)",
    "d_np": "S3 source-only, factor-reuse NNLS injected (one Cholesky, downdates)",
    "b_direct": "S3 source-only, library direct curvature kernel injected",
    "b_touched": "S3 source-only, touched-index two-stage curvature kernel injected",
}

#: Which curvature kernel each injected-kernel route installs, by
#: :data:`fixed_light_numpy_kernels.KERNELS` name. A route absent from this map
#: runs the library's own dispatcher untouched — which is what route ``b``, the
#: control of this A/B, must keep doing.
_ROUTE_CURVATURE_KERNELS = {
    "b_direct": "direct",
    "b_touched": "two_stage_touched",
}

#: The routes that are route ``b``'s problem in everything but the solver: same
#: dataset, same instances, same settings, same expected inversion class.
_ROUTES_LIKE_B = ("b", "d_np", "b_direct", "b_touched")


def _row_plan():
    """The (route, formalism) pairs in ``--row-order``.

    ``grouped`` is formalism-major (every dense row, then every sparse row) and
    is the default; ``interleaved`` is route-major, so the two formalisms of one
    route sit next to each other in time and a slow thermal drift cannot land
    entirely on one of them. The order is recorded per row, because it is a
    property of the measurement.
    """
    if ROW_ORDER == "interleaved":
        return [
            (route, formalism) for route in ROUTE_SELECTION for formalism in FORMALISM_SELECTION
        ]
    return [(route, formalism) for formalism in FORMALISM_SELECTION for route in ROUTE_SELECTION]


def _analysis_for(route, formalism):
    """The production entry point for one (route, formalism), built the numba way.

    The triple, in this order: the thread environment is already pinned (header),
    the dataset already carries its CPU operator (or deliberately does not), and
    the analysis is built with ``use_jax=False``.

    ``d_np``, ``b_direct`` and ``b_touched`` are built **exactly** as ``b`` is
    (:data:`_ROUTES_LIKE_B`) — same dataset, same adapt images, same positive-only
    settings. The kernel is swapped by patching the library's entry point for the
    duration of the row, not by building a different analysis, so nothing but the
    one function differs.
    """
    return al.AnalysisImaging(
        dataset=_ROUTE_DATASETS[(route, formalism)],
        adapt_images=adapt_images if route == "a" else adapt_images_s3,
        settings=_settings if route != "c" else _settings_positive_negative,
        use_jax=False,
    )


def _instance_for(route, index):
    return instances[index] if route == "a" else instances_s3[index]


def _assert_dispatch(route, formalism):
    """Build one eager fit and assert the factory chose the class the row claims."""
    analysis = _analysis_for(route, formalism)
    fit = analysis.fit_from(instance=_instance_for(route, 0))
    inversion = fit.inversion
    expected = _EXPECTED_CLASS[formalism]
    if not isinstance(inversion, expected):
        raise AssertionError(
            f"row {route}_{formalism}: the inversion factory dispatched "
            f"{type(inversion).__name__}, not {expected.__name__}. The row would be a "
            f"correct number wearing the wrong label."
        )
    if fit._xp is not np:
        raise AssertionError(
            f"row {route}_{formalism}: fit._xp is {fit._xp!r}, not numpy — this is not "
            f"the CPU path."
        )
    return type(inversion).__name__


#: P4's relative tolerance: the injected kernel must be the library's answer, not
#: an approximation of it. 1e-9 is the scale fp64 round-off in a ~1500-parameter
#: Cholesky reaches, and it is the tolerance the harness kernels are pinned at.
P4_RTOL = 1.0e-9


def _p4_equivalence(formalism):
    """P4: route ``d_np``'s injected solve reaches route ``b``'s evidence.

    One eager fit each way on the same instance and formalism — the same shape
    as P3, and for the same reason: a millisecond that came from a different
    minimiser is not a faster solve, it is a different answer. Both the Δ in nats
    and the maximum absolute reconstruction difference are recorded, because a
    figure of merit can agree while the reconstruction does not.
    """
    _clear_memos()
    _fit_b = _analysis_for("b", formalism).fit_from(instance=_instance_for("b", 0))
    _fom_b = float(_fit_b.figure_of_merit)
    _x_b = np.asarray(_fit_b.inversion.reconstruction, dtype=float)

    _clear_memos()
    with fixed_light_numpy_solvers.numpy_solver_injected(
        _factor_reuse_solver, label=f"P4_d_np_{formalism}"
    ) as _counts:
        _fit_d = _analysis_for("d_np", formalism).fit_from(instance=_instance_for("d_np", 0))
        _fom_d = float(_fit_d.figure_of_merit)
        _x_d = np.asarray(_fit_d.inversion.reconstruction, dtype=float)
    _clear_memos()

    if _counts["numpy"] <= 0:
        raise AssertionError(
            f"P4 ({formalism}): the injected solver was never called "
            f"({_counts}); the gate would have compared route b against itself."
        )

    _rel = abs(_fom_d - _fom_b) / max(abs(_fom_b), 1e-300)
    _max_abs = float(np.max(np.abs(_x_d - _x_b))) if _x_b.size else 0.0
    _pass = _rel <= P4_RTOL
    _status = ("PASS" if _pass else "FAIL") if PINS_ASSERT else "RECORDED"

    _record_gate(
        f"P4_d_np_equals_b_evidence_{formalism}",
        _status,
        {
            "figure_of_merit_b": _fom_b,
            "figure_of_merit_d_np": _fom_d,
            "d_log_evidence_nats": _fom_d - _fom_b,
            "rel_diff": _rel,
            "rtol": P4_RTOL,
            "max_abs_diff_reconstruction": _max_abs,
            "injected_solver_calls": dict(_counts, last_stats=None),
            "solver_stats": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in (_counts["last_stats"] or {}).items()
                if k != "passive_set"
            },
            "summary": (f"Δ {_fom_d - _fom_b:+.6e} nats (rel {_rel:.3e}), max |Δx| {_max_abs:.3e}"),
        },
    )
    if PINS_ASSERT and not _pass:
        raise AssertionError(
            f"P4 FAILED ({formalism}): route b figure_of_merit {_fom_b!r} vs injected "
            f"factor-reuse {_fom_d!r} (rel {_rel:.3e} > rtol {P4_RTOL:g}, max |Δx| "
            f"{_max_abs:.3e}). The d_np row would be a different minimiser wearing a "
            f"speedup's label."
        )
    return gates[f"P4_d_np_equals_b_evidence_{formalism}"]


def _clear_memos():
    """Everything that can carry state from one row into the next.

    The NNLS passive-set memo is the one that matters: dense and sparse-numba S3
    key IDENTICALLY (``_nnls_warm_start_fingerprint`` is built from shapes, not
    from the matrix), so a memo entry left behind by the dense row would seed the
    sparse row's solve with a 100 %-correct previous answer.

    Unconditional. ``--nnls-warm-start on`` does not reach this: seeding one
    formalism from the other is not a channel production has, and a row that
    inherited it would not be measuring the thing its label names.
    """
    nnls_memo._nnls_passive_set_memo.clear()


def _clear_memos_within_block():
    """The clears *inside* an ABBA block — the ones ``--nnls-warm-start`` gates.

    With the memo off (the default) every call in a block starts from the
    dense-sign seed, which is the cold production solve. With it on the calls
    seed each other, which is what a sampler's successive evaluations do — and
    clearing between them would leave the memo permanently empty and the flag
    doing nothing at all.
    """
    if NNLS_WARM_START_ON:
        return
    _clear_memos()


def _factor_reuse_solver(ZTZ, ZTx, *, stats=None):
    """Route ``d_np``'s kernel, resolved on the module at **call** time.

    ``numpy_solver_injected`` binds the callable it is handed once, when the
    injection is entered — which is outside ``call_accounting.install``. Passing
    ``nnls_factor_reuse`` directly would therefore bind the *unwrapped* function
    and the ``solver.nnls_factor_reuse`` site would report ``n_calls: 0`` while
    the kernel ran. This indirection resolves the module attribute per call, so
    whatever ``install`` has rebound it to is what runs.
    """
    return fixed_light_numpy_solvers.nnls_factor_reuse(ZTZ, ZTx, stats=stats)


def _operated_mapping_matrix_memo_size():
    """The size of the OTHER cross-evaluation memo, recorded rather than cleared.

    ``InversionImagingSparseNumba`` keeps a module-level memo of PSF-convolved
    MGE stacks (``imaging_numba/sparse.py``). It is production behaviour, so it
    is left alone — but it *is* a channel between rows, so its size is recorded
    before and after every row and a reader can see whether a row was served
    from it.
    """
    from autoarray.inversion.inversion.imaging_numba import sparse as _sparse_module

    memo = getattr(_sparse_module, "_operated_mapping_matrix_memo", None)
    return None if memo is None else len(memo)


#: Rotating index into the instance stream, so a row's calls walk it in order
#: whether they are clean calls, instrumented calls or warm-up calls.
_call_index = {"next": 1}


def _one_call(analysis, route):
    """One production likelihood call; returns ``(seconds, value)``."""
    one = _instance_for(route, _call_index["next"] % len(instances))
    _call_index["next"] += 1
    started = time.perf_counter()
    value = analysis.log_likelihood_function(instance=one)
    return time.perf_counter() - started, float(value)


def _warm_to_steady_state(analysis, route):
    """Call until the timing settles, and RECORD the whole sequence.

    Not a fixed warm-up count. The first call is where numba's lazy JIT
    compilation lands, but whether the *second* call is already at steady state
    is a property of the host, not something a cell can assume — so calls are
    repeated until the median of the last ``WARMUP_WINDOW`` agrees with the
    median of the previous ``WARMUP_WINDOW`` to ``WARMUP_TOLERANCE``, or
    ``WARMUP_MAX_CALLS`` is reached.

    The full per-call sequence is returned and written to the JSON either way. A
    warm-up that never settles is evidence about the machine the numbers were
    taken on; discarding it silently would leave a reader unable to tell a
    settled row from an unsettled one.

    THERE IS NO WARM-UP RAMP ON THIS CONFIGURATION — the hypothesis was tested
    and falsified. Fifteen consecutive calls after one warm-up (2026-09-15, HST /
    Delaunay / N=484, ``b_sparse_numba``) ran

        341, 310, 341, 376, 350, 337, 364, 344, 294, 279, 319, 338, 374, 367, 362 ms

    — flat, first/last 0.942, neither a smooth decay nor a step, but with +-15 %
    call-to-call scatter. An earlier run the same day had shown a 2.9x monotonic
    decay across its first ten calls (851, 1010, 762, 524, 495, 447, 391, 345,
    309, 292 ms), which looked exactly like a lazy-compilation or
    frequency-scaling ramp. It was neither: it did not reproduce, and the machine
    was carrying a load average of ~8 on 8 cores from other work at the time. It
    was queueing, not warming.

    So one warm-up call suffices *here*, and this loop exists for two reasons
    that survive that finding: the flatness is a measurement of one host at one
    size and not a general fact, and a run that does sit on a ramp must be
    visible as such rather than silently averaged. The scatter is the real
    lesson, and it is why the overhead gate is counterbalanced
    (:func:`_abba_blocks`) rather than a ratio of two sequential passes.
    """
    sequence = []
    steady = False
    while len(sequence) < WARMUP_MAX_CALLS:
        elapsed, _value = _one_call(analysis, route)
        sequence.append(elapsed)
        if len(sequence) >= 2 * WARMUP_WINDOW:
            recent = float(np.median(sequence[-WARMUP_WINDOW:]))
            previous = float(np.median(sequence[-2 * WARMUP_WINDOW : -WARMUP_WINDOW]))
            if abs(recent - previous) / max(previous, 1e-300) <= WARMUP_TOLERANCE:
                steady = True
                break
    return {
        "sequence_s": sequence,
        "n_calls": len(sequence),
        "steady": steady,
        "window": WARMUP_WINDOW,
        "tolerance": WARMUP_TOLERANCE,
        "max_calls": WARMUP_MAX_CALLS,
        # The first call carries numba's lazy compilation of every kernel this
        # row's formalism reaches.
        "first_call_incl_numba_compile_s": sequence[0],
        "note": (
            "Warm-up ran to a steady state, not to a fixed count: calls repeat until "
            "the median of the last `window` agrees with the median of the previous "
            "`window` to `tolerance`, or `max_calls` is hit. Every call is recorded. "
            "`steady: false` means the row's timings were taken on a host that never "
            "settled and should be read with the sequence in hand."
        ),
    }


def _abba_blocks(analysis, route, n_blocks, site_spec):
    """Counterbalanced clean-vs-instrumented timing. Cancels linear drift.

    Each block runs **A B B A** — one clean call, two instrumented calls, one
    clean call — and its ratio is ``mean(B) / mean(A)``. Under a drift that is
    linear in call index the two A's straddle the two B's symmetrically, so the
    drift cancels in the ratio exactly; a sequential "all clean, then all
    instrumented" design instead charges the entire drift over the pass to the
    instrumentation.

    That is not a hypothetical. On this configuration the sequential estimator
    returned 0.71, 0.98, 1.10, 1.11, 1.24 and 1.37 across six rows of one leg —
    including ratios *below* 1, which no real instrumentation can produce — while
    the counterbalanced estimator on the same row returned 1.0085.

    Returns the clean times, the instrumented times, the per-block ratios, the
    accumulated accounting snapshot and the solver stats.
    """
    clean: list[float] = []
    instrumented: list[float] = []
    block_ratios: list[float] = []
    values: list[float] = []
    totals: dict[str, dict] = {}
    solver_stats: dict[str, dict] = {}

    for _block in range(n_blocks):
        _clear_memos_within_block()
        a1, value = _one_call(analysis, route)
        values.append(value)

        _clear_memos_within_block()
        call_accounting.install(site_spec())
        try:
            b1, value_b1 = _one_call(analysis, route)
            b2, value_b2 = _one_call(analysis, route)
            snapshot = call_accounting.snapshot()
            solver_stats = call_accounting.captured_stats()
        finally:
            call_accounting.uninstall()
        values.extend([value_b1, value_b2])

        _clear_memos_within_block()
        a2, value = _one_call(analysis, route)
        values.append(value)

        # `install` resets the counters, so each block's snapshot covers exactly
        # its own two instrumented calls; they are accumulated here.
        for label, entry in snapshot.items():
            total = totals.setdefault(label, {"excl_s": 0.0, "incl_s": 0.0, "n_calls": 0})
            total["excl_s"] += entry["excl_s"]
            total["incl_s"] += entry["incl_s"]
            total["n_calls"] += entry["n_calls"]

        clean.extend([a1, a2])
        instrumented.extend([b1, b2])
        block_ratios.append(((b1 + b2) / 2.0) / max((a1 + a2) / 2.0, 1e-300))

    return {
        "clean_s": clean,
        "instrumented_s": instrumented,
        "block_ratios": block_ratios,
        "totals": totals,
        "solver_stats": solver_stats,
        "values": values,
        "n_blocks": n_blocks,
    }


def _load_average():
    """``/proc/loadavg``'s 1/5/15-minute figures, or ``None`` where unavailable.

    Recorded at the start and the end of the timed rows, as first-class data.
    Every timing in this JSON is a wall-clock measurement on a shared machine; a
    load average above the core count means the row was competing for those
    cores, and a reader who cannot see that has no way to tell a measurement
    from a queue.
    """
    try:
        return [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        return None


load_average_at_start = _load_average()

print("\n" + "=" * 70)
print("ROWS — one production log_likelihood_function call per row")
print("=" * 70)
if load_average_at_start is not None:
    print(f"  load average at start: {load_average_at_start} (cores: {os.cpu_count()})")

rows: dict[str, dict] = {}

for _route, _formalism in _row_plan():
    _key = f"{_route}_{_formalism}"
    print(f"\n--- {_key}: {_ROUTE_LABELS[_route]} [{_formalism}] ---")

    _clear_memos()
    _memo_before = _operated_mapping_matrix_memo_size()

    # --- P4, before anything is timed -------------------------------------
    # The gate runs its own, separate injection: it must compare the injected
    # solve against the LIBRARY's, and it cannot do that from inside a row whose
    # library entry point is already patched.
    _p4_gate = _p4_equivalence(_formalism) if _route == "d_np" else None

    # --- the row's injection ----------------------------------------------
    # Route d_np injects the factor-reuse kernel into the library's positive-only
    # entry point for EVERYTHING this row does — the dispatch assert, the warm-up
    # and every call of every ABBA block. It is entered OUTSIDE
    # `call_accounting.install` (which happens inside `_abba_blocks`), so the
    # accounting wrapper closes over the injected function and the
    # decomposition's solver site attributes the injected solve rather than the
    # library's.
    #
    # Entered and exited explicitly rather than with a `with` block: the row body
    # below is module-level code and every failure in it is a gate raising, which
    # ends the process. A patch leaked by a dying cell cannot reach another row,
    # and wrapping 200 lines in a `with` to say so would bury them.
    _injection = None
    _injection_counts = None
    if _route == "d_np":
        _injection = fixed_light_numpy_solvers.numpy_solver_injected(
            _factor_reuse_solver, label=_key
        )
        _injection_counts = _injection.__enter__()

    # Routes b_direct and b_touched inject a CURVATURE kernel rather than a
    # solver — one level below d_np, at
    # `inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from`,
    # which the inversion resolves by module attribute at `sparse.py:385`. Same
    # placement and the same reasoning as the solver injection above: entered
    # OUTSIDE `call_accounting.install` so the accounting wrapper closes over the
    # injected kernel and `sparse_numba.curvature_matrix` attributes it, and
    # entered/exited explicitly rather than with a `with` because the row body is
    # module-level code whose failures end the process.
    #
    # Route `b` is deliberately absent from `_ROUTE_CURVATURE_KERNELS`: it is the
    # production path un-patched, and it is the row this A/B is taken against.
    _kernel_injection = None
    _kernel_injection_counts = None
    if _route in _ROUTE_CURVATURE_KERNELS:
        _kernel_injection = fixed_light_numpy_kernels.curvature_kernel_injected(
            _ROUTE_CURVATURE_KERNELS[_route], label=_key
        )
        _kernel_injection_counts = _kernel_injection.__enter__()

    _inversion_class = _assert_dispatch(_route, _formalism)
    print(f"  inversion class: {_inversion_class}")

    _analysis = _analysis_for(_route, _formalism)

    # 1. Warm-up TO STEADY STATE. Discarded from the timings, recorded in full.
    _clear_memos()
    _warmup = _warm_to_steady_state(_analysis, _route)
    print(
        f"  warm-up: {_warmup['n_calls']} call(s), "
        f"{'steady' if _warmup['steady'] else 'NEVER SETTLED'}, "
        f"first {_warmup['first_call_incl_numba_compile_s']:.4f} s "
        f"(incl. numba compile), last {_warmup['sequence_s'][-1]:.4f} s"
    )

    _entry = {
        "route": _route,
        "formalism": _formalism,
        "label": f"{_ROUTE_LABELS[_route]} [{_formalism}]",
        "inversion_class": _inversion_class,
        "n_repeats": N_REPEATS,
        "warmup": _warmup,
        "warmup_incl_numba_compile_s": _warmup["first_call_incl_numba_compile_s"],
        "row_order": ROW_ORDER,
        "memo_cleared_between_rows": True,
        "memo_cleared_within_block": not NNLS_WARM_START_ON,
        "nnls_warm_start_mode": NNLS_WARM_START,
        "nnls_warm_start_env": os.environ.get("AUTOARRAY_NNLS_WARM_START"),
        "operated_mapping_matrix_memo_entries_before": _memo_before,
        "decomposed": False,
    }

    if _route == "c":
        _entry["route_c"] = route_c_block

    if _p4_gate is not None:
        _entry["p4_equivalence"] = _p4_gate

    if not DECOMPOSE:
        # No decomposition asked for: a plain clean pass, nothing to counterbalance.
        _clear_memos()
        _cpu_started = time.process_time()
        _clean_per_call = []
        _clean_values = []
        for _ in range(N_REPEATS):
            _elapsed, _value = _one_call(_analysis, _route)
            _clean_per_call.append(_elapsed)
            _clean_values.append(_value)
        _cpu_elapsed = time.process_time() - _cpu_started
        _clean_wall = sum(_clean_per_call)
        _instrumented_per_call = []
        _abba = None
    else:
        # 2. ABBA — counterbalanced clean/instrumented. `--n-repeats` is the
        #    number of CLEAN calls; each block contributes two of them.
        _n_blocks = max(1, -(-N_REPEATS // 2))
        _cpu_started = time.process_time()
        _abba = _abba_blocks(_analysis, _route, _n_blocks, _site_spec)
        _cpu_elapsed = time.process_time() - _cpu_started
        _clean_per_call = _abba["clean_s"]
        _instrumented_per_call = _abba["instrumented_s"]
        _clean_values = _abba["values"]
        _clean_wall = sum(_clean_per_call) + sum(_instrumented_per_call)

    _clean_mean = float(np.mean(_clean_per_call))
    _clean_median = float(np.median(_clean_per_call))
    _clean_spread = (max(_clean_per_call) - min(_clean_per_call)) / max(_clean_mean, 1e-300)

    print(
        f"  clean x{len(_clean_per_call)}: mean {_clean_mean * 1e3:.3f} ms, "
        f"median {_clean_median * 1e3:.3f} ms, spread {_clean_spread * 100:.1f} %"
    )

    _entry.update(
        {
            "call_ms": _clean_mean * 1e3,
            "call_ms_median": _clean_median * 1e3,
            "call_ms_min": float(np.min(_clean_per_call)) * 1e3,
            "call_ms_max": float(np.max(_clean_per_call)) * 1e3,
            "call_ms_sequence": [t * 1e3 for t in _clean_per_call],
            "clean_relative_spread": _clean_spread,
            "n_clean_calls": len(_clean_per_call),
            "log_likelihood": _clean_values[-1],
            "log_likelihood_sequence": _clean_values,
            "cpu_over_wall": _cpu_elapsed / max(_clean_wall, 1e-300),
            "cpu_over_wall_note": (
                "time.process_time() sums every thread of the process, so this is the "
                "effective thread count of the timed calls. At --threads 1 it should sit "
                "just above 1.0; anything higher means a pool this cell did not pin."
            ),
        }
    )

    if DECOMPOSE:
        _totals = _abba["totals"]
        _n_instrumented = len(_instrumented_per_call)
        _inst_mean = float(np.mean(_instrumented_per_call))

        # The ABBA ratio: the mean of the per-block, drift-cancelling ratios.
        # NOT instrumented_mean / clean_mean, which charges any drift over the
        # row to the instrumentation (measured: that estimator returned 0.71 to
        # 1.37 across six rows of one leg, on a quantity gated at 1.03).
        _block_ratios = _abba["block_ratios"]
        _overhead_ratio = float(np.mean(_block_ratios))
        # The gated quantity, in absolute milliseconds: the ratio's excess over 1
        # applied to THIS row's own measured clean mean. `_clean_mean` is in
        # seconds and is the same mean the row publishes as `call_ms`.
        _overhead_ms = (_overhead_ratio - 1.0) * _clean_mean * 1e3
        _overhead_assertable = _abba["n_blocks"] >= MIN_BLOCKS_FOR_OVERHEAD_ASSERT
        _overhead_status = (
            ("PASS" if _overhead_ms <= MAX_INSTRUMENTATION_OVERHEAD_MS else "FAIL")
            if _overhead_assertable
            else "RECORDED"
        )

        # Per-call exclusive time, then the honest remainder.
        _per_call_excl = {
            label: entry["excl_s"] / _n_instrumented for label, entry in _totals.items()
        }
        _attributed = float(sum(_per_call_excl.values()))
        _unattributed = _inst_mean - _attributed
        _unattributed_fraction = _unattributed / max(_inst_mean, 1e-300)

        # Rescale so the rows sum to the CLEAN call rather than the instrumented
        # one. The factor is written out so a reader can undo it.
        _rescale = _clean_mean / max(_inst_mean, 1e-300)

        _steps = {
            label: {
                "excl_s": value * _rescale,
                "excl_s_unscaled": value,
                "incl_s": _totals[label]["incl_s"] / _n_instrumented * _rescale,
                "n_calls": _totals[label]["n_calls"] / _n_instrumented,
                "group": call_accounting.site_group(label),
            }
            for label, value in _per_call_excl.items()
        }
        _steps[UNATTRIBUTED_LABEL] = {
            "excl_s": _unattributed * _rescale,
            "excl_s_unscaled": _unattributed,
            "incl_s": None,
            "n_calls": None,
            "group": "unattributed",
        }

        # Gate: every site the spec declares cached must be reached exactly once
        # per call. A cached site the library stopped caching moves milliseconds
        # between rows without changing any number a reader would look at.
        _cache_violations = {
            label: _totals[label]["n_calls"] / _n_instrumented
            for label in call_accounting.declared_cached_labels()
            if _totals[label]["n_calls"] not in (0, _n_instrumented)
        }

        _entry.update(
            {
                "decomposed": True,
                "call_ms_instrumented": _inst_mean * 1e3,
                "call_ms_instrumented_sequence": [t * 1e3 for t in _instrumented_per_call],
                "n_instrumented_calls": _n_instrumented,
                "abba": {
                    "n_blocks": _abba["n_blocks"],
                    "block_ratios": _block_ratios,
                    "ratio_mean": _overhead_ratio,
                    "ratio_median": float(np.median(_block_ratios)),
                    "ratio_min": float(np.min(_block_ratios)),
                    "ratio_max": float(np.max(_block_ratios)),
                    "design": "A B B A per block; ratio = mean(B) / mean(A)",
                    "why": (
                        "A B B A straddles the two instrumented calls with one clean call "
                        "either side, so a drift linear in call index cancels exactly in "
                        "the ratio. A sequential 'all clean then all instrumented' design "
                        "charges the whole drift over the row to the instrumentation: "
                        "measured on this configuration it returned 0.71, 0.98, 1.10, "
                        "1.11, 1.24 and 1.37 across six rows of one leg — including "
                        "ratios below 1, which no real instrumentation can produce."
                    ),
                },
                "instrumentation_overhead_ratio": _overhead_ratio,
                "instrumentation_overhead_ms": _overhead_ms,
                "instrumentation_overhead_status": _overhead_status,
                "instrumentation_overhead_assertable": _overhead_assertable,
                "instrumentation_overhead_threshold_ms": MAX_INSTRUMENTATION_OVERHEAD_MS,
                "instrumentation_overhead_reference_ratio": REFERENCE_OVERHEAD_RATIO,
                "instrumentation_overhead_gate_note": (
                    "The gate is the MILLISECOND budget, not the ratio. The instrument's "
                    "cost is a fixed number of wrapper invocations per call, so a ratio "
                    "gate tightens as the call gets shorter: 1.0147 at a 413 ms call is "
                    "6.1 ms of instrument and passed, 1.0366 at a 224 ms call is 8.2 ms "
                    "and was killed. The ratio is recorded beside the milliseconds so "
                    "the rows of levers 1-3 can still be read against it."
                ),
                "min_blocks_for_overhead_assert": MIN_BLOCKS_FOR_OVERHEAD_ASSERT,
                "decomposition_rescale_factor": _rescale,
                "attributed_ms": _attributed * 1e3,
                "unattributed_ms": _unattributed * 1e3,
                "unattributed_fraction": _unattributed_fraction,
                "steps": _steps,
                "n_sites_instrumented": len(_totals),
                "n_sites_reached": sum(1 for entry in _totals.values() if entry["n_calls"] > 0),
                "cached_site_call_count_violations": _cache_violations,
                "solver_stats": {
                    label: {k: v for k, v in stats.items() if k != "passive_set"}
                    for label, stats in _abba["solver_stats"].items()
                },
                "operated_mapping_matrix_memo_entries_after": (
                    _operated_mapping_matrix_memo_size()
                ),
            }
        )

        print(
            f"  instrumented x{_n_instrumented}: {_inst_mean * 1e3:.3f} ms; "
            f"ABBA overhead {_overhead_ms:+.2f} ms (x{_overhead_ratio:.4f}) "
            f"[{_overhead_status}] of a {MAX_INSTRUMENTATION_OVERHEAD_MS:.1f} ms budget "
            f"(blocks {[round(r, 4) for r in _block_ratios]})"
        )
        print(
            f"  attributed {_attributed * 1e3:.3f} ms, unattributed "
            f"{_unattributed * 1e3:.3f} ms ({_unattributed_fraction * 100:.2f} %) "
            f"over {_entry['n_sites_reached']}/{_entry['n_sites_instrumented']} sites"
        )
        if not _overhead_assertable:
            print(
                f"  overhead RECORDED, not asserted: {_abba['n_blocks']} block(s) < "
                f"{MIN_BLOCKS_FOR_OVERHEAD_ASSERT}; clean-call spread "
                f"{_clean_spread * 100:.1f} % ({_clean_spread * _clean_mean * 1e3:.2f} ms) "
                f"is the noise floor the "
                f"{MAX_INSTRUMENTATION_OVERHEAD_MS:.1f} ms budget would have to beat. "
                f"Re-run with --n-repeats "
                f"{2 * MIN_BLOCKS_FOR_OVERHEAD_ASSERT} or more to assert it."
            )

        if _cache_violations:
            raise AssertionError(
                f"row {_key}: sites declared cached were reached a different number of "
                f"times than once per call: {_cache_violations}. The decomposition's "
                f"meaning has changed and the spec must be updated on purpose."
            )
        if _overhead_status == "FAIL":
            raise AssertionError(
                f"row {_key}: ABBA instrumentation overhead {_overhead_ms:.2f} ms > "
                f"{MAX_INSTRUMENTATION_OVERHEAD_MS} ms (ratio {_overhead_ratio:.4f} on a "
                f"{_clean_mean * 1e3:.3f} ms clean call) over {_abba['n_blocks']} "
                f"counterbalanced blocks {[round(r, 4) for r in _block_ratios]} — the "
                f"harness is changing the number it is measuring. The budget is absolute "
                f"because the instrument's cost is: ~40 descriptor wrappers per call, "
                f"measured at 6-8 ms on every recorded row of this cell."
            )
        if _unattributed_fraction > MAX_UNATTRIBUTED_FRACTION:
            raise AssertionError(
                f"row {_key}: {_unattributed_fraction * 100:.2f} % of the call is "
                f"unattributed (> {MAX_UNATTRIBUTED_FRACTION * 100:.0f} %). The site spec "
                f"does not cover this formalism's call graph."
            )

    if _injection is not None:
        _injection.__exit__(None, None, None)
        _last_stats = _injection_counts["last_stats"] or {}
        _entry["injected_solver"] = {
            "kernel": "likelihood_breakdown.fixed_light_numpy_solvers.nnls_factor_reuse",
            "dotted_name_patched": fixed_light_numpy_solvers.LIBRARY_POSITIVE_ONLY_DOTTED,
            "installed_outside_call_accounting": True,
            "n_calls_numpy": _injection_counts["numpy"],
            "n_calls_jax": _injection_counts["jax"],
            "last_solve_stats": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in _last_stats.items()
                if k != "passive_set"
            },
        }
        print(
            f"  injected solver: {_injection_counts['numpy']} numpy call(s), "
            f"{_injection_counts['jax']} jax call(s); last solve "
            f"{_entry['injected_solver']['last_solve_stats']}"
        )
        if _injection_counts["numpy"] <= 0:
            raise AssertionError(
                f"row {_key}: the injected factor-reuse solver was never called "
                f"({_injection_counts['numpy']} numpy calls). This row's milliseconds "
                f"are route b's, wearing route d_np's label."
            )

    if _kernel_injection is not None:
        _kernel_injection.__exit__(None, None, None)
        _entry["injected_kernel"] = {
            "kernel": _kernel_injection_counts["dotted_name"],
            "kernel_name": _kernel_injection_counts["kernel"],
            "dotted_name_patched": fixed_light_numpy_kernels.LIBRARY_CURVATURE_DISPATCHER_DOTTED,
            "installed_outside_call_accounting": True,
            "n_calls": _kernel_injection_counts["n_calls"],
            "last_pix_pixels": _kernel_injection_counts["last_pix_pixels"],
        }
        print(
            f"  injected curvature kernel: "
            f"{_kernel_injection_counts['kernel']} — "
            f"{_kernel_injection_counts['n_calls']} call(s), last pix_pixels "
            f"{_kernel_injection_counts['last_pix_pixels']}"
        )
        if _kernel_injection_counts["n_calls"] <= 0:
            raise AssertionError(
                f"row {_key}: the injected curvature kernel "
                f"{_kernel_injection_counts['kernel']!r} was never called "
                f"({_kernel_injection_counts['n_calls']} calls). This row's milliseconds "
                f"are route b's, wearing route {_route}'s label."
            )

    rows[_key] = _entry

_clear_memos()
load_average_at_end = _load_average()


# ===================================================================
# PART D — Summary, JSON, PNG
# ===================================================================

al_version = al.__version__

print("\n" + "=" * 70)
print(f"NUMBA CPU FIXED-LENS-LIGHT — {instrument.upper()} / {MESH} — v{al_version}")
print("=" * 70)

_name_width = max(len(key) for key in rows) if rows else 10
for _key, _entry in rows.items():
    _line = f"  {_key:<{_name_width}}  {_entry['call_ms']:>10.3f} ms"
    if _entry["decomposed"]:
        _line += (
            f"   unattributed {_entry['unattributed_fraction'] * 100:>5.2f} %"
            f"   ABBA overhead {_entry['instrumentation_overhead_ms']:+.2f} ms"
            f" (x{_entry['instrumentation_overhead_ratio']:.4f})"
            f" [{_entry['instrumentation_overhead_status']}]"
        )
    print(_line)

# The group table of the first decomposed row, as a readable summary.
for _key, _entry in rows.items():
    if not _entry["decomposed"]:
        continue
    print(f"\n  --- decomposition [{_key}] ---")
    _by_group: dict[str, float] = {}
    for _label, _step in _entry["steps"].items():
        _by_group[_step["group"]] = _by_group.get(_step["group"], 0.0) + _step["excl_s"]
    for _group, _seconds in sorted(_by_group.items(), key=lambda kv: -kv[1]):
        _pretty = GROUP_LABELS.get(_group, _group)
        print(f"    {_pretty:<44} {_seconds * 1e3:>10.3f} ms")

device_block = {
    "use_jax": False,
    "backend": "numba_cpu",
    "inversion_paths": sorted({entry["inversion_class"] for entry in rows.values()}),
    "hostname": socket.gethostname(),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS") or None,
    "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS") or None,
    "cpu_count": os.cpu_count(),
    "note": (
        "Cell-local. _profile_cli.device_info_dict() is deliberately NOT called: it "
        "imports jax unconditionally (_profile_cli.py:392), and a JAX runtime in this "
        "process changes the thread pools the numba rows are measured through."
    ),
}

configuration = {
    "mesh": MESH,
    "dataset": DATASET,
    "pixel_scale_arcsec": pixel_scale,
    "mask_radius_arcsec": mask_radius,
    "image_pixels_masked": int(n_image_pixels),
    "over_sampled_pixels": int(n_over_sampled_pixels),
    "source_pixels": int(n_source_pixels),
    "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
    "routes_selected": list(ROUTE_SELECTION),
    "formalisms_selected": list(FORMALISM_SELECTION),
    "row_order": ROW_ORDER,
    "n_threads": N_THREADS,
    "n_repeats": N_REPEATS,
    "instances": INSTANCE_MODE,
    "iid_seed": _IID_SEED if INSTANCE_MODE == "iid" else None,
    "decompose": DECOMPOSE,
    "pins_mode": PINS_MODE,
    "use_jax": False,
    "thread_env": thread_env,
    "numba_thread_env": numba_thread_env,
    "nnls_warm_start_mode": NNLS_WARM_START,
    "nnls_warm_start_env": os.environ.get("AUTOARRAY_NNLS_WARM_START"),
    "nnls_warm_start_env_preexisting": _nnls_warm_start_before,
    "memo_cleared_between_rows": True,
    "memo_cleared_within_block": not NNLS_WARM_START_ON,
    "memo_note": (
        "The between-row clears are unconditional; only the WITHIN-BLOCK clears are "
        "gated by --nnls-warm-start. With the memo on, a block's calls seed each other "
        "(the sampler's regime); with it off every call is a cold dense-sign solve."
    ),
    "over_sample_size_lp_rule": {
        "sub_size_list": [4, 2, 2],
        "radial_list": [0.3, 0.6],
        "centre": [0.0, 0.0],
    },
}

# ---------------------------------------------------------------------------
# Contention and timing status — FIRST-CLASS DATA, not a footnote
# ---------------------------------------------------------------------------
# Two things a reader must not have to infer:
#
# 1. Whether the machine was busy. Every number here is wall clock on a shared
#    host; a load average above the core count means the row was queueing for
#    those cores. Recorded at both ends of the timed section.
# 2. Whether the run was ever MEANT to produce quotable numbers. A smoke leg
#    exists to prove the wiring and to make the gates fire; its milliseconds are
#    not a measurement of anything. That is stated in the artifact, in a field, so
#    a future reader cannot mistake one for the other by skipping a note.

_cores = os.cpu_count() or 1
_load_peak = max(
    [la[0] for la in (load_average_at_start, load_average_at_end) if la is not None] or [0.0]
)
_contended = _load_peak > _cores

#: A config name containing "smoke" declares the run's own purpose.
_is_smoke = bool(_cli.config_name and "smoke" in _cli.config_name.lower())

if _is_smoke:
    _timing_status = "wiring_only_not_measured"
    _timing_status_reason = (
        "This leg ran under a --config-name containing 'smoke'. Its purpose is to prove "
        "the cell end to end and to make the gates fire, NOT to produce timings. The "
        "call_ms values here are wiring evidence: do not quote them, do not put them in "
        "a note, do not compare them against another leg. The gate outcomes under "
        "`gates` and the per-row structural checks ARE valid — they are exact "
        "comparisons, not wall-clock measurements."
    )
elif _contended:
    _timing_status = "measured_under_contention"
    _timing_status_reason = (
        f"Load average reached {_load_peak:.2f} on {_cores} cores during the timed "
        f"rows, so these calls competed for the cores they were measured on. The "
        f"numbers are real but carry the host's queueing; a thread-scaling comparison "
        f"in particular (t1 vs t8) is actively misleading under oversubscription, "
        f"because the t1 leg's spare cores are not spare."
    )
else:
    _timing_status = "measured"
    _timing_status_reason = (
        f"Load average stayed at or below the core count ({_load_peak:.2f} on "
        f"{_cores} cores) across the timed rows."
    )

print(f"\n  timing_status: {_timing_status}")
if _timing_status != "measured":
    print(f"    {_timing_status_reason}")

breakdown_summary = {
    "autolens_version": al_version,
    "timing_status": _timing_status,
    "timing_status_reason": _timing_status_reason,
    "contention": {
        "load_average_at_start": load_average_at_start,
        "load_average_at_end": load_average_at_end,
        "cpu_count": _cores,
        "load_average_peak_1min": _load_peak,
        "contention_warning": _contended,
        "contention_note": (
            "load_average_at_* are /proc/loadavg's 1/5/15-minute figures, read either "
            "side of the timed rows. contention_warning is the 1-minute figure "
            "exceeding the core count: the rows were competing for the cores they were "
            "timed on, which widens the per-call scatter and can inject multi-call "
            "transients. It does not affect any gate under `gates` — those are exact "
            "comparisons of numbers, not of times."
        ),
    },
    "device": device_block,
    "machine": machine_info_dict(),
    "instrument": instrument,
    "configuration": configuration,
    "regularization": reg_provenance,
    "all_route_keys": list(ALL_ROUTE_KEYS),
    "default_route_keys": list(DEFAULT_ROUTE_KEYS),
    "all_formalism_keys": list(ALL_FORMALISM_KEYS),
    "systems": {
        "s0": {
            "n_params": int(system_s0.n_params),
            "n_mapper": int(system_s0.n_mapper),
            "n_funcs": int(system_s0.n_funcs),
            "edge_zeroed_pixels": int(system_s0.edge_zero_mask.sum()),
            "log_evidence_eager": log_evidence_s0,
        },
        "s3": {
            "n_params": int(system_s3_dense.n_params),
            "n_mapper": int(system_s3_dense.n_mapper),
            "n_funcs": int(system_s3_dense.n_funcs),
            "edge_zeroed_pixels": int(system_s3_dense.edge_zero_mask.sum()),
            "subtracted_light_flux": float(system_s3_dense.subtracted_light_flux),
            "log_evidence_eager_dense": _fom_s3_dense,
            "log_evidence_eager_sparse_numba": _fom_s3_sparse,
        },
    },
    "gates": gates,
    "gate_thresholds": {
        "P2_rtol": P2_RTOL,
        "P3_rtol": P3_RTOL,
        "P4_rtol": P4_RTOL,
        "mapper_logdet_rtol": MAPPER_LOGDET_RTOL,
        "max_instrumentation_overhead_ms": MAX_INSTRUMENTATION_OVERHEAD_MS,
        "reference_overhead_ratio": REFERENCE_OVERHEAD_RATIO,
        "min_blocks_for_overhead_assert": MIN_BLOCKS_FOR_OVERHEAD_ASSERT,
        "max_unattributed_fraction": MAX_UNATTRIBUTED_FRACTION,
        "warmup_window": WARMUP_WINDOW,
        "warmup_tolerance": WARMUP_TOLERANCE,
        "warmup_max_calls": WARMUP_MAX_CALLS,
    },
    "route_c": route_c_block,
    "rows": rows,
    "decomposition_method": (
        "likelihood_breakdown.call_accounting wraps ~40 named library descriptors and "
        "three module-level solver functions and the PRODUCTION "
        "AnalysisImaging.log_likelihood_function is then run untouched. Each row is "
        "inclusive time, exclusive (self) time and the number of times the library "
        "reached the site; exclusive times are additive by construction, so "
        "'unattributed' is a measured remainder and not an estimate. It is always an "
        "explicit row. Rows are rescaled by decomposition_rescale_factor (clean / "
        "instrumented) so they sum to the quoted, uninstrumented call. The clean and "
        "instrumented calls are COUNTERBALANCED (A B B A per block), so the overhead "
        "ratio is free of any drift linear in call index, and the warm-up runs to a "
        "steady state with its whole per-call sequence recorded under rows[*].warmup."
    ),
    "group_labels": GROUP_LABELS,
    "no_pin_note": (
        "NO PIN WAS EVER CALIBRATED HERE. No numba fixed-light configuration has a "
        "pinned log likelihood, log evidence or log determinant anywhere in this repo, "
        "so every numerical quantity in this JSON is RECORDED, not asserted, and "
        "pinned_expected is null. The only hard verdicts this leg carries are the "
        "structural gates under `gates` (P1, P2, P3, the S3-equals-S0 mapper block, the "
        "per-row dispatch, the cached-access counts, the instrumentation overhead and "
        "the coverage), none of which is a calibrated number."
    ),
}

_cell_name = f"fixed_light_numba_{MESH}"
if DATASET != "hst":
    _cell_name = f"{_cell_name}_{DATASET}"
if SOURCE_PIXELS_REQUESTED is not None:
    _cell_name = f"{_cell_name}_n{int(n_source_pixels)}"

# `cell=` is passed EXPLICITLY. Without it `resolve_output_paths` derives the
# cell name from the first underscore-separated token of the basename — here
# `fixed`, which would collide with `fixed_light*`'s artifacts under any shared
# --config-name. `_profile_cli.py:558-562` documents this exact bug happening
# before (autolens_profiling#219).
dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"{_cell_name}_breakdown_{instrument}_v{al_version}",
    cell=_cell_name,
)
dict_path.write_text(json.dumps(breakdown_summary, indent=2, default=str))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_labels = list(rows)
_times_ms = [rows[key]["call_ms"] for key in _labels]
_colors = []
for _key in _labels:
    if _key.startswith("a_"):
        _colors.append("#C44E52")
    elif _key.startswith("b_"):
        _colors.append("#DD8452")
    elif _key.startswith("d_np_"):
        _colors.append("#4C72B0")
    else:
        _colors.append("#55A868")

fig, ax = plt.subplots(figsize=(11, max(4.0, 0.6 * len(_labels))))
_y = np.arange(len(_labels), dtype=float)
_bars = ax.barh(_y, _times_ms, color=_colors, edgecolor="white", height=0.6)
for _bar, _t in zip(_bars, _times_ms):
    ax.text(
        _bar.get_width() + (max(_times_ms) if _times_ms else 1.0) * 0.01,
        _bar.get_y() + _bar.get_height() / 2,
        f"{_t:.2f} ms",
        va="center",
        fontsize=8,
    )
ax.set_yticks(_y)
ax.set_yticklabels(_labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Whole numba CPU likelihood call (ms)", fontsize=11)
fig.suptitle(
    f"Numba CPU fixed-lens-light likelihood — {MESH} — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_source_pixels} source pixels  |  {N_THREADS} thread(s)",
    fontsize=9,
)
ax.margins(x=0.20)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

# ===================================================================
# Pinned-value record — there is no pin
# ===================================================================
# `expected=None` is the honest state: no numba fixed-light configuration has
# ever been calibrated, so there is nothing to drift from. Recording it
# explicitly (rather than omitting the block) is what stops a reader — or
# PyAutoHeart's profiling-drift scan — from reading an absent block as a silent
# pass.

record_pinned_check(dict_path, None, [])
print(
    "\n  Pinned check: NO PIN WAS EVER CALIBRATED HERE "
    "(pinned_expected: null; every numerical quantity above is RECORDED)."
)

print("\nFinished.")
