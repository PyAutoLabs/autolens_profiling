"""
JAX Profiling: Fixed Lens Light — the XLA DEVICE TIMELINE of the production call
================================================================================

Phase 1 of the ``hst-gpu-non-solver-residue`` epic (autolens_profiling#268).

The ``fixed-lens-light-profiling`` epic ended with ~21 of the 25.4 ms certified
Delaunay A100 call at HST N=1500 **not** being the solver. But its largest term
— "~13.9 ms mesh / mapper / weights / imaging / blurring" — is attribution
*arithmetic* across two cells: phase-0 kernel rows subtracted from a phase-1
whole call. Every JAX decomposition this repo owns (``--split-setup``) is ~11
**separately compiled** prefix programs whose rows move with fusion; PyAutoArray
#531 made one of them negative. Nobody has ever measured where the *fused
production program* spends its device time.

This cell measures it. One program, one process, one timeline.

What it does
------------

1. Builds the same S3 system every cell in the family builds (lens light fixed
   after SLaM ``light[1]``, converted to regular profiles and subtracted), and
   times route ``b`` (library PDIP) and route ``d`` (certified active set at the
   PRODUCTION budget 7, PDIP fallback) exactly as ``fixed_light_library.py``
   does — so the whole-call wall reconciles with phase 5's row.
2. Lowers route ``d`` **once** and compiles that one lowering twice: with CUDA
   command buffers as XLA ships them (the production program) and with them
   disabled (the traceable program). Times both.
3. Traces ``--trace-calls`` steady calls of the command-buffers-OFF executable
   with ``jax.profiler``, joins every device kernel to its HLO instruction and
   every instruction to its library source stack, and writes a per-stage
   millisecond table that sums to the wall.
4. Takes the optimized HLO census: how many ``(n,n)`` adds of ``F + lambda*H``
   survive XLA, how many ``[ids][:, ids]`` gathers, and whether the PSF FFT of
   the mapping-matrix cube is compiled once or twice.

The mechanics of the join — the stack frame index, the ``requires`` rules that
tell the two log determinants apart, the mixed-fusion row — are documented in
``likelihood_breakdown.xla_attribution``, not repeated here.

Command buffers: what the stage rows actually attribute
-------------------------------------------------------

**The stage rows attribute the graph-less program.** With XLA's defaults the GPU
module is captured into a CUDA graph and every kernel reports
``hlo_op="command_buffer_1"``, which destroys the join (23 % of device duration
recoverable on the spike, against 100 % with the graphs off). So the traced
executable is compiled with ``xla_gpu_enable_command_buffer=""``.

That is a different program from the one production runs, and this cell refuses
to hide the difference. It times **both** executables — ``wall_ms`` per route
for command buffers ON (the production number, the one comparable to phase 5's
25.39 ms row) and OFF (the traced program) — and records
``command_buffer_delta_ms`` and its percentage. Kernel time is not what the
graphs change; launch overhead is, so the delta lands in the ``device_idle`` and
``host_outside_span`` rows rather than in any stage. Reconciliation is computed
**against the OFF wall**, because that is the program the kernels came from.

A short probe also traces the ON executable and records whether CUPTI still
emits per-kernel events inside the graph (``command_buffer_probe``), so the
size of the delta can be judged rather than assumed.

Flags (beyond the shared ``_profile_cli`` set)
----------------------------------------------

``--mesh {delaunay,delaunay_nn,rectangular}`` (required) — which cell's model to
rebuild.

``--border-relocator {cell,library,off}`` (default ``cell``) — ``cell`` passes
``use_border_relocator=True``, which every profiling cell in this family forces;
``library`` passes ``None`` and takes the shipped config default; ``off`` passes
``False``. The **resolved** boolean is recorded and is part of the output
filename, because on this workspace ``library`` resolves to **True**
(``config/general.yaml`` declares no ``inversion`` block, so autoconf falls
through to the packaged default) — so ``cell`` and ``library`` are the same
program here, and ``off`` is the leg that measures what the relocator costs.

``--trace-calls K`` (default 10) — steady calls taken inside the profiler.
``--source-pixels N`` — mesh size (fiducial 1500 for the Delaunay family).
``--routes b,d`` (default) — which routes are built and timed; the trace is
always taken on route ``d``, which must therefore be selected.
``--command-buffers {off,on}`` (default ``off``) — which executable the trace is
taken on. ``on`` is the diagnostic that shows the join failing; it is not a
configuration in which any stage row should be quoted.
``--trace-dir`` — where the profiler writes its xplane (a temporary directory by
default; traces are large and do not belong in the repo).

The batched mode (phase 2, autolens_profiling#273)
--------------------------------------------------

``--vmap-batch B`` turns this cell into the **matched vmap-vs-scalar**
experiment. Without it nothing below happens and the cell is phase 1 exactly,
down to the output filename.

Production does not run one likelihood at a time. Current ``Fitness._vmap``
(``PyAutoFit/autofit/non_linear/fitness.py``, PyAutoFit#1638) is
``jax.jit(jax.vmap(call))``,
and Nautilus drives it with ``use_jax_vmap=True``. The Delaunay qhull callback
is ``vmap_method="sequential"`` (``PyAutoArray .../interpolator/delaunay.py``),
so the host round trip phase 1 measured as a single 5.44 ms device-idle gap
runs **once per lane** and is the one term that cannot amortise over a batch.
Nobody had traced that program.

So this mode builds ``B`` lane parameter trees and runs **two arms on the same
lanes, in one process, under the same solver injection**:

``vmap``
    ``jax.jit(jax.vmap(fn))`` — the current production nesting. Array 343376
    measured the retired ``jax.vmap(jax.jit(fn))`` composition; those artifacts
    retain their original filenames and provenance and are not relabelled.
``scalar``
    ``jax.jit(fn)`` called ``B`` times per batch, each blocked. This stands for
    ``Nautilus(use_jax_vmap=False, use_jax_jit=True)``; plain
    ``use_jax_vmap=False`` is UNJITTED and is not what this arm measures.

``--lanes distinct`` (default) makes lane *k* the seeded random draw *k* of the
phase-3 draw set (``fixed_light_draws_steps.random_draws``) applied to the lens
mass, with the lens light fixed at S3 — a batch of *different* models, which is
what a search evaluates. ``--lanes identical`` broadcasts the fiducial and is
the **control**: identical lanes share a straggler and hide exactly the cost
this experiment is looking for.

``--fallback on`` (default) keeps the ``lax.cond`` PDIP fallback — under
``vmap`` the batched predicate turns it into a ``select`` and both branches
run, so that row is the certified solve *plus* PDIP. ``--fallback off`` is the
cond-free program. Both are rows, never a footnote.

Per lane the two arms' log likelihoods are pinned equal at **1e-9 relative**,
and each is checked against an independently compiled library-PDIP batch at the
same tolerance. The cell exits non-zero if any required pin fails: agreement
between two harness arms alone does not prove that either evaluates the library's
positive solution.

``--arms {vmap,scalar,both}`` (default ``both``) and ``--draw-seed`` (default
0) complete the set. Every one of these flags is recorded in
``configuration``; strict shared CLI parsing also rejects an old checkout that
does not define them.

The library-solver mode (certified-solver phase B, autolens_profiling#300)
--------------------------------------------------------------------------

``--solver-source library`` (with ``--vmap-batch``) replaces phase 2's harness
injection with the **library's own dispatch**: PyAutoArray #567 shipped the
certified active-set solver behind ``Settings(positive_only_solver="certified",
certified_fallback="pdip"|"none", certified_pass_budget=N)``, and this mode is
what production would actually run under ``Fitness._vmap``. Without the flag
(``--solver-source harness``, the default) nothing below happens and the batched
mode is phase 2 exactly, down to the output filename and the JSON key set.

- ``--solver {pdip,certified}`` (required in this mode) is
  ``Settings.positive_only_solver`` for both arms; ``--fallback on|off`` maps to
  ``certified_fallback`` ``pdip`` | ``none``; ``--certified-budget N`` is
  ``certified_pass_budget`` (default: the resolved config value, 16 as packaged).
  None of the three is accepted in harness mode, where they would be swallowed.
- Nothing is monkeypatched in PART V. Instead the cell asserts that the fiducial
  S3 inversion's ``positive_only_solver_used`` equals the requested solver, and
  that every JAX-path solve traced in the reporting pass received that
  ``solver``: a row that asked for ``certified`` and silently ran PDIP fails.
- PART B (routes ``b`` and ``d``, the fiducial ``d == b`` pin) is unchanged: it
  is phase-1 context in every mode, not a phase-B row.
- Per-lane ``certified`` / ``passes`` come from the library's own ``stats=``
  out-dict (#566), read in an UNTIMED, separately compiled reporting pass through
  ``library_solver_injection.library_solver_observed`` — a forwarding wrapper
  that changes nothing computed. ``uncertified_lanes`` counts the lanes whose
  certificate failed (the lanes that fell back, or with ``none`` returned an
  uncertified iterate).
- Rectangular runs here as well as Delaunay. It has no qhull callback: the cell
  asserts ZERO callbacks per arm and ``host_qhull_ms_per_lane`` is 0.

**The phase-B gate, PRE-REGISTERED before any A100 submission.** Phase 3
(``results/notes/hst_gpu_residue_phase3_psf_2026_09.md``) placed phase 2's
~2.5e-9 certified-vs-PDIP residual *between the compositions* — ``jit(vmap)``
against scalar ``jit`` — not in the solver. So library mode compiles two
library-PDIP references, one per composition, and per lane:

1. GATED at ``LANE_RTOL`` (1e-9 relative): the ``jit(vmap)`` arm against the
   ``jit(vmap)`` library-PDIP reference;
2. GATED at ``LANE_RTOL``: the scalar ``jit`` arm against the scalar ``jit``
   library-PDIP reference;
3. RECORDED, not gated: the ``jit(vmap)`` arm against the scalar arm, and the two
   references against each other, as relative differences and absolute nats;
4. RECORDED: ``uncertified_lanes`` for a certified row. A
   ``certified_fallback=none`` row is ``policy_eligible`` only if every lane
   certifies AND passes (1) and (2).

Timing is never gated. As in phase 2 the JSON and PNG are written first and the
cell then raises if any gated lane failed. Harness mode keeps phase 2's
three-way gate unchanged.

The captured-lane mode (certified-solver phase C1, autolens_profiling#304)
-------------------------------------------------------------------------

``--lanes captured --batches <npz>`` (library mode only) replaces phase B's
seeded draw family with the lanes of a REAL Nautilus run, recorded by
``nautilus_batch_capture.py`` at the production ``source_pix[1]`` settings
(``n_live=150``, ``n_batch=20``). A captured lane is the mass AND shear of one
proposal's physical parameter vector — mass, shear AND regularization for a
``pix1`` capture, the regularization alone for ``pix2`` — evaluated the way
``Fitness.call`` evaluates it (``_captured_likelihood_fn``: the capture's model,
rebuilt by the same ``nautilus_batches.capture_model_from``, instantiated from the
vector inside the trace), so every lane's own regularization reaches the
likelihood in both passes; lens light fixed at S3 as in every lane of this cell.
The file is refused unless its recorded S3 fingerprint — dataset sha256s, mesh,
source pixels, the regularization CLASS the capture freed
(``nautilus_batches.regularization_class_for(mesh, stage)``: AdaptSplit on Delaunay,
Constant / Adapt on rectangular pix1 / pix2) and the S0 fiducial scheme and
coefficients that built S3, border relocator, precision and the eager S3 figure of merit
(``EQUIVALENCE_RTOL``) — matches the system built here, and unless the rebuilt
model's parameter paths equal the capture's, in order. The per-lane
regularization coefficients are NOT identity: they are what the lanes vary. Without ``--lanes captured`` nothing below happens, and ``--batches``,
``--batch-sample`` and ``--captured-pass`` are rejected.

``--captured-pass time`` (default) is the phase-B harness exactly — PART V, both
arms, both per-composition references and the same PRE-REGISTERED gate at
``LANE_RTOL`` (1e-9, no tighter: Delaunay has a ~2e-10 run-to-run floor and a
~2.5e-9 cross-composition residual) — on ``B`` captured lanes chosen by
``nautilus_batches.timed_window``: consecutive captured calls from the first
post-prior batch, so at ``B == n_batch`` the batch IS one real Nautilus batch.
The JSON gains ``vmap.captured`` (the file, its fingerprint check, and the
window's library-PDIP values against the capture's own figures of merit —
RECORDED, it proves the rebuilt trees are the models the sampler evaluated).

``--captured-pass rate`` (needs ``--solver certified``) is PART R: UNTIMED, it
runs every captured lane — or every lane of ``--batch-sample K`` calls evenly
spaced over the run — through the library's certified solve under
``jax.jit(jax.vmap(fn))`` in chunks of ``--vmap-batch`` (the last chunk padded
by repeating its final lane, padded rows dropped), observed with
``library_solver_observed``. It reports the uncertified-lane rate and the pass
distribution against the budget, overall, by Nautilus phase and by call-index
half (the split rules are stated in ``nautilus_batches``: phase = the sampler's
bound count and ``explored`` flag when the batch was requested; early = call
index below the median replayed call). RECORDED beside it: every lane's certified
value against the capture's library-PDIP figure of merit, and every uncertified
lane (up to ``RATE_PDIP_RECHECK_MAX``) re-run through scalar library PDIP — what
a guard's second pass would return. It writes its own JSON after PART B and
exits; PART C and PART V do not run.

The PSF-cube mode (phase 3, autolens_profiling#295)
---------------------------------------------------

``--psf-candidate NAME`` turns this cell into the **PSF mapping-matrix cube**
experiment. Without it nothing below happens and the cell is phase 1 exactly.
It runs on the single-call path only and is rejected together with
``--vmap-batch`` (one experiment per invocation).

Phase 1 found the PSF convolution of the mapping-matrix cube to be the largest
computation of the fused call (7.12 ms of 31.64 ms on the A100). This mode
swaps that convolution for a candidate — a scoped harness rebinding of
``Convolver.convolved_mapping_matrix_from``
(``likelihood_breakdown.psf_cube_injection``) — **inside the same fused whole-call
jit**, and changes nothing else:

- route ``d`` (PART B) and the traced compile (PART C) run under the PSF
  injection **nested inside** the certified-solver injection;
- route ``b`` runs outside both — library PDIP and library convolution, the
  unmodified library answer every pin is taken against (``--routes`` must
  include ``b``);
- the injected convolution must actually be taken on the JAX path
  (``counts["jax"] > 0`` for every non-control candidate), or the row would be
  the library's convolution wearing the candidate's label.

The candidates (``psf_cube_injection.CANDIDATES``): ``control`` (no patch — the
library FFT path as shipped), ``frame_pow2``, ``layout_src_first``,
``real_space_direct``, ``conv_cudnn_batched``, and two DIAGNOSTIC precision rows,
``mp_cube_c64`` and ``c64_full``. **Diagnostic rows may not be quoted as
levers**: the campaign constraint is fp64 at a 1e-9 relative pin, and a
``fp64_exact=False`` candidate is measured for the milliseconds it would save and
the nats it costs, never promoted.

Pins. The fiducial ``d == b`` pin, plus ``--pin-draws N`` (default 8) seeded
distinct draws (``--draw-seed``, default 0 — the phase-2 lane family: the draw's
lens mass swapped into S3) evaluated by the SAME compiled route ``d`` and route
``b`` executables PART B timed. For a non-control candidate each draw is also
evaluated by route ``d`` compiled WITHOUT the PSF injection (certified solver,
library convolution — the "d-control"); ``rel_diff_vs_d_control`` isolates the
convolution from the certified-vs-PDIP solver difference.

The gate (``psf_cube_injection.psf_gate_rows``), **pre-registered before any
A100 data** and at ``EQUIVALENCE_RTOL`` (1e-9 relative) everywhere:

1. the fiducial pin, route ``d`` (candidate) vs route ``b`` (unmodified
   library), is GATED for every fp64 candidate, control included;
2. a non-control fp64 candidate's draw pins are GATED on
   ``rel_diff_vs_d_control``; each draw's ``rel_diff`` vs route ``b`` is
   recorded (``within_rtol_vs_b_library``), not gated;
3. ``control``'s draw pins are recorded vs route ``b``, not gated — there is no
   d-control to isolate against, and the fiducial gate already holds the
   library-answer constraint;
4. a diagnostic candidate's rows carry the relative error and the nats,
   labelled ``DIAGNOSTIC``, never gated, never PASS.

Why the draws are gated on the d-control: phase 2 (job 344635,
``results/notes/hst_gpu_residue_phase2_vmap_2026_09.md``) recorded a
2.0-2.6e-9 certified-vs-library-PDIP residual on seed-0 draw 7 on the A100. It
is a property of the SOLVER path (the RTX gives 2.7e-10 for the same draw), not
of the convolution; gating this phase's lever on it would fail every task,
control included, and measure nothing about the convolution. The
convolution-isolated pin tests what phase 3 changes; the fiducial pin against
the unmodified library holds the campaign's standard on every leg. Nothing was
relaxed after seeing A100 data — this was fixed before submission.

**The exit gate is ENFORCED in this mode**: the JSON and PNG are written first,
then the cell raises ``AssertionError`` if any gated fp64 pin failed, if ``|reconciliation_pct| > 5`` or if ``unjoined_ms > 0``. It never gates
on speed. The census also gains ``status`` / ``anchors`` /
``rows_excluded`` from ``xla_attribution.census_anchor_status()`` — the census
line anchors checked against the INSTALLED PyAutoArray — and a
``conv_mapping_matrix`` count for the real-space candidates.

Erratum (phase 1 prose and the phase-3 plan). ``mapping_matrix_native_from``
scatters straight into the PADDED FFT frame, not the image grid, and the source
axis is LAST: for this cell (masked dataset 141x141, 21x21 PSF) the frame
``ConvolverState`` builds is ``fft_shape = (180, 180)``, so the cube is
``(180, 180, 1500)`` fp64 (389 MB) — measured from ``psf_candidate.provenance``
on the RTX legs. It is neither ``(1500, 180, 180)`` nor the ``(200, 200, 1500)``
the phase-3 plan assumed. The ``ConvolverState`` docstring's "even FFT sizes are
incremented to odd sizes" note is STALE — the code does not do it.

Output
------

``results/breakdown/imaging/fixed_light_trace_<mesh>[_border_off][_n<N>][_jitvmap<B>_<lanes>_fb<on|off>][_psf_<candidate>]_<config>.{json,png}``.
Library mode replaces the batched token with
``_jitvmap<B>_<lanes>_lib<solver>_fb<on|off>_b<budget>`` (``<lanes>`` is
``captured_<stage>`` for a phase-C1 timed pass). The phase-C1 rate pass writes
``fixed_light_trace_<mesh>_captured_rate_lib<solver>_fb<on|off>_b<budget>_<stage>[_sample<K>]_<config>.json``
and no PNG.

The ``--config-name`` used for the phase-1 legs
(``local_rtx2060_fp64_fixed_light_trace``, ``hpc_a100_fp64_fixed_light_trace``)
sits **outside** ``build_readme.py``'s ``CONFIG_TAGGED_RE``, so none of this
reaches the dashboard: these are research rows for one note, not a tracked
profiling series.

What may not be quoted from this JSON
-------------------------------------

- A stage row as a *production* millisecond without saying it was measured on
  the command-buffers-OFF program, with ``command_buffer_delta_ms`` beside it.
- ``mixed_fusion.prorated_estimate_ms`` as a measurement. It is an equal split
  of a fused kernel across the stages its constituents span, and it is labelled
  ESTIMATE everywhere it appears.
- Any row at all if ``reconciliation_pct`` is outside 5 % or ``unjoined_ms`` is
  non-zero — those two say the join is incomplete, and an incomplete join
  produces a table that still adds up.
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
    _sys.path.insert(0, str(_misc_dir))


import argparse
import contextlib
import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autofit.jax import register_model as _register_model_pytrees

sys.path.insert(0, str(_profiling_root()))

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke). Verifies the import
# graph + module-level setup succeeded without running the profile. Placed
# before every argparse so a smoke run needs no flags at all.
import os as _smoke_os
import sys as _smoke_sys

from likelihood_breakdown import (  # noqa: E402
    active_set_steps,
    host_callback_probe,
    library_solver_injection,
    nautilus_batches,
    psf_cube_injection,
    timing,
    xla_attribution,
)
from likelihood_breakdown import (
    fixed_light_draws_steps as flds,
)

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _production_config import observe_thread_env as _observe_thread_env  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    delaunay_regularization,
    device_info_dict,
    machine_info_dict,
    parse_profile_cli,
    rect_mesh_classes,
    resolve_output_paths,
)

_cli = parse_profile_cli()

#: Phase 3's smallest **zero-fallback** pass budget per mesh — the budget a
#: PRODUCTION run fixes, and therefore the only budget this phase profiles.
#: ``fixed_light_library.CERTIFYING_BUDGET`` (2 for Delaunay) is phase 0's
#: fiducial and falls back on 67.5 % of a graded draw set; it is not a
#: production cost and is not what this cell measures.
PHASE3_SAFE_BUDGET = {"rectangular": 11, "delaunay": 7, "delaunay_nn": 7}

#: Source-pixel count each mesh's cell builds at its fiducial.
FIDUCIAL_SOURCE_PIXELS = {"rectangular": 39 * 39, "delaunay": 1500, "delaunay_nn": 1500}

#: Relative tolerance on the route d == route b equivalence pin. Inherited from
#: ``fixed_light_library.py``; the certified scheme is a different factorisation
#: order to PDIP, not a different answer.
EQUIVALENCE_RTOL = 1.0e-9

#: The command-buffer knob, passed per compilation so both programs live in one
#: process. ``""`` disables every command-buffer command type.
_COMMAND_BUFFER_OPTION = "xla_gpu_enable_command_buffer"

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument(
    "--mesh", choices=("rectangular", "delaunay", "delaunay_nn"), required=True
)
_cell_parser.add_argument("--border-relocator", choices=("cell", "library", "off"), default="cell")
_cell_parser.add_argument("--trace-calls", type=int, default=10)
_cell_parser.add_argument("--source-pixels", type=int, default=None)
_cell_parser.add_argument("--routes", default="b,d")
_cell_parser.add_argument("--command-buffers", choices=("off", "on"), default="off")
_cell_parser.add_argument("--trace-dir", default=None)
_cell_parser.add_argument("--safe-budget", type=int, default=None)
# --- phase 2 (#273): the matched vmap-vs-scalar mode -----------------------
# Every one of these is recorded into configuration when the mode is on, and
# NONE of them appears there when it is off, so a phase-1 JSON written by this
# checkout is key-for-key the JSON phase 1 wrote.
_cell_parser.add_argument("--vmap-batch", type=int, default=None)
_cell_parser.add_argument(
    "--lanes", choices=("distinct", "identical", "captured"), default="distinct"
)
_cell_parser.add_argument("--arms", choices=("vmap", "scalar", "both"), default="both")
_cell_parser.add_argument("--draw-seed", type=int, default=0)
_cell_parser.add_argument("--fallback", choices=("on", "off"), default="on")
# --- certified-solver phase B (#300): the LIBRARY-driven solver rows ---------
# ``harness`` (default) is phase 2 exactly. ``library`` selects the solver through
# ``al.Settings`` with no monkeypatch; it is only meaningful with --vmap-batch.
_cell_parser.add_argument("--solver-source", choices=("harness", "library"), default="harness")
_cell_parser.add_argument("--solver", choices=("pdip", "certified"), default=None)
_cell_parser.add_argument("--certified-budget", type=int, default=None)
# --- certified-solver phase C1 (#304): lanes replayed from a Nautilus capture --
# Only meaningful with --lanes captured (library mode); rejected otherwise.
_cell_parser.add_argument("--batches", default=None)
_cell_parser.add_argument("--batch-sample", type=int, default=None)
_cell_parser.add_argument("--captured-pass", choices=("time", "rate"), default=None)
# --- phase 3 (#295): the PSF mapping-matrix cube candidates -----------------
# The choices are a LITERAL (the contracts test execs these statements alone);
# test_psf_cube_injection pins them to psf_cube_injection.CANDIDATES.
_cell_parser.add_argument(
    "--psf-candidate",
    choices=(
        "control",
        "frame_pow2",
        "layout_src_first",
        "real_space_direct",
        "conv_cudnn_batched",
        "mp_cube_c64",
        "c64_full",
    ),
    default=None,
)
_cell_parser.add_argument("--pin-draws", type=int, default=None)
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh
BORDER_RELOCATOR_MODE = _cell_args.border_relocator
TRACE_CALLS = int(_cell_args.trace_calls)
TRACE_ON_COMMAND_BUFFERS = _cell_args.command_buffers == "on"

if TRACE_CALLS < 1:
    raise ValueError(f"--trace-calls must be >= 1 (got {TRACE_CALLS})")

#: ``None`` is phase 1. An int turns on the batched experiment (PART V).
VMAP_BATCH = _cell_args.vmap_batch
LANES_MODE = _cell_args.lanes
ARMS = ("vmap", "scalar") if _cell_args.arms == "both" else (_cell_args.arms,)
DRAW_SEED = int(_cell_args.draw_seed)
#: ``True`` is route d (``lax.cond`` PDIP fallback, a ``select`` under vmap);
#: ``False`` is route d0, the cond-free program.
FALLBACK_ON = _cell_args.fallback == "on"

#: Relative tolerance for both per-lane numerical gates: current production
#: ``jit(vmap)`` against B scalar ``jit`` calls, and each harness value against
#: the independently compiled library-PDIP reference on the same lane.
LANE_RTOL = 1.0e-9

if VMAP_BATCH is not None and VMAP_BATCH < 1:
    raise ValueError(f"--vmap-batch must be >= 1 when given (got {VMAP_BATCH})")
if VMAP_BATCH is None and _cell_args.arms != "both":
    raise ValueError(
        "--arms only means anything with --vmap-batch; without it this cell runs the "
        "phase-1 single-call trace and would silently ignore the flag"
    )

#: ``harness`` is phase 2 (the scoped monkeypatch). ``library`` is phase B
#: (#300): the solver is chosen through ``al.Settings`` and nothing is patched.
SOLVER_SOURCE = _cell_args.solver_source
LIBRARY_SOLVER_MODE = SOLVER_SOURCE == "library"
#: The library solver under test (library mode only).
LIBRARY_SOLVER = _cell_args.solver
#: ``Settings.certified_fallback``: ``--fallback on`` -> ``pdip``, ``off`` -> ``none``.
LIBRARY_FALLBACK = "pdip" if FALLBACK_ON else "none"

if LIBRARY_SOLVER_MODE:
    if VMAP_BATCH is None:
        raise ValueError(
            "--solver-source library runs in the batched mode only; pass --vmap-batch B "
            "(the single-call trace and the PSF mode keep the harness injection)"
        )
    if LIBRARY_SOLVER is None:
        raise ValueError(
            "--solver-source library needs --solver {pdip,certified}: the row must name the "
            "library solver it measures"
        )
else:
    for _flag, _value in (
        ("--solver", _cell_args.solver),
        ("--certified-budget", _cell_args.certified_budget),
    ):
        if _value is not None:
            raise ValueError(
                f"{_flag} only means anything with --solver-source library; the harness mode "
                f"would silently ignore it"
            )
if _cell_args.certified_budget is not None and _cell_args.certified_budget < 1:
    raise ValueError(f"--certified-budget must be >= 1 (got {_cell_args.certified_budget})")

#: Phase C1 (#304): ``--lanes captured`` replays a nautilus_batch_capture.py file.
CAPTURED_LANES = LANES_MODE == "captured"
#: ``time`` is the phase-B harness on captured lanes; ``rate`` is the untimed
#: per-lane certification pass over the whole capture (or --batch-sample calls).
CAPTURED_PASS = (_cell_args.captured_pass or "time") if CAPTURED_LANES else None
BATCH_SAMPLE = _cell_args.batch_sample
if CAPTURED_LANES:
    if not LIBRARY_SOLVER_MODE:
        raise ValueError(
            "--lanes captured runs in the library-solver mode only (--solver-source library): "
            "the replay measures the library's own certified solve"
        )
    if _cell_args.batches is None:
        raise ValueError("--lanes captured needs --batches <npz> from nautilus_batch_capture.py")
    if CAPTURED_PASS == "rate" and LIBRARY_SOLVER != "certified":
        raise ValueError(
            "--captured-pass rate measures the certified solver's uncertified-lane rate; it "
            "needs --solver certified"
        )
    if CAPTURED_PASS == "time" and BATCH_SAMPLE is not None:
        raise ValueError(
            "--batch-sample only means anything with --captured-pass rate; the timed pass "
            "replays one fixed window of B lanes (nautilus_batches.timed_window)"
        )
    if BATCH_SAMPLE is not None and BATCH_SAMPLE < 1:
        raise ValueError(f"--batch-sample must be >= 1 (got {BATCH_SAMPLE})")
else:
    for _flag, _value in (
        ("--batches", _cell_args.batches),
        ("--batch-sample", _cell_args.batch_sample),
        ("--captured-pass", _cell_args.captured_pass),
    ):
        if _value is not None:
            raise ValueError(
                f"{_flag} only means anything with --lanes captured; this leg would silently "
                f"ignore it"
            )

#: ``None`` is phase 1. A name turns on the PSF-cube experiment (#295).
PSF_CANDIDATE = _cell_args.psf_candidate
#: Seeded distinct draws pinned beside the fiducial in PSF mode.
PIN_DRAWS = (
    None
    if PSF_CANDIDATE is None
    else int(_cell_args.pin_draws if _cell_args.pin_draws is not None else 8)
)

if PSF_CANDIDATE is not None and VMAP_BATCH is not None:
    raise ValueError(
        "--psf-candidate runs on the single-call path only; it cannot be combined with "
        "--vmap-batch (one experiment per invocation)"
    )
if PSF_CANDIDATE is None and _cell_args.pin_draws is not None:
    raise ValueError(
        "--pin-draws only means anything with --psf-candidate; without it this cell runs "
        "the phase-1 single-call trace and would silently ignore the flag"
    )
if PIN_DRAWS is not None and PIN_DRAWS < 1:
    raise ValueError(f"--pin-draws must be >= 1 (got {PIN_DRAWS})")
if PSF_CANDIDATE is not None and _cli.use_mixed_precision:
    raise ValueError(
        "--psf-candidate is an fp64 experiment (the precision candidates are the diagnostic "
        "rows mp_cube_c64 and c64_full); do not combine it with --use-mixed-precision"
    )

PASS_BUDGET = int(
    _cell_args.safe_budget if _cell_args.safe_budget is not None else PHASE3_SAFE_BUDGET[MESH]
)

ROUTE_SELECTION = tuple(t.strip() for t in _cell_args.routes.split(",") if t.strip())
_unknown = set(ROUTE_SELECTION) - {"b", "d"}
if _unknown:
    raise ValueError(f"--routes accepts only b and d in this cell (got {sorted(_unknown)})")
if "d" not in ROUTE_SELECTION:
    raise ValueError("--routes must include d: the trace is taken on route d")
if PSF_CANDIDATE is not None and "b" not in ROUTE_SELECTION:
    raise ValueError(
        "--psf-candidate needs route b: it is the unmodified library answer every pin is "
        "taken against"
    )

SOURCE_PIXELS_REQUESTED = (
    _cell_args.source_pixels if _cell_args.source_pixels is not None else _cli.source_pixels
)

DATASET = "hst"
if _cli.instrument is not None and _cli.instrument != DATASET:
    raise ValueError(
        f"--instrument {_cli.instrument!r} is not this cell's dataset; phase 1 is HST only."
    )

Timer = timing.Timer
block = timing.block

timer = Timer()
jit_records: dict[str, dict] = {}


def jit_profile(func, label, *args, n_repeats=10):
    """Cell-local binding of ``timing.jit_profile`` (this cell's timer/records)."""
    return timing.jit_profile(
        func, label, *args, n_repeats=n_repeats, timer=timer, jit_records=jit_records
    )


def peak_bytes():
    """Device peak bytes, when the backend exposes them (``None`` on CPU)."""
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 — a backend without memory stats is not an error
        return None
    value = stats.get("peak_bytes_in_use")
    return int(value) if value is not None else None


def steady_wall_ms(executable, tree, n_repeats: int) -> float:
    """Mean per-call wall of *n_repeats* steady calls of an already-compiled executable.

    Every leaf of the returned pytree is blocked (``timing.block``), so a
    tuple-returning call cannot be timed asynchronously.
    """
    block(executable(tree))
    start = time.perf_counter()
    for _ in range(n_repeats):
        block(executable(tree))
    return (time.perf_counter() - start) / n_repeats * 1e3


# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================
# Deliberately identical to ``fixed_light_library.py`` PART A, which is itself
# the three pixelized breakdown cells' construction. If a sibling cell's model
# changes, this cell must change with it or its wall no longer reconciles with
# phase 5's row.

print("=" * 70)
print(f"FIXED-LIGHT TRACE — {MESH} — border relocator: {BORDER_RELOCATOR_MODE}")
print("=" * 70)

print(f"\n--- Dataset loading & masking [{DATASET}, mesh={MESH}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[DATASET]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / DATASET


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


auto_simulate_if_missing(
    dataset_path,
    dataset_type="imaging",
    instrument=DATASET,
    workspace_root=_workspace_root,
)

dataset_sha256 = {
    name: _sha256(dataset_path / name) for name in ("data.fits", "noise_map.fits", "psf.fits")
}
cell_source_sha256 = _sha256(Path(__file__).resolve())

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

if MESH == "rectangular":
    mesh_pixels_yx = (
        39 if SOURCE_PIXELS_REQUESTED is None else int(round(math.sqrt(SOURCE_PIXELS_REQUESTED)))
    )
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
    n_source_pixels = mesh_pixels_yx * mesh_pixels_yx
    n_mesh_vertices = None
else:
    mesh_pixels_yx = None
    mesh_shape = None
    n_mesh_vertices = 1500 if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    n_source_pixels = n_mesh_vertices

print(f"  Source pixels: {n_source_pixels} (fiducial: {FIDUCIAL_SOURCE_PIXELS[MESH]})")

print("\n--- Adapt image (lensed source) ---")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

image_plane_mesh_grid = None
if MESH != "rectangular":
    print("\n--- Image mesh construction (Hilbert) ---")
    with timer.section("image_mesh_hilbert"):
        image_mesh = al.image_mesh.Hilbert(
            pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0
        )
        image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
            mask=dataset.mask, adapt_data=adapt_image
        )
    print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

print("\n--- Model construction ---")

with timer.section("model_build"):
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
        if MESH == "delaunay":
            mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
        else:
            mesh_obj = al.mesh.DelaunayNN(pixels=n_mesh_vertices, areas_factor=0.5, zeroed_pixels=0)
        reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

    pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization: {reg_scheme} ({reg_provenance})")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

tracer = al.Tracer(galaxies=list(instance.galaxies))

_adapt_kwargs = {
    "galaxy_image_dict": {instance.galaxies.source: adapt_image},
    "galaxy_name_image_dict": {"('galaxies', 'source')": adapt_image},
}
if image_plane_mesh_grid is not None:
    _adapt_kwargs["galaxy_image_plane_mesh_grid_dict"] = {
        instance.galaxies.source: image_plane_mesh_grid
    }
    _adapt_kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
        "('galaxies', 'source')": image_plane_mesh_grid
    }
adapt_images = al.AdaptImages(**_adapt_kwargs)

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]

TAU_REL = active_set_steps.TAU_REL_DEFAULT

# ---------------------------------------------------------------------------
# The border relocator, as the cells force it and as the library ships it
# ---------------------------------------------------------------------------
# Every profiling cell in this family passes ``use_border_relocator=True``.
# Production passes nothing and takes the config default. They are different
# programs — the relocator is a whole stage of the trace — so the mode is a flag
# and the RESOLVED boolean is recorded, never inferred from the mode name.

_BORDER_RELOCATOR_SETTING = {"cell": True, "library": None, "off": False}[BORDER_RELOCATOR_MODE]
_settings = al.Settings(
    use_border_relocator=_BORDER_RELOCATOR_SETTING,
    use_mixed_precision=_cli.use_mixed_precision,
)
BORDER_RELOCATOR_RESOLVED = bool(_settings.use_border_relocator)

# ---------------------------------------------------------------------------
# Phase B (#300): the library solver settings, and the explicit-PDIP reference
# ---------------------------------------------------------------------------
# ``_settings`` is left exactly as phase 2 builds it, so PART B (routes b and d,
# the fiducial pin) is unchanged in every mode. In library mode PART V's arms
# are built from ``_settings_library`` — the solver named on the command line —
# and its two references from ``_settings_pdip_reference``, which names PDIP
# EXPLICITLY rather than trusting the config default to stay PDIP.
_settings_library = None
_settings_pdip_reference = _settings
CERTIFIED_BUDGET = None
CERTIFIED_BUDGET_BASIS = None
if LIBRARY_SOLVER_MODE:
    if _cell_args.certified_budget is not None:
        CERTIFIED_BUDGET = int(_cell_args.certified_budget)
        CERTIFIED_BUDGET_BASIS = "--certified-budget"
    else:
        # The resolved config value (the packaged general.yaml's inversion block,
        # unless this workspace's config shadows it) — what production would run.
        CERTIFIED_BUDGET = int(al.Settings().certified_pass_budget)
        CERTIFIED_BUDGET_BASIS = "config default (Settings().certified_pass_budget)"
    _settings_library = al.Settings(
        use_border_relocator=_BORDER_RELOCATOR_SETTING,
        use_mixed_precision=_cli.use_mixed_precision,
        positive_only_solver=LIBRARY_SOLVER,
        certified_fallback=LIBRARY_FALLBACK,
        certified_pass_budget=CERTIFIED_BUDGET,
    )
    _settings_pdip_reference = al.Settings(
        use_border_relocator=_BORDER_RELOCATOR_SETTING,
        use_mixed_precision=_cli.use_mixed_precision,
        positive_only_solver="pdip",
    )
    if bool(_settings_library.use_border_relocator) != BORDER_RELOCATOR_RESOLVED:
        raise AssertionError("the library-solver Settings resolved a different border relocator")

# MEASURED, not assumed: ``library`` resolved to **True** on this workspace
# (autolens_profiling/config/general.yaml declares no ``inversion`` block, so
# autoconf falls through to the packaged default, which is on). The premise
# "the cells force the relocator on, production ships it off" is therefore false
# here, and ``cell`` and ``library`` compile the same program. ``off`` is the
# flag that actually answers what the relocator costs.
if BORDER_RELOCATOR_MODE == "library" and BORDER_RELOCATOR_RESOLVED:
    print(
        "  NOTE: --border-relocator library resolved to True from the config, so this leg is "
        "the same program as --border-relocator cell. Use --border-relocator off for the "
        "relocator-free comparison."
    )

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {DATASET}")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Source pixels:           {n_source_pixels}")
print(f"  Routes:                  {','.join(ROUTE_SELECTION)}")
print(f"  Pass budget (route d):   {PASS_BUDGET}  [phase 3 production budget]")
print(f"  Border relocator:        {BORDER_RELOCATOR_MODE} -> {BORDER_RELOCATOR_RESOLVED}")
print(f"  Trace calls:             {TRACE_CALLS}")
print(f"  tau_rel:                 {TAU_REL:.6e}")
if PSF_CANDIDATE is not None:
    _psf_spec = psf_cube_injection.CANDIDATES[PSF_CANDIDATE]
    print(f"  PSF candidate:           {PSF_CANDIDATE} ({_psf_spec.kind}) — {_psf_spec.label}")
    print(f"  Pin draws:               {PIN_DRAWS} (seed {DRAW_SEED})")
if VMAP_BATCH is None:
    print("  Batched mode:            off (phase-1 single-call trace)")
else:
    print(f"  Batched mode:            --vmap-batch {VMAP_BATCH}  lanes {LANES_MODE}")
    print(f"  Arms:                    {','.join(ARMS)}")
    print(f"  Draw seed:               {DRAW_SEED}")
    print(
        f"  Solver fallback:         {'on (route d, lax.cond -> select under vmap)' if FALLBACK_ON else 'off (route d0, cond-free)'}"
    )
    print(f"  Solver source:           {SOLVER_SOURCE}")
    if LIBRARY_SOLVER_MODE:
        print(
            f"  Library solver:          {LIBRARY_SOLVER}  certified_fallback {LIBRARY_FALLBACK}  "
            f"certified_pass_budget {CERTIFIED_BUDGET} ({CERTIFIED_BUDGET_BASIS})"
        )

# ---------------------------------------------------------------------------
# S3 — the source-only system every route in this cell fits
# ---------------------------------------------------------------------------

print("\n--- S0 -> S3: lens light converted to regular profiles + subtracted (eager) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=_settings,
        xp=np,
    )
    log_evidence_ref = float(fit.figure_of_merit)

with timer.section("s3_system_build"):
    system_s3 = active_set_steps.fixed_light_system_from(
        fit,
        dataset,
        adapt_images=adapt_images,
        settings=_settings,
        name="S3_mge_converted_to_regular",
    )

print(
    f"  S3: n={system_s3.n_params} (mapper {system_s3.n_mapper} + funcs {system_s3.n_funcs}), "
    f"edge-zeroed {int(system_s3.edge_zero_mask.sum())}"
)
log_evidence_s3_library = float(system_s3.fit.figure_of_merit)
print(f"  S3 figure_of_merit (library) = {log_evidence_s3_library}")

# ---------------------------------------------------------------------------
# Phase C1 (#304): the captured batches, and the proof they fit THIS S3 system
# ---------------------------------------------------------------------------
# nautilus_batch_capture.py copies PART A and records the fingerprint of the S3
# system it fitted. A lane is a parameter vector, which means nothing on a
# different dataset, mesh, regularization or relocator — so every one of those is
# re-derived here and a mismatch refuses the file.
captured = None
captured_record: dict | None = None
if CAPTURED_LANES:
    _batches_path = Path(_cell_args.batches)
    captured = nautilus_batches.load(_batches_path)
    _fp = captured["meta"]["s3_fingerprint"]
    CAPTURED_STAGE = captured["meta"]["stage"]
    _mismatch = {
        _key: (_want, _got)
        for _key, _want, _got in (
            ("mesh", _fp["mesh"], MESH),
            ("source_pixels", int(_fp["source_pixels"]), int(n_source_pixels)),
            ("dataset_sha256", _fp["dataset_sha256"], dataset_sha256),
            # The regularization COEFFICIENTS are free parameters of the capture (and
            # of every replayed lane), so they are not the system's identity. The
            # CLASS the capture freed is — stage-dependent on rectangular (Constant in
            # pix1, Adapt in pix2; nautilus_batches.regularization_class_for) — and so
            # is the S0 fiducial the S3 subtraction was built with (its scheme and
            # coefficients here, its figure of merit in log_evidence_s3_library below).
            (
                "regularization_class",
                _fp["regularization_class"],
                nautilus_batches.regularization_class_for(MESH, CAPTURED_STAGE),
            ),
            ("regularization_s0_fiducial", _fp["regularization_s0_fiducial"], reg_provenance),
            ("border_relocator", bool(_fp["border_relocator"]), BORDER_RELOCATOR_RESOLVED),
            (
                "use_mixed_precision",
                bool(_fp["use_mixed_precision"]),
                bool(_cli.use_mixed_precision),
            ),
        )
        if _want != _got
    }
    _s3_rel = abs(float(_fp["log_evidence_s3_library"]) - log_evidence_s3_library) / max(
        abs(log_evidence_s3_library), 1e-300
    )
    if _s3_rel > EQUIVALENCE_RTOL:
        _mismatch["log_evidence_s3_library"] = (
            float(_fp["log_evidence_s3_library"]),
            log_evidence_s3_library,
        )
    if _mismatch:
        raise AssertionError(
            f"{_batches_path} was captured on a different S3 system: {_mismatch}. Its lanes "
            f"are parameter vectors of that system and are not this cell's lanes."
        )
    # The replay model: the SAME builder the capture used, so a lane vector means
    # the same parameters here as it did to the sampler. Its order is asserted.
    _replay_model = nautilus_batches.capture_model_from(
        CAPTURED_STAGE,
        mesh=MESH,
        mesh_obj=mesh_obj,
        lens_fixed=list(system_s3.source_only_tracer.galaxies)[0],
    )
    _replay_paths = [".".join(_p) for _p in _replay_model.paths]
    if _replay_paths != list(captured["meta"]["parameter_paths"]):
        raise AssertionError(
            f"{_batches_path} parameter paths {captured['meta']['parameter_paths']} are not "
            f"this checkout's {CAPTURED_STAGE} model {_replay_paths} (order included)"
        )
    _reg_cols = [
        _replay_paths.index(_p) for _p in nautilus_batches.regularization_paths(_replay_paths)
    ]
    captured_record = {
        "batches_path": str(_batches_path),
        "batches_sha256": _sha256(_batches_path),
        "schema_version": int(captured["schema_version"]),
        "n_calls": int(captured["call_size"].shape[0]),
        "n_lanes": int(captured["parameters"].shape[0]),
        "call_sizes_seen": sorted({int(_c) for _c in captured["call_size"]}),
        "calls_by_phase": {
            _p: int((captured["call_phase"] == _p).sum())
            for _p in dict.fromkeys(captured["call_phase"].tolist())
        },
        "parameter_paths": list(captured["meta"]["parameter_paths"]),
        "stage": CAPTURED_STAGE,
        "regularization": captured["meta"]["regularization"],
        "nautilus": captured["meta"]["nautilus"],
        "capture_fit_status": captured["meta"].get("fit_status"),
        "capture_library_revisions": captured["meta"].get("library_revisions"),
        "fingerprint_check": {
            "fields": [
                "mesh",
                "source_pixels",
                "dataset_sha256",
                "regularization_class",
                "regularization_s0_fiducial",
                "border_relocator",
                "use_mixed_precision",
                "log_evidence_s3_library",
            ],
            "not_identity": (
                "the regularization coefficients: free per lane since the capture frees them"
            ),
            "log_evidence_s3_library_rel_diff": _s3_rel,
            "passed": True,
        },
    }
    print(
        f"  captured batches: {captured_record['n_calls']} calls, "
        f"{captured_record['n_lanes']} lanes ({captured_record['calls_by_phase']}); "
        f"S3 fingerprint matches (rel {_s3_rel:.2e})"
    )

peak_after_setup = peak_bytes()

# ===================================================================
# PART B — the library path, one whole likelihood call per route
# ===================================================================

instance_s3 = copy.deepcopy(instance)
_source_only_galaxies = list(system_s3.source_only_tracer.galaxies)
instance_s3.galaxies.lens = _source_only_galaxies[0]
instance_s3.galaxies.source = _source_only_galaxies[1]
params_tree_s3 = jax.tree_util.tree_map(jnp.asarray, instance_s3)

# ---------------------------------------------------------------------------
# Draw trees — shared by the phase-2 vmap lanes and the phase-3 draw pins
# ---------------------------------------------------------------------------
# A draw IS a seeded random draw of the phase-3 family applied to the lens
# MASS, with the lens light fixed at S3 — the same construction
# ``fixed_light_draws.py`` uses (``flds.mass_from(BASE_MASS, offsets)``). The S3
# dataset is the FIDUCIAL light-subtracted one and is shared by every draw.
BASE_MASS = flds.fiducial_mass_values(instance)

_lens_s3 = instance_s3.galaxies.lens
# The Galaxy's profiles by the names it holds them under — the same rule
# ``fixed_light_system._profile_attrs`` uses, because af.Model paths and the
# adapt-image dictionaries are keyed on those names.
_LENS_ATTRS = {
    _k: _v
    for _k, _v in vars(_lens_s3).items()
    if not _k.startswith("_") and _k not in {"id", "redshift"}
}
if "mass" not in _LENS_ATTRS:
    raise AssertionError(
        "the S3 source-only lens galaxy has no `mass` attribute — the lane "
        "construction would swap a profile that is not there"
    )


def _lane_tree(draw):
    """The params pytree of one lane: S3 with draw's mass swapped in.

    SHALLOW copies, deliberately. ``copy.deepcopy(instance_s3)`` duplicates
    the source galaxy's ``Pixelization`` and ``Regularization``, which sit in
    the pytree's **static aux data** and compare by object identity — two
    deepcopies of the same instance therefore have unequal treedefs and
    cannot be stacked into a batch at all. Sharing those objects across every
    lane is also the truth of the experiment: production batches one dataset
    and one pixelization over B parameter vectors.
    """
    if not draw.offsets:
        return params_tree_s3
    _inst = copy.copy(instance_s3)
    _inst.galaxies = copy.copy(instance_s3.galaxies)
    _attrs = dict(_LENS_ATTRS)
    _attrs["mass"] = flds.mass_from(BASE_MASS, draw.offsets, mass_cls=draw.mass_cls)
    _inst.galaxies.lens = al.Galaxy(redshift=float(_lens_s3.redshift), **_attrs)
    return jax.tree_util.tree_map(jnp.asarray, _inst)


def _regularization_range(rows) -> dict:
    """Min / max of every free regularization parameter over the capture *rows*."""
    _paths = captured["meta"]["parameter_paths"]
    _vals = captured["parameters"][np.asarray(rows, dtype=int)]
    return {
        _paths[_c]: {"min": float(_vals[:, _c].min()), "max": float(_vals[:, _c].max())}
        for _c in _reg_cols
    }


def _psf_context():
    """The phase-3 PSF injection, or a no-op yielding zero counts in every other mode.

    Always entered INSIDE the certified-solver injection, and always around a
    fresh ``jax.jit``: the Convolver is a closed-over constant, so the patch
    acts at trace time.
    """
    if PSF_CANDIDATE is None:
        return contextlib.nullcontext({"jax": 0, "delegated": 0})
    return psf_cube_injection.psf_convolution_injected(PSF_CANDIDATE)


def _captured_likelihood_fn(dataset_for_route, settings_for_route):
    """``parameter vector -> log likelihood``, exactly as ``Fitness.call`` does it.

    Phase C1 (#304): a captured lane is a physical parameter vector of the
    capture's model (mass, shear AND regularization in pix1; regularization only
    in pix2). ``_replay_model.instance_from_vector(..., xp=jnp)`` inside the trace
    is the Fitness JAX path (``ignore_assertions=True``), so every lane's own
    regularization coefficients reach the likelihood as traced values — which a
    pytree of the S3 instance cannot do: its Pixelization sits in static aux data.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset_for_route,
        adapt_images=adapt_images,
        settings=settings_for_route,
        use_jax=True,
    )

    def _likelihood(vector):
        instance = _replay_model.instance_from_vector(vector=vector, ignore_assertions=True, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    return _likelihood


def _likelihood_fn(dataset_for_route, settings_for_route):
    """``params tree -> log likelihood`` through the library's analysis path."""
    analysis = al.AnalysisImaging(
        dataset=dataset_for_route,
        adapt_images=adapt_images,
        settings=settings_for_route,
        use_jax=True,
    )

    def _likelihood(tree):
        return analysis.log_likelihood_function(instance=tree)

    return _likelihood


#: The likelihood builder PART V and PART R use for their LANES: the S3 tree
#: function in every phase-B/phase-2 mode, the Fitness-style vector function for
#: captured lanes. PART B (the fiducial routes) always uses ``_likelihood_fn``.
_lane_likelihood_fn = _captured_likelihood_fn if CAPTURED_LANES else _likelihood_fn

print("\n" + "=" * 70)
print("LIBRARY PATH — one full likelihood call per route")
print("=" * 70)

#: ``(key, label, injection)``. ``injection`` is ``None`` for the library solver
#: as shipped, or ``(pass_budget, fallback)`` for the harness-injected certified
#: scheme. Route ``d`` runs at the PRODUCTION budget, never phase 0's fiducial.
_ROUTE_SPECS: dict[str, tuple] = {
    "b": ("b_s3_pdip", "S3 PDIP (source-only)", None),
    "d": (
        "d_s3_certified_fallback",
        f"S3 certified active set, budget {PASS_BUDGET}, PDIP fallback",
        (PASS_BUDGET, True),
    ),
}

routes: dict[str, dict] = {}
#: The compiled executable of each route, kept so the phase-3 draw pins evaluate
#: the SAME programs PART B timed (a compiled executable accepts any tree with
#: the fiducial's structure).
_compiled_routes: dict = {}

for _token in ROUTE_SELECTION:
    _key, _label, _injection = _ROUTE_SPECS[_token]
    print(f"\n--- {_key}: {_label} ---")
    _fn = _likelihood_fn(system_s3.dataset, _settings)
    _entry: dict = {"label": _label, "status": "ok"}

    if _injection is None:
        _compiled, _value = jit_profile(_fn, f"{_key}_library_likelihood_jit", params_tree_s3)
        _entry["solver"] = "library"
    else:
        _budget, _fallback = _injection
        # The PSF injection (phase 3) nests INSIDE the solver injection and is a
        # no-op in every other mode; route b above never sees either.
        with (
            library_solver_injection.certified_solver_injected(
                _budget, fallback=_fallback, tau_rel=TAU_REL
            ) as _counts,
            _psf_context() as _psf_counts,
        ):
            _compiled, _value = jit_profile(_fn, f"{_key}_library_likelihood_jit", params_tree_s3)
        _entry["solver"] = "harness-injected certified active set"
        if PSF_CANDIDATE is not None:
            _entry["psf_candidate"] = PSF_CANDIDATE
            _entry["injected_psf_calls_jax"] = int(_psf_counts["jax"])
            _entry["injected_psf_calls_delegated"] = int(_psf_counts["delegated"])
            if PSF_CANDIDATE != "control" and int(_psf_counts["jax"]) == 0:
                raise AssertionError(
                    f"{_key}: the injected PSF candidate {PSF_CANDIDATE!r} was never called on "
                    f"the JAX path — this row would be the library's convolution wearing the "
                    f"wrong label."
                )
        _entry["pass_budget"] = int(_budget)
        _entry["fallback"] = bool(_fallback)
        _entry["injected_solver_calls_jax"] = int(_counts["jax"])
        _entry["injected_solver_calls_numpy"] = int(_counts["numpy"])
        if int(_counts["jax"]) == 0:
            raise AssertionError(
                f"{_key}: the injected solver was never called on the JAX path — "
                f"this row would be the library's own PDIP wearing the wrong label."
            )

    _entry["ms"] = timer.records[-1][1] / 10 * 1e3
    _entry["log_likelihood"] = float(_value)
    print(f"  {_entry['ms']:.3f} ms  -> log likelihood {_entry['log_likelihood']:.6f}")
    routes[_key] = _entry
    _compiled_routes[_key] = _compiled

peak_after_routes = peak_bytes()

# ---------------------------------------------------------------------------
# The equivalence pin — route d IS route b, or the trace decomposes a wrong answer
# ---------------------------------------------------------------------------

equivalence_pins: list[dict] = []

if "b" in ROUTE_SELECTION:
    _got = routes["d_s3_certified_fallback"]["log_likelihood"]
    _ref = routes["b_s3_pdip"]["log_likelihood"]
    _rel = abs(_got - _ref) / max(abs(_ref), 1e-300)
    _pin = {
        "pin": "route d (certified, fallback) == route b (library PDIP)",
        "route": "d_s3_certified_fallback",
        "reference": "b_s3_pdip",
        "expectation": "the certified active set returns the library's own positive solution",
        "rtol": EQUIVALENCE_RTOL,
        "got": _got,
        "reference_value": _ref,
        "rel_diff": _rel,
        "status": "PASS" if _rel <= EQUIVALENCE_RTOL else "FAIL",
    }
    equivalence_pins.append(_pin)
    print(f"\n  [{_pin['status']:>7}] {_pin['pin']}  rel {_rel:.3e}")
else:
    print("\n  route b not selected — the d == b equivalence pin is SKIPPED, not passed.")

# ---------------------------------------------------------------------------
# Phase 3 (#295): the fiducial pin labelled per candidate, and the draw pins
# ---------------------------------------------------------------------------
# Route b (library PDIP + library convolution) is the unmodified library answer.
# The gate (psf_cube_injection.psf_gate_rows, PRE-REGISTERED before any A100
# data): the fiducial d == b pin is gated at EQUIVALENCE_RTOL for every fp64
# candidate including control; a non-control fp64 candidate's draws are gated
# on rel_diff_vs_d_control (the convolution-isolated pin) and their rel_diff vs
# route b is recorded; control's draws are recorded vs route b; a diagnostic
# candidate is labelled DIAGNOSTIC and never PASSes. The draws are evaluated by
# the SAME compiled executables PART B timed.

psf_pin_draw_rows: list[dict] = []
psf_d_control_record: dict | None = None
psf_gated_rows: list[dict] = []
psf_pins_failed: list[str] = []

if PSF_CANDIDATE is not None:
    for _pin in equivalence_pins:
        _pin["pin"] = f"route d (certified, fallback, psf {PSF_CANDIDATE}) == route b (library)"
        _pin["abs_diff_nats"] = abs(_pin["got"] - _pin["reference_value"])

    # For a non-control candidate, route d compiled WITHOUT the PSF injection
    # (certified solver + library convolution) isolates the convolution from the
    # certified-vs-PDIP solver difference. Its draws are the GATED draw pins.
    _compiled_d_control = None
    if PSF_CANDIDATE != "control":
        with library_solver_injection.certified_solver_injected(
            PASS_BUDGET, fallback=True, tau_rel=TAU_REL
        ) as _dc_counts:
            _fn_dc = _likelihood_fn(system_s3.dataset, _settings)
            with timer.section("psf_pin_d_control_compile"):
                _compiled_d_control = jax.jit(_fn_dc).lower(params_tree_s3).compile()
        if int(_dc_counts["jax"]) == 0:
            raise AssertionError("the d-control reference never called the injected solver")
        _dc_fiducial = float(block(_compiled_d_control(params_tree_s3)))
        _d_fiducial = routes["d_s3_certified_fallback"]["log_likelihood"]
        psf_d_control_record = {
            "program": "route d (certified solver, budget and fallback as route d) with the "
            "LIBRARY convolution — compiled without the PSF injection",
            "compile_s": float(timer.records[-1][1]),
            "fiducial_log_likelihood": _dc_fiducial,
            "fiducial_rel_diff_candidate_vs_d_control": abs(_d_fiducial - _dc_fiducial)
            / max(abs(_dc_fiducial), 1e-300),
            "gated": False,
            "note": (
                "Isolates the convolution: the candidate and this program share the solver, "
                "so their difference is the convolution alone. The draw pins are GATED on it "
                "(rel_diff_vs_d_control, pre-registered before any A100 data); the fiducial "
                "pin stays gated against route b (the unmodified library)."
            ),
        }

    _pin_draws = flds.random_draws(DRAW_SEED, PIN_DRAWS)
    _fiducial_structure = jax.tree_util.tree_structure(params_tree_s3)
    with timer.section("psf_pin_draws"):
        for _k, _draw in enumerate(_pin_draws):
            _tree = _lane_tree(_draw)
            if jax.tree_util.tree_structure(_tree) != _fiducial_structure:
                raise AssertionError(
                    f"pin draw {_k} has a different pytree structure from the fiducial S3 tree"
                )
            _ll_d = float(block(_compiled_routes["d_s3_certified_fallback"](_tree)))
            _ll_b = float(block(_compiled_routes["b_s3_pdip"](_tree)))
            _abs = abs(_ll_d - _ll_b)
            _rel = _abs / max(abs(_ll_b), 1e-300)
            _row = {
                "draw": _k,
                "draw_name": _draw.name,
                "draw_kind": _draw.kind,
                "offsets": {_p: float(_o) for _p, _o in _draw.offsets.items()},
                "log_likelihood_d_candidate": _ll_d,
                "log_likelihood_b_library": _ll_b,
                "abs_diff_nats": _abs,
                "rel_diff": _rel,
                "rtol": EQUIVALENCE_RTOL,
            }
            if _compiled_d_control is not None:
                _ll_dc = float(block(_compiled_d_control(_tree)))
                _row["log_likelihood_d_control"] = _ll_dc
                _row["rel_diff_vs_d_control"] = abs(_ll_d - _ll_dc) / max(abs(_ll_dc), 1e-300)
                _row["rel_diff_d_control_vs_b_library"] = abs(_ll_dc - _ll_b) / max(
                    abs(_ll_b), 1e-300
                )
            psf_pin_draw_rows.append(_row)

    psf_gated_rows, psf_pins_failed = psf_cube_injection.psf_gate_rows(
        PSF_CANDIDATE, equivalence_pins, psf_pin_draw_rows, EQUIVALENCE_RTOL
    )
    for _pin in equivalence_pins:
        print(f"  [{_pin['status']:>10}] {_pin['pin']}  rel {_pin['rel_diff']:.3e}")
    for _row in psf_pin_draw_rows:
        _k, _ll_d, _ll_b = (
            _row["draw"],
            _row["log_likelihood_d_candidate"],
            _row["log_likelihood_b_library"],
        )
        _rel, _abs = _row["rel_diff"], _row["abs_diff_nats"]
        print(
            f"  [{_row['status']:>10}] draw {_k} ({_row['draw_name']}): d {_ll_d:.9f} vs b "
            f"{_ll_b:.9f}  rel {_rel:.3e}  abs {_abs:.3e} nats"
            + (
                f"  | vs d-control rel {_row['rel_diff_vs_d_control']:.3e}"
                if "rel_diff_vs_d_control" in _row
                else ""
            )
        )

# ===================================================================
# PART R — the captured-lane RATE pass (phase C1, #304), untimed
# ===================================================================
# ``--lanes captured --captured-pass rate``: every captured lane (or every lane of
# ``--batch-sample K`` evenly spaced calls) through the library's certified solve
# under the production composition ``jax.jit(jax.vmap(fn))``, in chunks of
# ``--vmap-batch`` lanes (at B = n_batch a chunk IS one real Nautilus batch), with
# the solve OBSERVED (``library_solver_observed``: the library's own stats=
# out-dict through an ordered debug callback, nothing computed changed). Nothing
# here is timed; the cell writes its own JSON and exits before PART C / PART V.


def _c1_library_revisions() -> dict:
    """``git rev-parse HEAD`` of every PyAuto library this process imported."""
    import subprocess as _subprocess

    import autoarray as _autoarray
    import autogalaxy as _autogalaxy
    import autonerves as _autonerves

    def _rev(module) -> dict:
        _repo = Path(module.__file__).resolve().parents[1]
        try:
            _sha = _subprocess.run(
                ["git", "-C", str(_repo), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, _subprocess.CalledProcessError):
            _sha = None
        return {"head": _sha, "path": str(_repo), "module_file": module.__file__}

    return {
        "PyAutoNerves": _rev(_autonerves),
        "PyAutoFit": _rev(af),
        "PyAutoArray": _rev(_autoarray),
        "PyAutoGalaxy": _rev(_autogalaxy),
        "PyAutoLens": _rev(al),
    }


#: At most this many uncertified lanes get a scalar library-PDIP re-evaluation in
#: the rate pass (RECORDED): the rare-event check must not become the pass.
RATE_PDIP_RECHECK_MAX = 256

if CAPTURED_LANES and "shear" not in _LENS_ATTRS:
    raise AssertionError(
        "the S3 source-only lens has no `shear` attribute — a captured lane's shear would "
        "have nowhere to go"
    )

if CAPTURED_PASS == "rate":
    print("\n" + "=" * 70)
    print(
        f"CAPTURED-LANE RATE PASS — library {LIBRARY_SOLVER}, certified_fallback "
        f"{LIBRARY_FALLBACK}, budget {CERTIFIED_BUDGET}, chunks of {VMAP_BATCH}"
    )
    print("=" * 70)

    _call_index = captured["call_index"]
    _calls = nautilus_batches.sample_calls(int(captured["call_size"].shape[0]), BATCH_SAMPLE)
    _lanes = nautilus_batches.lanes_of_calls(_call_index, _calls)
    _chunks = nautilus_batches.chunked(_lanes, VMAP_BATCH)
    print(
        f"  replaying {_lanes.shape[0]} lanes from {_calls.shape[0]} of "
        f"{captured_record['n_calls']} calls in {len(_chunks)} chunks"
    )

    _rate_rows: list[tuple[int, bool]] = []

    def _collect_rate(passes, certified):
        _rate_rows.append((int(passes), bool(certified)))

    _ll_rate = np.empty(_lanes.shape[0], dtype=float)
    _certified = np.empty(_lanes.shape[0], dtype=bool)
    _passes = np.empty(_lanes.shape[0], dtype=int)
    _pos = 0
    _t_rate = time.perf_counter()
    with library_solver_injection.library_solver_observed(report=_collect_rate) as _observed:
        _fn_rate = jax.jit(jax.vmap(_lane_likelihood_fn(system_s3.dataset, _settings_library)))
        with timer.section("captured_rate_pass"):
            for _ci, (_chunk, _n_real) in enumerate(_chunks):
                _batched = jnp.asarray(captured["parameters"][_chunk])
                _before = len(_rate_rows)
                _ll = np.asarray(block(_fn_rate(_batched)), dtype=float)
                jax.effects_barrier()
                _rows = _rate_rows[_before:]
                if len(_rows) != VMAP_BATCH or _ll.shape != (VMAP_BATCH,):
                    raise AssertionError(
                        f"chunk {_ci}: {len(_rows)} certification rows and log likelihoods of "
                        f"shape {_ll.shape} for a batch of {VMAP_BATCH} — the report is not "
                        f"one row per lane"
                    )
                _ll_rate[_pos : _pos + _n_real] = _ll[:_n_real]
                _passes[_pos : _pos + _n_real] = [_r[0] for _r in _rows[:_n_real]]
                _certified[_pos : _pos + _n_real] = [_r[1] for _r in _rows[:_n_real]]
                _pos += _n_real
                if _ci % 50 == 0 or _ci == len(_chunks) - 1:
                    print(
                        f"    chunk {_ci + 1}/{len(_chunks)}: {_pos} lanes, "
                        f"{int((~_certified[:_pos]).sum())} uncertified, "
                        f"max passes {int(_passes[:_pos].max())}  "
                        f"[{time.perf_counter() - _t_rate:.1f} s]"
                    )
    _jax_solvers = sorted(set(_observed["solvers_jax"]))
    if _jax_solvers != [LIBRARY_SOLVER]:
        raise AssertionError(
            f"the rate pass traced the positive-only solve with solver={_jax_solvers}, not "
            f"[{LIBRARY_SOLVER!r}] — the rate would be some other solver's"
        )

    # Per-lane labels: the Nautilus phase of the lane's call, and early / late.
    _lane_calls = _call_index[_lanes]
    _lane_phase = captured["call_phase"][_lane_calls]
    _lane_half = nautilus_batches.early_late(_lane_calls, _calls)

    # RECORDED, not gated: this pass's certified log likelihood against the value
    # the capture's own library-PDIP Fitness returned for the same lane (the
    # Fitness jit(vmap) composition over parameter vectors, not this cell's tree
    # composition). Resampled lanes (-1e99 sentinel, non-finite) are excluded.
    _fom = captured["figure_of_merit"][_lanes]
    _valid = np.isfinite(_fom) & (_fom > -1.0e98) & np.isfinite(_ll_rate)
    _rel_vs_capture = np.abs(_ll_rate - _fom) / np.maximum(np.abs(_fom), 1e-300)
    _vs_capture = {
        "reference": (
            "the capture's own figure of merit for the lane: library PDIP through "
            "Fitness.call_wrap's jax.jit(jax.vmap(call)) over parameter vectors"
        ),
        "n_compared": int(_valid.sum()),
        "n_excluded_resampled_or_nonfinite": int((~_valid).sum()),
        "max_rel_diff": float(_rel_vs_capture[_valid].max()) if _valid.any() else None,
        "median_rel_diff": float(np.median(_rel_vs_capture[_valid])) if _valid.any() else None,
        "max_abs_diff_nats": (
            float(np.abs(_ll_rate - _fom)[_valid].max()) if _valid.any() else None
        ),
        "n_above_lane_rtol": int((_rel_vs_capture[_valid] > LANE_RTOL).sum()),
        "n_above_lane_rtol_uncertified": int(
            (_rel_vs_capture[_valid & ~_certified] > LANE_RTOL).sum()
        ),
        "lane_rtol": LANE_RTOL,
        "gated": False,
        "note": (
            "RECORDED, NOT GATED: a cross-composition comparison (Fitness over vectors vs this "
            "cell over trees) and, for certified lanes, certified vs PDIP. The gated 1e-9 pin is "
            "the timed pass's (same composition, same lanes). It still shows whether a "
            "certified_fallback=none lane that FAILED its certificate returned a wrong answer."
        ),
    }

    # RECORDED: every uncertified lane (up to RATE_PDIP_RECHECK_MAX) re-run through
    # scalar library PDIP — what a guard's host-side second pass would return.
    _uncert_idx = np.flatnonzero(~_certified)
    _recheck_rows: list[dict] = []
    if _uncert_idx.size:
        _pdip_scalar = jax.jit(_lane_likelihood_fn(system_s3.dataset, _settings_pdip_reference))
        with timer.section("captured_rate_uncertified_pdip_recheck"):
            for _j in _uncert_idx[:RATE_PDIP_RECHECK_MAX]:
                _k = int(_lanes[_j])
                _ref = float(block(_pdip_scalar(jnp.asarray(captured["parameters"][_k]))))
                _recheck_rows.append(
                    {
                        "lane_row": _k,
                        "call_index": int(_call_index[_k]),
                        "lane_in_call": int(captured["lane_in_call"][_k]),
                        "phase": str(_lane_phase[_j]),
                        "passes": int(_passes[_j]),
                        "log_likelihood_certified_none": float(_ll_rate[_j]),
                        "log_likelihood_scalar_library_pdip": _ref,
                        "rel_diff": abs(float(_ll_rate[_j]) - _ref) / max(abs(_ref), 1e-300),
                        "abs_diff_nats": abs(float(_ll_rate[_j]) - _ref),
                    }
                )

    rate_block = {
        "pass": "rate",
        "solver": LIBRARY_SOLVER,
        "certified_fallback": LIBRARY_FALLBACK,
        "certified_pass_budget": CERTIFIED_BUDGET,
        "certified_pass_budget_basis": CERTIFIED_BUDGET_BASIS,
        "composition": "jax.jit(jax.vmap(fn)) — the current Fitness._vmap composition",
        "chunk_size": int(VMAP_BATCH),
        "chunk_rule": (
            "the replayed lanes in capture order, split into chunks of --vmap-batch; the last "
            "chunk is padded by repeating its final lane and the padded rows are dropped. At "
            "--vmap-batch == n_batch every chunk is exactly one captured Nautilus batch."
        ),
        "batch_sample": BATCH_SAMPLE,
        "stage": CAPTURED_STAGE,
        "regularization_range": _regularization_range(_lanes),
        "calls_replayed": int(_calls.shape[0]),
        "calls_rule": (
            "all captured calls"
            if BATCH_SAMPLE is None or BATCH_SAMPLE >= captured_record["n_calls"]
            else f"{int(_calls.shape[0])} calls evenly spaced over the run "
            f"(nautilus_batches.sample_calls: first and last call included)"
        ),
        "report_source": (
            "library solve observed through its stats= out-dict (PyAutoArray #566) and "
            "jax.debug.callback(..., ordered=True); untimed; one row per lane checked per chunk"
        ),
        "traced_solver_kwargs_jax": _jax_solvers,
        "overall": nautilus_batches.rate_summary(_certified, _passes, CERTIFIED_BUDGET),
        "by_nautilus_phase": nautilus_batches.grouped_rate_summaries(
            _certified, _passes, CERTIFIED_BUDGET, _lane_phase
        ),
        "by_call_half": nautilus_batches.grouped_rate_summaries(
            _certified, _passes, CERTIFIED_BUDGET, _lane_half
        ),
        "phase_rule": (
            "by_nautilus_phase: the sampler state when the batch was requested "
            "(nautilus_batches.phase_of) — prior = only the unit-cube bound (prior draws), "
            "exploration = more bounds and Sampler.explored False, sampling = explored True. "
            "by_call_half: early = call index below the median replayed call index, late = the "
            "rest."
        ),
        "vs_capture_pdip": _vs_capture,
        "uncertified_pdip_recheck": {
            "n_uncertified": int(_uncert_idx.size),
            "n_rechecked": len(_recheck_rows),
            "cap": RATE_PDIP_RECHECK_MAX,
            "max_rel_diff": max((_r["rel_diff"] for _r in _recheck_rows), default=None),
            "rows": _recheck_rows,
            "gated": False,
        },
        "lane_certified": _certified.astype(int).tolist(),
        "lane_passes": _passes.tolist(),
        "lane_rows": _lanes.tolist(),
        "rate_pass_wall_s": time.perf_counter() - _t_rate,
    }
    _ov = rate_block["overall"]
    print(
        f"\n  RATE: {_ov['n_uncertified']}/{_ov['n_lanes']} uncertified "
        f"(rate {_ov['uncertified_rate']:.3e}); max passes {_ov['max_passes']} vs budget "
        f"{CERTIFIED_BUDGET}; {_ov['n_at_budget']} lanes at the budget"
    )
    for _label, _row in {**rate_block["by_nautilus_phase"], **rate_block["by_call_half"]}.items():
        print(
            f"    {_label:<12} {_row['n_uncertified']:>6}/{_row['n_lanes']:<7} "
            f"max passes {_row['max_passes']}"
        )
    print(
        f"  vs capture PDIP (recorded): max rel {_vs_capture['max_rel_diff']}, "
        f"{_vs_capture['n_above_lane_rtol']} lanes above {LANE_RTOL:.0e}"
    )

    rate_summary_json = {
        "device": device_info_dict(),
        "machine": machine_info_dict(),
        "precision": "mixed" if _cli.use_mixed_precision else "fp64",
        "configuration": {
            "mesh": MESH,
            "source_pixels": int(n_source_pixels),
            "dataset": DATASET,
            "dataset_sha256": dataset_sha256,
            "cell_source_sha256": cell_source_sha256,
            "border_relocator_mode": BORDER_RELOCATOR_MODE,
            "border_relocator": BORDER_RELOCATOR_RESOLVED,
            "regularization": reg_provenance,
            "vmap_batch": int(VMAP_BATCH),
            "lanes": LANES_MODE,
            "captured_pass": CAPTURED_PASS,
            "batch_sample": BATCH_SAMPLE,
            "solver_source": SOLVER_SOURCE,
            "solver": LIBRARY_SOLVER,
            "certified_fallback": LIBRARY_FALLBACK,
            "certified_pass_budget": CERTIFIED_BUDGET,
            "certified_tau_rel": float(_settings_library.certified_tau_rel),
            "use_mixed_precision": bool(_cli.use_mixed_precision),
            "thread_env": _observe_thread_env(),
        },
        "captured": captured_record,
        "rate": rate_block,
        "routes": routes,
        "equivalence_pins": equivalence_pins,
        "reference": {"log_evidence_s3_library": log_evidence_s3_library},
        "library_revisions": _c1_library_revisions(),
        "timer": {_label: _secs for _label, _secs in timer.records},
        "note": (
            "Phase C1 (#304) rate pass: untimed. No millisecond here is a result. The rate is "
            "the fraction of captured lanes whose certificate failed under the named "
            "certified_fallback; the phase-C2 verdict takes it together with the timed pass."
        ),
    }
    _rate_name = (
        f"fixed_light_trace_{MESH}_captured_rate_lib{LIBRARY_SOLVER}"
        f"_fb{'on' if FALLBACK_ON else 'off'}_b{CERTIFIED_BUDGET}_{CAPTURED_STAGE}"
        + (f"_sample{int(BATCH_SAMPLE)}" if BATCH_SAMPLE is not None else "")
    )
    _rate_json, _ = resolve_output_paths(
        _cli,
        default_dir=_workspace_root / "results" / "breakdown" / "imaging",
        default_basename=f"{_rate_name}_trace_{DATASET}_v{al.__version__}",
        cell=_rate_name,
    )
    _rate_json.write_text(json.dumps(rate_summary_json, indent=2, default=str))
    print(f"\n  Rate JSON saved to: {_rate_json}")
    sys.exit(0)

# ===================================================================
# PART C — one lowering, two executables, one timeline
# ===================================================================

print("\n" + "=" * 70)
print("XLA TRACE — the production program's device timeline")
print("=" * 70)

trace_dir = (
    Path(_cell_args.trace_dir)
    if _cell_args.trace_dir
    else Path(tempfile.mkdtemp(prefix="fixed_light_trace_"))
)
trace_dir.mkdir(parents=True, exist_ok=True)
print(f"  trace log dir: {trace_dir}")

trace_block: dict = {}
census_block: dict = {}
command_buffer_probe: dict = {}
vmap_block: dict | None = None

if VMAP_BATCH is None:
    # Phase 3: the PSF candidate nests INSIDE the solver injection around a
    # fresh jax.jit; outside PSF mode ``_psf_context`` is a no-op.
    with (
        library_solver_injection.certified_solver_injected(
            PASS_BUDGET, fallback=True, tau_rel=TAU_REL
        ) as _trace_counts,
        _psf_context() as _trace_psf_counts,
    ):
        _fn = _likelihood_fn(system_s3.dataset, _settings)

        with timer.section("trace_lower"):
            _lowered = jax.jit(_fn).lower(params_tree_s3)

        # ONE lowering, TWO compilations. The command-buffer knob is a per-compile
        # option, so both programs live in this process and both walls are measured
        # on the same lowering rather than across two runs.
        with timer.section("trace_compile_command_buffers_on"):
            _ex_on = _lowered.compile()
        with timer.section("trace_compile_command_buffers_off"):
            _ex_off = _lowered.compile(compiler_options={_COMMAND_BUFFER_OPTION: ""})

        _value_on = _ex_on(params_tree_s3)
        block(_value_on)
        _value_off = _ex_off(params_tree_s3)
        block(_value_off)

        if int(_trace_counts["jax"]) == 0:
            raise AssertionError(
                "the injected solver was never called on the JAX path while compiling the "
                "traced executable — the trace would decompose the library's own PDIP."
            )
        if (
            PSF_CANDIDATE is not None
            and PSF_CANDIDATE != "control"
            and int(_trace_psf_counts["jax"]) == 0
        ):
            raise AssertionError(
                f"the injected PSF candidate {PSF_CANDIDATE!r} was never called on the JAX path "
                f"while compiling the traced executable — the trace would decompose the "
                f"library's own convolution."
            )

        wall_on_ms = steady_wall_ms(_ex_on, params_tree_s3, TRACE_CALLS)
        wall_off_ms = steady_wall_ms(_ex_off, params_tree_s3, TRACE_CALLS)
        print(f"  command buffers ON  (production program): {wall_on_ms:8.3f} ms/call")
        print(f"  command buffers OFF (traced program):     {wall_off_ms:8.3f} ms/call")
        _delta_ms = wall_off_ms - wall_on_ms
        print(
            f"  delta (OFF - ON):                         {_delta_ms:8.3f} ms "
            f"({100.0 * _delta_ms / wall_on_ms:+.1f} %)"
        )

        _traced_ex = _ex_on if TRACE_ON_COMMAND_BUFFERS else _ex_off
        _traced_wall_ms = wall_on_ms if TRACE_ON_COMMAND_BUFFERS else wall_off_ms

        hlo_text = _traced_ex.as_text()
        module_proto = (
            _traced_ex.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()
        )
        # The optimized HLO and the module proto are written beside the trace, so the
        # census and the stage map can be re-derived from the SAME program later
        # without a 5-minute recompile — and so a disputed census row can be checked
        # against the text that produced it.
        (trace_dir / "hlo_optimized.txt").write_text(hlo_text)
        (trace_dir / "hlo_module.pb").write_bytes(module_proto)
        print(f"  optimized HLO + module proto written to {trace_dir}")

        stack_index = xla_attribution.StackFrameIndex.from_module_proto(module_proto)
        index = xla_attribution.hlo_index(hlo_text, stack_index)
        print(
            f"  optimized HLO: {len(index)} instructions, "
            f"{sum(1 for i in index.values() if i.frames)} resolved to source"
        )

        # The traced calls, and the SAME number of untraced calls immediately after,
        # so the profiler's own overhead is recorded rather than assumed negligible.
        _main_dir = trace_dir / ("traced_on" if TRACE_ON_COMMAND_BUFFERS else "traced_off")
        # The clock starts INSIDE the profiler context. ``jax.profiler.trace``'s own
        # start-up and its serialisation on exit are seconds, not milliseconds: timing
        # the whole context and dividing by K reported 145 ms/call against a 61 ms
        # untraced call on the shakeout, which is the profiler being switched on, not
        # a traced likelihood evaluation.
        with timer.section("trace_calls"):
            with jax.profiler.trace(
                str(_main_dir), create_perfetto_link=False, create_perfetto_trace=False
            ):
                # No warm-up call in here: the executable is already warm (both
                # command-buffer walls were measured above), and an extra call would
                # make the trace hold K+1 executions, which ``split_calls`` would
                # then refuse to split into K.
                _trace_t0 = time.perf_counter()
                for _ in range(TRACE_CALLS):
                    block(_traced_ex(params_tree_s3))
                traced_wall_ms = (time.perf_counter() - _trace_t0) / TRACE_CALLS * 1e3
        untraced_wall_ms = steady_wall_ms(_traced_ex, params_tree_s3, TRACE_CALLS)

        print(f"  traced   wall: {traced_wall_ms:8.3f} ms/call")
        print(f"  untraced wall: {untraced_wall_ms:8.3f} ms/call")

        events = xla_attribution.device_events(_main_dir)
        # Reconciled against the TRACED wall: these kernel durations came from the
        # traced run, and reconciling them against the untraced wall folds the
        # profiler's own overhead into the residual (it made host_outside_span_ms
        # negative on the first shakeout). The untraced wall is the production-
        # comparable number and is recorded beside it with its own percentage.
        trace_block = xla_attribution.attribute(
            events,
            index,
            wall_ms=traced_wall_ms,
            calls=TRACE_CALLS,
            untraced_wall_ms=untraced_wall_ms,
        )
        census_block = xla_attribution.hlo_census(index)
        if PSF_CANDIDATE is not None:
            # Phase 3: the census anchors are CHECKED against the installed
            # PyAutoArray rather than assumed (phase 2 had to exclude the whole
            # census when they moved). ``status`` covers the PSF rows; any other
            # row whose anchor moved is listed in ``rows_excluded``.
            census_block.update(xla_attribution.census_anchor_status())

        # The qhull pure_callback is a host round-trip. Look for it by name on the
        # host planes; when the profiler does not name it, the device-idle gap it
        # creates is the only evidence and ``largest_idle_gaps`` above carries it.
        #
        # Host events NEST: the shakeout showed ``pure_callback.4`` (4.03 ms)
        # containing ``_wrapped_callback`` containing ``pure_callback_impl``
        # containing ``scipy_delaunay_tri_only`` (2.51 ms, the actual qhull). Summing
        # every matching event counted the same work five times over (12.78 ms on a
        # 61 ms call). So the events are aggregated BY NAME, and two headline numbers
        # are reported: the outermost ``pure_callback*`` span (the whole round trip)
        # and the qhull call inside it.
        _callback_hosts = [
            e
            for e in xla_attribution.host_events(_main_dir)
            if any(k in e["name"].lower() for k in ("callback", "qhull", "delaunay", "host_send"))
        ]
        _by_name: dict[str, dict] = {}
        for _e in _callback_hosts:
            _row = _by_name.setdefault(_e["name"], {"count": 0, "ms": 0.0})
            _row["count"] += 1
            _row["ms"] += _e["dur_ns"] / 1e6 / TRACE_CALLS

        def _sum_host_ms(predicate) -> float:
            return sum(v["ms"] for k, v in _by_name.items() if predicate(k.lower()))

        host_callback_ms = _sum_host_ms(lambda n: n.startswith("pure_callback"))
        trace_block["host_callback_ms"] = host_callback_ms
        trace_block["host_callback"] = {
            "pure_callback_span_ms": host_callback_ms,
            "qhull_ms": _sum_host_ms(lambda n: "scipy_delaunay" in n),
            "events_by_name": dict(sorted(_by_name.items(), key=lambda kv: -kv[1]["ms"])),
            "note": (
                "Host-plane events, aggregated BY NAME because they NEST: pure_callback.N contains "
                "_wrapped_callback contains pure_callback_impl contains scipy_delaunay_tri_only "
                "(the qhull call). pure_callback_span_ms is the outermost round trip; qhull_ms is "
                "the triangulation inside it. These are HOST milliseconds and are NOT a row of the "
                "device table — the device pays for them as the idle gap at pure_callback.N, which "
                "largest_idle_gaps names."
            ),
        }

        # The command-buffer probe: does CUPTI still emit per-kernel events inside a
        # CUDA graph? Answering this with data is what lets the ON/OFF delta be
        # judged instead of assumed.
        _probe_dir = trace_dir / "probe_command_buffers_on"
        _probe_ex = _ex_off if TRACE_ON_COMMAND_BUFFERS else _ex_on
        block(_probe_ex(params_tree_s3))
        with jax.profiler.trace(
            str(_probe_dir), create_perfetto_link=False, create_perfetto_trace=False
        ):
            for _ in range(3):
                block(_probe_ex(params_tree_s3))
        _probe_events = xla_attribution.device_events(_probe_dir)
        _probe_index = xla_attribution.hlo_index(_probe_ex.as_text())
        _probe_total = sum(e["dur_ns"] for e in _probe_events) or 1
        _probe_joined = sum(e["dur_ns"] for e in _probe_events if e["hlo_op"] in _probe_index)
        command_buffer_probe = {
            "probed_program": "command_buffers_off"
            if TRACE_ON_COMMAND_BUFFERS
            else "command_buffers_on",
            "events": len(_probe_events),
            "kernel_names_present": len({e["name"] for e in _probe_events}) > 1,
            "distinct_hlo_op": len({e["hlo_op"] for e in _probe_events}),
            "joined_duration_pct": 100.0 * _probe_joined / _probe_total,
            "top_hlo_op_by_duration": sorted(
                {
                    str(e["hlo_op"]): sum(
                        x["dur_ns"] for x in _probe_events if x["hlo_op"] == e["hlo_op"]
                    )
                    / 1e6
                    for e in _probe_events
                }.items(),
                key=lambda kv: -kv[1],
            )[:6],
            "note": (
                "CUPTI emits per-kernel events inside a CUDA graph, but their hlo_op stat is the "
                "graph node (command_buffer_N), not the instruction. joined_duration_pct is how "
                "much of the device time the name-based join recovers on that program."
            ),
        }

else:
    # ===================================================================
    # PART V — the matched vmap-vs-scalar experiment (#273)
    # ===================================================================
    # Two arms, the SAME B lanes, one process, one solver injection. The whole
    # point is that the pair is matched: a vmap millisecond and a scalar
    # millisecond measured in different processes, on different models, or
    # under different fallback semantics are not a comparison.

    print("\n" + "-" * 70)
    print(
        f"BATCHED MODE — B={VMAP_BATCH} {LANES_MODE} lanes, arms {','.join(ARMS)}, "
        f"fallback {'on' if FALLBACK_ON else 'off'}"
        + (
            f", LIBRARY solver {LIBRARY_SOLVER} (budget {CERTIFIED_BUDGET})"
            if LIBRARY_SOLVER_MODE
            else ""
        )
    )
    print("-" * 70)

    # --- the lanes ---------------------------------------------------------
    # Lane k IS draw k of the phase-3 seeded random family, applied to the lens
    # MASS, with the lens light fixed at S3 — the same construction
    # ``fixed_light_draws.py`` uses (``flds.mass_from(BASE_MASS, offsets)``).
    # The S3 dataset is the FIDUCIAL light-subtracted one and is shared by every
    # lane, exactly as production shares one dataset across a batch.
    captured_window = None
    if LANES_MODE == "distinct":
        lane_draws = flds.random_draws(DRAW_SEED, VMAP_BATCH)
    elif LANES_MODE == "captured":
        # Phase C1 (#304): B lanes of REAL Nautilus proposal batches — the window
        # rule is nautilus_batches.timed_window (consecutive calls from the first
        # post-prior batch; at B == n_batch exactly one captured batch). The Draw
        # records the mass offsets from the fiducial for the lane table; the lane
        # TREE (mass and shear) is built from the captured vector itself.
        captured_window = nautilus_batches.timed_window(
            captured["call_index"], captured["call_phase"], VMAP_BATCH
        )
        _paths = captured["meta"]["parameter_paths"]
        _mass_keys = {
            "galaxies.lens.mass.centre.centre_0": "centre_0",
            "galaxies.lens.mass.centre.centre_1": "centre_1",
            "galaxies.lens.mass.ell_comps.ell_comps_0": "ell_comps_0",
            "galaxies.lens.mass.ell_comps.ell_comps_1": "ell_comps_1",
            "galaxies.lens.mass.einstein_radius": "einstein_radius",
        }
        lane_draws = []
        for _k in captured_window:
            _vals = dict(zip(_paths, (float(_x) for _x in captured["parameters"][_k])))
            lane_draws.append(
                flds.Draw(
                    name=(
                        f"call{int(captured['call_index'][_k]):05d}"
                        f"_lane{int(captured['lane_in_call'][_k]):02d}"
                    ),
                    kind="captured",
                    offsets={
                        _short: _vals[_long] - BASE_MASS[_short]
                        for _long, _short in _mass_keys.items()
                        if _long in _vals
                    },
                    index=int(_k),
                    mass_cls="Isothermal",
                )
            )
    else:
        lane_draws = [
            flds.Draw(name="fiducial", kind="fiducial", offsets={}, mass_cls="Isothermal")
            for _ in range(VMAP_BATCH)
        ]
    lane_draws_sha256 = hashlib.sha256(
        json.dumps(
            [
                {
                    "name": draw.name,
                    "kind": draw.kind,
                    "offsets": draw.offsets,
                    "mass_cls": draw.mass_cls,
                }
                for draw in lane_draws
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if captured_window is not None:
        # The captured lanes carry the shear too, which the Draw offsets do not.
        lane_draws_sha256 = hashlib.sha256(
            np.ascontiguousarray(captured["parameters"][captured_window]).tobytes()
        ).hexdigest()

    # The lane trees come from ``_lane_tree`` (hoisted above PART B so the
    # phase-3 draw pins build the SAME trees).

    with timer.section("vmap_lane_build"):
        if captured_window is None:
            lane_trees = [_lane_tree(_d) for _d in lane_draws]
        else:
            # Captured lanes are parameter VECTORS (see _captured_likelihood_fn).
            lane_trees = [jnp.asarray(captured["parameters"][_k]) for _k in captured_window]
        _fiducial_structure = (
            jax.tree_util.tree_structure(params_tree_s3)
            if captured_window is None
            else jax.tree_util.tree_structure(lane_trees[0])
        )
        for _i, _t in enumerate(lane_trees):
            if jax.tree_util.tree_structure(_t) != _fiducial_structure:
                raise AssertionError(
                    f"lane {_i} has a different pytree structure from the fiducial S3 tree — "
                    f"the stacked batch would not be the same model family. (Static aux data "
                    f"such as the Pixelization and Regularization objects compares by IDENTITY, "
                    f"so every lane must SHARE them, not hold a copy.)"
                )
        batched_tree = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lane_trees)

    print(
        f"  lanes built: {len(lane_trees)} ({LANES_MODE}, "
        + (f"seed {DRAW_SEED})" if captured_window is None else "captured window)")
    )
    for _d in lane_draws[: min(4, len(lane_draws))]:
        _shown = {_p: round(float(_o), 5) for _p, _o in _d.offsets.items()}
        print(f"    {_d.name:<14} {_shown}")
    if len(lane_draws) > 4:
        print(f"    ... {len(lane_draws) - 4} more")

    arm_blocks: dict[str, dict] = {}
    lane_rows: list[dict] = []
    lane_pins: list[dict] = []
    report_rows: list[tuple[int, bool]] = []
    report_source = None

    def _steady_invoke_ms(invoke, n_repeats: int) -> float:
        """Mean per-INVOCATION wall of *n_repeats* blocked invocations.

        One invocation is one batch: a single call for the vmap arm, ``B``
        sequential blocked calls for the scalar arm.
        """
        block(invoke())
        _start = time.perf_counter()
        for _ in range(n_repeats):
            block(invoke())
        return (time.perf_counter() - _start) / n_repeats * 1e3

    def _host_callback_rows(log_dir) -> dict:
        """The phase-1 host-plane aggregation, per traced call.

        Host events NEST (``pure_callback.N`` contains ``_wrapped_callback``
        contains ``pure_callback_impl`` contains the qhull body), so they are
        aggregated BY NAME and two headline numbers are reported rather than a
        sum that counts the same work several times over.
        """
        _hosts = [
            _e
            for _e in xla_attribution.host_events(log_dir)
            if any(
                _k in _e["name"].lower() for _k in ("callback", "qhull", "delaunay", "host_send")
            )
        ]
        _by_name: dict[str, dict] = {}
        for _e in _hosts:
            _row = _by_name.setdefault(_e["name"], {"count": 0, "ms": 0.0})
            _row["count"] += 1
            _row["ms"] += _e["dur_ns"] / 1e6 / TRACE_CALLS
        _span = sum(
            _v["ms"] for _k, _v in _by_name.items() if _k.lower().startswith("pure_callback")
        )
        # The probe replaces the library body, and the profiler names a host event
        # from the CODE OBJECT rather than __name__ -- so under the probe there is
        # no "scipy_delaunay" event at all and a phase-1 matcher reports 0.0 ms of
        # qhull in silence. The probe's own event names are matched here as well,
        # and the finer qhull-vs-tables split comes from the probe's records.
        _body_keys = ("scipy_delaunay", *host_callback_probe.PROBE_HOST_EVENT_FRAGMENTS)
        return {
            "pure_callback_span_ms": _span,
            "qhull_ms": max(
                (
                    _v["ms"]
                    for _k, _v in _by_name.items()
                    if any(_f in _k.lower() for _f in _body_keys)
                ),
                default=0.0,
            ),
            "callback_body_ms_note": (
                "the OUTERMOST matching body event, not a sum: under the probe the body "
                "appears twice (the wrapper and the timed re-implementation it calls) and "
                "summing them would count the same work twice"
            ),
            "events_by_name": dict(sorted(_by_name.items(), key=lambda kv: -kv[1]["ms"])),
            "note": (
                "HOST milliseconds per traced BATCH (the whole invocation), aggregated by name "
                "because the events nest. Not a row of the device table: the device pays for "
                "them as the idle gaps largest_idle_gaps names."
            ),
        }

    def _run_arm(arm: str, scalar_fn, vmap_fn) -> dict:
        """Compile, time and trace one arm on the B lanes.

        *scalar_fn* is ``jax.jit(fn)`` and *vmap_fn* is
        ``jax.jit(jax.vmap(fn))``: the current ``Fitness._vmap`` composition.
        """
        print(f"\n  --- arm {arm} ---")

        if arm == "vmap":
            # The current production nesting (PyAutoFit#1638).
            _prod = vmap_fn

            def _invoke_prod():
                return _prod(batched_tree)

            # The production wrapper is itself jitted, so its lowering is the
            # exact current batched program and accepts the command-buffer knob.
            with timer.section("vmap_arm_lower"):
                _lowered = _prod.lower(batched_tree)
            _hlo_source = (
                "jax.jit(jax.vmap(fn)).lower(batched_tree) — the exact current "
                "Fitness._vmap composition (PyAutoFit#1638)"
            )

            def _invoke_ex(ex):
                return ex(batched_tree)

            _expected_calls = TRACE_CALLS
            _executions_per_invoke = 1
        else:
            _prod = scalar_fn

            def _invoke_prod():
                return [_prod(_t) for _t in lane_trees]

            with timer.section("scalar_arm_lower"):
                _lowered = scalar_fn.lower(lane_trees[0])
            _hlo_source = "jax.jit(fn).lower(lane_trees[0]) — the same program phase 1 traced"

            def _invoke_ex(ex):
                return [ex(_t) for _t in lane_trees]

            # One traced invocation is B executions of one program, so the trace
            # is split by TRACE_CALLS * B. DECIDED IN CODE: pooling the lanes
            # gives B times the blocks and therefore a per-stage median over
            # every lane, which is what a straggler shows up in; splitting per
            # lane would give B tables of TRACE_CALLS blocks and no pooled
            # median at all.
            _expected_calls = TRACE_CALLS * VMAP_BATCH
            _executions_per_invoke = VMAP_BATCH

        with timer.section(f"{arm}_arm_compile_command_buffers_on"):
            _ex_on = _lowered.compile()
        with timer.section(f"{arm}_arm_compile_command_buffers_off"):
            _ex_off = _lowered.compile(compiler_options={_COMMAND_BUFFER_OPTION: ""})

        _wall_prod = _steady_invoke_ms(_invoke_prod, TRACE_CALLS)
        _wall_on = _steady_invoke_ms(lambda: _invoke_ex(_ex_on), TRACE_CALLS)
        _wall_off = _steady_invoke_ms(lambda: _invoke_ex(_ex_off), TRACE_CALLS)
        print(f"    batch wall, production nesting:      {_wall_prod:9.3f} ms")
        print(f"    batch wall, AOT command buffers ON:  {_wall_on:9.3f} ms")
        print(f"    batch wall, AOT command buffers OFF: {_wall_off:9.3f} ms")

        _hlo_text = _ex_off.as_text()
        _proto = _ex_off.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()
        (trace_dir / f"hlo_optimized_{arm}.txt").write_text(_hlo_text)
        (trace_dir / f"hlo_module_{arm}.pb").write_bytes(_proto)
        _index = xla_attribution.hlo_index(
            _hlo_text, xla_attribution.StackFrameIndex.from_module_proto(_proto)
        )
        print(
            f"    optimized HLO: {len(_index)} instructions, "
            f"{sum(1 for _i in _index.values() if _i.frames)} resolved to source"
        )

        _arm_dir = trace_dir / f"traced_{arm}"
        probe.reset()
        with timer.section(f"trace_calls_{arm}"):
            with jax.profiler.trace(
                str(_arm_dir), create_perfetto_link=False, create_perfetto_trace=False
            ):
                _t0 = time.perf_counter()
                for _ in range(TRACE_CALLS):
                    block(_invoke_ex(_ex_off))
                _traced_batch_ms = (time.perf_counter() - _t0) / TRACE_CALLS * 1e3
        _callback_records = [_r.as_dict() for _r in probe.records]
        _probe_summary = probe.summary()
        _callbacks_per_call = probe.calls / TRACE_CALLS
        if MESH == "rectangular" and probe.calls != 0:
            # The rectangular mesh has no qhull host callback at all. A non-zero
            # count means the program is not the one this row claims to be.
            raise AssertionError(
                f"arm {arm}: {probe.calls} qhull callbacks on the RECTANGULAR mesh, which has "
                f"none — the traced program is not the rectangular likelihood"
            )
        _untraced_batch_ms = _steady_invoke_ms(lambda: _invoke_ex(_ex_off), TRACE_CALLS)

        _events = xla_attribution.device_events(_arm_dir)
        _block = xla_attribution.attribute(
            _events,
            _index,
            wall_ms=_traced_batch_ms / _executions_per_invoke,
            calls=_expected_calls,
            untraced_wall_ms=_untraced_batch_ms / _executions_per_invoke,
        )
        # The vmap arm's "call" is the whole batch, so its stage rows are per
        # BATCH; the scalar arm's are already per likelihood. Both are divided
        # to a per-LANE row so the two arms can be put side by side.
        _per_lane_divisor = VMAP_BATCH if arm == "vmap" else 1
        _block["per_stage_ms_per_lane"] = {
            _label: _row["median_ms"] / _per_lane_divisor
            for _label, _row in _block["per_stage_ms"].items()
        }
        _block["per_lane_divisor"] = _per_lane_divisor
        _block["host_callback"] = _host_callback_rows(_arm_dir)
        _block["host_callback_ms"] = _block["host_callback"]["pure_callback_span_ms"]

        _row = {
            "arm": arm,
            "program": (
                "jax.jit(jax.vmap(fn))" if arm == "vmap" else "jax.jit(fn), called B times"
            ),
            "stands_for": (
                "Nautilus(use_jax_vmap=True) — Fitness._vmap"
                if arm == "vmap"
                else "Nautilus(use_jax_vmap=False, use_jax_jit=True). Plain use_jax_vmap=False "
                "is UNJITTED and is NOT this arm"
            ),
            "hlo_source": _hlo_source,
            "batch_wall_ms": _wall_prod,
            "wall_production_nesting_ms": _wall_prod,
            "wall_aot_command_buffers_on_ms": _wall_on,
            "wall_aot_command_buffers_off_ms": _wall_off,
            "command_buffer_delta_ms": _wall_off - _wall_on,
            "production_nesting_vs_aot_on_delta_ms": _wall_prod - _wall_on,
            "production_nesting_vs_aot_on_delta_pct": 100.0 * (_wall_prod - _wall_on) / _wall_on,
            "wall_per_lane_ms": _wall_prod / VMAP_BATCH,
            "wall_per_lane_command_buffers_off_ms": _wall_off / VMAP_BATCH,
            "traced_batch_wall_ms": _traced_batch_ms,
            "untraced_batch_wall_ms": _untraced_batch_ms,
            "traced_wall_per_lane_ms": _traced_batch_ms / VMAP_BATCH,
            "callback_count_per_call": _callbacks_per_call,
            "callback_count_per_lane": _callbacks_per_call / VMAP_BATCH,
            "callback_expectation": (
                "0 — the rectangular mesh has no qhull host callback (asserted)"
                if MESH == "rectangular"
                else (
                    "B per batched call — the qhull pure_callback is vmap_method='sequential'"
                    if arm == "vmap"
                    else "1 per likelihood evaluation, B per batch"
                )
            ),
            "host_qhull_ms": _probe_summary["qhull_ms_total"] / TRACE_CALLS,
            "host_tables_ms": _probe_summary["tables_ms_total"] / TRACE_CALLS,
            "host_qhull_ms_per_lane": _probe_summary["qhull_ms_total"] / TRACE_CALLS / VMAP_BATCH,
            "host_tables_ms_per_lane": _probe_summary["tables_ms_total"] / TRACE_CALLS / VMAP_BATCH,
            "host_callback_probe": _probe_summary,
            "host_callback_records_first": _callback_records[: min(4, len(_callback_records))],
            "device_idle_ms": _block["device_idle_ms"],
            "device_idle_ms_per_lane": _block["device_idle_ms"] / _per_lane_divisor,
            "trace": _block,
            "hlo_census": {
                "status": "excluded",
                "reason": (
                    "The legacy census anchors PyAutoArray source line numbers that moved after "
                    "the phase-1 run. No batched attribution is published from unsupported "
                    "anchors; the device-event table and wall reconciliation remain valid."
                ),
            },
            "trace_split": {
                "expected_calls": _expected_calls,
                "executions_per_invocation": _executions_per_invoke,
                "method": _block["call_split"],
                "note": (
                    "DECIDED IN CODE: the scalar arm's trace is split by trace_calls * B, so "
                    "every lane's execution is its own block and the per-stage median is taken "
                    "over all of them. The vmap arm's is split by trace_calls, one block per "
                    "batched execution, and per_stage_ms_per_lane divides those rows by B."
                ),
            },
        }
        print(
            f"    callbacks/call {_callbacks_per_call:6.2f}  "
            f"qhull {_row['host_qhull_ms']:7.3f} ms  tables {_row['host_tables_ms']:7.3f} ms  "
            f"device_idle {_block['device_idle_ms']:7.3f} ms  "
            f"reconciliation {_block['reconciliation_pct']:+.2f} %"
        )
        return _row

    # Compile the independent library-PDIP reference before installing either
    # diagnostic monkeypatch. This is an untimed numerical gate over the exact
    # same lane trees; agreement between the two harness arms is insufficient.
    # (Library mode builds it from ``_settings_pdip_reference``, which names PDIP
    # explicitly; harness mode passes ``_settings`` exactly as phase 2 did.)
    _fn_library_pdip = _lane_likelihood_fn(system_s3.dataset, _settings_pdip_reference)
    _library_pdip_fn = jax.jit(jax.vmap(_fn_library_pdip))
    with timer.section("vmap_lane_library_pdip_reference"):
        _ll_library_pdip = np.asarray(block(_library_pdip_fn(batched_tree)), dtype=float)
    if _ll_library_pdip.shape != (VMAP_BATCH,):
        raise AssertionError(
            f"the library-PDIP reference returned shape {_ll_library_pdip.shape}, not "
            f"({VMAP_BATCH},)"
        )

    # --- phase B (#300): the per-composition references and the solver check ---
    # Phase 3 placed phase 2's ~2.5e-9 residual BETWEEN the jit(vmap) and scalar
    # jit compositions, not in the solver. So library mode pins each arm against
    # a library-PDIP reference of ITS OWN composition: the jit(vmap) reference
    # above for the vmap arm, and this scalar-jit reference for the scalar arm.
    _ll_library_pdip_scalar = None
    library_solver_record: dict | None = None
    if LIBRARY_SOLVER_MODE:
        _library_pdip_scalar_fn = jax.jit(_fn_library_pdip)
        with timer.section("vmap_lane_library_pdip_scalar_reference"):
            _ll_library_pdip_scalar = np.asarray(
                [float(block(_library_pdip_scalar_fn(_t))) for _t in lane_trees], dtype=float
            )

        # The dispatch the library itself reports on the fiducial inversion. This
        # replaces the harness injection's call counter: a row that asked for the
        # certified solver and silently ran PDIP fails here, loudly.
        def _fiducial_solver_used(settings_for_check) -> str:
            _analysis = al.AnalysisImaging(
                dataset=system_s3.dataset,
                adapt_images=adapt_images,
                settings=settings_for_check,
                use_jax=True,
            )
            _fit = _analysis.fit_from(instance=params_tree_s3)
            return str(_fit.inversion.positive_only_solver_used)

        _solver_used = _fiducial_solver_used(_settings_library)
        _reference_solver_used = _fiducial_solver_used(_settings_pdip_reference)
        if _solver_used != LIBRARY_SOLVER:
            raise AssertionError(
                f"--solver {LIBRARY_SOLVER} was requested, but the fiducial inversion's "
                f"positive_only_solver_used is {_solver_used!r} — every row of this table "
                f"would be the wrong solver wearing the requested label"
            )
        if _reference_solver_used != "pdip":
            raise AssertionError(
                f"the library-PDIP reference dispatches {_reference_solver_used!r}, not 'pdip'"
            )
        library_solver_record = {
            "solver_used": _solver_used,
            "solver_used_source": (
                "AbstractInversion.positive_only_solver_used on the fiducial S3 inversion, "
                "built eagerly on the JAX backend (AnalysisImaging(use_jax=True).fit_from) "
                "with the arms' Settings"
            ),
            "reference_solver_used": _reference_solver_used,
        }
        print(
            f"  library solver dispatch: {_solver_used} (requested {LIBRARY_SOLVER}); "
            f"reference {_reference_solver_used}"
        )

    # One probe and one solver over BOTH arms: the host wrapper and the solver
    # are identical for the pair, or the pair is not matched. Harness mode
    # installs the scoped injection; library mode installs NOTHING — the solver
    # is the library's own, selected by ``_settings_library``.
    _solver_context = (
        contextlib.nullcontext({"jax": 0, "numpy": 0})
        if LIBRARY_SOLVER_MODE
        else library_solver_injection.certified_solver_injected(
            PASS_BUDGET, fallback=FALLBACK_ON, tau_rel=TAU_REL
        )
    )
    _arm_settings = _settings_library if LIBRARY_SOLVER_MODE else _settings
    with (
        host_callback_probe.qhull_probe() as probe,
        _solver_context as _trace_counts,
    ):
        # The wrappers reproduce current Fitness._vmap and the scalar-jit
        # control. Both close over the same likelihood function.
        _fn_v = _lane_likelihood_fn(system_s3.dataset, _arm_settings)
        _scalar_pin_fn = jax.jit(_fn_v)
        _vmap_pin_fn = jax.jit(jax.vmap(_fn_v))

        # --- the per-lane pin, un-timed ------------------------------------
        with timer.section("vmap_lane_log_likelihoods"):
            _ll_vmap = np.asarray(block(_vmap_pin_fn(batched_tree)), dtype=float)
            _ll_scalar = np.asarray(
                [float(block(_scalar_pin_fn(_t))) for _t in lane_trees], dtype=float
            )
        if _ll_vmap.shape != (VMAP_BATCH,):
            raise AssertionError(
                f"the vmap arm returned shape {_ll_vmap.shape}, not ({VMAP_BATCH},) — "
                f"the batch axis is not the lane axis"
            )

        if not LIBRARY_SOLVER_MODE:
            for _k, _draw in enumerate(lane_draws):
                _abs = abs(_ll_vmap[_k] - _ll_scalar[_k])
                _rel = _abs / max(abs(_ll_scalar[_k]), 1e-300)
                _rel_vmap_pdip = abs(_ll_vmap[_k] - _ll_library_pdip[_k]) / max(
                    abs(_ll_library_pdip[_k]), 1e-300
                )
                _rel_scalar_pdip = abs(_ll_scalar[_k] - _ll_library_pdip[_k]) / max(
                    abs(_ll_library_pdip[_k]), 1e-300
                )
                _status = (
                    "PASS" if max(_rel, _rel_vmap_pdip, _rel_scalar_pdip) <= LANE_RTOL else "FAIL"
                )
                lane_rows.append(
                    {
                        "lane": _k,
                        "draw_name": _draw.name,
                        "draw_kind": _draw.kind,
                        "offsets": {_p: float(_o) for _p, _o in _draw.offsets.items()},
                        "log_likelihood_vmap": float(_ll_vmap[_k]),
                        "log_likelihood_scalar": float(_ll_scalar[_k]),
                        "log_likelihood_library_pdip": float(_ll_library_pdip[_k]),
                        "abs_diff_nats": float(_abs),
                        "rel_diff": float(_rel),
                        "rel_diff_vmap_vs_library_pdip": float(_rel_vmap_pdip),
                        "rel_diff_scalar_vs_library_pdip": float(_rel_scalar_pdip),
                        "status": _status,
                        # Filled by the reporting pass below when it runs.
                        "certified": None,
                        "pass_at_certification": None,
                        "pdip_iter": None,
                    }
                )
            lane_pins = []
            for _r in lane_rows:
                for _label, _got_key, _ref_key, _rel_key in (
                    (
                        "current jit(vmap) == scalar jit",
                        "log_likelihood_vmap",
                        "log_likelihood_scalar",
                        "rel_diff",
                    ),
                    (
                        "current jit(vmap) == library PDIP",
                        "log_likelihood_vmap",
                        "log_likelihood_library_pdip",
                        "rel_diff_vmap_vs_library_pdip",
                    ),
                    (
                        "scalar jit == library PDIP",
                        "log_likelihood_scalar",
                        "log_likelihood_library_pdip",
                        "rel_diff_scalar_vs_library_pdip",
                    ),
                ):
                    lane_pins.append(
                        {
                            "pin": f"lane {_r['lane']}: {_label}",
                            "rtol": LANE_RTOL,
                            "got": _r[_got_key],
                            "reference_value": _r[_ref_key],
                            "rel_diff": _r[_rel_key],
                            "status": "PASS" if _r[_rel_key] <= LANE_RTOL else "FAIL",
                        }
                    )
            _n_failed = sum(1 for _r in lane_rows if _r["status"] == "FAIL")
            _worst_rel = max(
                max(
                    _r["rel_diff"],
                    _r["rel_diff_vmap_vs_library_pdip"],
                    _r["rel_diff_scalar_vs_library_pdip"],
                )
                for _r in lane_rows
            )
            print(
                f"\n  per-lane harness and library-PDIP pins <= {LANE_RTOL:.0e} relative: "
                f"{len(lane_rows) - _n_failed}/{len(lane_rows)} PASS  "
                f"(worst rel {_worst_rel:.3e})"
            )
        else:
            # The PRE-REGISTERED phase-B gate (module docstring, and the submit
            # header): per lane, each arm against the library-PDIP reference of
            # its OWN composition, at LANE_RTOL relative — GATED. The two
            # cross-composition differences (arm vs arm, reference vs reference)
            # are RECORDED with their nats and are NOT gated.
            def _rel_of(got, ref):
                return abs(got - ref) / max(abs(ref), 1e-300)

            for _k, _draw in enumerate(lane_draws):
                _ref_v = float(_ll_library_pdip[_k])
                _ref_s = float(_ll_library_pdip_scalar[_k])
                _got_v = float(_ll_vmap[_k])
                _got_s = float(_ll_scalar[_k])
                _rel_vv = _rel_of(_got_v, _ref_v)
                _rel_ss = _rel_of(_got_s, _ref_s)
                lane_rows.append(
                    {
                        "lane": _k,
                        "draw_name": _draw.name,
                        "draw_kind": _draw.kind,
                        "offsets": {_p: float(_o) for _p, _o in _draw.offsets.items()},
                        "log_likelihood_vmap": _got_v,
                        "log_likelihood_scalar": _got_s,
                        "log_likelihood_library_pdip": _ref_v,
                        "log_likelihood_library_pdip_scalar": _ref_s,
                        # GATED — same composition.
                        "rel_diff_vmap_vs_library_pdip": _rel_vv,
                        "rel_diff_scalar_vs_library_pdip_scalar": _rel_ss,
                        # RECORDED — across compositions.
                        "abs_diff_nats": abs(_got_v - _got_s),
                        "rel_diff": _rel_of(_got_v, _got_s),
                        "rel_diff_scalar_vs_library_pdip": _rel_of(_got_s, _ref_v),
                        "abs_diff_library_pdip_vmap_vs_scalar_nats": abs(_ref_v - _ref_s),
                        "rel_diff_library_pdip_vmap_vs_scalar": _rel_of(_ref_v, _ref_s),
                        "gated": [
                            "rel_diff_vmap_vs_library_pdip",
                            "rel_diff_scalar_vs_library_pdip_scalar",
                        ],
                        "status": "PASS" if max(_rel_vv, _rel_ss) <= LANE_RTOL else "FAIL",
                        # Filled by the observed reporting pass below.
                        "certified": None,
                        "passes": None,
                        "pdip_iter": None,
                    }
                )
            lane_pins = []
            for _r in lane_rows:
                for _label, _got_key, _ref_key, _rel_key, _gated in (
                    (
                        "jit(vmap) arm == jit(vmap) library PDIP",
                        "log_likelihood_vmap",
                        "log_likelihood_library_pdip",
                        "rel_diff_vmap_vs_library_pdip",
                        True,
                    ),
                    (
                        "scalar jit arm == scalar jit library PDIP",
                        "log_likelihood_scalar",
                        "log_likelihood_library_pdip_scalar",
                        "rel_diff_scalar_vs_library_pdip_scalar",
                        True,
                    ),
                    (
                        "jit(vmap) arm vs scalar jit arm (cross-composition)",
                        "log_likelihood_vmap",
                        "log_likelihood_scalar",
                        "rel_diff",
                        False,
                    ),
                    (
                        "jit(vmap) library PDIP vs scalar jit library PDIP (cross-composition)",
                        "log_likelihood_library_pdip",
                        "log_likelihood_library_pdip_scalar",
                        "rel_diff_library_pdip_vmap_vs_scalar",
                        False,
                    ),
                ):
                    _within = _r[_rel_key] <= LANE_RTOL
                    lane_pins.append(
                        {
                            "pin": f"lane {_r['lane']}: {_label}",
                            "rtol": LANE_RTOL,
                            "gated": _gated,
                            "got": _r[_got_key],
                            "reference_value": _r[_ref_key],
                            "rel_diff": _r[_rel_key],
                            "abs_diff_nats": abs(_r[_got_key] - _r[_ref_key]),
                            "within_rtol": bool(_within),
                            "status": ("PASS" if _within else "FAIL") if _gated else "RECORDED",
                        }
                    )
            _n_failed = sum(1 for _r in lane_rows if _r["status"] == "FAIL")
            _worst_gated = max(
                max(
                    _r["rel_diff_vmap_vs_library_pdip"],
                    _r["rel_diff_scalar_vs_library_pdip_scalar"],
                )
                for _r in lane_rows
            )
            _worst_cross = max(_r["rel_diff"] for _r in lane_rows)
            print(
                f"\n  per-lane same-composition library-PDIP pins <= {LANE_RTOL:.0e} relative "
                f"(GATED): {len(lane_rows) - _n_failed}/{len(lane_rows)} PASS  "
                f"(worst rel {_worst_gated:.3e}); cross-composition vmap vs scalar (recorded): "
                f"worst rel {_worst_cross:.3e}, "
                f"worst {max(_r['abs_diff_nats'] for _r in lane_rows):.3e} nats"
            )

        # --- the arms ------------------------------------------------------
        for _arm_name in ARMS:
            arm_blocks[_arm_name] = _run_arm(_arm_name, _scalar_pin_fn, _vmap_pin_fn)

        if not LIBRARY_SOLVER_MODE and int(_trace_counts["jax"]) == 0:
            raise AssertionError(
                "the injected solver was never called on the JAX path — every row of this "
                "table would be the library's own PDIP wearing the certified label."
            )

    # --- the per-lane certification report, un-timed -----------------------
    # A SEPARATE injection context with the report hook, and therefore SEPARATE
    # jax.jit objects: a jitted function compiled under one injection is cached,
    # so reusing the timed arm's wrapper here would hand back the program
    # compiled WITHOUT the hook (or, run the other way round, would leave the
    # ordered host callback inside every timed call).
    _report_arm = "vmap" if "vmap" in ARMS else "scalar"

    def _collect_report(pass_at, certified):
        report_rows.append((int(pass_at), bool(certified)))

    if not LIBRARY_SOLVER_MODE:
        with library_solver_injection.certified_solver_injected(
            PASS_BUDGET, fallback=FALLBACK_ON, tau_rel=TAU_REL, report=_collect_report
        ) as _report_counts:
            _fn_report = _likelihood_fn(system_s3.dataset, _settings)
            with timer.section("vmap_certification_report"):
                if _report_arm == "vmap":
                    block(jax.jit(jax.vmap(_fn_report))(batched_tree))
                else:
                    _rep_fn = jax.jit(_fn_report)
                    for _t in lane_trees:
                        block(_rep_fn(_t))
        report_source = (
            f"{_report_arm} arm, jax.debug.callback(..., ordered=True) inside the injected "
            f"certified solver; ordered because the unordered batching rule returns the lanes "
            f"permuted"
        )

        if len(report_rows) == len(lane_rows):
            for _r, (_pass_at, _cert) in zip(lane_rows, report_rows):
                _r["certified"] = bool(_cert)
                _r["pass_at_certification"] = int(_pass_at) if _pass_at > 0 else None
        print(
            f"  certification report: {len(report_rows)} rows for {len(lane_rows)} lanes "
            f"({sum(1 for _, _c in report_rows if _c)} certified within budget {PASS_BUDGET})"
        )
    else:
        # Library mode: the library's own solve, OBSERVED — a forwarding wrapper
        # that hands the library a ``stats`` out-dict (PyAutoArray #566) and emits
        # its traced ``passes``/``certified`` through an ordered debug callback.
        # It changes nothing that is computed, and it runs on a separately
        # compiled program that is never timed.
        with library_solver_injection.library_solver_observed(report=_collect_report) as _observed:
            _fn_report = _lane_likelihood_fn(system_s3.dataset, _settings_library)
            with timer.section("vmap_certification_report"):
                if _report_arm == "vmap":
                    _ll_report = np.asarray(
                        block(jax.jit(jax.vmap(_fn_report))(batched_tree)), dtype=float
                    )
                else:
                    _rep_fn = jax.jit(_fn_report)
                    _ll_report = np.asarray(
                        [float(block(_rep_fn(_t))) for _t in lane_trees], dtype=float
                    )
        _jax_solvers = sorted(set(_observed["solvers_jax"]))
        if _jax_solvers != [LIBRARY_SOLVER]:
            raise AssertionError(
                f"the library's positive-only solve was traced with solver={_jax_solvers}, not "
                f"[{LIBRARY_SOLVER!r}] — the dispatch the fiducial check reported is not the "
                f"one the batched program ran"
            )
        _ll_arm = _ll_vmap if _report_arm == "vmap" else _ll_scalar
        _report_rel = float(
            np.max(np.abs(_ll_report - _ll_arm) / np.maximum(np.abs(_ll_arm), 1e-300))
        )
        report_source = (
            f"{_report_arm} arm, library solve observed through its stats= out-dict "
            f"(PyAutoArray #566) and jax.debug.callback(..., ordered=True); ordered because "
            f"the unordered batching rule returns the lanes permuted. Untimed, separately "
            f"compiled; max rel diff of its log likelihoods vs the timed arm's: "
            f"{_report_rel:.3e}"
        )
        if LIBRARY_SOLVER == "certified" and len(report_rows) == len(lane_rows):
            for _r, (_passes, _cert) in zip(lane_rows, report_rows):
                _r["certified"] = bool(_cert)
                _r["passes"] = int(_passes)
        library_solver_record.update(
            {
                "traced_solver_kwargs_jax": _jax_solvers,
                "observed_calls_jax": int(_observed["jax"]),
                "report_rows": len(report_rows),
                "report_vs_timed_arm_max_rel_diff": _report_rel,
            }
        )
        if LIBRARY_SOLVER == "certified":
            print(
                f"  certification report: {len(report_rows)} rows for {len(lane_rows)} lanes "
                f"({sum(1 for _, _c in report_rows if _c)} certified within budget "
                f"{CERTIFIED_BUDGET}; fallback {LIBRARY_FALLBACK})"
            )
        else:
            print("  certification report: not applicable (library PDIP row)")

    # --- what the phase-1 keys mean in this mode ---------------------------
    # The scalar arm IS the phase-1 program (jax.jit(fn)), so it fills the
    # top-level trace/census/command-buffer keys and the printed stage table
    # below; the vmap arm's own full block lives under `vmap`. When only the
    # vmap arm was asked for, those keys carry the vmap arm and say so.
    _headline = arm_blocks.get("scalar") or arm_blocks["vmap"]
    trace_block = _headline["trace"]
    census_block = _headline["hlo_census"]
    wall_on_ms = _headline["wall_aot_command_buffers_on_ms"]
    wall_off_ms = _headline["wall_aot_command_buffers_off_ms"]
    traced_wall_ms = trace_block["wall_ms"]
    untraced_wall_ms = trace_block["untraced_wall_ms"]
    command_buffer_probe = {
        "note": (
            "Not taken in --vmap-batch mode: the CUPTI-inside-a-CUDA-graph probe is a phase-1 "
            "diagnostic about the JOIN, and this mode already compiles and times both the "
            "command-buffers ON and OFF executable of every arm."
        )
    }

    vmap_block = {
        "batch": int(VMAP_BATCH),
        "lanes_mode": LANES_MODE,
        "arms": list(ARMS),
        "draw_seed": DRAW_SEED,
        "fallback": FALLBACK_ON,
        "composition": "jax.jit(jax.vmap(fn))",
        "composition_provenance": (
            "Current Fitness._vmap after PyAutoFit#1638. Array 343376 used the retired "
            "jax.vmap(jax.jit(fn)) composition and is historical evidence only."
        ),
        "headline_arm": "scalar" if "scalar" in arm_blocks else "vmap",
        "lane_rtol": LANE_RTOL,
        "lanes": lane_rows,
        "lane_pins": lane_pins,
        "lanes_failed": sum(1 for _r in lane_rows if _r["status"] == "FAIL"),
        "arm_rows": arm_blocks,
        "certification_report": {
            "source": report_source,
            "rows": len(report_rows),
            "pdip_iter": (
                "NOT AVAILABLE per lane without changing library numerics. The fallback branch "
                "calls the library's own reconstruction_positive_only_from, which returns the "
                "solution alone — jax_nnls.solve_nnls' iteration count is discarded inside the "
                "library's custom_vjp primal. The PDIP while_loop's cost is in the pdip_solve "
                "stage row of each arm's device table instead, which is a max-over-lanes "
                "quantity under vmap because every lane runs the same fixed kernel sequence."
            ),
        },
        "lane_construction": {
            "draws_sha256": lane_draws_sha256,
            "base_mass": {_k: float(_v) for _k, _v in BASE_MASS.items()},
            "sigma_scale": flds.RANDOM_SIGMA_SCALE,
            "parameters": list(flds.RANDOM_PARAMETERS),
            "note": (
                "Lane k is random draw k of the phase-3 seeded family applied to the lens mass "
                "(Isothermal), with the lens light FIXED at S3 and the S3 light-subtracted "
                "dataset shared by every lane — one dataset, B parameter vectors, as production "
                "batches. 'identical' broadcasts the fiducial and is the CONTROL: identical "
                "lanes share a straggler and hide the cost this experiment looks for."
            ),
        },
    }
    if captured_window is not None:
        # Phase C1 (#304). Only in captured mode, so a phase-B JSON written by this
        # checkout keeps its key set.
        _win_calls = captured["call_index"][captured_window]
        _fom = captured["figure_of_merit"][captured_window]
        _valid = np.isfinite(_fom) & (_fom > -1.0e98)
        _rel_capture = np.abs(_ll_library_pdip - _fom) / np.maximum(np.abs(_fom), 1e-300)
        vmap_block["lane_construction"] = {
            "draws_sha256": lane_draws_sha256,
            "lane_rows": captured_window.tolist(),
            "calls": sorted({int(_c) for _c in _win_calls}),
            "stage": CAPTURED_STAGE,
            "regularization_range": _regularization_range(captured_window),
            "phases": sorted({str(captured["call_phase"][_c]) for _c in _win_calls}),
            "window_rule": (
                "nautilus_batches.timed_window: consecutive captured calls, in call order, from "
                "the FIRST call that is not a prior batch, concatenated and truncated to B; at B "
                "== n_batch exactly one real Nautilus batch. Falls back to the latest B lanes if "
                "the post-prior lanes run out; never repeats a lane."
            ),
            "note": (
                "Lane k is captured lane lane_rows[k]: the mass AND shear of a real Nautilus "
                "proposal (nautilus_batch_capture.py), lens light fixed at S3, the S3 "
                "light-subtracted dataset shared by every lane. `offsets` in the lane table are "
                "the mass offsets from the fiducial; the shear is in the capture file."
            ),
        }
        vmap_block["captured"] = {
            **captured_record,
            "timed_window_vs_capture_pdip": {
                "reference": (
                    "the capture's own figure of merit (library PDIP, Fitness.call_wrap "
                    "jax.jit(jax.vmap(call)) over parameter vectors) for the same lanes"
                ),
                "compared_against": "this cell's jit(vmap) library-PDIP reference",
                "n_compared": int(_valid.sum()),
                "max_rel_diff": float(_rel_capture[_valid].max()) if _valid.any() else None,
                "max_abs_diff_nats": (
                    float(np.abs(_ll_library_pdip - _fom)[_valid].max()) if _valid.any() else None
                ),
                "gated": False,
                "note": (
                    "RECORDED, NOT GATED: proves the lane trees rebuilt here are the models the "
                    "sampler evaluated (a wrong parameter mapping shows up as nats, not 1e-10)."
                ),
            },
        }
        print(
            f"  captured window vs capture PDIP (recorded): max rel "
            f"{vmap_block['captured']['timed_window_vs_capture_pdip']['max_rel_diff']}"
        )

    if "vmap" in arm_blocks and "scalar" in arm_blocks:
        _v_ms = arm_blocks["vmap"]["wall_per_lane_ms"]
        _s_ms = arm_blocks["scalar"]["wall_per_lane_ms"]
        vmap_block["matched"] = {
            "vmap_wall_per_lane_ms": _v_ms,
            "scalar_wall_per_lane_ms": _s_ms,
            "speedup_vmap_over_scalar": _s_ms / _v_ms,
            "note": (
                "Both walls are the PRODUCTION nesting of their arm, measured in one process on "
                "the same B lanes under the same solver injection. A speedup below 1 means the "
                "batch is slower per lane than B sequential jitted calls."
            ),
        }
        print(
            f"\n  MATCHED: vmap {_v_ms:.3f} ms/lane vs scalar {_s_ms:.3f} ms/lane "
            f"-> {_s_ms / _v_ms:.3f}x"
        )

    if LIBRARY_SOLVER_MODE:
        # Phase B (#300). These keys exist ONLY in library mode, so a harness-mode
        # JSON written by this checkout is key-for-key the phase-2 JSON.
        import subprocess as _subprocess

        import autoarray as _autoarray
        import autofit as _autofit
        import autogalaxy as _autogalaxy

        def _library_revision(module) -> dict:
            _repo = Path(module.__file__).resolve().parents[1]
            try:
                _sha = _subprocess.run(
                    ["git", "-C", str(_repo), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            except (OSError, _subprocess.CalledProcessError):
                _sha = None
            return {"head": _sha, "path": str(_repo), "module_file": module.__file__}

        _certified_lanes = [_r["certified"] for _r in lane_rows]
        _uncertified = (
            sum(1 for _c in _certified_lanes if _c is False)
            if LIBRARY_SOLVER == "certified" and None not in _certified_lanes
            else None
        )
        vmap_block.update(
            {
                "solver_source": SOLVER_SOURCE,
                "solver": LIBRARY_SOLVER,
                "certified_fallback": LIBRARY_FALLBACK,
                "certified_pass_budget": CERTIFIED_BUDGET,
                "certified_pass_budget_basis": CERTIFIED_BUDGET_BASIS,
                "certified_tau_rel": float(_settings_library.certified_tau_rel),
                "solver_used": library_solver_record["solver_used"],
                "solver_check": library_solver_record,
                "lanes_certified": [_r["certified"] for _r in lane_rows],
                "lanes_passes": [_r["passes"] for _r in lane_rows],
                "uncertified_lanes": _uncertified,
                "per_lane_certification_note": (
                    "certified / passes come from the library's own solve through its stats= "
                    "out-dict (PyAutoArray #566): passes is the number of restricted active-set "
                    "passes the library ran (0 when the unconstrained solve was already "
                    "non-negative), certified is its KKT certificate. With certified_fallback="
                    "pdip a False lane returned the PDIP solution; with none it returned the "
                    "last uncertified iterate. null for a library-PDIP row (no certificate)."
                    if LIBRARY_SOLVER == "certified"
                    else "Not applicable: a library-PDIP row has no certificate or pass count."
                ),
                "gate": {
                    "pre_registered": True,
                    "gated": [
                        "per lane: jit(vmap) arm vs jit(vmap) library PDIP <= lane_rtol relative",
                        "per lane: scalar jit arm vs scalar jit library PDIP <= lane_rtol relative",
                    ],
                    "recorded_not_gated": [
                        "per lane: jit(vmap) arm vs scalar jit arm (rel_diff, abs_diff_nats)",
                        "per lane: jit(vmap) library PDIP vs scalar jit library PDIP",
                        "uncertified_lanes (certified_fallback=none rows)",
                    ],
                    "lanes_failed": sum(1 for _r in lane_rows if _r["status"] == "FAIL"),
                    "passed": all(_r["status"] == "PASS" for _r in lane_rows),
                    "policy_eligible": (
                        all(_r["status"] == "PASS" for _r in lane_rows)
                        and (LIBRARY_FALLBACK != "none" or _uncertified == 0)
                    ),
                    "note": (
                        "Pre-registered before submission (module docstring; submit header). "
                        "Phase 3 (results/notes/hst_gpu_residue_phase3_psf_2026_09.md) placed "
                        "phase 2's ~2.5e-9 certified-vs-PDIP residual BETWEEN the jit(vmap) and "
                        "scalar-jit compositions, not in the solver — so each arm is gated "
                        "against the library-PDIP reference of its OWN composition, and the "
                        "cross-composition difference is recorded with its nats. A "
                        "certified_fallback=none row is policy-eligible only if every lane "
                        "certifies AND passes the gate. Timing is never gated."
                    ),
                },
                "library_revisions": {
                    "PyAutoArray": _library_revision(_autoarray),
                    "PyAutoFit": _library_revision(_autofit),
                    "PyAutoGalaxy": _library_revision(_autogalaxy),
                    "PyAutoLens": _library_revision(al),
                    "autoarray_file": _autoarray.__file__,
                },
            }
        )
        vmap_block["certification_report"]["passes_semantics"] = (
            "library `passes`: restricted active-set passes run (jax_active_set.solve_certified)"
        )
        if "matched" in vmap_block:
            vmap_block["matched"]["note"] = (
                "Both walls are the PRODUCTION nesting of their arm, measured in one process on "
                "the same B lanes with the SAME library Settings (no injection). A speedup below "
                "1 means the batch is slower per lane than B sequential jitted calls."
            )
        print(
            f"  phase-B gate: {'PASS' if vmap_block['gate']['passed'] else 'FAIL'}  "
            f"solver_used {vmap_block['solver_used']}  uncertified lanes {_uncertified}  "
            f"policy-eligible {vmap_block['gate']['policy_eligible']}"
        )

peak_after_all = peak_bytes()

if vmap_block is None:
    print(
        "\n  --- per-stage table (ms/call, command buffers "
        f"{'ON' if TRACE_ON_COMMAND_BUFFERS else 'OFF'}) ---"
    )
else:
    print(
        f"\n  --- per-stage table (ms per likelihood, command buffers OFF) — "
        f"{vmap_block['headline_arm']} arm ---"
    )
for _label in trace_block["stage_order"]:
    _row = trace_block["per_stage_ms"][_label]
    print(
        f"    {_label:<34} {_row['median_ms']:8.3f}  [{_row['min_ms']:.3f} .. {_row['max_ms']:.3f}]"
    )
print(f"    {'device_idle':<34} {trace_block['device_idle_ms']:8.3f}")
print(f"    {'-' * 44}")
print(f"    {'sum (kernels + idle)':<34} {trace_block['sum_ms']:8.3f}")
print(f"    {'wall (traced, same executable)':<34} {trace_block['wall_ms']:8.3f}")
print(f"    {'wall (untraced, same executable)':<34} {trace_block['untraced_wall_ms']:8.3f}")
print(f"    {'host outside device span':<34} {trace_block['host_outside_span_ms']:8.3f}")
print(
    f"    reconciliation: {trace_block['reconciliation_pct']:+.2f} % (vs traced wall)  "
    f"| {trace_block['reconciliation_vs_untraced_pct']:+.2f} % (vs untraced)  "
    f"| unjoined {trace_block['unjoined_ms']:.3f} ms  "
    f"| other {trace_block['other_ms']:.3f} ms"
)

print(
    f"    {'host pure_callback (HOST, not a device row)':<34} "
    f"{trace_block['host_callback']['pure_callback_span_ms']:8.3f}  "
    f"(qhull {trace_block['host_callback']['qhull_ms']:.3f})"
)

if vmap_block is not None:
    print("\n  --- matched arms (ms per LANE, command buffers OFF, traced program) ---")
    _stage_labels = sorted(
        {
            _lbl
            for _arm_row in vmap_block["arm_rows"].values()
            for _lbl in _arm_row["trace"]["per_stage_ms_per_lane"]
        },
        key=lambda _lbl: (
            -max(
                _arm_row["trace"]["per_stage_ms_per_lane"].get(_lbl, 0.0)
                for _arm_row in vmap_block["arm_rows"].values()
            )
        ),
    )
    print(f"    {'stage':<34} {'vmap':>10} {'scalar':>10}")
    for _lbl in _stage_labels:
        _cells = []
        for _arm_name in ("vmap", "scalar"):
            _arm_row = vmap_block["arm_rows"].get(_arm_name)
            _cells.append(
                f"{_arm_row['trace']['per_stage_ms_per_lane'][_lbl]:10.3f}"
                if _arm_row and _lbl in _arm_row["trace"]["per_stage_ms_per_lane"]
                else f"{'-':>10}"
            )
        print(f"    {_lbl:<34} {_cells[0]} {_cells[1]}")
    for _label, _key in (
        ("device_idle", "device_idle_ms_per_lane"),
        ("host qhull (probe)", "host_qhull_ms_per_lane"),
        ("host tables (probe)", "host_tables_ms_per_lane"),
        ("qhull callbacks per batch", "callback_count_per_call"),
        ("qhull callbacks per lane", "callback_count_per_lane"),
        ("WALL per lane (production)", "wall_per_lane_ms"),
    ):
        _cells = []
        for _arm_name in ("vmap", "scalar"):
            _arm_row = vmap_block["arm_rows"].get(_arm_name)
            _cells.append(f"{_arm_row[_key]:10.3f}" if _arm_row else f"{'-':>10}")
        print(f"    {_label:<34} {_cells[0]} {_cells[1]}")

print("\n  --- HLO census ---")
if VMAP_BATCH is not None:
    print(f"    EXCLUDED: {census_block['reason']}")
else:
    print(
        f"    (n,n) add of F + lambda*H at abstract.py:371: "
        f"{census_block['curvature_reg_add_nn']['count']}"
        f"  -> {census_block['curvature_reg_add_nn']['verdict']}"
    )
    print(
        f"    gathers at abstract.py:613 (edge subset):     "
        f"{census_block['edge_subset_gathers_613']['count']}"
    )
    print(
        f"    gathers at abstract.py:397 (reduced):         "
        f"{census_block['curvature_reg_reduced_gathers_397']['count']}"
    )
    print(
        f"    FFTs sourced at convolver.py:                 "
        f"{census_block['fft_convolver_total']['count']}"
        f"  (mapping matrix {census_block['fft_mapping_matrix']['count']}, "
        f"image {census_block['fft_image']['count']})"
    )
    print(
        f"    real-space convolutions of the cube:          "
        f"{census_block['conv_mapping_matrix']['count']}"
    )
    if "status" in census_block:
        print(
            f"    census anchors: {census_block['status']}  "
            f"(rows excluded: {', '.join(census_block['rows_excluded']) or 'none'})"
        )
    print(
        f"    cholesky factorizations:                      {census_block['cholesky']['count']}"
        f"  (triangular solves {census_block['triangular_solve']['count']})"
    )
    _where = census_block["curvature_reg_add_nn"]["where_the_sum_lives"]
    print(
        f"    opcodes written at abstract.py:371:           "
        f"{census_block['curvature_reg_add_nn']['opcodes_at_line_371']}"
    )
    print(
        f"    producers shared by >1 consumer of F+lambda*H: "
        f"{_where['producers_shared_by_more_than_one_consumer']}"
    )
    for _r in _where["consumers_of_F_plus_lambda_H"]:
        print(
            f"        {_r['consumer']:26} <- {_r['producer']:30} "
            f"{_r['producer_opcode']:12} {(_r['producer_source'] or '').split('/')[-1]}"
        )
    print(f"    VERDICT: {census_block['curvature_reg_add_nn']['verdict']}")

# ===================================================================
# Summary + JSON + PNG
# ===================================================================

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

# ---------------------------------------------------------------------------
# Phase 3 (#295): the PSF-candidate block and its ENFORCED gate
# ---------------------------------------------------------------------------
psf_block: dict | None = None
if PSF_CANDIDATE is not None:
    _spec = psf_cube_injection.CANDIDATES[PSF_CANDIDATE]
    _psf_label = "psf_convolution_mapping_matrix"
    _psf_stage_ms = trace_block["per_stage_ms"].get(_psf_label, {}).get("median_ms", 0.0)
    _psf_mixed = {
        _k: _v
        for _k, _v in trace_block["mixed_fusion"]["constituent_stage_sets"].items()
        if _psf_label in _k.split(" + ")
    }

    def _section_s(label: str):
        _hits = [_t for _l, _t in timer.records if _l == label]
        return float(_hits[-1]) if _hits else None

    _fp64_pins = psf_gated_rows
    _pins_failed = psf_pins_failed
    _recon = float(trace_block["reconciliation_pct"])
    _unjoined = float(trace_block["unjoined_ms"])
    _failures = []
    if _pins_failed:
        _failures.append(
            f"{len(_pins_failed)} gated fp64 pin(s) above {EQUIVALENCE_RTOL:.0e} relative "
            f"(fiducial vs route b; lever draws vs d-control): {', '.join(_pins_failed)}"
        )
    if not abs(_recon) <= 5.0:
        _failures.append(f"reconciliation {_recon:+.2f} % is outside +-5 %")
    if _unjoined > 0.0:
        _failures.append(f"unjoined_ms {_unjoined:.3f} > 0 — the join is incomplete")

    _rels = [_r["rel_diff"] for _r in psf_pin_draw_rows] + [
        _p["rel_diff"] for _p in equivalence_pins
    ]
    _nats = [_r["abs_diff_nats"] for _r in psf_pin_draw_rows] + [
        _p["abs_diff_nats"] for _p in equivalence_pins
    ]
    psf_block = {
        "candidate": PSF_CANDIDATE,
        "kind": _spec.kind,
        "fp64_exact": _spec.fp64_exact,
        "row_label": (
            "DIAGNOSTIC — may not be quoted as a lever (fp64 campaign constraint)"
            if not _spec.fp64_exact
            else ("CONTROL" if _spec.kind == "control" else "LEVER candidate")
        ),
        "provenance": psf_cube_injection.candidate_provenance(
            PSF_CANDIDATE,
            system_s3.dataset.psf,
            system_s3.dataset.mask,
            n_src=int(system_s3.n_mapper),
        ),
        "injection_counts": {
            "route_d_jax": routes["d_s3_certified_fallback"].get("injected_psf_calls_jax"),
            "route_d_delegated": routes["d_s3_certified_fallback"].get(
                "injected_psf_calls_delegated"
            ),
            "traced_program_jax": int(_trace_psf_counts["jax"]),
            "traced_program_delegated": int(_trace_psf_counts["delegated"]),
        },
        "whole_call_ms": {
            "route_d_jit_profile_ms": routes["d_s3_certified_fallback"]["ms"],
            "route_b_jit_profile_ms": routes["b_s3_pdip"]["ms"],
            "traced_program_command_buffers_on_ms": wall_on_ms,
            "traced_program_command_buffers_off_ms": wall_off_ms,
            "note": (
                "route_d_jit_profile_ms is the whole fused call of the candidate program (10 "
                "steady calls, timing.jit_profile — the phase-1/2 basis). The traced program is "
                "the same lowering compiled with command buffers on and off."
            ),
        },
        "compile_s": {
            "route_d": jit_records.get("d_s3_certified_fallback_library_likelihood_jit", {}).get(
                "compile_s"
            ),
            "route_b": jit_records.get("b_s3_pdip_library_likelihood_jit", {}).get("compile_s"),
            "traced_command_buffers_on": _section_s("trace_compile_command_buffers_on"),
            "traced_command_buffers_off": _section_s("trace_compile_command_buffers_off"),
        },
        "psf_rows_ms": {
            _psf_label: _psf_stage_ms,
            "mixed_fusion_sets_containing_psf": _psf_mixed,
            "mixed_fusion_note": (
                "Fused kernels whose constituents span the PSF stage AND another stage are in "
                "mixed_fusion, not in the PSF row; listed here for readability, never summed."
            ),
            "top_instructions": trace_block["stage_audit"].get(_psf_label, []),
        },
        "pins": {
            "rtol": EQUIVALENCE_RTOL,
            "reference": "route b — library PDIP + library convolution (the unmodified library)",
            "fiducial": equivalence_pins,
            "draws": psf_pin_draw_rows,
            "draw_seed": DRAW_SEED,
            "pin_draws": PIN_DRAWS,
            "max_rel_diff": max(_rels) if _rels else None,
            "max_abs_diff_nats": max(_nats) if _nats else None,
            "d_control": psf_d_control_record,
            "max_rel_diff_vs_d_control": max(
                (
                    _r["rel_diff_vs_d_control"]
                    for _r in psf_pin_draw_rows
                    if "rel_diff_vs_d_control" in _r
                ),
                default=None,
            ),
        },
        "census_status": census_block.get("status"),
        "gate": {
            "enforced": True,
            "fp64_pins_checked": len(_fp64_pins),
            "fp64_pins_failed": _pins_failed,
            "reconciliation_pct": _recon,
            "reconciliation_ok": abs(_recon) <= 5.0,
            "unjoined_ms": _unjoined,
            "unjoined_ok": _unjoined <= 0.0,
            "failures": _failures,
            "passed": not _failures,
            "gated_pins": [_p.get("pin") or f"draw {_p['draw']}" for _p in psf_gated_rows],
            "note": (
                "ENFORCED in PSF mode: the cell writes this JSON and its PNG, then raises if any "
                "gated fp64 pin failed, |reconciliation_pct| > 5 or unjoined_ms > 0. Speed is "
                "never gated. " + psf_cube_injection.PSF_GATE_NOTE
            ),
        },
    }

trace_summary = {
    "device": device_info_dict(),
    "machine": machine_info_dict(),
    "precision": "mixed" if _cli.use_mixed_precision else "fp64",
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "mesh": MESH,
        "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
        "source_pixels": int(n_source_pixels),
        "source_pixels_requested": (
            int(SOURCE_PIXELS_REQUESTED) if SOURCE_PIXELS_REQUESTED is not None else None
        ),
        "dataset": DATASET,
        "dataset_sha256": dataset_sha256,
        "cell_source_sha256": cell_source_sha256,
        "inversion_path": "dense",
        "pass_budget": PASS_BUDGET,
        "pass_budget_basis": (
            "phase 3 (autolens_profiling#255): the smallest ZERO-fallback budget per mesh, the "
            "budget a production run fixes. NOT phase 0's certifying budget (2 for Delaunay), "
            "which falls back on 67.5 % of a graded draw set and is not a production cost."
        ),
        "border_relocator_mode": BORDER_RELOCATOR_MODE,
        "border_relocator": BORDER_RELOCATOR_RESOLVED,
        "border_relocator_note": (
            "'cell' is use_border_relocator=True, which every profiling cell in this family "
            "forces; 'library' passes None and takes the shipped config default, which is what "
            "production runs. The resolved boolean is what this leg actually ran."
        ),
        "routes_selected": list(ROUTE_SELECTION),
        "trace_calls": TRACE_CALLS,
        "command_buffers_traced": "on" if TRACE_ON_COMMAND_BUFFERS else "off",
        "use_mixed_precision": bool(_cli.use_mixed_precision),
        "tau_rel": TAU_REL,
        "regularization": reg_provenance,
        "thread_env": _observe_thread_env(),
        "over_sample_size_lp_rule": {
            "sub_size_list": [4, 2, 2],
            "radial_list": [0.3, 0.6],
            "centre": [0.0, 0.0],
        },
    },
    "solver_injection": {
        "patched": library_solver_injection.LIBRARY_POSITIVE_ONLY_DOTTED,
        "pass_budget": PASS_BUDGET,
        "fallback": True if vmap_block is None else FALLBACK_ON,
        "calls_jax": int(_trace_counts["jax"]),
        "calls_numpy": int(_trace_counts["numpy"]),
        "note": (
            "Routes d and the traced program use a scoped harness monkeypatch of the library's "
            "own positive-only entry point. Everything else in the call — mapper, F + lambda*H, "
            "both log determinants, the evidence — is the library's own code, unmodified."
        ),
    },
    "routes": routes,
    "equivalence_pins": equivalence_pins,
    "equivalence_rtol": EQUIVALENCE_RTOL,
    "reference": {
        "log_evidence_s0_library": log_evidence_ref,
        "log_evidence_s3_library": log_evidence_s3_library,
        "s3": {
            "n_params": int(system_s3.n_params),
            "n_mapper": int(system_s3.n_mapper),
            "n_funcs": int(system_s3.n_funcs),
            "edge_zeroed_pixels": int(system_s3.edge_zero_mask.sum()),
            "subtracted_light_flux": float(system_s3.subtracted_light_flux),
        },
    },
    "command_buffers": {
        "wall_on_ms": wall_on_ms,
        "wall_off_ms": wall_off_ms,
        "command_buffer_delta_ms": wall_off_ms - wall_on_ms,
        "command_buffer_delta_pct": 100.0 * (wall_off_ms - wall_on_ms) / wall_on_ms,
        "traced": "on" if TRACE_ON_COMMAND_BUFFERS else "off",
        "compiler_option": _COMMAND_BUFFER_OPTION,
        "probe": command_buffer_probe,
        "note": (
            "ONE lowering, TWO compilations, one process. wall_on_ms is the PRODUCTION program "
            "(CUDA graphs as XLA ships them) and is the number comparable to phase 5's row; "
            "wall_off_ms is the graph-less program the stage rows attribute. The delta is the "
            "launch overhead the graphs remove — it lands in device_idle and host_outside_span, "
            "not in any stage. Reconciliation is computed against wall_off_ms."
        ),
    },
    "trace": trace_block,
    "hlo_census": census_block,
    "stage_map_provenance": xla_attribution.stage_map_provenance(),
    "trace_timing": {
        "traced_wall_ms": traced_wall_ms,
        "untraced_wall_ms": untraced_wall_ms,
        "profiler_overhead_ms": traced_wall_ms - untraced_wall_ms,
        "profiler_overhead_pct": 100.0 * (traced_wall_ms - untraced_wall_ms) / untraced_wall_ms,
    },
    "jit_phases": jit_records,
    "peak_bytes": {
        "after_setup": peak_after_setup,
        "after_routes": peak_after_routes,
        "after_all": peak_after_all,
        "note": (
            "jax.devices()[0].memory_stats()['peak_bytes_in_use']; None on a backend that does "
            "not report it (CPU). Process-wide high-water marks."
        ),
    },
    "note": (
        "Every stage row is a MEASURED sum of GPU kernel durations joined to library source "
        "through the optimized HLO of ONE program — not arithmetic across compilations. The "
        "rows attribute the command-buffers-OFF program: kernel time is the same as production, "
        "the launch overhead the CUDA graphs remove is not, and command_buffers.* carries both "
        "walls and their delta. sum_ms = kernel rows + measured device_idle_ms; the residual "
        "against the wall is host_outside_span_ms and is reported, not absorbed. "
        "mixed_fusion.prorated_estimate_ms is an ESTIMATE and is never part of the table. "
        "No row may be quoted if reconciliation_pct is outside 5 % or unjoined_ms is non-zero."
    ),
}

if vmap_block is not None:
    # The new flags go in ``configuration`` ONLY in this mode, so a phase-1 JSON
    # written by this checkout is key-for-key the JSON phase 1 wrote — while a
    # batched JSON carries the exact current composition as provenance.
    trace_summary["configuration"].update(
        {
            "vmap_batch": int(VMAP_BATCH),
            "lanes": LANES_MODE,
            "arms": list(ARMS),
            "fallback": "on" if FALLBACK_ON else "off",
            "draw_seed": DRAW_SEED,
            "vmap_composition": "jax.jit(jax.vmap(fn))",
            "lane_rtol": LANE_RTOL,
            "vmap_flags_note": (
                "Strict ProfileCLI.parse_cell_args rejects unknown flags. These keys record "
                "the exact batching program and prevent historical vmap(jit) artifacts from "
                "being mistaken for current jit(vmap) measurements."
            ),
        }
    )
    if LIBRARY_SOLVER_MODE:
        trace_summary["configuration"].update(
            {
                "solver_source": SOLVER_SOURCE,
                "solver": LIBRARY_SOLVER,
                "certified_fallback": LIBRARY_FALLBACK,
                "certified_pass_budget": CERTIFIED_BUDGET,
            }
        )
        if CAPTURED_LANES:
            trace_summary["configuration"].update(
                {"captured_pass": CAPTURED_PASS, "batches": str(_cell_args.batches)}
            )
        trace_summary["solver_injection"] = {
            "patched": None,
            "part_v": "none — library mode selects the solver through al.Settings",
            "part_b_route_d": {
                "patched": library_solver_injection.LIBRARY_POSITIVE_ONLY_DOTTED,
                "pass_budget": PASS_BUDGET,
                "fallback": True,
            },
            "note": (
                "Library mode (phase B, #300): PART V's arms and references run the library's "
                "own positive-only solve, selected by Settings(positive_only_solver, "
                "certified_fallback, certified_pass_budget); nothing is injected. PART B's "
                "single-call route d (the fiducial d == b pin) still uses the phase-1 harness "
                "injection at the phase-3 budget and is not a phase-B row. The untimed "
                "certification report observes the library solve (library_solver_observed) "
                "without changing it."
            ),
        }
    trace_summary["vmap"] = vmap_block
    trace_summary["headline_arm_note"] = (
        f"`trace`, `hlo_census`, `command_buffers` and `trace_timing` above describe the "
        f"{vmap_block['headline_arm']} arm — the program phase 1 traced. Both arms' full "
        f"blocks, the lane table and the matched comparison are under `vmap`."
    )

if psf_block is not None:
    # Only in PSF mode, so a phase-1 JSON written by this checkout is unchanged.
    trace_summary["configuration"].update(
        {
            "psf_candidate": PSF_CANDIDATE,
            "pin_draws": PIN_DRAWS,
            "draw_seed": DRAW_SEED,
        }
    )
    trace_summary["psf_candidate"] = psf_block

# The suffix carries the MODE and the RESOLVED value. Keying it on the resolved
# value alone let the ``cell`` and ``library`` legs write the same filename on a
# workspace where the config default is already True — the second leg silently
# clobbered the first.
_cell_name = f"fixed_light_trace_{MESH}"
if BORDER_RELOCATOR_MODE == "off":
    # ``off`` states its own resolved value; ``_border_off_off`` said it twice.
    _cell_name = f"{_cell_name}_border_off"
elif BORDER_RELOCATOR_MODE == "library":
    # ``library`` does NOT: it resolves from the config, and on this workspace it
    # resolved to True. The resolved value is in the name so a library leg can
    # never be mistaken for, or clobber, the leg it happened to agree with.
    _cell_name = f"{_cell_name}_border_library_{'on' if BORDER_RELOCATOR_RESOLVED else 'off'}"
if SOURCE_PIXELS_REQUESTED is not None:
    _cell_name = f"{_cell_name}_n{int(n_source_pixels)}"
if VMAP_BATCH is not None:
    # The batched legs never collide with the phase-1 files in the same
    # directory: batch size, lane family and fallback semantics are three
    # different programs and all three are in the name.
    if not LIBRARY_SOLVER_MODE:
        _cell_name = (
            f"{_cell_name}_jitvmap{int(VMAP_BATCH)}_{LANES_MODE}_fb{'on' if FALLBACK_ON else 'off'}"
        )
    else:
        # Phase B (#300): the library solver, its fallback and its budget are three
        # more program choices, so all three are in the name — and the ``_lib``
        # token keeps these files disjoint from every harness-mode file.
        _cell_name = (
            f"{_cell_name}_jitvmap{int(VMAP_BATCH)}_{LANES_MODE}"
            + (f"_{CAPTURED_STAGE}" if CAPTURED_LANES else "")
            + f"_lib{LIBRARY_SOLVER}_fb{'on' if FALLBACK_ON else 'off'}_b{CERTIFIED_BUDGET}"
        )

if PSF_CANDIDATE is not None:
    # One file pair per candidate, never colliding with phase 1's own file.
    _cell_name = f"{_cell_name}_psf_{PSF_CANDIDATE}"

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"{_cell_name}_trace_{DATASET}_v{al_version}",
    cell=_cell_name,
)
dict_path.write_text(json.dumps(trace_summary, indent=2, default=str))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart -------------------------------------------------------
# Phase 1: the per-stage table, largest first. Batched mode: the two arms side
# by side, per stage, in milliseconds PER LANE — the only basis on which a
# batched program and B sequential ones can be compared at all.

if vmap_block is not None:
    _arm_names = [_a for _a in ("vmap", "scalar") if _a in vmap_block["arm_rows"]]
    _labels = sorted(
        {
            _lbl
            for _a in _arm_names
            for _lbl in vmap_block["arm_rows"][_a]["trace"]["per_stage_ms_per_lane"]
        },
        key=lambda _lbl: max(
            vmap_block["arm_rows"][_a]["trace"]["per_stage_ms_per_lane"].get(_lbl, 0.0)
            for _a in _arm_names
        ),
    )
    _labels = [
        _lbl
        for _lbl in _labels
        if max(
            vmap_block["arm_rows"][_a]["trace"]["per_stage_ms_per_lane"].get(_lbl, 0.0)
            for _a in _arm_names
        )
        > 0.0
    ]
    _labels = _labels + ["device_idle"]

    def _value(arm_name: str, label: str) -> float:
        _row = vmap_block["arm_rows"][arm_name]
        if label == "device_idle":
            return _row["device_idle_ms_per_lane"]
        return _row["trace"]["per_stage_ms_per_lane"].get(label, 0.0)

    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.42 * len(_labels))))
    _y = np.arange(len(_labels), dtype=float)
    _height = 0.38
    _arm_colors = {"vmap": "#4C72B0", "scalar": "#DD8452"}
    _max_ms = 1.0
    for _i, _a in enumerate(_arm_names):
        _vals = [_value(_a, _lbl) for _lbl in _labels]
        _max_ms = max(_max_ms, max(_vals, default=0.0))
        _offset = (_i - (len(_arm_names) - 1) / 2.0) * _height
        ax.barh(
            _y + _offset,
            _vals,
            height=_height,
            color=_arm_colors[_a],
            edgecolor="white",
            label=f"{_a} ({vmap_block['arm_rows'][_a]['wall_per_lane_ms']:.2f} ms/lane wall)",
        )
        for _yy, _v in zip(_y + _offset, _vals):
            if _v > 0.0:
                ax.text(_v + _max_ms * 0.01, _yy, f"{_v:.3f}", va="center", fontsize=7)

    ax.set_yticks(_y)
    ax.set_yticklabels(_labels, fontsize=9)
    ax.set_xlabel("device time per LANE (ms) — command buffers OFF, traced program")
    ax.set_xlim(0, _max_ms * 1.22)
    _cb_ms = [
        f"{_a} {vmap_block['arm_rows'][_a]['callback_count_per_call']:.1f}" for _a in _arm_names
    ]
    if not LIBRARY_SOLVER_MODE:
        ax.set_title(
            f"Fixed-light {MESH} N={n_source_pixels} — vmap vs scalar at B={VMAP_BATCH} "
            f"({LANES_MODE} lanes, fallback {'on' if FALLBACK_ON else 'off'})\n"
            f"budget {PASS_BUDGET}, border relocator {BORDER_RELOCATOR_RESOLVED} | "
            f"qhull callbacks per batched call: {', '.join(_cb_ms)} | "
            f"lanes pinned (all relative checks <= {LANE_RTOL:.0e}) "
            f"{len(lane_rows) - vmap_block['lanes_failed']}/{len(lane_rows)}",
            fontsize=10,
        )
    else:
        ax.set_title(
            f"Fixed-light {MESH} N={n_source_pixels} — LIBRARY {LIBRARY_SOLVER} "
            f"(fallback {LIBRARY_FALLBACK}, budget {CERTIFIED_BUDGET}), vmap vs scalar at "
            f"B={VMAP_BATCH} ({LANES_MODE} lanes)\n"
            f"border relocator {BORDER_RELOCATOR_RESOLVED} | "
            f"qhull callbacks per batched call: {', '.join(_cb_ms)} | "
            f"same-composition pins (<= {LANE_RTOL:.0e}) "
            f"{len(lane_rows) - vmap_block['lanes_failed']}/{len(lane_rows)} | "
            f"uncertified lanes {vmap_block['uncertified_lanes']}",
            fontsize=10,
        )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    print(f"  Chart saved to:        {chart_path}")

    timer.summary()

    if vmap_block["lanes_failed"]:
        _failed = [_r for _r in lane_rows if _r["status"] == "FAIL"]
        for _r in _failed:
            print(
                f"  [   FAIL] lane {_r['lane']} ({_r['draw_name']}): "
                f"vmap {_r['log_likelihood_vmap']!r} vs scalar "
                f"{_r['log_likelihood_scalar']!r}  "
                f"abs {_r['abs_diff_nats']:.3e} nats (rel {_r['rel_diff']:.3e})"
                + (
                    f"  | GATED vmap-vs-ref {_r['rel_diff_vmap_vs_library_pdip']:.3e}, "
                    f"scalar-vs-ref {_r['rel_diff_scalar_vs_library_pdip_scalar']:.3e}"
                    if LIBRARY_SOLVER_MODE
                    else ""
                )
            )
        if LIBRARY_SOLVER_MODE:
            raise AssertionError(
                f"{len(_failed)} of {len(lane_rows)} lanes disagree with the library-PDIP "
                f"reference of their own composition by more than {LANE_RTOL:.0e} relative "
                f"(the pre-registered phase-B gate). Nothing in this table supports a "
                f"production recommendation. The JSON and PNG were written first and hold "
                f"the evidence."
            )
        raise AssertionError(
            f"{len(_failed)} of {len(lane_rows)} lanes disagree between the vmap and scalar "
            f"arms or against the library-PDIP reference by more than {LANE_RTOL:.0e} relative. "
            f"The numerical gate failed, so nothing in this table supports a production "
            f"recommendation. The JSON and PNG were written first and hold the evidence."
        )

    sys.exit(0)

_rows = [
    (label, trace_block["per_stage_ms"][label]["median_ms"]) for label in trace_block["stage_order"]
]
_rows.append(("device_idle", trace_block["device_idle_ms"]))
_rows = [(label, ms) for label, ms in _rows if ms > 0.0]
_rows.sort(key=lambda kv: kv[1])

_colors = []
for label, _ in _rows:
    if label in (xla_attribution.OTHER, xla_attribution.UNJOINED):
        _colors.append("#C44E52")  # the rows that say the join is incomplete
    elif label == xla_attribution.MIXED_FUSION:
        _colors.append("#8172B3")
    elif label in ("device_idle", xla_attribution.MEMSET):
        _colors.append("#937860")
    elif label == "pdip_solve":
        _colors.append("#55A868")
    else:
        _colors.append("#4C72B0")

fig, ax = plt.subplots(figsize=(11, max(4.0, 0.34 * len(_rows))))
_y = np.arange(len(_rows), dtype=float)
_bars = ax.barh(_y, [ms for _, ms in _rows], color=_colors, edgecolor="white", height=0.66)
_max_ms = max((ms for _, ms in _rows), default=1.0)
for _bar, (_, _ms) in zip(_bars, _rows):
    ax.text(
        _bar.get_width() + _max_ms * 0.01,
        _bar.get_y() + _bar.get_height() / 2,
        f"{_ms:.3f} ms",
        va="center",
        fontsize=8,
    )

ax.set_yticks(_y)
ax.set_yticklabels([label for label, _ in _rows], fontsize=9)
ax.set_xlabel("device time per likelihood call (ms)")
ax.set_xlim(0, _max_ms * 1.18)
_psf_title = (
    ""
    if psf_block is None
    else f" — PSF candidate {PSF_CANDIDATE} [{psf_block['row_label'].split(' ')[0]}]"
)
ax.set_title(
    f"Fixed-light {MESH} N={n_source_pixels} — XLA device timeline of the production call"
    f"{_psf_title}\n"
    f"budget {PASS_BUDGET}, border relocator {BORDER_RELOCATOR_RESOLVED}, "
    f"command buffers {'ON' if TRACE_ON_COMMAND_BUFFERS else 'OFF'} | "
    f"wall {trace_block['wall_ms']:.2f} ms, sum {trace_block['sum_ms']:.2f} ms "
    f"({trace_block['reconciliation_pct']:+.1f} %)",
    fontsize=10,
)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
print(f"  Chart saved to:        {chart_path}")

timer.summary()

if psf_block is not None:
    # The ENFORCED phase-3 gate, AFTER the evidence is on disk.
    _gate = psf_block["gate"]
    print(
        f"\n  PSF GATE [{PSF_CANDIDATE}, {psf_block['row_label']}]: "
        f"{'PASS' if _gate['passed'] else 'FAIL'}  "
        f"(fp64 pins {_gate['fp64_pins_checked'] - len(_gate['fp64_pins_failed'])}/"
        f"{_gate['fp64_pins_checked']}, reconciliation {_gate['reconciliation_pct']:+.2f} %, "
        f"unjoined {_gate['unjoined_ms']:.3f} ms)"
    )
    if not _gate["passed"]:
        raise AssertionError(
            f"PSF candidate {PSF_CANDIDATE!r} failed the phase-3 gate: "
            + "; ".join(_gate["failures"])
            + ". The JSON and PNG were written first and hold the evidence."
        )
