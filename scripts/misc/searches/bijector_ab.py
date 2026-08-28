"""Phase 8B — bijector (log-coordinate) A/B on the free-regularization NaN
wall / conditioning.

W5 of the inference programme (``results/notes/inference/PROGRAMME.md`` §"Phase
8 — 8B", issue #162). The PyAutoFit half (`bijector.py`, `MultiStartGradient
(bijector=...)`, opt-in `record_lane_nan_history` / `trace_param_indices`
diagnostics) is PyAutoFit PR#1525, merged to main. This script is the
autolens_profiling half: it runs the actual A/B and scores it against the
pre-registered falsification criteria below.

PRE-REGISTRATION (``PROGRAMME.md`` §"8B", stated before any run)
------------------------------------------------------------------
8B is a **category-1 reparameterization**: gradient searches step in ``log
lambda`` (the LogUniform prior's own coordinate) instead of ``lambda`` itself.
The physical AdaptSplit likelihood is evaluated completely unchanged (see
``autofit/non_linear/bijector.py``'s module docstring, "the equivalence
argument" — a MAP search's objective needs no Jacobian, so the map cannot move
the argmax). Measured: NaN-wall position in steps, free-AdaptSplit convergence
speed (historical: **2,200 steps free vs 98 steps fixed-reg**,
``PROGRAMME.md:579``), lambda-trajectory behaviour (time at high coefficient,
clip rate, resurrection count).

WHICH CELL WALLS — same split as Phase 8A (``slogdet_ab.py``)
-----------------------------------------------------------------
- ``delaunay_adapt_split`` — Delaunay mesh + free AdaptSplit. The actual NaN
  wall (Phase 8A/CP-4 measured 90/416 Cholesky NaNs on this cell; only 58-73%
  rescued by slogdet alone). This is where a criterion-1 (NaN wall position)
  result can be non-vacuous.
- ``knn`` — KNN mesh + free AdaptSplit. The cell the 8B pre-registration text
  names. Its high-coefficient region is a finite, escapable floor (Phase 8A:
  ZERO Cholesky NaNs over 416 draws on this cell — VOID for a NaN-wall
  question, but the cell whose gradient-work slogdet/bijector recommendation
  is actually about). Both cells are run; ``delaunay_adapt_split`` for F1,
  ``knn`` for F2/F3/the ``logit`` secondary arm.
- ``mge`` — no pixelization/regularization at all, hence zero LogUniform
  priors under a ``"regularization."`` path. The ``log_reg`` arm therefore
  resolves to an EMPTY per-path map (``searches._samplers._bijector_object``)
  and must be bit-identical to ``none`` — this is the falsification control
  (F4).

ARMS (39 tasks total — matches ``hpc/batch_gpu/submit_phase8b_bijector_a100``)
--------------------------------------------------------------------------------
- ``delaunay_adapt_split``: {cholesky, slogdet} x {none, log_reg} x seed 0-4
  = 20. ``log_det_method`` is varied here (not on knn/mge) because this is
  the cell Phase 8A measured the NaN wall on, and 8B's log-coordinate step
  and 8A's log-det method are independent levers over the SAME wall.
- ``knn``: {none, log_reg} x seed 0-4 = 10, plus ``logit`` (secondary arm,
  F4's other half) x seed 0-4 = 5. Total 15. ``log_det_method`` is left at
  its W8-resolved default (slogdet on GPU, cholesky on CPU) throughout.
- ``mge``: {none, log_reg} x seed {0, 1} = 4 (control; F4).

Every arm: ``multi_start_prodigy`` (genuinely fixed-step:
``check_for_convergence=False``, resurrection on by default),
``SEARCHES_N_STARTS=16``, ``SEARCHES_N_STEPS=3000`` (the #117-validated
pixelized-cell budget — a long plateau on these cells is a reg mode, not
convergence, see ``_samplers.py``), ``SEARCHES_BATCH_SIZE=4`` (mandatory on
pixelized cells, PyAutoFit#1374 tiling is numerically inert),
``SEARCHES_CLIPPER=prior_box``, ``SEARCHES_SCALER=none``,
``SEARCHES_LANE_HISTORY=1``. ``SEARCHES_TRACE_PARAMS`` is set to the two
regularization coefficient paths on ``knn``/``delaunay_adapt_split`` (absent
on ``mge``, which has none). Every arm's ``--config-name`` carries
``log_det_method`` + ``bijector`` + ``seed`` so no two arms' results JSONs
collide (``_runner.resolve_output_paths`` derives the JSON basename from
``--config-name`` alone, not from ``unique_tag``).

READOUTS (per arm, from the ``diagnostics`` block PyAutoFit#1525 + this
repo's ``_per_lane.per_lane_block`` write into every MultiStart* results JSON)
--------------------------------------------------------------------------------
- first value-NaN step (global: earliest step at which ANY lane's
  ``lane_value_nan_history`` entry is True) and the traced regularization
  coefficient value(s) at that step, from ``trace_history``.
- lambda_inner / lambda_outer trajectories (median across lanes per step).
- steps-to-within-10-nats of a reference log-posterior. **There is no
  dedicated fixed-regularization control arm in this campaign** (the
  historical 98-step figure is from a different, undocumented run cited only
  as a step count in ``PROGRAMME.md``, not as a log-posterior value this
  script can replay against) — so the reference used here is the maximum
  ``lane_best_log_posterior`` observed across every PHYSICALLY VALID arm in
  that (cell, log_det_method) group, every bijector included (ruled
  2026-08-28, see "Scorer amendments 2026-08-28" below). This is a
  DOCUMENTED DEVIATION from a literal "vs the historical fixed-reg run"
  reading; the resolved reference value, the row it came from and every
  excluded row are recorded in the verdict artifact so it can be checked,
  and F2 is scored as "log_reg reduces steps-to-THIS-reference by >= 2x
  relative to none, at matched seeds" rather than against the bare "2,200 /
  98" figures (which the artifact still cites for context).
- lane deaths/resurrections (``n_resurrections``), clip rate
  (``n_clipped_lane_steps / (total_steps * n_starts)``), fraction of
  (step, lane) pairs with either traced coefficient > 1e4.
- final `d` (Prodigy's internal distance/step-scale estimate,
  ``optax.contrib.ProdigyState.estim_lr``) is **NOT** recorded here: it lives
  in ``search_internal["opt_state"]``, which ``_per_lane.per_lane_block``
  does not serialize into the results JSON (an unbounded, non-JSON-safe JAX
  pytree) and this driver runs each arm in an isolated subprocess (matching
  how the real SLURM array submit runs them), so there is no in-process
  handle to it either. Recorded as ``null`` with this note in every row
  rather than silently omitted.

PRE-REGISTERED FALSIFICATION (any two -> 8B falsified; record and close, no
rescoping to logit)
------------------------------------------------------------------------------
- **F1** — median first-NaN step under ``log_reg`` is NOT earlier than
  ``none`` (on ``delaunay_adapt_split``), OR value-NaN lane-steps fall by
  LESS than 50%.
- **F2** — steps-to-reference is NOT reduced >= 2x at matched seeds (on
  ``knn``; historical framing: 2,200 -> <= 1,100).
- **F3** — ``log_reg`` lanes spend >= the same fraction of steps at
  lambda > 1e4 as ``none`` (on either wall cell).
- **F4** — the MGE control's **winning lane** disagrees between ``none`` and
  ``log_reg`` on ``best_fom`` OR ``max_log_likelihood`` by more than fp64
  relative 1e-9 at a matched seed, OR the ``knn`` ``logit`` arm reproduces the
  pinned-lane-to-infinity pathology (a lane parked on the logit box boundary
  at completion). *(Amended 2026-08-27 — was byte-identity of every per-lane
  final/best parameter vector; see "Scorer amendments 2026-08-27" below.)*
  Every criterion also has a third state, **UNSCORABLE**: the inputs it needs
  are absent, so it neither fired nor did not fire. The verdict stage refuses
  to conclude while any criterion is unscorable and fewer than two have fired.

- **F5** — figure of merit at matched physical points (the shared initial
  broad-start draw, i.e. the step-0 global-best fom) differs by more than
  1e-9 relative between arms at a matched seed. *(DEMOTED 2026-08-28 from a
  HALT to a reported fp-reproducibility diagnostic — see "Scorer amendments
  2026-08-28" below.)*

Scorer amendments 2026-08-27 (issue #182)
------------------------------------------------------------------------------
Two repairs, both to the scorer only — the arm table, the readouts and the
"any two -> falsified" threshold are untouched:

1. **UNSCORABLE is a first-class state.** ``score_f1`` collapsed missing
   inputs to ``False`` (silent PASS) and ``score_f2`` collapsed them to
   ``True`` (silent FAIL): the same absence produced opposite confident
   answers. Every criterion now returns ``{"scorable": False, "falsified":
   None, "reason": ...}`` when it cannot be asked, and ``score_rows`` returns
   ``INCONCLUSIVE`` (``falsified: None``) rather than a verdict while any
   criterion is unscorable and fewer than two have fired. A criterion whose
   disjunction has already fired stays conclusive — one fired limb settles it
   regardless of the others.
2. **F4 no longer demands byte-identity.** The MGE control's ``log_reg`` map
   is empty, so both arms compute the same objective — but they are two
   separate GPU runs, and byte-identity across all 16 lanes' parameter
   vectors is a claim about float reduction order, resurrection draws and
   lane batching, not about the bijector. F5 already proves the objective is
   inert at matched physical points. F4 now asks whether the **winning
   lane's** ``best_fom`` and ``max_log_likelihood`` agree within fp64
   relative 1e-9; the old byte-identity result is kept as an informational
   field (``mge_per_seed_byte_identical``), reported and never scored.

Scorer amendments 2026-08-28 (issue #185; mirrors the dated
``results/notes/inference/DECISIONS.md`` entry of the same day, which is the
authoritative record — this section is its scorer-side restatement)
------------------------------------------------------------------------------
Four repairs, all to the scorer. The arm table, the readouts and the "any two
-> falsified" threshold are again untouched:

1. **The F2 reference is the group-wide PHYSICALLY VALID maximum.** It was the
   max ``lane_best_log_posterior`` over the ``none`` arm only — a target
   defined by the control arm's own stalling, which is circular when the
   question is whether ``none`` stalls. It is now the max over ALL arms in the
   (cell, log_det_method) group (``none``/``log_reg``/``logit`` alike),
   restricted to rows that are neither VOID (``diagnostics.valid`` false,
   total wall < ``VOID_MIN_WALL_S``, or no ``schema_version``) nor
   NON-PHYSICAL (best-point ``ell_comps`` magnitude >=
   ``ELL_COMPS_MAX_MAGNITUDE``, i.e. outside the unit disk). Without the
   physicality filter the max-over-arms rule would be set by a box corner:
   ``delaunay_adapt_split·slogdet·log_reg·seed1`` reports 2.1e53 at a point
   pinned to (±1, ±1). The magnitude comes from
   ``recovered_offline_verification.best_point_ell_comps_magnitude`` where the
   row carries it, else is computed from the winning lane's
   ``lane_best_params`` via ``diagnostics.ell_comps_pairs``; each row records
   which in ``ell_comps_source``. Tolerance is unchanged at
   ``REFERENCE_TOLERANCE_NATS``.
2. **F2 "never reached" has explicit semantics.** At a matched seed, within
   the step budget: ``none`` never within tolerance while ``log_reg`` is ->
   ratio ``+inf`` (counts as >= 2x); ``log_reg`` never while ``none`` is ->
   ratio ``0`` (counts against); both never -> that seed is unscorable and
   drops out. Median over the scorable seeds, as before. Previously "never
   reached" produced no ratio at all, so the strongest possible pass —
   ``none`` never converges, ``log_reg`` always does — scored as UNSCORABLE.
3. **F5 is DEMOTED from a HALT to a reported diagnostic.** As written it
   compares the step-0 global-best fom between two SEPARATE GPU runs: that
   measures floating-point reproducibility across processes, not the
   objective. The MGE control's ``log_reg`` map is provably EMPTY (an identity
   reparameterization) and its two arms still diverge 1.7e-2 relative by step
   3000. F5 is still computed and reported with the same 1e-9 number, now
   labelled an "fp-reproducibility diagnostic" under the verdict's
   ``diagnostics.f5`` key with ``halts: false``. The sound F5 — evaluate one
   physical point under both parameterizations IN ONE PROCESS and assert
   bit-equality — is a PyAutoFit unit test, filed separately; a two-run A/B
   cannot measure it.
4. **F4's fp-equivalence limb is informational only.** The winning-lane
   ``best_fom`` / ``max_log_likelihood`` comparison is the same cross-run
   quantity F5 measures and fails for the same reason, so it is reported and
   never sets ``falsified``. F4 now trips on ONE thing: the ``knn`` ``logit``
   arm reproducing the pinned-lane-to-boundary pathology.

Usage (from the ``autolens_profiling/`` root)::

    python3 scripts/misc/searches/bijector_ab.py --stage run
    python3 scripts/misc/searches/bijector_ab.py --stage score
    python3 scripts/misc/searches/bijector_ab.py --stage run --dry-run   # print commands only
    python3 scripts/misc/searches/bijector_ab.py --score --smoke         # exercise the scorer on
                                                                          # two already-run knn rows

Writes:

    results/notes/inference/phase_08_regularization/bijector_ab/verdict_<hardware>.json
    results/notes/inference/phase_08_regularization/bijector_ab/rows_<hardware>.npz

``<hardware>`` names the machine that ran the SCORER (``hardware_label()``
reads the local JAX backend), not the machine that ran the arms — the rows are
loaded from JSONs that may have been produced anywhere. The ``rows_*.npz`` dump
is a re-derivable restatement of those JSONs and is gitignored; only the
verdict JSON is committed.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
for _p in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

import numpy as np  # noqa: E402

# The two regularization coefficient paths, shared by ``knn`` and
# ``delaunay_adapt_split`` (both build ``af.Model(al.reg.AdaptSplit)`` under
# ``galaxies.source.pixelization.regularization`` -- see
# ``searches._setup._knn_model`` / ``_delaunay_adapt_split_model``). Kept as a
# module constant here (rather than re-derived via ``_samplers._probe_model``)
# so this driver has no import-time dependency on ``autolens``/``autofit`` --
# it only needs them once a ``run`` or non-smoke ``score`` actually executes.
REG_COEFFICIENT_PATHS = (
    "galaxies.source.pixelization.regularization.inner_coefficient",
    "galaxies.source.pixelization.regularization.outer_coefficient",
)

LEAF_SCRIPT = {
    "delaunay_adapt_split": "scripts/imaging/searches/multi_start_prodigy/delaunay_adapt_split.py",
    "knn": "scripts/imaging/searches/multi_start_prodigy/knn.py",
    "mge": "scripts/imaging/searches/multi_start_prodigy/mge.py",
}
TRACED_CELLS = {"delaunay_adapt_split", "knn"}

N_STARTS = 16
N_STEPS = 3000
BATCH_SIZE = 4
HIGH_LAMBDA_THRESHOLD = 1e4
REFERENCE_TOLERANCE_NATS = 10.0
STEPS_TO_REFERENCE_FACTOR = 2.0
# A row is VOID (cannot contribute a reference) if its diagnostics say so, if
# it never ran long enough to mean anything, or if it carries no schema stamp.
# 2 minutes: the shortest real arm in this campaign is the ~5 min MGE control,
# and every pixelized arm is >= 1.7 h -- anything under 2 min never fit.
VOID_MIN_WALL_S = 120.0
# ell_comps is an independent per-component box prior, so 21.5% of its volume
# is outside the unit disk and NON-PHYSICAL. A best point there cannot define
# a reference log-posterior (2026-08-28 ruling, DECISIONS.md).
ELL_COMPS_MAX_MAGNITUDE = 1.0
# Cited from PROGRAMME.md:579 ("historical: 2,200 steps vs 98 fixed") -- kept
# for the artifact/RESULTS.md framing; NOT used as the numeric reference in
# `score_rows` (see the module docstring's "READOUTS" section on why).
HISTORICAL_FREE_REG_STEPS = 2200
HISTORICAL_FIXED_REG_STEPS = 98


# -----------------------------------------------------------------------------
# Arm table
# -----------------------------------------------------------------------------


def build_arms(*, smoke: bool = False) -> list[dict]:
    """The 39 pre-registered (cell, log_det_method, bijector, seed) arms.

    ``smoke=True`` returns a tiny 2-arm subset (knn, seed 0, none/log_reg,
    4 starts / 20 steps) -- NOT used by ``--stage run`` in CI (that would
    still cost a real dataset build + fit); it exists only so a caller that
    wants to sanity-check the arm table shape without paying for a real
    campaign has something smaller than 39 to iterate.
    """
    if smoke:
        return [
            dict(
                cell="knn",
                model_type="knn",
                log_det_method=None,
                bijector=bijector,
                seed=0,
                n_starts=4,
                n_steps=20,
            )
            for bijector in ("none", "log_reg")
        ]

    arms: list[dict] = []
    for log_det_method in ("cholesky", "slogdet"):
        for bijector in ("none", "log_reg"):
            for seed in range(5):
                arms.append(
                    dict(
                        cell="delaunay_adapt_split",
                        model_type="delaunay_adapt_split",
                        log_det_method=log_det_method,
                        bijector=bijector,
                        seed=seed,
                        n_starts=N_STARTS,
                        n_steps=N_STEPS,
                    )
                )
    for bijector in ("none", "log_reg", "logit"):
        for seed in range(5):
            arms.append(
                dict(
                    cell="knn",
                    model_type="knn",
                    log_det_method=None,
                    bijector=bijector,
                    seed=seed,
                    n_starts=N_STARTS,
                    n_steps=N_STEPS,
                )
            )
    for bijector in ("none", "log_reg"):
        for seed in (0, 1):
            arms.append(
                dict(
                    cell="mge",
                    model_type="mge",
                    log_det_method=None,
                    bijector=bijector,
                    seed=seed,
                    n_starts=N_STARTS,
                    n_steps=N_STEPS,
                )
            )
    return arms


def arm_config_name(arm: dict) -> str:
    """``--config-name`` for one arm -- the ONLY thing ``resolve_output_paths``
    keys the results-JSON basename off, so this must be unique per arm."""
    log_det = arm["log_det_method"] or "auto"
    return f"{arm['cell']}_{log_det}_{arm['bijector']}_seed{arm['seed']}"


def results_dir() -> Path:
    return _ROOT / "results" / "notes" / "inference" / "phase_08_regularization" / "bijector_ab"


def search_output_dir(cell: str) -> Path:
    return (
        _ROOT
        / "results"
        / "searches"
        / "multi_start_prodigy"
        / "imaging"
        / cell
        / "hst"
        / "phase8b"
    )


def hardware_label() -> str:
    try:
        import jax

        if jax.default_backend() == "gpu":
            return f"gpu_{jax.devices()[0].device_kind.replace(' ', '_')}"
    except Exception:
        pass
    return f"cpu_{platform.machine()}"


# -----------------------------------------------------------------------------
# Run stage — subprocess per arm, one leaf script invocation each
# -----------------------------------------------------------------------------


def run_arm(arm: dict, *, dry_run: bool = False) -> Path | None:
    """Invoke the leaf script for one arm via subprocess, matching exactly how
    the real SLURM array submit (``hpc/batch_gpu/submit_phase8b_bijector_a100``)
    runs each task -- one isolated process per arm, so JAX/matplotlib/autofit
    global state never leaks between arms.
    """
    leaf = _ROOT / LEAF_SCRIPT[arm["cell"]]
    config_name = arm_config_name(arm)
    out_dir = search_output_dir(arm["cell"])

    env = dict(os.environ)
    env["SEARCHES_CLIPPER"] = "prior_box"
    env["SEARCHES_SCALER"] = "none"
    env["SEARCHES_N_STARTS"] = str(arm["n_starts"])
    env["SEARCHES_N_STEPS"] = str(arm["n_steps"])
    env["SEARCHES_BATCH_SIZE"] = str(BATCH_SIZE)
    env["SEARCHES_LANE_HISTORY"] = "1"
    env["SEARCHES_SEED"] = str(arm["seed"])
    env["SEARCHES_BIJECTOR"] = arm["bijector"]
    if arm["log_det_method"] is not None:
        env["SEARCHES_LOG_DET_METHOD"] = arm["log_det_method"]
    else:
        env.pop("SEARCHES_LOG_DET_METHOD", None)
    if arm["cell"] in TRACED_CELLS:
        env["SEARCHES_TRACE_PARAMS"] = ",".join(REG_COEFFICIENT_PATHS)
    else:
        env.pop("SEARCHES_TRACE_PARAMS", None)

    cmd = [
        sys.executable,
        str(leaf),
        "--instrument",
        "hst",
        "--config-name",
        config_name,
        "--output-dir",
        str(out_dir),
    ]
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return None

    t0 = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=str(_ROOT))
    print(f"  arm {config_name} done in {time.time() - t0:.1f}s", flush=True)

    matches = sorted(out_dir.glob(f"*_{config_name}.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise RuntimeError(
            f"arm {config_name!r} completed but no results JSON matching "
            f"'*_{config_name}.json' was found under {out_dir} -- the leaf "
            f"script's own --config-name/--output-dir resolution must have "
            f"diverged from this driver's."
        )
    return matches[-1]


def run_all(arms: list[dict], *, dry_run: bool = False) -> list[Path]:
    paths = []
    for i, arm in enumerate(arms):
        print(f"\n=== arm {i + 1}/{len(arms)}: {arm_config_name(arm)} ===", flush=True)
        path = run_arm(arm, dry_run=dry_run)
        if path is not None:
            paths.append(path)
    return paths


# -----------------------------------------------------------------------------
# Readouts — one results JSON -> one row
# -----------------------------------------------------------------------------


def _nan_safe_array(nested) -> np.ndarray | None:
    """``None`` (JSON null) entries -> ``np.nan``; ``None`` input -> ``None``."""
    if nested is None:
        return None

    def convert(x):
        if isinstance(x, list):
            return [convert(v) for v in x]
        return np.nan if x is None else x

    return np.asarray(convert(nested), dtype=float)


def _best_point_ell_comps(
    payload: dict, per_lane: list[dict], winning_lane: int | None
) -> tuple[dict | None, str | None]:
    """``ell_comps`` magnitude at this row's best point, and the field it came
    from (2026-08-28 ruling — see the module docstring's amendments section).

    Preferred source is the recovery verifier's own recorded value; rows that
    completed normally do not carry it, so it is recomputed from the winning
    lane's best parameter vector through ``diagnostics.ell_comps_pairs``. The
    two agree exactly on every recovered row in this campaign, so the fallback
    is the same measurement, not a looser one.
    """
    recorded = (payload.get("recovered_offline_verification") or {}).get(
        "best_point_ell_comps_magnitude"
    )
    if recorded:
        return (
            {k: float(v) for k, v in recorded.items()},
            "recovered_offline_verification.best_point_ell_comps_magnitude",
        )

    pairs = (payload.get("diagnostics") or {}).get("ell_comps_pairs") or {}
    if not pairs or winning_lane is None or winning_lane >= len(per_lane):
        return None, None
    params = (per_lane[winning_lane] or {}).get("lane_best_params")
    if not params:
        return None, None
    magnitudes = {}
    for path, indices in pairs.items():
        if len(indices) != 2 or max(indices) >= len(params):
            continue
        magnitudes[path] = float(np.hypot(params[indices[0]], params[indices[1]]))
    if not magnitudes:
        return None, None
    return magnitudes, "diagnostics.ell_comps_pairs + per_lane[winning].lane_best_params"


def row_from_payload(payload: dict, *, source: Path | None = None) -> dict:
    """Extract this experiment's readouts from one MultiStart* results JSON
    (``_runner._build_summary`` / ``_per_lane.per_lane_block`` shape).
    """
    diag = payload.get("diagnostics") or {}
    cfg = payload.get("sampler_config") or {}
    counters = diag.get("counters") or {}
    per_lane = diag.get("per_lane") or []

    n_starts = diag.get("n_starts_configured")
    total_steps = counters.get("total_steps")
    n_clipped = counters.get("n_clipped_lane_steps")
    n_value_nan = counters.get("n_value_nan_lane_steps")

    value_nan_hist = diag.get("lane_value_nan_history")
    value_nan_arr = None if value_nan_hist is None else np.asarray(value_nan_hist, dtype=bool)
    trace_hist = _nan_safe_array(diag.get("trace_history"))
    trace_idx = diag.get("trace_param_indices")
    fom_hist = _nan_safe_array(diag.get("fom_history_global_best"))

    first_nan_step = None
    coeff_at_first_nan = None
    if value_nan_arr is not None and value_nan_arr.size:
        any_per_step = value_nan_arr.any(axis=1)
        hits = np.flatnonzero(any_per_step)
        if hits.size:
            first_nan_step = int(hits[0])
            if trace_hist is not None and trace_hist.shape[0] > first_nan_step:
                nan_lanes = np.flatnonzero(value_nan_arr[first_nan_step])
                if nan_lanes.size:
                    coeff_at_first_nan = trace_hist[first_nan_step, nan_lanes[0]].tolist()

    frac_high_lambda = None
    if trace_hist is not None and trace_hist.size:
        with np.errstate(invalid="ignore"):
            high = np.any(trace_hist > HIGH_LAMBDA_THRESHOLD, axis=-1)
        frac_high_lambda = float(np.nanmean(high.astype(float)))

    lane_bests = [lane.get("lane_best_log_posterior") for lane in per_lane]
    lane_bests_finite = [v for v in lane_bests if v is not None]
    best_log_posterior = max(lane_bests_finite) if lane_bests_finite else None
    winning_lane = (
        int(np.argmax([-np.inf if v is None else v for v in lane_bests]))
        if lane_bests_finite
        else None
    )

    pinned_final_counts = [
        lane.get("n_pinned_final") for lane in per_lane if lane.get("n_pinned_final")
    ]

    ell_comps, ell_comps_source = _best_point_ell_comps(payload, per_lane, winning_lane)
    ell_comps_max = max(ell_comps.values()) if ell_comps else None

    total_wall_s = (payload.get("performance") or {}).get("total_wall_s")
    void_reasons = []
    if diag.get("valid") is False:
        void_reasons.append(f"diagnostics.valid=false {diag.get('invalid_reasons') or []}")
    if total_wall_s is not None and total_wall_s < VOID_MIN_WALL_S:
        void_reasons.append(f"total_wall_s={total_wall_s:.1f} < {VOID_MIN_WALL_S:g}")
    if payload.get("schema_version") is None:
        void_reasons.append("no schema_version")

    return {
        "source": str(source) if source is not None else None,
        "cell": payload.get("model"),
        "log_det_method": payload.get("log_det_method"),
        "bijector": cfg.get("bijector"),
        "seed": cfg.get("seed"),
        "record_lane_nan_history": cfg.get("record_lane_nan_history"),
        "n_starts": n_starts,
        "total_steps": total_steps,
        "n_resurrections": counters.get("n_resurrections"),
        "n_clipped_lane_steps": n_clipped,
        "n_value_nan_lane_steps": n_value_nan,
        "clip_rate": (
            n_clipped / (total_steps * n_starts)
            if n_clipped is not None and total_steps and n_starts
            else None
        ),
        "first_value_nan_step": first_nan_step,
        "coefficient_at_first_nan": coeff_at_first_nan,
        "frac_steps_high_lambda": frac_high_lambda,
        "best_log_posterior": best_log_posterior,
        "winning_lane_index": winning_lane,
        # Physical-validity inputs to the 2026-08-28 F2 reference rule.
        "total_wall_s": total_wall_s,
        "diagnostics_valid": diag.get("valid"),
        "void_reasons": void_reasons,
        "best_point_ell_comps_magnitude": ell_comps,
        "best_point_ell_comps_magnitude_max": ell_comps_max,
        "ell_comps_source": ell_comps_source,
        # The two winning-lane scalars F4 compares between bijector arms:
        # the global best figure of merit the search reached, and the best
        # point's log-likelihood as PyAutoFit recorded it.
        "best_fom": counters.get("best_fom"),
        "max_log_likelihood": (payload.get("results") or {}).get("max_log_likelihood"),
        "fom_history_global_best": None if fom_hist is None else fom_hist.tolist(),
        "step0_fom": (
            float(fom_hist[0])
            if fom_hist is not None and fom_hist.size and np.isfinite(fom_hist[0])
            else None
        ),
        "n_lanes_pinned_final": len(pinned_final_counts),
        "max_pinned_final_count": max(pinned_final_counts) if pinned_final_counts else 0,
        "final_params_per_lane": [lane.get("final_params") for lane in per_lane],
        "lane_best_params_per_lane": [lane.get("lane_best_params") for lane in per_lane],
        "trace_param_indices": trace_idx,
        # See the module docstring's READOUTS section: NOT recoverable from
        # the results JSON this driver reads (opt_state is not serialized).
        "final_d": None,
        "final_d_note": (
            "not exposed by _per_lane.per_lane_block (search_internal['opt_state'] "
            "is not JSON-serialized); would need an in-process capture, which "
            "this driver deliberately does not do (see module docstring)."
        ),
    }


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows.append(row_from_payload(payload, source=path))
    return rows


def discover_rows(*, smoke: bool = False) -> list[dict]:
    """Load every row this driver's own arm table could have produced, by
    globbing the standard output dirs for each cell's ``phase8b`` results.
    """
    if smoke:
        out_dir = search_output_dir("knn")
        matches = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if len(matches) < 2:
            raise SystemExit(
                f"--score --smoke needs at least 2 results JSONs under {out_dir} "
                f"(from the two knn.py smoke runs described in the module "
                f"docstring's Usage section) — found {len(matches)}."
            )
        return load_rows(matches[-2:])

    paths = []
    for cell in LEAF_SCRIPT:
        out_dir = search_output_dir(cell)
        paths.extend(sorted(out_dir.glob("*.json")))
    if not paths:
        raise SystemExit(
            f"no results JSONs found under {_ROOT / 'results' / 'searches' / 'multi_start_prodigy' / 'imaging'}"
            f"/*/hst/phase8b/ — run `--stage run` first."
        )
    return load_rows(paths)


# -----------------------------------------------------------------------------
# Verdict — F1-F5, scored on already-loaded rows (pure, so it is testable on
# synthetic rows without running a search)
# -----------------------------------------------------------------------------


def _group(rows: list[dict], *, cell: str, log_det_method: str | None = None) -> list[dict]:
    out = [r for r in rows if r["cell"] == cell]
    if log_det_method is not None:
        out = [r for r in out if r["log_det_method"] == log_det_method]
    tiers = {r["log_det_method"] for r in out}
    if len(tiers) > 1:
        raise ValueError(
            f"refusing to compare cell={cell!r} rows with mixed log_det_method "
            f"{sorted(str(t) for t in tiers)}: pass log_det_method= to select one."
        )
    return out


def _tiers_for_cell(rows: list[dict], cell: str) -> list[str | None]:
    """Every distinct ``log_det_method`` present for one cell's rows, sorted
    with ``None`` first. Used to score a cell PER tier rather than pooling —
    ``delaunay_adapt_split`` legitimately has both ``cholesky`` and
    ``slogdet`` rows in the real campaign, and ``_group`` refuses to mix
    them."""
    tiers = {r["log_det_method"] for r in rows if r["cell"] == cell}
    return sorted(tiers, key=lambda t: (t is not None, t))


def _by_bijector(rows: list[dict], label: str) -> list[dict]:
    return sorted((r for r in rows if r["bijector"] == label), key=lambda r: r["seed"])


def _row_label(row: dict) -> str:
    return f"{row.get('cell')}·{row.get('log_det_method') or 'auto'}·{row.get('bijector')}·seed{row.get('seed')}"


def _reference_exclusion(row: dict) -> str | None:
    """Why this row may not define a reference log-posterior, or ``None`` if
    it may (2026-08-28 ruling — module docstring, amendment 1)."""
    if row.get("best_log_posterior") is None:
        return "no lane_best_log_posterior recorded"
    void = row.get("void_reasons") or []
    if void:
        return "void: " + "; ".join(void)
    magnitude = row.get("best_point_ell_comps_magnitude_max")
    if magnitude is not None and magnitude >= ELL_COMPS_MAX_MAGNITUDE:
        return (
            f"non-physical: best-point ell_comps magnitude {magnitude:.6g} >= "
            f"{ELL_COMPS_MAX_MAGNITUDE:g} (outside the unit disk), from "
            f"{row.get('ell_comps_source')}"
        )
    return None


def _resolve_reference(group_rows: list[dict]) -> tuple[float | None, dict]:
    """The empirical reference log-posterior for one (cell, log_det_method)
    group, plus the audit trail behind it.

    **Amended 2026-08-28 (issue #185).** The maximum is taken over EVERY arm
    in the group -- ``none``, ``log_reg`` and ``logit`` alike -- restricted to
    physically valid rows (see ``_reference_exclusion``). It used to be the
    max over the ``none`` arm only, which made the control arm's own stalling
    define the target. See the module docstring's READOUTS section for why
    this is not the literal historical fixed-reg value either way.
    """
    considered, excluded = [], []
    for row in group_rows:
        reason = _reference_exclusion(row)
        if reason is None:
            considered.append(row)
        else:
            excluded.append(
                {
                    "row": _row_label(row),
                    "best_log_posterior": row.get("best_log_posterior"),
                    "reason": reason,
                }
            )

    best = max(considered, key=lambda r: r["best_log_posterior"]) if considered else None
    detail = {
        "rule": (
            "max lane_best_log_posterior over ALL bijector arms in this (cell, "
            "log_det_method) group, restricted to physically valid rows "
            "(2026-08-28 ruling, DECISIONS.md / issue #185)"
        ),
        "reference_row": _row_label(best) if best is not None else None,
        "n_rows_in_group": len(group_rows),
        "n_rows_considered": len(considered),
        "rows_considered": [_row_label(r) for r in considered],
        "rows_excluded": excluded,
        "ell_comps_max_magnitude": ELL_COMPS_MAX_MAGNITUDE,
        "void_min_wall_s": VOID_MIN_WALL_S,
    }
    return (best["best_log_posterior"] if best is not None else None), detail


def _steps_to_reference(row: dict, reference: float | None, tolerance: float) -> int | None:
    if reference is None or row["fom_history_global_best"] is None:
        return None
    fom = np.asarray(row["fom_history_global_best"], dtype=float)
    log_post = np.where(np.isfinite(fom), -0.5 * fom, np.nan)
    hits = np.flatnonzero(np.abs(log_post - reference) <= tolerance)
    return int(hits[0]) if hits.size else None


# -----------------------------------------------------------------------------
# UNSCORABLE — the third state (2026-08-27, issue #182)
# -----------------------------------------------------------------------------
#
# Every criterion below returns THREE states, never two:
#
#   {"scorable": True,  "falsified": True}   the criterion fired
#   {"scorable": True,  "falsified": False}  the criterion did not fire
#   {"scorable": False, "falsified": None, "reason": ...}   it could not be asked
#
# The scorer used to collapse the third into one of the first two, and in
# OPPOSITE directions: ``score_f1`` computed ``bool(None) or bool(None)`` ->
# ``False`` (missing data read as "criterion did not fire" = a silent PASS),
# while ``score_f2`` computed ``median_ratio is None or ...`` -> ``True``
# (missing data read as "criterion fired" = a silent FAIL). A campaign with
# arms still running, or with the diagnostics it needs switched off, therefore
# produced a confident verdict about data that did not exist. A criterion that
# cannot be asked must say so, and ``score_rows`` must refuse to conclude
# while any criterion is in that state.


def _unscorable(reason: str, **extra) -> dict:
    """The third state: this criterion could not be evaluated at all."""
    return {"falsified": None, "scorable": False, "reason": reason, **extra}


def score_f1(delaunay_rows: list[dict]) -> dict:
    """F1 — median first-NaN step / value-NaN lane-steps, none vs log_reg, on
    delaunay_adapt_split (the actual wall cell)."""
    none_rows = _by_bijector(delaunay_rows, "none")
    log_reg_rows = _by_bijector(delaunay_rows, "log_reg")

    def _median_first_nan(rows):
        vals = [r["first_value_nan_step"] for r in rows if r["first_value_nan_step"] is not None]
        return float(np.median(vals)) if vals else None

    def _total_value_nan(rows):
        vals = [
            r["n_value_nan_lane_steps"] for r in rows if r["n_value_nan_lane_steps"] is not None
        ]
        return sum(vals) if vals else None

    none_median = _median_first_nan(none_rows)
    log_reg_median = _median_first_nan(log_reg_rows)
    none_total = _total_value_nan(none_rows)
    log_reg_total = _total_value_nan(log_reg_rows)

    not_earlier = (
        None if none_median is None or log_reg_median is None else log_reg_median >= none_median
    )
    not_reduced_50 = None if not none_total else (log_reg_total or 0) >= 0.5 * none_total

    measured = {
        "none_median_first_nan_step": none_median,
        "log_reg_median_first_nan_step": log_reg_median,
        "none_total_value_nan_lane_steps": none_total,
        "log_reg_total_value_nan_lane_steps": log_reg_total,
        "not_earlier": not_earlier,
        "not_reduced_50pct": not_reduced_50,
        "n_none_rows": len(none_rows),
        "n_log_reg_rows": len(log_reg_rows),
    }
    # F1 is a disjunction: either limb firing falsifies it, so a True limb is
    # conclusive even when the other limb is missing. Only when NO limb fires
    # and at least one is unmeasurable is the criterion unscorable.
    if not_earlier is True or not_reduced_50 is True:
        return {**measured, "falsified": True, "scorable": True}
    if not_earlier is None or not_reduced_50 is None:
        return _unscorable(
            "F1 needs both a first-value-NaN step and a value-NaN lane-step total for the "
            "none AND log_reg arms on delaunay_adapt_split; at least one is missing "
            "(no rows for an arm, or SEARCHES_LANE_HISTORY was off so no NaN history was "
            "recorded).",
            **measured,
        )
    return {**measured, "falsified": False, "scorable": True}


def score_f2(knn_rows: list[dict]) -> dict:
    """F2 — steps-to-reference, none vs log_reg, at matched seeds, on knn.

    **Amended 2026-08-28 (issue #185).** The reference is now the group-wide
    physically-valid maximum (``_resolve_reference``), and "never reached
    within the step budget" has explicit semantics rather than silently
    producing no ratio: ``none`` never / ``log_reg`` does -> ``+inf`` (counts
    as >= 2x); ``log_reg`` never / ``none`` does -> ``0`` (counts against);
    both never -> that seed is unscorable and drops out of the median.
    """
    none_rows = {r["seed"]: r for r in _by_bijector(knn_rows, "none")}
    log_reg_rows = {r["seed"]: r for r in _by_bijector(knn_rows, "log_reg")}
    reference, reference_detail = _resolve_reference(knn_rows)

    per_seed = {}
    ratios = []
    for seed in sorted(set(none_rows) & set(log_reg_rows)):
        n_steps_none = _steps_to_reference(none_rows[seed], reference, REFERENCE_TOLERANCE_NATS)
        n_steps_log_reg = _steps_to_reference(
            log_reg_rows[seed], reference, REFERENCE_TOLERANCE_NATS
        )
        if n_steps_none is None and n_steps_log_reg is None:
            ratio, note = None, "neither arm reached the reference — seed unscorable"
        elif n_steps_none is None:
            ratio, note = (
                float("inf"),
                "none never reached the reference while log_reg did — counts as >= 2x",
            )
        elif n_steps_log_reg is None:
            ratio, note = (
                0.0,
                "log_reg never reached the reference while none did — counts against",
            )
        else:
            ratio, note = n_steps_none / max(n_steps_log_reg, 1), None
        per_seed[seed] = {
            "steps_none": n_steps_none,
            "steps_log_reg": n_steps_log_reg,
            "reduction_ratio": ratio,
            "note": note,
        }
        if ratio is not None:
            ratios.append(ratio)

    median_ratio = float(np.median(ratios)) if ratios else None
    measured = {
        "reference_log_posterior": reference,
        "reference_note": (
            "max best_log_posterior over ALL bijector arms in this (cell, "
            "log_det_method) group, restricted to physically valid rows "
            "(2026-08-28 ruling) — see module docstring, not the literal "
            f"historical fixed-reg run cited as {HISTORICAL_FIXED_REG_STEPS} steps."
        ),
        "reference_resolution": reference_detail,
        "per_seed": per_seed,
        "median_reduction_ratio": median_ratio,
        "required_ratio": STEPS_TO_REFERENCE_FACTOR,
        "n_matched_seeds": len(per_seed),
        "n_scorable_seeds": len(ratios),
    }
    if reference is None:
        return _unscorable(
            "F2 has no reference log-posterior: no physically valid knn row recorded a "
            "lane_best_log_posterior to resolve one from.",
            **measured,
        )
    if not per_seed:
        return _unscorable(
            "F2 needs matched seeds present in BOTH the none and log_reg knn arms; there are none.",
            **measured,
        )
    if median_ratio is None:
        return _unscorable(
            "F2 could not compute a steps-to-reference ratio on any matched seed — on every "
            f"matched seed NEITHER arm reached within {REFERENCE_TOLERANCE_NATS} nats of the "
            "reference, or fom_history_global_best was not recorded. 'neither reached the "
            "reference' is not the same measurement as 'reached it too slowly', so this is "
            "unscorable rather than falsified.",
            **measured,
        )
    return {**measured, "falsified": median_ratio < STEPS_TO_REFERENCE_FACTOR, "scorable": True}


def score_f3_group(group_rows: list[dict]) -> dict:
    """F3 for one already (cell, log_det_method)-filtered group: fraction of
    steps at lambda > 1e4, log_reg vs none."""
    none_fracs = [
        r["frac_steps_high_lambda"]
        for r in _by_bijector(group_rows, "none")
        if r["frac_steps_high_lambda"] is not None
    ]
    log_reg_fracs = [
        r["frac_steps_high_lambda"]
        for r in _by_bijector(group_rows, "log_reg")
        if r["frac_steps_high_lambda"] is not None
    ]
    none_mean = float(np.mean(none_fracs)) if none_fracs else None
    log_reg_mean = float(np.mean(log_reg_fracs)) if log_reg_fracs else None
    measured = {
        "none_mean_frac_high_lambda": none_mean,
        "log_reg_mean_frac_high_lambda": log_reg_mean,
    }
    if none_mean is None or log_reg_mean is None:
        return _unscorable(
            "F3 needs a traced high-lambda fraction for BOTH arms in this group; at least "
            "one arm has no row with trace_history recorded (SEARCHES_TRACE_PARAMS off, or "
            "the arm has not run).",
            **measured,
        )
    return {**measured, "falsified": log_reg_mean >= none_mean, "scorable": True}


def score_f3(rows: list[dict]) -> dict:
    """F3 — fraction of steps at lambda > 1e4, log_reg vs none, on either
    wall cell (falsified if EITHER cell/tier fails). ``delaunay_adapt_split``
    is scored PER log_det_method tier (never pooled across cholesky/slogdet
    -- same "refuse to mix tiers" rule ``_group`` enforces for F1/F2)."""
    per_cell: dict = {}
    falsified = False
    unscorable_reasons: list[str] = []

    delaunay_tiers = _tiers_for_cell(rows, "delaunay_adapt_split")
    per_tier = {}
    for tier in delaunay_tiers:
        tier_rows = _group(rows, cell="delaunay_adapt_split", log_det_method=tier)
        result = score_f3_group(tier_rows)
        label = tier if tier is not None else "auto"
        per_tier[label] = result
        if result["falsified"]:
            falsified = True
        elif not result.get("scorable"):
            unscorable_reasons.append(f"delaunay_adapt_split[{label}]: {result['reason']}")
    if not delaunay_tiers:
        unscorable_reasons.append("delaunay_adapt_split: no rows")
    per_cell["delaunay_adapt_split"] = {"per_log_det_method": per_tier}

    knn_rows = _group(rows, cell="knn")
    knn_result = score_f3_group(knn_rows) if knn_rows else None
    per_cell["knn"] = knn_result
    if knn_result is None:
        unscorable_reasons.append("knn: no rows")
    elif knn_result["falsified"]:
        falsified = True
    elif not knn_result.get("scorable"):
        unscorable_reasons.append(f"knn: {knn_result['reason']}")

    # Same disjunction rule as F1: EITHER cell/tier firing falsifies F3, so a
    # fired limb is conclusive even with the rest unscorable.
    if falsified:
        return {"per_cell": per_cell, "falsified": True, "scorable": True}
    if unscorable_reasons:
        return _unscorable("; ".join(unscorable_reasons), per_cell=per_cell)
    return {"per_cell": per_cell, "falsified": False, "scorable": True}


FP64_RELATIVE_TOLERANCE = 1e-9


def _rel_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def score_f4(mge_rows: list[dict], knn_rows: list[dict]) -> dict:
    """F4 — MGE control equivalence (none vs log_reg) and the knn `logit`
    pinned-lane pathology.

    **Criterion amended 2026-08-27 (issue #182).** It used to be byte-identity
    of every per-lane final/best parameter vector. That is a stronger claim
    than the pre-registration's own equivalence argument supports: the MGE
    control's ``log_reg`` map is EMPTY, so the two arms compute the same
    objective, but they are still two separate GPU runs, and non-associative
    float reduction, resurrection draws and lane-batch ordering move
    trailing bits in lanes that never mattered to the answer. Byte-identity
    across all 16 lanes therefore fails for reasons that have nothing to do
    with the bijector, and F5 already proves the objective itself is inert at
    matched physical points.

    **Amended again 2026-08-28 (issue #185): the fp-equivalence limb is now
    INFORMATIONAL too.** The winning lane's ``best_fom`` /
    ``max_log_likelihood`` comparison is a cross-run reproducibility
    measurement — exactly what F5 measures, and it fails for exactly the same
    reason (the MGE control's map is EMPTY and its two arms still diverge
    1.7e-2 relative by step 3000). It is reported and never sets
    ``falsified``. F4 now trips on ONE thing: the ``knn`` ``logit`` arm
    reproducing the pinned-lane-to-boundary pathology. Both the fp-equivalence
    result and the pre-amendment byte-identity result
    (``mge_per_seed_byte_identical``) ride along as informational fields.
    """
    none_rows = {r["seed"]: r for r in _by_bijector(mge_rows, "none")}
    log_reg_rows = {r["seed"]: r for r in _by_bijector(mge_rows, "log_reg")}

    def _identical(a: dict, b: dict) -> bool | None:
        pa, pb = a["final_params_per_lane"], b["final_params_per_lane"]
        if pa is None or pb is None:
            return None
        return pa == pb and a["lane_best_params_per_lane"] == b["lane_best_params_per_lane"]

    per_seed_identity = {}
    per_seed_equivalence = {}
    for seed in sorted(set(none_rows) & set(log_reg_rows)):
        a, b = none_rows[seed], log_reg_rows[seed]
        per_seed_identity[seed] = _identical(a, b)
        fom_rel = _rel_diff(a["best_fom"], b["best_fom"])
        ll_rel = _rel_diff(a["max_log_likelihood"], b["max_log_likelihood"])
        agree = (
            None
            if fom_rel is None or ll_rel is None
            else (fom_rel <= FP64_RELATIVE_TOLERANCE and ll_rel <= FP64_RELATIVE_TOLERANCE)
        )
        per_seed_equivalence[seed] = {
            "none_best_fom": a["best_fom"],
            "log_reg_best_fom": b["best_fom"],
            "best_fom_rel_diff": fom_rel,
            "none_max_log_likelihood": a["max_log_likelihood"],
            "log_reg_max_log_likelihood": b["max_log_likelihood"],
            "max_log_likelihood_rel_diff": ll_rel,
            "none_winning_lane_index": a["winning_lane_index"],
            "log_reg_winning_lane_index": b["winning_lane_index"],
            "agree_within_fp64": agree,
        }
    mge_differs = any(v["agree_within_fp64"] is False for v in per_seed_equivalence.values())
    mge_checked = any(v["agree_within_fp64"] is not None for v in per_seed_equivalence.values())

    logit_rows = _by_bijector(knn_rows, "logit")
    # A crude, documented threshold for "reproduces the pinned-lane pathology":
    # more than half a lane's traced coordinates parked exactly on the
    # (finite, logit-mapped) box bound at completion. `n_pinned_final` counts
    # ALL parameters, not just the traced regularization ones, so this is a
    # necessary-not-sufficient signal, flagged as such.
    logit_pinned = [
        {"seed": r["seed"], "max_pinned_final_count": r["max_pinned_final_count"]}
        for r in logit_rows
    ]
    logit_pathology = (
        any(r["max_pinned_final_count"] >= 1 for r in logit_rows) if logit_rows else None
    )

    measured = {
        "criterion": (
            "the knn logit arm reproduces the pinned-lane-to-boundary pathology "
            "(amended 2026-08-28, issue #185 — the MGE winning-lane best_fom / "
            "max_log_likelihood fp-equivalence limb is INFORMATIONAL only: it is "
            "the same cross-run fp-reproducibility quantity F5 measures, not an "
            "objective difference the bijector could cause)"
        ),
        "fp64_relative_tolerance": FP64_RELATIVE_TOLERANCE,
        # INFORMATIONAL ONLY since 2026-08-28 — reported, never scored.
        "mge_per_seed_equivalence": per_seed_equivalence,
        # INFORMATIONAL ONLY — the pre-amendment byte-identity check. Reported
        # so a reader can see how far the two arms' per-lane vectors drifted,
        # never scored (see this function's docstring).
        "mge_per_seed_byte_identical": per_seed_identity,
        "mge_checked": mge_checked,
        "mge_differs": mge_differs,
        "knn_logit_pinned_final_counts": logit_pinned,
        "knn_logit_pathology_suspected": logit_pathology,
    }
    # 2026-08-28: only the logit limb scores. mge_differs stays in `measured`
    # as a reported diagnostic and can no longer set `falsified`.
    if logit_pathology:
        return {**measured, "falsified": True, "scorable": True}
    if logit_pathology is None:
        return _unscorable(
            "F4 has no knn logit rows, so its only scoring limb (the pinned-lane-to-boundary "
            "pathology) is unmeasured. The MGE fp-equivalence limb is informational since "
            "2026-08-28 and cannot settle F4 on its own.",
            **measured,
        )
    return {**measured, "falsified": False, "scorable": True}


def score_f5(rows: list[dict]) -> dict:
    """F5 — fom at the shared initial broad-start draw (step-0 global-best
    fom) between arms at a matched seed, compared at 1e-9 relative.

    **DEMOTED 2026-08-28 (issue #185) from a HALT to a reported diagnostic.**
    The two arms are two SEPARATE GPU processes, so this compares floating-
    point reproducibility across runs, not the objective: the MGE control's
    ``log_reg`` map is provably EMPTY (an identity reparameterization) and its
    two arms still diverge 1.7e-2 relative by step 3000. The same 1e-9 number
    is still computed and reported, now as an fp-reproducibility diagnostic
    with ``halts: False``. The sound version — one process, one physical
    point, both parameterizations, assert bit-equality — is a PyAutoFit unit
    test, filed separately; a two-run A/B cannot measure it.
    """
    problems = []
    for cell in {r["cell"] for r in rows}:
        for log_det in {r["log_det_method"] for r in rows if r["cell"] == cell}:
            cell_rows = _group(rows, cell=cell, log_det_method=log_det)
            by_bijector = {}
            for r in cell_rows:
                by_bijector.setdefault(r["bijector"], {})[r["seed"]] = r
            labels = sorted(by_bijector)
            for i, label_a in enumerate(labels):
                for label_b in labels[i + 1 :]:
                    for seed in sorted(set(by_bijector[label_a]) & set(by_bijector[label_b])):
                        a = by_bijector[label_a][seed]["step0_fom"]
                        b = by_bijector[label_b][seed]["step0_fom"]
                        if a is None or b is None:
                            continue
                        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
                        if rel > 1e-9:
                            problems.append(
                                {
                                    "cell": cell,
                                    "log_det_method": log_det,
                                    "seed": seed,
                                    "bijector_a": label_a,
                                    "bijector_b": label_b,
                                    "fom_a": a,
                                    "fom_b": b,
                                    "rel_diff": rel,
                                }
                            )
    return {
        "role": "fp-reproducibility diagnostic (DEMOTED from HALT, 2026-08-28, issue #185)",
        "note": (
            "compares the step-0 global-best fom between two separate GPU runs at a "
            "matched seed. A difference is cross-run floating-point non-reproducibility, "
            "not evidence the bijector moved the objective — the MGE control's map is "
            "EMPTY and still diverges. Reported, never scored, never halts."
        ),
        "problems": problems,
        "n_problems": len(problems),
        "max_rel_diff": max((p["rel_diff"] for p in problems), default=0.0),
        "reproducible_within_1e-9": not problems,
        "halts": False,
    }


def score_rows(rows: list[dict]) -> dict:
    """Run F1-F5 against a set of already-loaded rows. Pure function -- takes
    no filesystem/network dependency, so it is exercised directly by
    ``test_searches_bijector.py`` on synthetic rows.
    """
    # F5 is a reported diagnostic since 2026-08-28 (issue #185): it is computed
    # and carried in the verdict under `diagnostics.f5`, and never halts.
    f5 = score_f5(rows)

    knn_rows = _group(rows, cell="knn")
    mge_rows = _group(rows, cell="mge")

    # delaunay_adapt_split legitimately carries BOTH cholesky and slogdet
    # rows in the real campaign; score F1 per tier (never pooled — same
    # "refuse to mix tiers" rule _group enforces elsewhere) and combine
    # conservatively: falsified if EITHER tier falsifies.
    delaunay_tiers = _tiers_for_cell(rows, "delaunay_adapt_split")
    f1_per_tier = {}
    for tier in delaunay_tiers:
        tier_rows = _group(rows, cell="delaunay_adapt_split", log_det_method=tier)
        f1_per_tier[tier if tier is not None else "auto"] = score_f1(tier_rows)
    if not f1_per_tier:
        f1 = _unscorable("no delaunay_adapt_split rows")
    elif any(bool(v["falsified"]) for v in f1_per_tier.values()):
        f1 = {"per_log_det_method": f1_per_tier, "falsified": True, "scorable": True}
    elif all(v.get("scorable") for v in f1_per_tier.values()):
        f1 = {"per_log_det_method": f1_per_tier, "falsified": False, "scorable": True}
    else:
        f1 = _unscorable(
            "; ".join(
                f"delaunay_adapt_split[{tier}]: {v['reason']}"
                for tier, v in f1_per_tier.items()
                if not v.get("scorable")
            ),
            per_log_det_method=f1_per_tier,
        )
    f2 = score_f2(knn_rows) if knn_rows else _unscorable("no knn rows")
    f3 = score_f3(rows)
    f4 = score_f4(mge_rows, knn_rows) if mge_rows else _unscorable("no mge rows")

    criteria = {
        "f1_nan_wall_position": f1,
        "f2_steps_to_reference": f2,
        "f3_time_at_high_lambda": f3,
        "f4_mge_control_and_logit_pathology": f4,
    }
    falsified_count = sum(bool(c.get("falsified")) for c in criteria.values())
    unscorable = {k: v.get("reason") for k, v in criteria.items() if not v.get("scorable")}

    # The pre-registration says "any TWO criteria falsified -> 8B falsified".
    # That threshold cannot be evaluated while a criterion is unscorable: two
    # already-fired criteria settle it either way, but anything less is a
    # verdict about data that does not exist. INCONCLUSIVE is a real outcome
    # here, not a failure of the scorer (2026-08-27, issue #182).
    if unscorable and falsified_count < 2:
        verdict = "INCONCLUSIVE"
        falsified = None
    else:
        falsified = falsified_count >= 2
        verdict = "FALSIFIED" if falsified else "NOT FALSIFIED"

    return {
        **criteria,
        "diagnostics": {"f5": f5},
        "falsified_criteria_count": falsified_count,
        "unscorable_criteria": unscorable,
        "halted": False,
        "falsified": falsified,
        "verdict": verdict,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _json_safe(obj):
    """``inf``/``-inf``/``nan`` -> JSON-valid sentinels.

    F2's ratios are legitimately ``+inf`` (``none`` never reached the
    reference, ``log_reg`` did), and ``json.dumps`` would emit a bare
    ``Infinity``, which is not valid JSON. The value is kept exact in the
    scored dict (so tests and callers can compare it numerically) and only
    stringified on the way into the artifact.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else None)
    return obj


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # "verdict" is a documented synonym for "score" -- the campaign notes and
    # DECISIONS entries refer to this step as `--stage verdict`.
    p.add_argument("--stage", choices=("run", "score", "verdict"), default=None)
    p.add_argument("--score", action="store_true", help="shorthand for --stage score")
    p.add_argument("--smoke", action="store_true", help="tiny arm table / two-row scorer smoke")
    p.add_argument("--dry-run", action="store_true", help="print run commands without executing")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    stage = args.stage or ("score" if args.score else None)
    if stage == "verdict":
        stage = "score"
    if stage is None:
        raise SystemExit(
            "pass --stage run|score|verdict (or --score as a shorthand for --stage score)"
        )

    if stage == "run":
        arms = build_arms(smoke=args.smoke)
        print(f"=== RUN — {len(arms)} arms ===", flush=True)
        run_all(arms, dry_run=args.dry_run)
        return

    print("=== SCORE ===", flush=True)
    rows = discover_rows(smoke=args.smoke)
    print(f"  loaded {len(rows)} rows", flush=True)
    verdict = score_rows(rows)

    artifact = {
        "schema_version": 1,
        "experiment": "phase_8b_bijector_ab",
        "issue": 162,
        "hardware": hardware_label(),
        # The label above names the machine that ran the SCORER, not the arms
        # -- rows are loaded from JSONs that may have been produced anywhere.
        "hardware_note": (
            "hardware/`<hardware>` in the artifact filename is the scoring host, resolved "
            "from the local JAX backend. Each row's own run hardware is in its results JSON "
            "under `device` / `hardware`."
        ),
        "n_rows": len(rows),
        "n_rows_expected": len(build_arms()),
        # A verdict scored on fewer than the full pre-registered arm table is
        # PRELIMINARY and must be re-run when the missing arms land.
        "preliminary": len(rows) < len(build_arms()),
        "historical_reference": {
            "free_reg_steps": HISTORICAL_FREE_REG_STEPS,
            "fixed_reg_steps": HISTORICAL_FIXED_REG_STEPS,
            "source": "results/notes/inference/PROGRAMME.md:579",
        },
        "verdict": verdict,
        "pass_criteria_preregistered": [
            "F1: median first-NaN step under log_reg not later than none, "
            "OR value-NaN lane-steps fall < 50%",
            "F2 (reference amended 2026-08-28, issue #185): steps-to-reference not reduced "
            ">= 2x at matched seeds, where the reference is the group-wide max "
            "lane_best_log_posterior over ALL bijector arms restricted to physically valid "
            "rows, and 'never reached' scores as +inf / 0 / seed-unscorable",
            "F3: log_reg lanes spend >= same fraction of steps at lambda > 1e4",
            "F4 (amended 2026-08-27 #182, again 2026-08-28 #185): the knn logit arm "
            "reproduces the pinned-lane pathology. The MGE control's winning-lane "
            f"best_fom / max_log_likelihood fp64 agreement ({FP64_RELATIVE_TOLERANCE:g} "
            "relative) and the older per-lane byte-identity check are both retained as "
            "informational fields, not criteria.",
            "F5 (demoted 2026-08-28, issue #185): fom at matched physical points differs "
            "> 1e-9 relative. Reported as an fp-reproducibility diagnostic under "
            "verdict.diagnostics.f5; it does not halt and is not a criterion.",
            "Every criterion has a third state, UNSCORABLE: its inputs are absent, so it "
            "neither fired nor did not fire. The verdict is INCONCLUSIVE while any "
            "criterion is unscorable and fewer than two have fired.",
        ],
        "scorer_amendments": {
            "2026-08-27": (
                "UNSCORABLE state added to F1-F4 (they previously collapsed missing data "
                "to a silent PASS in F1 and a silent FAIL in F2); F4 changed from "
                "per-lane byte-identity to winning-lane fp64 equivalence. Issue #182."
            ),
            "2026-08-28": (
                "F2's reference became the group-wide max over ALL bijector arms restricted "
                "to physically valid rows (not void, best-point ell_comps magnitude < 1) "
                "instead of the max over the none arm alone; F2 'never reached' given "
                "explicit +inf / 0 / seed-unscorable semantics; F5 demoted from HALT to a "
                "reported fp-reproducibility diagnostic; F4's fp-equivalence limb made "
                "informational so F4 trips only on the knn logit pinned-lane pathology. "
                "Issue #185; the authoritative record is the dated "
                "results/notes/inference/DECISIONS.md entry."
            ),
        },
        "campaign_notes": {
            "stack_version_split": (
                "The arms scored here ran on 2026.8.17.1 with pre-#1536 PyAutoFit; the 15 "
                "resubmitted arms (RAL job 341978, array indices 0,2,3,14,17,19,21,24,25,"
                "26,27,30,32,33,34) run on PyAutoFit f466dce1a / PyAutoGalaxy 0fbe863d / "
                "PyAutoLens b23ee53e9. The likelihood code is unchanged between them "
                "(#1536/#713/#1538/#589 touch results-writing and an OPT-IN clipper only, "
                "and no 8B arm opts in), so the two halves are pooled — but this rests on "
                "a reading of four diffs, not a measurement, and is flagged."
            ),
            "clipper": (
                "No 8B arm used ClipperPriorBoxJoint: with a bijector set, the joint disk "
                "clipper is refused at construction (PyAutoFit "
                "multi_start_gradient/search.py:368-383). Every arm ran the per-component "
                "prior_box clipper, which is the mechanism that lets best points settle "
                "outside the unit disk. Filed as a PyAutoFit follow-up."
            ),
        },
    }

    dest_dir = results_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else f"_{hardware_label()}"
    dest = dest_dir / f"verdict{suffix}.json"
    dest.write_text(json.dumps(_json_safe(artifact), indent=2) + "\n")
    print(f"\nwritten: {dest}")

    rows_npz = dest_dir / f"rows{suffix}.npz"
    np.savez(
        rows_npz,
        rows_json=json.dumps(_json_safe(rows)),
    )
    print(f"written: {rows_npz}")

    _print_verdict(artifact)


def _print_verdict(artifact: dict) -> None:
    v = artifact["verdict"]
    print("\n" + "=" * 78)
    print("PHASE 8B — bijector A/B")
    print("=" * 78)
    print(
        f"rows: {artifact['n_rows']} / {artifact['n_rows_expected']}  "
        f"hardware: {artifact['hardware']}"
    )
    if artifact.get("preliminary"):
        print(
            f"*** PRELIMINARY — scored on {artifact['n_rows']} of "
            f"{artifact['n_rows_expected']} pre-registered arms; re-run when the rest land ***"
        )
    f5 = v["diagnostics"]["f5"]
    print(
        f"  [diagnostic] f5 fp-reproducibility: {f5['n_problems']} matched-seed pair(s) "
        f"above 1e-9 (max rel {f5['max_rel_diff']:.3g}) — reported, does not halt"
    )
    for key in (
        "f1_nan_wall_position",
        "f2_steps_to_reference",
        "f3_time_at_high_lambda",
        "f4_mge_control_and_logit_pathology",
    ):
        block = v[key]
        state = "UNSCORABLE" if not block.get("scorable") else f"falsified={block['falsified']}"
        print(f"  {key}: {state}")
        if not block.get("scorable"):
            print(f"      reason: {block.get('reason')}")
    print(f"\nfalsified criteria: {v['falsified_criteria_count']} / 4")
    if v.get("unscorable_criteria"):
        print(f"unscorable criteria: {len(v['unscorable_criteria'])} / 4")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
