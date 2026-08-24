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
  ``lane_best_log_posterior`` observed across every ``none``-bijector seed
  for that (cell, log_det_method) group. This is a DOCUMENTED DEVIATION from
  a literal "vs the historical fixed-reg run" reading; the resolved
  reference value is recorded in the verdict artifact so it can be checked,
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
- **F4** — the MGE control differs by ANY bit between ``none`` and
  ``log_reg`` (per-lane final/best params, byte-for-byte), OR the ``knn``
  ``logit`` arm reproduces the pinned-lane-to-infinity pathology (a lane
  parked on the logit box boundary at completion).
- **F5** — figure of merit at matched physical points (the shared initial
  broad-start draw, i.e. the step-0 global-best fom) differs by more than
  1e-9 relative between arms at a matched seed -> this is a BUG (the
  bijector must not change the physical objective — see
  ``autofit.non_linear.bijector``'s equivalence argument), not a science
  finding, and HALTS the verdict rather than counting toward the other four.

Usage (from the ``autolens_profiling/`` root)::

    python3 scripts/misc/searches/bijector_ab.py --stage run
    python3 scripts/misc/searches/bijector_ab.py --stage score
    python3 scripts/misc/searches/bijector_ab.py --stage run --dry-run   # print commands only
    python3 scripts/misc/searches/bijector_ab.py --score --smoke         # exercise the scorer on
                                                                          # two already-run knn rows

Writes:

    results/notes/inference/phase_08_regularization/bijector_ab/verdict_<hardware>.json
    results/notes/inference/phase_08_regularization/bijector_ab/rows_<hardware>.npz
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

    pinned_final_counts = [
        lane.get("n_pinned_final") for lane in per_lane if lane.get("n_pinned_final")
    ]

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


def _resolve_reference(none_rows: list[dict]) -> float | None:
    """The empirical reference log-posterior for a (cell, log_det_method)
    group -- see the module docstring's READOUTS section for why this is not
    the literal historical fixed-reg value.
    """
    values = [r["best_log_posterior"] for r in none_rows if r["best_log_posterior"] is not None]
    return max(values) if values else None


def _steps_to_reference(row: dict, reference: float | None, tolerance: float) -> int | None:
    if reference is None or row["fom_history_global_best"] is None:
        return None
    fom = np.asarray(row["fom_history_global_best"], dtype=float)
    log_post = np.where(np.isfinite(fom), -0.5 * fom, np.nan)
    hits = np.flatnonzero(np.abs(log_post - reference) <= tolerance)
    return int(hits[0]) if hits.size else None


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
    falsified = bool(not_earlier) or bool(not_reduced_50)
    return {
        "none_median_first_nan_step": none_median,
        "log_reg_median_first_nan_step": log_reg_median,
        "none_total_value_nan_lane_steps": none_total,
        "log_reg_total_value_nan_lane_steps": log_reg_total,
        "not_earlier": not_earlier,
        "not_reduced_50pct": not_reduced_50,
        "falsified": falsified,
    }


def score_f2(knn_rows: list[dict]) -> dict:
    """F2 — steps-to-reference, none vs log_reg, at matched seeds, on knn."""
    none_rows = {r["seed"]: r for r in _by_bijector(knn_rows, "none")}
    log_reg_rows = {r["seed"]: r for r in _by_bijector(knn_rows, "log_reg")}
    reference = _resolve_reference(list(none_rows.values()))

    per_seed = {}
    ratios = []
    for seed in sorted(set(none_rows) & set(log_reg_rows)):
        n_steps_none = _steps_to_reference(none_rows[seed], reference, REFERENCE_TOLERANCE_NATS)
        n_steps_log_reg = _steps_to_reference(
            log_reg_rows[seed], reference, REFERENCE_TOLERANCE_NATS
        )
        ratio = (
            None
            if not n_steps_none or n_steps_log_reg is None
            else n_steps_none / max(n_steps_log_reg, 1)
        )
        per_seed[seed] = {
            "steps_none": n_steps_none,
            "steps_log_reg": n_steps_log_reg,
            "reduction_ratio": ratio,
        }
        if ratio is not None:
            ratios.append(ratio)

    median_ratio = float(np.median(ratios)) if ratios else None
    falsified = median_ratio is None or median_ratio < STEPS_TO_REFERENCE_FACTOR
    return {
        "reference_log_posterior": reference,
        "reference_note": (
            "max best_log_posterior across none-arm seeds for this (cell, "
            "log_det_method) group — see module docstring, not the literal "
            f"historical fixed-reg run cited as {HISTORICAL_FIXED_REG_STEPS} steps."
        ),
        "per_seed": per_seed,
        "median_reduction_ratio": median_ratio,
        "required_ratio": STEPS_TO_REFERENCE_FACTOR,
        "falsified": falsified,
    }


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
    falsified = None if none_mean is None or log_reg_mean is None else log_reg_mean >= none_mean
    return {
        "none_mean_frac_high_lambda": none_mean,
        "log_reg_mean_frac_high_lambda": log_reg_mean,
        "falsified": falsified,
    }


def score_f3(rows: list[dict]) -> dict:
    """F3 — fraction of steps at lambda > 1e4, log_reg vs none, on either
    wall cell (falsified if EITHER cell/tier fails). ``delaunay_adapt_split``
    is scored PER log_det_method tier (never pooled across cholesky/slogdet
    -- same "refuse to mix tiers" rule ``_group`` enforces for F1/F2)."""
    per_cell: dict = {}
    falsified = False

    delaunay_tiers = _tiers_for_cell(rows, "delaunay_adapt_split")
    per_tier = {}
    for tier in delaunay_tiers:
        tier_rows = _group(rows, cell="delaunay_adapt_split", log_det_method=tier)
        result = score_f3_group(tier_rows)
        per_tier[tier if tier is not None else "auto"] = result
        if result["falsified"]:
            falsified = True
    per_cell["delaunay_adapt_split"] = {"per_log_det_method": per_tier}

    knn_rows = _group(rows, cell="knn")
    knn_result = score_f3_group(knn_rows) if knn_rows else None
    per_cell["knn"] = knn_result
    if knn_result and knn_result["falsified"]:
        falsified = True

    return {"per_cell": per_cell, "falsified": falsified}


def score_f4(mge_rows: list[dict], knn_rows: list[dict]) -> dict:
    """F4 — MGE control byte-identity (none vs log_reg) and the knn `logit`
    pinned-lane pathology."""
    none_rows = {r["seed"]: r for r in _by_bijector(mge_rows, "none")}
    log_reg_rows = {r["seed"]: r for r in _by_bijector(mge_rows, "log_reg")}

    def _identical(a: dict, b: dict) -> bool | None:
        pa, pb = a["final_params_per_lane"], b["final_params_per_lane"]
        if pa is None or pb is None:
            return None
        return pa == pb and a["lane_best_params_per_lane"] == b["lane_best_params_per_lane"]

    per_seed_identity = {}
    for seed in sorted(set(none_rows) & set(log_reg_rows)):
        per_seed_identity[seed] = _identical(none_rows[seed], log_reg_rows[seed])
    mge_differs = any(v is False for v in per_seed_identity.values())
    mge_checked = any(v is not None for v in per_seed_identity.values())

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

    falsified = bool(mge_differs) or bool(logit_pathology)
    return {
        "mge_per_seed_identical": per_seed_identity,
        "mge_checked": mge_checked,
        "mge_differs": mge_differs,
        "knn_logit_pinned_final_counts": logit_pinned,
        "knn_logit_pathology_suspected": logit_pathology,
        "falsified": falsified,
    }


def score_f5(rows: list[dict]) -> dict:
    """F5 — fom at the shared initial broad-start draw (step-0 global-best
    fom) must match between arms at a matched seed within 1e-9 relative. A
    trip here HALTS the verdict (bug, not a science finding)."""
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
    return {"problems": problems, "falsified": bool(problems), "halts": bool(problems)}


def score_rows(rows: list[dict]) -> dict:
    """Run F1-F5 against a set of already-loaded rows. Pure function -- takes
    no filesystem/network dependency, so it is exercised directly by
    ``test_searches_bijector.py`` on synthetic rows.
    """
    f5 = score_f5(rows)
    if f5["halts"]:
        return {
            "f5": f5,
            "halted": True,
            "verdict": "HALT — F5 tripped (bijector changed the physical objective; this is a bug)",
        }

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
    f1 = (
        {
            "per_log_det_method": f1_per_tier,
            "falsified": any(bool(v["falsified"]) for v in f1_per_tier.values()),
        }
        if f1_per_tier
        else {"falsified": None, "note": "no delaunay_adapt_split rows"}
    )
    f2 = score_f2(knn_rows) if knn_rows else {"falsified": None, "note": "no knn rows"}
    f3 = score_f3(rows)
    f4 = score_f4(mge_rows, knn_rows) if mge_rows else {"falsified": None, "note": "no mge rows"}

    falsified_count = sum(bool(c.get("falsified")) for c in (f1, f2, f3, f4))
    falsified = falsified_count >= 2
    return {
        "f1_nan_wall_position": f1,
        "f2_steps_to_reference": f2,
        "f3_time_at_high_lambda": f3,
        "f4_mge_control_and_logit_pathology": f4,
        "f5_physical_point_equality": f5,
        "falsified_criteria_count": falsified_count,
        "halted": False,
        "falsified": falsified,
        "verdict": "FALSIFIED" if falsified else "NOT FALSIFIED",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("run", "score"), default=None)
    p.add_argument("--score", action="store_true", help="shorthand for --stage score")
    p.add_argument("--smoke", action="store_true", help="tiny arm table / two-row scorer smoke")
    p.add_argument("--dry-run", action="store_true", help="print run commands without executing")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    stage = args.stage or ("score" if args.score else None)
    if stage is None:
        raise SystemExit("pass --stage run|score (or --score as a shorthand for --stage score)")

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
        "n_rows": len(rows),
        "historical_reference": {
            "free_reg_steps": HISTORICAL_FREE_REG_STEPS,
            "fixed_reg_steps": HISTORICAL_FIXED_REG_STEPS,
            "source": "results/notes/inference/PROGRAMME.md:579",
        },
        "verdict": verdict,
        "pass_criteria_preregistered": [
            "F1: median first-NaN step under log_reg not later than none, "
            "OR value-NaN lane-steps fall < 50%",
            "F2: steps-to-reference not reduced >= 2x at matched seeds",
            "F3: log_reg lanes spend >= same fraction of steps at lambda > 1e4",
            "F4: MGE control differs by any bit, OR knn logit arm reproduces "
            "the pinned-lane pathology",
            "F5: fom at matched physical points differs > 1e-9 relative -> bug, halts",
        ],
    }

    dest_dir = results_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else f"_{hardware_label()}"
    dest = dest_dir / f"verdict{suffix}.json"
    dest.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nwritten: {dest}")

    rows_npz = dest_dir / f"rows{suffix}.npz"
    np.savez(
        rows_npz,
        rows_json=json.dumps(rows),
    )
    print(f"written: {rows_npz}")

    _print_verdict(artifact)


def _print_verdict(artifact: dict) -> None:
    v = artifact["verdict"]
    print("\n" + "=" * 78)
    print("PHASE 8B — bijector A/B")
    print("=" * 78)
    print(f"rows: {artifact['n_rows']}  hardware: {artifact['hardware']}")
    if v.get("halted"):
        print(f"\n{v['verdict']}")
        for problem in v["f5"]["problems"]:
            print(f"  {problem}")
        return
    for key in (
        "f1_nan_wall_position",
        "f2_steps_to_reference",
        "f3_time_at_high_lambda",
        "f4_mge_control_and_logit_pathology",
    ):
        block = v[key]
        print(f"  {key}: falsified={block.get('falsified')}")
    print(f"\nfalsified criteria: {v['falsified_criteria_count']} / 4")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
