"""Calibrate and independently evaluate a profiling-only CPU memo precheck (#280).

All headline timings include the entire Analysis likelihood and policy costs.
Calibration is persisted and locked before holdouts are prepared or evaluated.
Existing phase-5 preparation and production solvers remain unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]
from _production_config import pin_thread_env  # noqa: E402

THREAD_ENV = pin_thread_env(1)
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
from likelihood_breakdown import fixed_light_numba_draws_steps as replay  # noqa: E402
from likelihood_breakdown import fixed_light_numba_memo_policy_steps as policy  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "phase5", ROOT / "scripts/imaging/likelihood_breakdown/fixed_light_numba_draws.py"
)
phase5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase5)
LANES = ("cold", "memo", "guarded")
SCHEDULE = [
    ("cold", "memo", "guarded"),
    ("memo", "guarded", "cold"),
    ("guarded", "cold", "memo"),
    ("guarded", "memo", "cold"),
    ("memo", "cold", "guarded"),
    ("cold", "guarded", "memo"),
]


def write_json(path, payload):
    # Arrays remain on one line: preserve exact numbers without enormous diffs.
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")


@contextlib.contextmanager
def lane_scope(lane, threshold, diagnostics=False):
    with replay.memo_scope(lane != "cold"):
        if lane == "guarded":
            with policy.policy_scope(threshold, diagnostics=diagnostics) as state:
                yield state
        else:
            yield None


def matched_counterfactual(state):
    """Solver-only paired diagnostic from the exact pre-decision memo snapshot.

    Called after the actual fit and outside its observer. These measurements
    cannot feed the eligibility decision or calibration threshold selection.
    """
    from autoarray.inversion.inversion import inversion_util

    if not state.had_memo_entry:
        return {"eligible": False}
    samples = {"cold": [], "memo": []}
    reconstructions = {}
    solvers = {}
    for repeat in range(4):
        for enabled in (False, True) if repeat % 2 == 0 else (True, False):
            lane = "memo" if enabled else "cold"
            # Observe separately from clean timings to avoid counting logging costs.
            with state.counterfactual_scope(enabled):
                start = time.perf_counter_ns()
                x = inversion_util.reconstruction_positive_only_from(
                    *state.last_solve_args, **state.last_solve_kwargs
                )
                elapsed = (time.perf_counter_ns() - start) / 1e6
                samples[lane].append(elapsed)
                reconstructions[lane] = np.asarray(x).copy()
    for enabled in (False, True):
        lane = "memo" if enabled else "cold"
        with state.counterfactual_scope(enabled), replay.observe_solver() as calls:
            inversion_util.reconstruction_positive_only_from(
                *state.last_solve_args, **state.last_solve_kwargs
            )
            solvers[lane] = calls[-1]
    ratio = float(np.median(samples["memo"]) / np.median(samples["cold"]))
    active_equal = sorted(solvers["memo"]["passive_set"]) == sorted(solvers["cold"]["passive_set"])
    delta = np.max(np.abs(reconstructions["memo"] - reconstructions["cold"]))
    scale = max(float(np.max(np.abs(reconstructions["cold"]))), 1e-300)
    return {
        "eligible": True,
        "scope": "production solver only; identical pre-decision system and memo; excludes precheck cost",
        "samples_ms": samples,
        "memo_over_cold": ratio,
        "warm_beneficial_3pct": ratio < 0.97,
        "warm_harmful_3pct": ratio > 1.03,
        "active_set_equal": active_equal,
        "reconstruction_max_relative_difference": float(delta / scale),
        "solvers": solvers,
    }


def diagnostics(cells, indices, lane, threshold, counterfactual=False):
    from autoarray.inversion.inversion.imaging_numba.sparse import InversionImagingSparseNumba

    result = {}
    with lane_scope(lane, threshold, diagnostics=True) as state:
        for index in indices:
            analysis, instance = cells[index]
            with replay.observe_solver() as calls:
                fit = analysis.fit_from(instance=instance)
                evidence = float(fit.figure_of_merit)
                if not isinstance(fit.inversion, InversionImagingSparseNumba):
                    raise AssertionError("Expected production CPU sparse inversion")
                if len(calls) != 1:
                    raise AssertionError("Expected one source-only solve")
                result[index] = {
                    "evidence": evidence,
                    "reconstruction": np.asarray(fit.inversion.reconstruction).copy(),
                    "passive_set": calls[-1]["passive_set"],
                    "solver": calls[-1],
                    "precheck": dict(state.records[-1]) if state else None,
                }
            if counterfactual and state:
                result[index]["matched_counterfactual"] = matched_counterfactual(state)
    return result


def timed(cells, indices, lane, threshold):
    samples = []
    with lane_scope(lane, threshold):
        for index in indices:
            analysis, instance = cells[index]
            start = time.perf_counter_ns()
            evidence = float(analysis.log_likelihood_function(instance=instance))
            elapsed = (time.perf_counter_ns() - start) / 1e6
            samples.append({"index": index, "evidence": evidence, "elapsed_ms": elapsed})
    return samples


def compile_cells(cells):
    # Discard all warm-up state. Also primes every Analysis dataset property.
    timed(cells, list(range(len(cells))), "cold", 0.0)


def evaluate(cells, indices, threshold, repeats, counterfactual=False):
    observed = {
        lane: diagnostics(cells, indices, lane, threshold, counterfactual) for lane in LANES
    }
    comparisons = {
        lane: {
            str(i): replay.compare_solutions(observed[lane][i], observed["cold"][i])
            for i in indices
        }
        for lane in ("memo", "guarded")
    }
    samples = {lane: [] for lane in LANES}
    schedules = []
    for repeat in range(repeats):
        order = SCHEDULE[repeat % len(SCHEDULE)]
        schedules.append(order)
        for lane in order:
            phase5.log(f"timing repeat={repeat + 1} lane={lane} threshold={threshold:g}")
            samples[lane].append(timed(cells, indices, lane, threshold))
    observer_agreement = all(
        np.isfinite(row["evidence"])
        and abs(row["evidence"] - observed[lane][row["index"]]["evidence"])
        <= 1e-9 * max(abs(observed[lane][row["index"]]["evidence"]), 1e-300)
        for lane in LANES
        for traversal in samples[lane]
        for row in traversal
    )
    totals = {lane: [sum(r["elapsed_ms"] for r in run) for run in samples[lane]] for lane in LANES}
    # Reconstruction arrays are used above, then differences retained for compact artifacts.
    for lane in LANES:
        for row in observed[lane].values():
            row.pop("reconstruction")
    summaries = {lane: phase5.summarize(totals[lane]) for lane in LANES}
    counts = {}
    for lane in LANES:
        solvers = [observed[lane][i]["solver"] for i in indices]
        counts[lane] = {
            "memo_hits": sum(s["seed_source"] == "memo" for s in solvers),
            "postguard_invalidations": sum(s["warm_start_fallback"] for s in solvers),
            "exception_retries": sum(len(s["kernel_attempts"]) > 1 for s in solvers),
            "outer_iterations": sum(s["outer_iterations"] for s in solvers),
            "inner_iterations": sum(s["inner_iterations"] for s in solvers),
        }
    checks = [observed["guarded"][i]["precheck"] for i in indices]
    counts["guarded"]["precheck_accepts"] = sum(c["accepted"] for c in checks)
    counts["guarded"]["precheck_rejects"] = sum(
        not c["accepted"] and c["reason"] != "missing_entry" for c in checks
    )
    counts["guarded"]["missing_entries"] = sum(c["reason"] == "missing_entry" for c in checks)
    counts["guarded"]["precheck_total_ms_diagnostic"] = sum(c["precheck_ms"] for c in checks)
    per_draw = {
        str(i): {
            lane: phase5.summarize(
                [row["elapsed_ms"] for run in samples[lane] for row in run if row["index"] == i]
            )
            for lane in LANES
        }
        for i in indices
    }
    matched = [
        observed["guarded"][i]
        for i in indices
        if observed["guarded"][i].get("matched_counterfactual", {}).get("eligible")
    ]
    matched_decisions = {
        "scope": "solver-only matched pre-decision system and memo; four paired repeats; 3% neutral band; excludes eligibility overhead",
        "eligible_transitions": len(matched),
        "accepted": sum(r["precheck"]["accepted"] for r in matched),
        "rejected": sum(not r["precheck"]["accepted"] for r in matched),
        "false_accepts": sum(
            r["precheck"]["accepted"] and r["matched_counterfactual"]["warm_harmful_3pct"]
            for r in matched
        ),
        "false_rejects": sum(
            not r["precheck"]["accepted"] and r["matched_counterfactual"]["warm_beneficial_3pct"]
            for r in matched
        ),
        "active_sets_equal": all(r["matched_counterfactual"]["active_set_equal"] for r in matched),
    }
    descriptive_errors = {
        "qualification": "Cross-lane descriptive timing classifications, not causal false-decision rates: lanes have different memo histories; no counterfactual oracle was used.",
        "accepted_but_guarded_over_3pct_slower_than_cold": [
            i
            for i in indices
            if observed["guarded"][i]["precheck"]["accepted"]
            and per_draw[str(i)]["guarded"]["median_ms"]
            > 1.03 * per_draw[str(i)]["cold"]["median_ms"]
        ],
        "rejected_but_unchanged_memo_over_3pct_faster_than_cold": [
            i
            for i in indices
            if observed["guarded"][i]["precheck"]["reason"] == "score_above_threshold"
            and per_draw[str(i)]["memo"]["median_ms"] < 0.97 * per_draw[str(i)]["cold"]["median_ms"]
        ],
    }
    return {
        "indices": indices,
        "counts": counts,
        "matched_decisions": matched_decisions,
        "per_draw": per_draw,
        "descriptive_decision_errors": descriptive_errors,
        "threshold": threshold,
        "samples": samples,
        "diagnostics": observed,
        "comparisons": comparisons,
        "numerical_passed": bool(
            observer_agreement
            and matched_decisions["active_sets_equal"]
            and all(c["passed"] for rows in comparisons.values() for c in rows.values())
        ),
        "observer_agreement": bool(observer_agreement),
        "lane_schedule": schedules,
        "sequence_totals_ms": totals,
        "sequence_summary": summaries,
        "guarded_over_cold": summaries["guarded"]["median_ms"] / summaries["cold"]["median_ms"],
        "guarded_over_memo": summaries["guarded"]["median_ms"] / summaries["memo"]["median_ms"],
    }


def calibration(args, stem):
    declaration = dict(policy.CALIBRATION)
    declaration["selection_scope"] = "full Analysis likelihood timings; calibration only"
    declaration["clipping"] = (
        "none; cumulative nearby Gaussian increments in original prior-sigma units"
    )
    declaration["holdout_seed"] = 1
    declaration["lane_schedule"] = SCHEDULE
    declaration["threshold_order"] = "ascending declared grid"
    declaration["matched_diagnostics"] = (
        "holdout only; four alternating cold/memo solver-only pairs from identical pre-decision memo; 3% neutral band"
    )
    declaration["smoke"] = args.smoke
    write_json(stem.with_suffix(".declaration.json"), declaration)
    # This declaration exists on disk before preparing or timing calibration systems.
    groups = {}
    candidate_rows = []
    thresholds = declaration["thresholds"]
    for group, nearby in (("nearby", True), ("broad", False)):
        n = 2 if args.smoke else declaration[f"{group}_n"]
        rows = policy.build_rows(declaration["seed"], n, nearby=nearby)
        cells, metadata, promotion = phase5.prepare(rows, args.source_pixels)
        compile_cells(cells)
        runs = []
        for threshold in thresholds:
            phase5.log(f"calibration {group}: threshold={threshold:g}")
            run = evaluate(
                cells, list(range(n)), threshold, 1 if args.smoke else declaration["repeats"]
            )
            run["numerical_passed"] &= promotion["passed"]
            runs.append(run)
            write_json(
                stem.with_suffix(f".calibration_{group}.json"), {"draws": metadata, "runs": runs}
            )
        groups[group] = {"draws": metadata, "runs": runs, "slope_promotion": promotion}
        del cells
        gc.collect()
    for i, threshold in enumerate(thresholds):
        broad, nearby = groups["broad"]["runs"][i], groups["nearby"]["runs"][i]
        candidate_rows.append(
            {
                "threshold": threshold,
                "numerical_passed": broad["numerical_passed"] and nearby["numerical_passed"],
                "broad_ratio": broad["guarded_over_cold"],
                "nearby_ratio": nearby["guarded_over_memo"],
            }
        )
    threshold = policy.select_threshold(candidate_rows)
    locked = {
        "threshold": threshold,
        "candidates": candidate_rows,
        "declaration": declaration,
        "locked_before_holdout": True,
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(stem.with_suffix(".locked.json"), locked)
    phase5.log(f"LOCKED threshold={threshold:g}; holdouts have not been prepared")
    return locked, groups


def plot(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    names = list(result["evaluations"])
    for j, lane in enumerate(LANES):
        means = [
            result["evaluations"][name]["sequence_summary"][lane]["median_ms"]
            / len(result["evaluations"][name]["indices"])
            for name in names
        ]
        x = np.arange(len(names)) + (j - 1) * 0.24
        ax.bar(x, means, width=0.23, label=lane)
        for xi, name in zip(x, names):
            run = result["evaluations"][name]
            ax.plot(
                [xi] * len(run["sequence_totals_ms"][lane]),
                np.asarray(run["sequence_totals_ms"][lane]) / len(run["indices"]),
                "k.",
                ms=3,
            )
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("Whole likelihood ms / model (sequence median and repeats)")
    ax.set_title(
        f"CPU pre-solve memo check: {result['verdict']}; threshold={result['locked']['threshold']:g}"
    )
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pixels", type=int, default=1500)
    parser.add_argument("--n-repeats", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/breakdown/imaging")
    parser.add_argument("--config-name", default="local_cpu_fp64_fixed_light_numba_s5b")
    args = parser.parse_args(argv)
    if args.source_pixels < 10 or args.n_repeats < 1:
        parser.error("source pixels >=10 and repeats >=1 required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"fixed_light_numba_memo_policy_delaunay_{args.config_name}"
    started = time.perf_counter()
    locked, calibration_runs = calibration(args, stem)
    threshold = locked["threshold"]
    lock_hash = hashlib.sha256(stem.with_suffix(".locked.json").read_bytes()).hexdigest()
    import autoarray
    import autofit
    import autogalaxy
    import autolens
    import numba
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    for pool in pools:
        if "filepath" in pool:
            pool["filepath"] = Path(pool["filepath"]).name
    original, manifest = replay.load_manifest(ROOT / phase5.MANIFEST)
    result = {
        "study": "fixed-light-numba-s5b",
        "locked": locked,
        "calibration": calibration_runs,
        "configuration": {
            "source_pixels": args.source_pixels,
            "repeats": args.n_repeats,
            "thread_env": THREAD_ENV,
            "numba_threads": numba.get_num_threads(),
            "threadpools": pools,
            "blas_observation": "available" if pools else "unavailable",
            "device": "cpu",
            "precision": "fp64",
            "formalism": "sparse_numba",
            "timing_scope": "whole Analysis likelihood including policy matvec and memo maintenance",
            "memo_reset": "empty at each traversal boundary, warm-up discarded",
            "lens_light": "per-model S0 solve and subtraction outside timing",
        },
        "manifest": manifest,
        "machine": platform.node(),
        "load_average_before": list(os.getloadavg()),
        "provenance": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "repo_revision": phase5.revision(ROOT),
            "locked_sha256": lock_hash,
            "libraries": {
                m.__name__: {
                    "version": m.__version__,
                    "revision": phase5.revision(Path(m.__file__).parent.parent),
                }
                for m in (autoarray, autofit, autogalaxy, autolens)
            },
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(policy.__file__),
                    Path(replay.__file__),
                    Path(phase5.__file__),
                )
            },
        },
        "draws": {},
        "evaluations": {},
        "slope_promotion": {},
    }
    groups = {
        "original": original[:2] if args.smoke else original,
        "nearby_holdout": policy.build_rows(1, 2 if args.smoke else 32, nearby=True),
        "broad_holdout": policy.build_rows(1, 2 if args.smoke else 24, nearby=False),
    }
    for group, rows in groups.items():
        cells, metadata, promotion = phase5.prepare(rows, args.source_pixels)
        result["draws"][group] = metadata
        result["slope_promotion"][group] = promotion
        compile_cells(cells)
        orders = (
            {
                "graded": list(range(len(rows))),
                "permuted": np.random.default_rng(278).permutation(len(rows)).tolist(),
            }
            if group == "original"
            else {group: list(range(len(rows)))}
        )
        for name, indices in orders.items():
            phase5.log(f"holdout {name}: locked threshold={threshold:g}")
            result["evaluations"][name] = evaluate(
                cells, indices, threshold, args.n_repeats, counterfactual=True
            )
            write_json(stem.with_suffix(".partial.json"), result)
        del cells
        gc.collect()
    runs = result["evaluations"]
    result["gates"] = {
        "numerical": all(r["numerical_passed"] for r in runs.values()),
        "slope_promotion": all(p["passed"] for p in result["slope_promotion"].values()),
        "lock_unchanged": lock_hash
        == hashlib.sha256(stem.with_suffix(".locked.json").read_bytes()).hexdigest(),
        "thread_configuration": numba.get_num_threads() == 1
        and all(p["num_threads"] == 1 for p in pools),
        "complete": not args.smoke and args.source_pixels == 1500 and args.n_repeats >= 5,
    }
    result["performance_targets"] = {
        "graded_5pct_vs_memo": runs["graded"]["guarded_over_memo"] <= 0.95,
        "permuted_5pct_vs_memo": runs["permuted"]["guarded_over_memo"] <= 0.95,
        "graded_3pct_vs_cold": runs["graded"]["guarded_over_cold"] <= 1.03,
        "permuted_3pct_vs_cold": runs["permuted"]["guarded_over_cold"] <= 1.03,
        "broad_holdout_3pct_vs_cold": runs["broad_holdout"]["guarded_over_cold"] <= 1.03,
        "nearby_holdout_3pct_vs_memo": runs["nearby_holdout"]["guarded_over_memo"] <= 1.03,
    }
    result["verdict"] = (
        ("GO" if all(result["performance_targets"].values()) else "NO_LEVER")
        if all(result["gates"].values())
        else "INCOMPLETE_OR_FAIL"
    )
    result["wall_seconds"] = time.perf_counter() - started
    result["load_average_after"] = list(os.getloadavg())
    write_json(stem.with_suffix(".json"), result)
    plot(result, stem.with_suffix(".png"))
    phase5.log(f"{result['verdict']}: {stem.name}")
    return 0 if all(v for k, v in result["gates"].items() if k != "complete") else 1


if __name__ == "__main__" and os.environ.get("AUTOLENS_PROFILING_SMOKE") != "1":
    raise SystemExit(main())
