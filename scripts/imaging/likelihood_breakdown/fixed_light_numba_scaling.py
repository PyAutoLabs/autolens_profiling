"""Phase 6 CPU source-pixel scaling; each invocation measures one N in a fresh process.

Clean Analysis timings, instrumented Analysis timings and numerical diagnostics
are separate traversals. All start with an empty production NNLS memo. The
seed-601 manifest is written before model preparation. No policy is tuned.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
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
from likelihood_breakdown import fixed_light_numba_memo_policy_steps as draws  # noqa: E402
from likelihood_breakdown import fixed_light_numba_scaling_steps as scaling  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "phase5", ROOT / "scripts/imaging/likelihood_breakdown/fixed_light_numba_draws.py"
)
phase5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase5)
PIXELS = (500, 1000, 1500, 2500, 4000)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


def declaration(n_draws):
    return {
        "seed": 601,
        "requested_pixels": list(PIXELS),
        "draws_per_sequence": n_draws,
        "groups": {
            name: draws.build_rows(601, n_draws, nearby=nearby)
            for name, nearby in (("nearby", True), ("broad", False))
        },
        "nearby_sigma": 0.05,
        "broad_sigma": 5.0,
        "timing": "six counterbalanced fresh-cache traversals per lane/sequence",
        "precision": "fp64",
        "formalism": "sparse_numba",
        "threads": 1,
    }


def timed(cells, enabled, instrumented=False):
    def evaluate(index):
        analysis, instance = cells[index]
        return float(analysis.log_likelihood_function(instance=instance))

    if not instrumented:
        return replay.replay_sequence(evaluate, range(len(cells)), enabled)
    rows = []
    with replay.memo_scope(enabled):
        for index in range(len(cells)):
            with scaling.timing_scope() as timing:
                evidence = evaluate(index)
            rows.append(
                {
                    "index": index,
                    "evidence": evidence,
                    "elapsed_ms": timing.total_ms,
                    "exclusive_ms": dict(timing.exclusive_ms),
                    "inclusive_ms": dict(timing.inclusive_ms),
                }
            )
    return rows


def diagnostics(cells, enabled):
    from autoarray.inversion.inversion.imaging_numba.sparse import InversionImagingSparseNumba

    result = {}
    with replay.memo_scope(enabled), replay.observe_solver() as calls:
        for index, (analysis, instance) in enumerate(cells):
            before = len(calls)
            fit = analysis.fit_from(instance=instance)
            evidence = float(fit.figure_of_merit)
            inversion = fit.inversion
            if not isinstance(inversion, InversionImagingSparseNumba) or len(calls) != before + 1:
                raise AssertionError("Expected one production sparse-numba source-only solve")
            ids = inversion.solve_ids_to_keep
            full_dimension = len(inversion.reconstruction)
            result[index] = {
                "evidence": evidence,
                "reconstruction": np.asarray(inversion.reconstruction).copy(),
                "passive_set": calls[-1]["passive_set"],
                "solver": calls[-1],
                "mapper_dimension": full_dimension,
                "solve_dimension": len(ids) if ids is not None else full_dimension,
                "matrix_bytes": int(inversion.curvature_reg_matrix.nbytes),
            }
    return result


def evaluate_group(cells, repeats):
    observed = {
        lane: diagnostics(cells, enabled) for lane, enabled in (("cold", False), ("memo", True))
    }
    comparisons = {
        str(i): replay.compare_solutions(observed["memo"][i], observed["cold"][i])
        for i in range(len(cells))
    }
    samples = {lane: [] for lane in observed}
    breakdown = {lane: [] for lane in observed}
    # Counterbalance instrumentation order as well as memo lane order. Each
    # traversal remains separate with fresh caches; instrumentation is never
    # installed in a clean timing. This limits long-block thermal/load drift.
    for repeat in range(repeats):
        modes = ((samples, False), (breakdown, True))
        if repeat % 2:
            modes = modes[::-1]
        for target, instrumented in modes:
            for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                lane = "memo" if enabled else "cold"
                phase5.log(f"{'breakdown' if instrumented else 'clean'} repeat={repeat + 1} {lane}")
                target[lane].append(timed(cells, enabled, instrumented))
    agreement = all(
        np.isfinite(row["evidence"])
        and abs(row["evidence"] - observed[lane][row["index"]]["evidence"])
        <= 1e-9 * max(abs(observed[lane][row["index"]]["evidence"]), 1e-300)
        for collection in (samples, breakdown)
        for lane, runs in collection.items()
        for run in runs
        for row in run
    )
    summaries = {}
    for lane in observed:
        totals = [sum(row["elapsed_ms"] for row in run) for run in samples[lane]]
        observed_totals = [sum(row["elapsed_ms"] for row in run) for run in breakdown[lane]]
        exclusive = {
            key: float(
                np.median([sum(row["exclusive_ms"][key] for row in run) for run in breakdown[lane]])
            )
            / len(cells)
            for key in breakdown[lane][0][0]["exclusive_ms"]
        }
        clean_median = float(np.median(totals))
        observed_median = float(np.median(observed_totals))
        summaries[lane] = {
            "sequence_totals_ms": totals,
            "observed_sequence_totals_ms": observed_totals,
            "ms_per_model": clean_median / len(cells),
            "exclusive_ms_per_model": exclusive,
            "observer_overhead_relative": abs(observed_median / clean_median - 1),
            "breakdown_reconciles": abs(observed_median / clean_median - 1) <= 0.05,
            "memo_hits": sum(r["solver"]["seed_source"] == "memo" for r in observed[lane].values()),
            "memo_drops": sum(r["solver"]["warm_start_fallback"] for r in observed[lane].values()),
            "solve_dimensions": [r["solve_dimension"] for r in observed[lane].values()],
        }
        for row in observed[lane].values():
            row.pop("reconstruction")
    return {
        "samples": samples,
        "breakdown_samples": breakdown,
        "diagnostics": observed,
        "comparisons": comparisons,
        "summary": summaries,
        "numerical_passed": all(row["passed"] for row in comparisons.values()) and agreement,
        "observer_agreement": agreement,
        "breakdown_passed": all(row["breakdown_reconciles"] for row in summaries.values()),
    }


def plot(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    names = [f"{group}/{lane}" for group in result["groups"] for lane in ("cold", "memo")]
    values = [
        result["groups"][group]["summary"][lane]
        for group in result["groups"]
        for lane in ("cold", "memo")
    ]
    axes[0].bar(names, [value["ms_per_model"] for value in values])
    axes[0].set_ylabel("Clean whole likelihood ms / model")
    bottom = np.zeros(len(names))
    for key in values[0]["exclusive_ms_per_model"]:
        heights = np.array([value["exclusive_ms_per_model"][key] for value in values])
        axes[1].bar(names, heights, bottom=bottom, label=key)
        bottom += heights
    axes[1].set_ylabel("Exclusive diagnostic ms / model")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle(f"CPU source scaling N={result['requested_pixels']} — {result['status']}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aggregate(directory, config_name):
    """Retain absent/failed sizes and fit only numerically valid measured cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {}
    for n in PIXELS:
        path = directory / f"fixed_light_numba_scaling_delaunay_{config_name}_n{n}.json"
        cells[n] = json.loads(path.read_text()) if path.exists() else None
    report = {"expected_pixels": list(PIXELS), "lanes": {}, "input_sha256": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for group in ("nearby", "broad"):
        for lane in ("cold", "memo"):
            rows = []
            for n, cell in cells.items():
                good = (
                    cell
                    and not cell.get("smoke")
                    and cell.get("gates", {}).get("numerical")
                    and cell.get("gates", {}).get("single_thread")
                    and cell.get("gates", {}).get("complete")
                )
                summary = (
                    cell.get("groups", {}).get(group, {}).get("summary", {}).get(lane, {})
                    if cell
                    else {}
                )
                rows.append(
                    {
                        "source_pixels": n,
                        "total_ms": summary.get("ms_per_model"),
                        "status": "measured" if good else "failed" if cell else "missing",
                    }
                )
            report["lanes"][f"{group}/{lane}"] = scaling.summarize_scaling(rows)
            measured = [r for r in rows if r["status"] == "measured"]
            axes[0].plot(
                [r["source_pixels"] for r in measured],
                [r["total_ms"] for r in measured],
                ".-",
                label=f"{group}/{lane}",
            )
    for n, cell in cells.items():
        if cell:
            path = directory / f"fixed_light_numba_scaling_delaunay_{config_name}_n{n}.json"
            report["input_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["solve_curvature_crossover_brackets"] = {}
    for group in ("nearby", "broad"):
        for lane in ("cold", "memo"):
            brackets = []
            for low, high in zip(PIXELS[:-1], PIXELS[1:]):
                pair = [cells[low], cells[high]]
                if not all(c and all(c.get("gates", {}).values()) for c in pair):
                    continue
                values = [
                    c["groups"][group]["summary"][lane]["exclusive_ms_per_model"] for c in pair
                ]
                differences = [v["solve"] - v["curvature"] for v in values]
                if (
                    differences[0] == 0
                    or differences[1] == 0
                    or differences[0] * differences[1] < 0
                ):
                    brackets.append(
                        {"requested_pixels": [low, high], "solve_minus_curvature_ms": differences}
                    )
            report["solve_curvature_crossover_brackets"][f"{group}/{lane}"] = brackets
    report["crossover_scope"] = (
        "Adjacent measured passing cells only; empty means no observed crossing, not no crossing exists."
    )
    report["memory"] = [
        {
            "requested_pixels": n,
            "peak_rss_kib": cell.get("peak_rss_kib") if cell else None,
            "status": cell.get("status") if cell else "missing",
        }
        for n, cell in cells.items()
    ]
    memory = [r for r in report["memory"] if r["peak_rss_kib"] is not None]
    axes[1].plot(
        [r["requested_pixels"] for r in memory], [r["peak_rss_kib"] / 1024**2 for r in memory], ".-"
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Requested source pixels")
    axes[0].set_ylabel("Whole likelihood ms/model")
    axes[0].legend()
    axes[1].set_ylabel("Process peak RSS GiB (includes preparation)")
    stem = directory / f"fixed_light_numba_scaling_summary_{config_name}"
    write_json(stem.with_suffix(".json"), report)
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pixels", type=int)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--n-repeats", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/breakdown/imaging")
    parser.add_argument("--config-name", default="hpc_ral_cpu_fp64_fixed_light_numba_s6")
    args = parser.parse_args(argv)
    if args.aggregate:
        return aggregate(args.output_dir, args.config_name)
    if args.source_pixels is None or args.source_pixels < 10 or args.n_repeats < 1:
        parser.error("source pixels >=10 and positive repeats required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        args.output_dir
        / f"fixed_light_numba_scaling_delaunay_{args.config_name}_n{args.source_pixels}"
    )
    declared = declaration(2 if args.smoke else 8)
    write_json(stem.with_suffix(".declaration.json"), declared)
    declaration_hash = hashlib.sha256(
        stem.with_suffix(".declaration.json").read_bytes()
    ).hexdigest()
    started = time.perf_counter()
    result = {
        "study": "fixed-light-numba-s6",
        "requested_pixels": args.source_pixels,
        "declaration": declared,
        "declaration_sha256": declaration_hash,
        "repeats": args.n_repeats,
        "smoke": args.smoke,
        "groups": {},
        "load_before": list(os.getloadavg()),
        "status": "RUNNING",
    }
    write_json(stem.with_suffix(".json"), result)
    try:
        for group, rows in declared["groups"].items():
            cells, metadata, promotion = phase5.prepare(rows, args.source_pixels)
            timed(cells, False)  # discarded compilation and lazy dataset warmup
            group_result = evaluate_group(cells, args.n_repeats)
            group_result.update(
                draws=metadata,
                slope_promotion=promotion,
                preparation_inclusive_peak_rss_kib=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            )
            result["groups"][group] = group_result
            write_json(stem.with_suffix(".json"), result)
            del cells
            gc.collect()
    except Exception as error:
        result.update(
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write_json(stem.with_suffix(".json"), result)
        raise
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
    result["configuration"] = {
        "thread_env": THREAD_ENV,
        "numba_threads": numba.get_num_threads(),
        "threadpools": pools,
        "blas_observation": "available" if pools else "unavailable",
        "machine": platform.node(),
        "precision": "fp64",
        "formalism": "sparse_numba",
    }
    result["provenance"] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "repo_revision": phase5.revision(ROOT),
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
                Path(scaling.__file__),
                Path(replay.__file__),
                Path(draws.__file__),
                Path(phase5.__file__),
            )
        },
        "dataset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "dataset/imaging/hst").glob("*.fits"))
        },
    }
    result["gates"] = {
        "numerical": all(
            g["numerical_passed"] and g["slope_promotion"]["passed"]
            for g in result["groups"].values()
        ),
        "breakdown": all(g["breakdown_passed"] for g in result["groups"].values()),
        "single_thread": numba.get_num_threads() == 1 and all(p["num_threads"] == 1 for p in pools),
        "complete": not args.smoke and args.n_repeats >= 6 and args.source_pixels in PIXELS,
    }
    result.update(
        status="PASS" if all(result["gates"].values()) else "INCOMPLETE_OR_FAIL",
        wall_seconds=time.perf_counter() - started,
        load_after=list(os.getloadavg()),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_scope="process high-water RSS including imports, compilation, all preparations and both groups; not solver-only",
    )
    write_json(stem.with_suffix(".json"), result)
    plot(result, stem.with_suffix(".png"))
    phase5.log(f"{result['status']}: {stem.name}")
    return 0 if result["gates"]["numerical"] and result["gates"]["single_thread"] else 1


if __name__ == "__main__" and os.environ.get("AUTOLENS_PROFILING_SMOKE") != "1":
    raise SystemExit(main())
