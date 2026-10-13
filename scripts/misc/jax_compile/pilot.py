"""Plan or explicitly execute a raw, fixed-parameter HST/MGE CPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
NAMESPACE = "hst_mge_pilot_v1"
OBJECTIVE = "imaging-mge-runtime-v1"
SCHEMA = "profiling-pilot-observation"
BLOCKERS = [
    "Independent scientific correctness reference and reviewed tolerances absent",
    "Background load excluding campaign work is not measured",
    "Frozen host/stack/dataset/configuration and collection approval not verified",
]
CACHE_EVENT = re.compile(
    r"^(Persistent compilation cache hit|PERSISTENT COMPILATION CACHE MISS) "
    r"for '([^']+)' with key '([^']+)'$"
)


def stamp():
    return datetime.now(UTC).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cache_evidence(messages, target):
    """Only compiler events for the exact lowered module are evidence."""
    events = []
    for message in messages:
        match = CACHE_EVENT.fullmatch(message)
        if match and target and match[2] == target:
            events.append({"kind": "hit" if match[1].endswith("hit") else "miss", "key": match[3]})
    # Multiple or conflicting events are ambiguous, not a vote.
    event = events[0] if len(events) == 1 else None
    return {
        "target": target,
        "state": event["kind"] if event else "unknown",
        "key": event["key"] if event else None,
        "messages": messages,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--run", action="store_true", help="execute; default only prints a plan")
    return parser.parse_args(argv)


def make_plan(args, root=ROOT):
    if not 1 <= args.repetitions <= 10:
        raise ValueError("repetitions must be 1..10")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 1800:
        raise ValueError("process timeout must be finite and in (0, 1800] seconds")
    if not hasattr(os, "sched_getaffinity") or args.cpu not in os.sched_getaffinity(0):
        raise ValueError("select one CPU from the current process affinity")
    namespace = (root / "output" / NAMESPACE).absolute()
    output = args.output_dir.absolute()
    if output.resolve() != output or namespace.resolve() != namespace:
        raise ValueError("output path must not traverse symlinks")
    if not output.is_relative_to(namespace) or output == namespace or output.exists():
        raise ValueError(f"output must be a new run directory below output/{NAMESPACE}")
    dataset = args.dataset_path.resolve(strict=True)
    inputs = {name: digest(dataset / name) for name in ("data.fits", "noise_map.fits", "psf.fits")}
    steps = []
    for index in range(args.repetitions):
        # Cold/seed/warm groups are interleaved, with order recorded exactly.
        for mode in ("cold", "seed", "warm"):
            steps.append(
                {
                    "id": f"{index:02d}-{mode}",
                    "mode": mode,
                    "cache": f"cache/{index:02d}-{'cold' if mode == 'cold' else 'pair'}",
                    "seed_id": f"{index:02d}-seed" if mode == "warm" else None,
                }
            )
    steps.append({"id": "runtime", "mode": "runtime", "cache": "cache/runtime", "seed_id": None})
    return {
        "schema": "profiling-pilot-plan",
        "version": 1,
        "objective": OBJECTIVE,
        "dataset_path": str(dataset),
        "dataset_sha256": inputs,
        "output_dir": str(output),
        "cpu": args.cpu,
        "process_timeout_s": args.timeout,
        "max_processes": len(steps),
        "max_worker_wall_s": len(steps) * args.timeout,
        "runtime_warmups": 3,
        "runtime_samples": 10,
        "steps": steps,
        "accepted": False,
        "qualification_blockers": BLOCKERS,
        "resource_note": "One CPU sequentially; scheduler RAM allocation is external, not enforced here",
    }


def worker_environment(cache):
    env = os.environ.copy()
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
        JAX_COMPILATION_CACHE_DIR=str(cache),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false",
    )
    return env


def run_process(command, *, env, timeout, stdout, stderr, cwd=ROOT):
    """Bound a child and kill its process group on timeout/cancellation."""
    with Path(stdout).open("x") as out, Path(stderr).open("x") as err:
        start = time.perf_counter()
        process = subprocess.Popen(
            command, env=env, cwd=cwd, stdout=out, stderr=err, start_new_session=True
        )
        try:
            code = process.wait(timeout=max(0, timeout - (time.perf_counter() - start)))
            return {
                "outcome": "completed" if code == 0 else "failed",
                "returncode": code,
                "worker_wall_s": time.perf_counter() - start,
            }
        except subprocess.TimeoutExpired:
            return {"outcome": "timeout", "worker_wall_s": time.perf_counter() - start}
        finally:
            # Reap the direct child; remove any descendants even if it exited first.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def require_clean_sources(metadata):
    """A dirty flag alone cannot fingerprint differing source contents."""
    project = metadata["profiling"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", project.get("revision") or "")
        or project.get("tracked_dirty") is not False
    ):
        raise ValueError("profiling source requires a known revision and clean tracked tree")
    software = metadata["workload"]["software"]
    for name in ("autonerves", "autofit", "autoarray", "autogalaxy", "autolens"):
        state = software.get(name, {})
        if (
            not re.fullmatch(r"[0-9a-f]{40}", state.get("git_revision") or "")
            or state.get("git_dirty") is not False
        ):
            raise ValueError(f"{name} source requires a known revision and clean tracked tree")


def validate_record(record, step, plan, previous):
    if (
        record.get("schema") != SCHEMA
        or record.get("version") != 1
        or record.get("accepted") is not False
        or record.get("objective") != OBJECTIVE
        or record.get("worker_id") != step["id"]
        or record.get("mode") != step["mode"]
        or record.get("plan_sha256") != fingerprint(plan)
        or record.get("outcome") != "measured"
    ):
        raise ValueError("worker result identity/outcome mismatch")
    identity = record["provenance"]
    require_clean_sources(identity)
    if (
        identity["workload"]["dataset_sha256"] != plan["dataset_sha256"]
        or identity["cpu_affinity"] != [plan["cpu"]]
        or identity["jax_enable_x64"] is not True
        or not identity["devices"]
        or any(d["platform"] != "cpu" for d in identity["devices"])
    ):
        raise ValueError("actual inputs/backend/precision/affinity differ from plan")
    if fingerprint(identity["comparison_context"]) != record["context_sha256"]:
        raise ValueError("comparison context hash mismatch")
    if previous and record["context_sha256"] != previous[0]["context_sha256"]:
        raise ValueError("incompatible actual contexts; refusing to combine workers")
    timings = record["timings"]
    for name in ("setup_s", "trace_lower_s", "compile_s", "first_s"):
        if (
            not isinstance(timings[name], (int, float))
            or not math.isfinite(timings[name])
            or timings[name] < 0
        ):
            raise ValueError(f"invalid duration {name}")
    samples = timings["steady_samples_s"]
    count = plan["runtime_samples"] if step["mode"] == "runtime" else 0
    if len(samples) != count or any(not math.isfinite(v) or v < 0 for v in samples):
        raise ValueError("invalid steady sample count/value")
    expected_warmups = plan["runtime_warmups"] if step["mode"] == "runtime" else 0
    if timings["warmup_calls"] != expected_warmups or timings["synchronized"] is not True:
        raise ValueError("warmup/synchronization protocol mismatch")
    if step["mode"] == "warm":
        seed = next((r for r in previous if r["worker_id"] == step["seed_id"]), None)
        evidence = record["cache_evidence"]
        if (
            seed is None
            or record["seed_id"] != seed["worker_id"]
            or evidence["state"] != "hit"
            or not evidence["key"]
            or evidence["key"] != seed["cache_evidence"]["key"]
            or evidence["target"] != seed["cache_evidence"]["target"]
            or seed["cache_evidence"]["state"] != "miss"
        ):
            raise ValueError("warm cache hit is unknown or not linked to matching seed")
    elif step["mode"] in ("cold", "seed") and record["cache_evidence"]["state"] != "miss":
        raise ValueError("empty-cache target miss not demonstrated")


def summary(records):
    if len({r["context_sha256"] for r in records}) > 1:
        raise ValueError("incompatible observations cannot share a timing summary")
    groups = {}
    for record in records:
        for metric in ("trace_lower_s", "compile_s", "first_s", "steady_samples_s"):
            value = record["timings"][metric]
            samples = value if isinstance(value, list) else [value]
            groups.setdefault(f"{record['mode']}/{metric}", []).extend(samples)
    result = {}
    for name, samples in groups.items():
        if not samples:
            continue
        quartiles = (
            statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else None
        )
        result[name] = {
            "samples_s": samples,
            "n": len(samples),
            "median_s": statistics.median(samples),
            "min_s": min(samples),
            "max_s": max(samples),
            "iqr_s": quartiles[2] - quartiles[0] if quartiles else None,
        }
    return result


def execute(plan, *, runner=run_process):
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.json", plan)
    records, outcomes = [], []
    outcome = "incomplete"
    try:
        for step in plan["steps"]:
            directory = output / step["id"]
            directory.mkdir()
            cache = output / step["cache"]
            if step["mode"] != "warm":
                cache.mkdir(parents=True, exist_ok=False)
            elif not cache.is_dir() or cache.is_symlink():
                raise ValueError("warm cache absent or symlinked")
            request = {"plan": plan, "step": step}
            request_path = directory / "request.json"
            write_new(request_path, request)
            command = [
                sys.executable,
                str(Path(__file__).with_name("pilot_worker.py")),
                str(request_path),
            ]
            status = {"worker_id": step["id"], "outcome": "interrupted"}
            outcomes.append(status)
            try:
                status.update(
                    runner(
                        command,
                        env=worker_environment(cache),
                        timeout=plan["process_timeout_s"],
                        stdout=directory / "stdout.log",
                        stderr=directory / "stderr.log",
                    )
                )
                if status["outcome"] != "completed":
                    break
                path = directory / "observation.json"
                record = json.loads(path.read_text())
                status["artifact"] = str(path.relative_to(output))
                status["sha256"] = digest(path)
                validate_record(record, step, plan, records)
                records.append(record)
            except Exception as exc:
                status.update(outcome="invalid", error=f"{type(exc).__name__}: {exc}")
                break
            finally:
                write_new(directory / "status.json", status)
        else:
            outcome = "recorded"
    finally:
        receipt = {
            "schema": "profiling-pilot-receipt",
            "version": 1,
            "observed_at": stamp(),
            "plan_sha256": fingerprint(plan),
            "outcome": outcome,
            "accepted": False,
            "qualification_blockers": BLOCKERS,
            "workers": outcomes,
            "unstarted_workers": [s["id"] for s in plan["steps"][len(outcomes) :]],
            "summary": summary(records),
        }
        write_new(output / "receipt.json", receipt)
    return receipt


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"pilot interrupted by signal {signum}")


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    previous_handler = signal.signal(signal.SIGTERM, interrupt) if args.run else None
    try:
        result = execute(plan) if args.run else plan
    finally:
        if args.run:
            signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(args.run and result["outcome"] != "recorded")


if __name__ == "__main__":
    raise SystemExit(main())
