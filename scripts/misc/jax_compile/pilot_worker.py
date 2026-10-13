"""Private fresh-process worker for pilot.py; imports science only after setup."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts/misc"))
from jax_compile import current, pilot, probe


class CompileEvents(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def measure(objective, params, *, jax, runtime, clock=time.perf_counter, observe=lambda: None):
    """Keep all phases separate; synchronize every timed execution."""
    jax.block_until_ready(params)
    started = clock()
    lowered = jax.jit(objective).lower(params)
    trace_lower_s = clock() - started
    try:
        target = str(lowered.compiler_ir().operation.attributes["sym_name"].value)
    except (AttributeError, KeyError, TypeError):
        target = None
    events = CompileEvents()
    logger = logging.getLogger("jax._src.compiler")
    old_level = logger.level
    logger.addHandler(events)
    logger.setLevel(logging.DEBUG)
    try:
        started = clock()
        compiled = lowered.compile()
        compile_s = clock() - started
    finally:
        logger.removeHandler(events)
        logger.setLevel(old_level)
    evidence = pilot.cache_evidence(events.messages, target)
    load = [observe()]
    jax.block_until_ready(params)
    started = clock()
    output = jax.block_until_ready(compiled(params))
    first_s = clock() - started
    samples = []
    warmups = 3 if runtime else 0
    for _ in range(warmups):
        output = jax.block_until_ready(compiled(params))
    if runtime:
        for _ in range(10):
            jax.block_until_ready((params, output))
            load.append(observe())
            started = clock()
            output = jax.block_until_ready(compiled(params))
            samples.append(clock() - started)
    return (
        {
            "trace_lower_s": trace_lower_s,
            "compile_s": compile_s,
            "first_s": first_s,
            "steady_samples_s": samples,
            "warmup_calls": warmups,
            "synchronized": True,
        },
        evidence,
        output,
        load,
    )


def load_sample():
    return {"observed_at": pilot.stamp(), "host_load_average": list(os.getloadavg())}


def git_identity(directory):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def host_identity():
    cpuinfo = Path("/proc/cpuinfo")
    models = (
        sorted(
            {
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            }
        )
        if cpuinfo.exists()
        else []
    )
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "cpu_models": models,
        "logical_cpus": os.cpu_count(),
        "ram_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
    }


def configure(request):
    plan, step = request["plan"], request["step"]
    output = Path(plan["output_dir"])
    cache = output / step["cache"]
    if step not in plan["steps"] or output.resolve() != output:
        raise ValueError("invalid worker request")
    if not cache.is_relative_to(output) or cache.resolve() != cache:
        raise ValueError("cache must be isolated inside the run directory")
    if not cache.is_dir() or (step["mode"] != "warm" and any(cache.iterdir())):
        raise ValueError("cold/seed/runtime worker requires an empty cache")
    expected = pilot.worker_environment(cache)
    for key in (
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "XLA_FLAGS",
    ):
        if os.environ.get(key) != expected[key]:
            raise ValueError(f"worker environment mismatch: {key}")
    os.sched_setaffinity(0, {plan["cpu"]})
    if os.sched_getaffinity(0) != {plan["cpu"]}:
        raise ValueError("fixed CPU affinity unavailable")
    return cache


def run(request):
    start = time.perf_counter()
    plan, step = request["plan"], request["step"]
    cache = configure(request)
    import jax
    import numpy as np

    if not jax.config.jax_enable_x64 or any(d.platform != "cpu" for d in jax.devices()):
        raise ValueError("CPU/fp64 required; no fallback")
    # Explicitly control both settings; environment support varies by JAX release.
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    args = probe.parse_args(
        [
            "--objective",
            current.OBJECTIVE,
            "--transforms",
            "jit",
            "--dataset-path",
            plan["dataset_path"],
            "--output",
            str(ROOT / "output" / current.OUTPUT_NAMESPACE / "pilot-unused.json"),
        ]
    )
    # current.build_workload validates but never writes the unused probe output.
    workload = current.build_workload(args, ROOT)
    for leaf in jax.tree_util.tree_leaves(workload.params):
        if np.issubdtype(leaf.dtype, np.floating) and leaf.dtype != np.dtype("float64"):
            raise ValueError("floating input leaf is not fp64")
    metadata = current.provenance(args, ROOT, workload)
    if metadata["workload"]["dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("dataset changed since plan")
    environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("JAX_", "XLA_"))
        or key
        in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "PYTHONHASHSEED",
        )
    }
    packages = sorted(
        {(d.metadata.get("Name", "unknown"), d.version) for d in importlib.metadata.distributions()}
    )
    config_hashes = {
        str(p.relative_to(ROOT)): pilot.digest(p)
        for p in sorted((ROOT / "config").rglob("*"))
        if p.is_file()
    }
    metadata.update(
        host=host_identity(),
        profiling=git_identity(ROOT),
        config_file_sha256=config_hashes,
        installed_packages=packages,
        environment=environment,
        effective_cache_dir=str(jax.config.jax_compilation_cache_dir),
        producer_sha256={
            name: pilot.digest(Path(__file__).with_name(name))
            for name in ("pilot.py", "pilot_worker.py")
        },
    )
    pilot.require_clean_sources(metadata)
    # Paths differ intentionally per cache pair; effective non-path settings must match.
    metadata["comparison_context"] = {
        "workload": metadata["workload"],
        "host": metadata["host"],
        "profiling": metadata["profiling"],
        "config_file_sha256": config_hashes,
        "installed_packages": packages,
        "environment": {k: v for k, v in environment.items() if k != "JAX_COMPILATION_CACHE_DIR"},
        "producer_sha256": metadata["producer_sha256"],
        "devices": metadata["devices"],
        "cpu_affinity": metadata["cpu_affinity"],
        "jax_enable_x64": metadata["jax_enable_x64"],
    }
    setup_s = time.perf_counter() - start
    timings, evidence, output, load = measure(
        workload.objective,
        workload.params,
        jax=jax,
        runtime=step["mode"] == "runtime",
        observe=load_sample,
    )
    # Detect tracked source changes during this worker, not just between workers.
    if (
        git_identity(ROOT) != metadata["profiling"]
        or current.software_state() != metadata["workload"]["software"]
    ):
        raise ValueError("source identities changed during measurement")
    value = float(output)
    if not math.isfinite(value):
        raise ValueError("nonfinite likelihood output")
    timings["setup_s"] = setup_s
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "schema": pilot.SCHEMA,
        "version": 1,
        "objective": pilot.OBJECTIVE,
        "worker_id": step["id"],
        "mode": step["mode"],
        "seed_id": step["seed_id"],
        "pid": os.getpid(),
        "observed_at": pilot.stamp(),
        "plan_sha256": pilot.fingerprint(plan),
        "context_sha256": pilot.fingerprint(metadata["comparison_context"]),
        "outcome": "measured",
        "accepted": False,
        "qualification_blockers": pilot.BLOCKERS
        + ["Resolved solver defaults have not been independently audited"],
        "timings": timings,
        "provenance": metadata,
        "cache_evidence": evidence,
        "load": load,
        "output_value": value,
        "correctness": {"status": "unverified", "independent_reference": None},
        "host_peak_rss": {
            "value": rss,
            "os_unit": "KiB" if sys.platform == "linux" else "bytes",
            "scope": "worker since process start, sampled before JSON write",
        },
        "worker_internal_wall_s": time.perf_counter() - start,
    }


def main():
    request_path = Path(sys.argv[1]).resolve(strict=True)
    request = json.loads(request_path.read_text())
    result = run(request)
    pilot.write_new(request_path.parent / "observation.json", result)


if __name__ == "__main__":
    main()
