"""
Where does JAX *tracing* time go? (cold-compile research, issue #74)

Tracing (`jax.jit(fn).lower(x)`) is pure Python, recurs every process, and is
the one cost the persistent compilation cache cannot remove (#71). This probe
cProfiles a single `.lower()` call and aggregates cumulative time by library,
so the uncacheable floor can be attributed (autofit model mapping? autoarray
grids? jax machinery?) before anyone tries to reduce it.

Usage (from the ``autolens_profiling/`` root)::

    python jax_compile/trace_profile.py --model-type mge --transform vag
    python jax_compile/trace_profile.py --model-type pixelization --transform jit --top 40

Prints the per-library rollup and the top-N cumulative functions, and appends a
JSON record under ``jax_compile/results/<hardware>/trace_profile.json``.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from jax_compile.probe import build_objective, transformed_fn_and_arg, hardware_label

LIBRARY_KEYS = (
    "autofit",
    "autoconf",
    "autoarray",
    "autogalaxy",
    "autolens",
    "jax",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset-class", default="imaging")
    p.add_argument("--model-type", default="mge")
    p.add_argument("--instrument", default="hst")
    p.add_argument("--transform", default="vag")
    p.add_argument("--n-batch", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def library_of(filename: str) -> str:
    for key in LIBRARY_KEYS:
        if f"/{key}/" in filename:
            return key
    return "other"


def main():
    args = parse_args()

    import jax

    f, x0, ndim = build_objective(args)
    fn, arg = transformed_fn_and_arg(args.transform, f, x0, args.n_batch, args.batch_size)
    jitted = jax.jit(fn)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    jitted.lower(arg)
    profiler.disable()
    trace_s = time.perf_counter() - t0

    stats = pstats.Stats(profiler)

    # Rollup: exclusive (tottime) per library — sums to the trace wall time, so
    # the attribution is exact rather than double-counted cumulative time.
    per_library = defaultdict(float)
    for (filename, _, _), (_, _, tottime, _, _) in stats.stats.items():
        per_library[library_of(filename)] += tottime

    print(f"[trace_profile] {args.model_type} / {args.transform}: trace {trace_s:.2f}s")
    print(f"[trace_profile] exclusive-time rollup (sums to ~trace time):")
    rollup = dict(sorted(per_library.items(), key=lambda kv: -kv[1]))
    for lib, seconds in rollup.items():
        print(f"[trace_profile]   {lib:<10} {seconds:7.2f}s  ({100 * seconds / trace_s:4.1f}%)")

    print(f"[trace_profile] top {args.top} functions by cumulative time:")
    stats.sort_stats("cumulative").print_stats(args.top)

    record = {
        "model_type": args.model_type,
        "transform": args.transform,
        "ndim": ndim,
        "trace_s": round(trace_s, 3),
        "rollup_tottime_s": {k: round(v, 3) for k, v in rollup.items()},
        "jax_version": jax.__version__,
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_dir = _WORKSPACE_ROOT / "jax_compile" / "results" / hardware_label(jax)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trace_profile.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing.append(record)
    out_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"[trace_profile] wrote record -> {out_path}")


if __name__ == "__main__":
    main()
