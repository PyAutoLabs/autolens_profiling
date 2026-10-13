"""
Compile-time probe: where does JAX/XLA compilation time go, per likelihood and
per transform?

Research instrument for https://github.com/PyAutoLabs/autolens_profiling/issues/71
(core question: do we need jit boundaries inside the source to break up
compilation, or do settings / small changes suffice?). The companion feature
task (`PyAutoMind draft/feature/autolens_profiling/jax_compile_time_profiling.md`)
industrializes this across the full cell grid once the research settles the
method.

Unlike ``likelihood_runtime/`` (steady-state per-call cost) this measures the
**one-off costs** separately, via the AOT API:

    traced   = jax.jit(fn).lower(x)      # -> trace_s   (Python tracing)
    compiled = traced.compile()          # -> compile_s (XLA compilation)
    compiled(x) + block_until_ready      # -> first_s   (residual first-call)
    N more calls                         # -> steady_s  (per-call runtime)

Transforms mirror how samplers actually consume the likelihood:

    jit         jax.jit(f)                          -- jit-only samplers (Nautilus row)
    grad        jax.jit(jax.grad(f))
    vag         jax.jit(jax.value_and_grad(f))      -- single-start optimizers
    vmap        jax.jit(jax.vmap(f))                -- batched samplers, n_batch starts
    vmap_vag    jax.jit(jax.vmap(value_and_grad))   -- MultiStartAdam, no batch_size
    laxmap_vag  jax.jit(lax.map(value_and_grad, batch_size=)) -- MultiStartAdam batched
    pyloop_vag  jax.jit(jax.vmap(value_and_grad)) over one batch_size chunk,
                called n_batch/batch_size times from Python -- the batching
                boundary hoisted OUT of XLA (candidate laxmap_vag replacement;
                steady_s reports one full n_batch sweep for comparability)

The legacy flat-vector posterior builder remains unavailable. The explicit
``--objective imaging-mge-runtime-v1 --transforms jit`` path measures the current
HST/MGE fp64 runtime objective with pytree inputs. It requires a new output file
under ``output/compile_probe_runtime_v1/``; see README.md. Its results must never
be compared with historical compile pins.

"""

from __future__ import annotations

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


import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

_WORKSPACE_ROOT = _profiling_root()  # autolens_profiling/
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

N_BATCH_DEFAULT = 16
BATCH_SIZE_DEFAULT = 4
STEADY_CALLS = 3

TRANSFORMS = ("jit", "grad", "vag", "vmap", "vmap_vag", "laxmap_vag", "pyloop_vag")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset-class", default="imaging")
    p.add_argument("--model-type", default="mge", help="mge | pixelization | delaunay")
    p.add_argument("--instrument", default="hst")
    p.add_argument(
        "--transforms",
        default=",".join(TRANSFORMS),
        help=f"comma-separated subset of {TRANSFORMS}",
    )
    p.add_argument("--n-batch", type=int, default=N_BATCH_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument(
        "--cache-dir",
        default=None,
        help="enable the persistent compilation cache at this dir (run twice for cold/warm)",
    )
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--tag", default="", help="free-form tag recorded in the JSON")
    p.add_argument("--objective", help="explicit workload identity (see README.md)")
    p.add_argument("--output", help="new JSON file under output/compile_probe_runtime_v1/")
    p.add_argument("--dataset-path", help="existing HST FITS directory; never auto-simulated")
    return p.parse_args(argv)


def hardware_label(jax) -> str:
    kind = jax.default_backend()
    if kind == "gpu":
        dev = jax.devices()[0]
        return f"local_gpu_{dev.device_kind.replace(' ', '_')}"
    return f"local_{kind}"


def build_objective(args):
    """Retired flat-vector posterior API; never substitute the runtime likelihood."""
    raise NotImplementedError(
        "The legacy flat-vector posterior builder is unavailable. "
        "Use probe.py --objective imaging-mge-runtime-v1 --transforms jit "
        "with an isolated --output for the different current runtime objective. "
        "Legacy export and trace probes do not support this objective."
    )


def transformed_fn_and_arg(name, f, x0, n_batch, batch_size):
    import jax
    import jax.numpy as jnp

    if name == "jit":
        return f, x0
    xb = jnp.tile(x0, (n_batch, 1))
    if name == "grad":
        return jax.grad(f), x0
    if name == "vag":
        return jax.value_and_grad(f), x0
    if name == "vmap":
        return jax.vmap(f), xb
    if name == "vmap_vag":
        return jax.vmap(jax.value_and_grad(f)), xb
    if name == "laxmap_vag":
        vag = jax.value_and_grad(f)
        return (lambda X: jax.lax.map(vag, X, batch_size=batch_size)), xb
    if name == "pyloop_vag":
        return jax.vmap(jax.value_and_grad(f)), jnp.tile(x0, (batch_size, 1))
    raise ValueError(f"unknown transform {name!r}")


def cache_entry_count(cache_dir) -> int:
    """How many entries the persistent compilation cache currently holds.

    Counting files is enough to separate a cache MISS from a HIT: XLA writes one
    entry per compiled module, so a compile that wrote nothing reused what was
    already there.
    """
    if not cache_dir:
        return 0
    root = Path(cache_dir)
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def cache_state_from(before: int, after: int, cache_dir) -> str:
    """cold | warm | none | unknown — derived from what the compile DID.

    Warmness used to be recoverable only from the free-text ``tag``, which
    carries ~40 ad-hoc spellings across the corpus and cannot be parsed. Worse,
    ``cache_dir`` is non-empty on cold runs too — the cold run is the one that
    POPULATES the cache — so the obvious substring check is a trap.

    Derived per transform, not per run: each transform compiles its own module,
    so one invocation can legitimately miss on one and hit on another. A
    per-run tag could not express that at all.
    """
    if not cache_dir:
        return "none"
    if after > before:
        return "cold"  # wrote a new entry => this compile was a miss
    if before > 0:
        return "warm"  # wrote nothing into a populated cache => hit
    return "unknown"  # cache configured, empty, and still nothing written


def host_state() -> dict:
    """Load provenance. XLA compiles on the HOST cores, so this is load-bearing.

    jax_compile/README.md records the first measurements being wrong by up to
    7x (851 s vs 117 s for the same compile) purely from host load, and the
    corpus already mixes a 32-core RAL allocation with laptop rows. Without
    this, "is a compile slower?" is unanswerable after the fact.
    """
    try:
        load1, _, _ = os.getloadavg()
    except (OSError, AttributeError):  # not available on every platform
        load1 = None
    return {"cpu_count": os.cpu_count(), "load_avg_1m": round(load1, 2) if load1 else None}


def measure(name, f, x0, n_batch, batch_size, cache_dir=None):
    """AOT-split timings for one transform. Returns a dict of seconds."""
    import jax

    fn, arg = transformed_fn_and_arg(name, f, x0, n_batch, batch_size)
    jitted = jax.jit(fn)

    t0 = time.perf_counter()
    lowered = jitted.lower(arg)
    trace_s = time.perf_counter() - t0

    cache_before = cache_entry_count(cache_dir)
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    cache_after = cache_entry_count(cache_dir)

    t0 = time.perf_counter()
    jax.block_until_ready(compiled(arg))
    first_s = time.perf_counter() - t0

    # pyloop_vag compiles one batch_size chunk; a full n_batch sweep is
    # n_batch/batch_size sequential calls, so steady_s stays comparable to the
    # all-n_batch transforms (vmap_vag / laxmap_vag).
    calls_per_eval = n_batch // batch_size if name == "pyloop_vag" else 1

    t0 = time.perf_counter()
    for _ in range(STEADY_CALLS):
        for _ in range(calls_per_eval):
            out = compiled(arg)
        jax.block_until_ready(out)
    steady_s = (time.perf_counter() - t0) / STEADY_CALLS

    return {
        "transform": name,
        "trace_s": round(trace_s, 3),
        "compile_s": round(compile_s, 3),
        "first_s": round(first_s, 3),
        "steady_s": round(steady_s, 4),
        "cache_state": cache_state_from(cache_before, cache_after, cache_dir),
    }


def main():
    args = parse_args()
    from jax_compile.current import build_workload, provenance, validate_request, write_result

    out_path = validate_request(args, _WORKSPACE_ROOT)

    if args.cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.cache_dir)
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    import jax

    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    workload = build_workload(args, _WORKSPACE_ROOT)
    f, x0 = workload.objective, workload.params
    metadata = provenance(args, _WORKSPACE_ROOT, workload)
    hardware = hardware_label(jax)

    records = []
    for name in args.transforms.split(","):
        name = name.strip()
        print(f"[probe] {args.model_type} / {name} ...", flush=True)
        rec = measure(name, f, x0, args.n_batch, args.batch_size, cache_dir=args.cache_dir)
        # A populated cache with no new files does not prove a cache hit.
        # Keep the legacy heuristic visible without qualifying a new warm baseline.
        rec["cache_entry_heuristic"] = rec.pop("cache_state")
        rec["cache_state"] = "unknown" if args.cache_dir else "none"
        rec.update(
            record_kind="compile_probe_runtime_v1",
            dataset_class=args.dataset_class,
            model_type=args.model_type,
            instrument=args.instrument,
            model_free_parameters=workload.model.total_free_parameters,
            input_representation="registered-model-instance-pytree",
            objective=args.objective,
            steady_calls=STEADY_CALLS,
            hardware=hardware,
            jax_version=jax.__version__,
            cache_dir=args.cache_dir or "",
            mixed_precision=args.mixed_precision,
            host_state=host_state(),
            tag=args.tag,
            hostname=platform.node(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        records.append(rec)
        print(
            f"[probe]   trace {rec['trace_s']}s  compile {rec['compile_s']}s  "
            f"first {rec['first_s']}s  steady {rec['steady_s']}s  "
            f"cache {rec['cache_state']}",
            flush=True,
        )

    write_result(out_path, records, metadata)
    print(f"[probe] wrote {len(records)} records -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
