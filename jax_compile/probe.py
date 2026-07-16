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

Usage (from the ``autolens_profiling/`` root)::

    python jax_compile/probe.py --model-type mge
    python jax_compile/probe.py --model-type pixelization --transforms jit,vag
    python jax_compile/probe.py --model-type mge --cache-dir /tmp/jax_cache  # run twice: cold vs warm

Each run appends one JSON record per transform under
``jax_compile/results/<hardware>/<model_type>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]  # autolens_profiling/
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

N_BATCH_DEFAULT = 16
BATCH_SIZE_DEFAULT = 4
STEADY_CALLS = 3

TRANSFORMS = ("jit", "grad", "vag", "vmap", "vmap_vag", "laxmap_vag", "pyloop_vag")


def parse_args():
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
    return p.parse_args()


def hardware_label(jax) -> str:
    kind = jax.default_backend()
    if kind == "gpu":
        dev = jax.devices()[0]
        return f"local_gpu_{dev.device_kind.replace(' ', '_')}"
    return f"local_{kind}"


def build_objective(args):
    """Dataset/model/analysis via the searches/ builders; returns (f, x0, ndim).

    ``f`` is the physical-parameter negative log posterior, the same objective
    the gradient samplers optimize (searches_minimal/_grad_setup.py pattern).
    """
    import jax.numpy as jnp

    from searches._setup import build_for_cell

    dataset, model, analysis = build_for_cell(
        dataset_class=args.dataset_class,
        model_type=args.model_type,
        instrument=args.instrument,
        use_jax=True,
        use_mixed_precision=args.mixed_precision,
    )

    def f(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        log_l = analysis.log_likelihood_function(instance=instance)
        log_p = jnp.sum(
            jnp.asarray(model.log_prior_list_from_vector(vector=params, xp=jnp))
        )
        return -(log_l + log_p)

    x0 = jnp.asarray(model.vector_from_unit_vector([0.5] * model.prior_count))
    return f, x0, model.prior_count


def transformed_fn_and_arg(name, f, x0, n_batch, batch_size):
    import jax
    import jax.numpy as jnp

    xb = jnp.tile(x0, (n_batch, 1))
    if name == "jit":
        return f, x0
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


def measure(name, f, x0, n_batch, batch_size):
    """AOT-split timings for one transform. Returns a dict of seconds."""
    import jax

    fn, arg = transformed_fn_and_arg(name, f, x0, n_batch, batch_size)
    jitted = jax.jit(fn)

    t0 = time.perf_counter()
    lowered = jitted.lower(arg)
    trace_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0

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
    }


def main():
    args = parse_args()

    if args.cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.cache_dir)
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    import jax

    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    f, x0, ndim = build_objective(args)
    hardware = hardware_label(jax)

    records = []
    for name in args.transforms.split(","):
        name = name.strip()
        print(f"[probe] {args.model_type} / {name} ...", flush=True)
        rec = measure(name, f, x0, args.n_batch, args.batch_size)
        rec.update(
            dataset_class=args.dataset_class,
            model_type=args.model_type,
            instrument=args.instrument,
            ndim=ndim,
            n_batch=args.n_batch,
            batch_size=args.batch_size,
            hardware=hardware,
            jax_version=jax.__version__,
            cache_dir=args.cache_dir or "",
            mixed_precision=args.mixed_precision,
            tag=args.tag,
            hostname=platform.node(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        records.append(rec)
        print(
            f"[probe]   trace {rec['trace_s']}s  compile {rec['compile_s']}s  "
            f"first {rec['first_s']}s  steady {rec['steady_s']}s",
            flush=True,
        )

    out_dir = _WORKSPACE_ROOT / "jax_compile" / "results" / hardware
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_type}.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing.extend(records)
    out_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"[probe] wrote {len(records)} records -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
