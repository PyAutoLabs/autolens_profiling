"""
One hang-matrix cell: run one reproducer N times in a process that imports only jax.

Run with ``python -I`` from a venv holding nothing but the jax version under test, with
``XLA_FLAGS`` set by the parent (:mod:`run_matrix`). It prints machine-readable lines on
stdout, flushed one per event, so the parent can tell a deadlock (no new line, 0 % CPU)
from a slow call:

    HEADER {json}          versions, flags, cores
    COMPILED <seconds>
    CALL <i> <seconds>     one per completed call
    DONE

Cases (no PyAuto import in any of them; issue PyAutoLabs/PyAutoNerves#213):

``fft``     XLA:CPU ``FftThunk`` re-entering its own Eigen pool (PyAutoFit#1530). The graph
            is the one in autolens_workspace_test ``.github/scripts/xla_fft_pool_reentrancy_repro.py``:
            N scatter-fed FFT convolutions at S x S.
``lapack``  Batched (vmapped) Cholesky + cho_solve, the path behind the 0.10.x / 0.11.0
            exclusion (PyAutoHeart#274).
``export``  A serialized ``jax.export`` module and its inputs, written by
            :mod:`export_reproducers` from a real PyAuto likelihood (the 2d deadlock,
            PyAutoPulse ``knowledge/xla_cpu_deadlock_jaxlib_0_11_2.md``).
"""

import argparse
import json
import os
import pickle
import platform
import sys
import time


def _header(case, module):
    import importlib.metadata as md

    import jax

    return {
        "case": case,
        "module": os.path.basename(module) if module else None,
        "jax": jax.__version__,
        "jaxlib": md.version("jaxlib"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "cpu_count": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(),
        "backend": jax.default_backend(),
    }


def _fft_case():
    import jax
    import jax.numpy as jnp

    n = int(os.environ.get("FFT_N", "48"))
    s = int(os.environ.get("FFT_S", "512"))
    m = (s * s) // 2
    vals = jax.random.normal(jax.random.key(0), (n, m), dtype=jnp.float64)
    psfs = jax.random.normal(jax.random.key(1), (n, s, s), dtype=jnp.float64)
    idx = jnp.arange(m)

    def f(vals, psfs):
        total = 0.0
        flat = jnp.zeros((s * s,), dtype=jnp.float64)
        for i in range(n):
            grid = flat.at[idx].set(vals[i]).reshape(s, s)
            total = total + jnp.fft.irfft2(jnp.fft.rfft2(grid) * jnp.fft.rfft2(psfs[i]), s=(s, s)).sum()
        return total

    return f, (vals, psfs)


def _lapack_case():
    import jax
    import jax.numpy as jnp

    batch = int(os.environ.get("LAPACK_BATCH", "64"))
    size = int(os.environ.get("LAPACK_SIZE", "400"))
    a = jax.random.normal(jax.random.key(2), (batch, size, size), dtype=jnp.float64)
    spd = a @ jnp.swapaxes(a, -1, -2) + size * jnp.eye(size)
    rhs = jax.random.normal(jax.random.key(3), (batch, size), dtype=jnp.float64)

    def one(matrix, b):
        chol = jnp.linalg.cholesky(matrix)
        x = jax.scipy.linalg.cho_solve((chol, True), b)
        return b @ x + 2 * jnp.log(jnp.diag(chol)).sum()

    def f(spd, rhs):
        return jax.vmap(one)(spd, rhs).sum()

    return f, (spd, rhs)


def _export_case(path):
    import jax
    import jax.numpy as jnp

    with open(path, "rb") as fh:
        d = pickle.load(fh)
    exp = jax.export.deserialize(d["exp"])
    params = [jnp.asarray(p) for p in d["params"]]
    return exp.call, (params,)


def main():
    p = argparse.ArgumentParser(description="one hang-matrix cell (pure jax)")
    p.add_argument("--case", choices=("fft", "lapack", "export"), required=True)
    p.add_argument("--module", default=None, help="export case: the pickled jax.export module")
    p.add_argument("--calls", type=int, default=300)
    a = p.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    print("HEADER " + json.dumps(_header(a.case, a.module)), flush=True)

    try:
        if a.case == "fft":
            f, args = _fft_case()
        elif a.case == "lapack":
            f, args = _lapack_case()
        else:
            f, args = _export_case(a.module)
    except Exception as exc:  # an unloadable export is a cell outcome, not a crash
        print(f"UNLOADABLE {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        sys.exit(4)

    t0 = time.perf_counter()
    compiled = jax.jit(f).lower(*args).compile()
    print(f"COMPILED {time.perf_counter() - t0:.4f}", flush=True)
    for i in range(a.calls):
        t = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        print(f"CALL {i} {time.perf_counter() - t:.6f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
