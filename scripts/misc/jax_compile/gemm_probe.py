"""
XLA GPU autotuning / Triton-GEMM probe: one dense fp64 GEMM, one arm per run.

Research instrument for the 2026-09-08 A/B that resolved why the 2026-09-05
autotune pair in ``results/notes/delaunay_nn_breakdown.md`` (Curvature matrix F
at 25.63 ms with ``--xla_gpu_autotune_level=0`` vs 4.90 ms at level 4) appeared
to be contradicted by four later level-0 runs that measured F at ~4.8 ms.
The finding, the measured table and the cure are written up in
``results/notes/xla_autotune_triton_gemm.md``.

The probe strips the likelihood away and times only the shape that moved:

    F = (M * w[:, None]).T @ (M * w[:, None])

with ``M`` a (15361, 1560) float64 array from ``numpy.random.default_rng(0)`` —
the masked-pixel x mesh-vertex mapping-matrix shape of the HST / Delaunay-1500
imaging cell. It compiles that under ``jax.jit`` via the AOT API (``lower`` then
``compile``), times the first call and ten steady-state calls, and prints
``F[0, 0]`` at full float64 precision so the numerics of the two backends can be
compared.

**Arms.** The script itself takes a single positional argument, the arm label,
and reads everything else from the environment — the caller (see
``hpc/batch_gpu/submit_xla_autotune_gemm_probe``) sets ``XLA_FLAGS`` and
``JAX_COMPILATION_CACHE_DIR`` per arm. What matters is the pairing of the two:

    A  level 0, fresh cache          -- the shipped default on a virgin node
    B  level 4, fresh cache          -- XLA's own default, writes autotune entries
    C  level 0, B's cache            -- level 0 reading an autotune cache B seeded
    D  level 0, A's cache            -- level 0 re-reading its own (empty) cache
    E  level 0 + triton_gemm=false   -- the cure, no autotuning needed

Arms C and D are what make this a controlled experiment: JAX persists XLA's
per-fusion autotune cache under ``JAX_COMPILATION_CACHE_DIR`` (see
``jax_persistent_cache_enable_xla_caches``, printed below), keyed by fusion
fingerprint rather than by the autotune flag, so a level-0 run on a node whose
cache a level-4 run already seeded silently gets the autotuned kernel. A valid
autotune A/B therefore needs a **fresh cache directory per arm**, or a node no
level-4 run has ever touched.

**Reading the output.** Each run ends with one machine-greppable line:

    RESULT arm=<arm> level=<autotune level parsed from XLA_FLAGS> \
        compile_s=... first_ms=... steady_mean_ms=... steady_min_ms=... \
        cache_dir=...

``compile_s`` is the tell for what happened: ~2 s means the autotuner actually
ran, ~0.4 s means a real compile without it, and ~0.07 s (or less) means a cache
hit. ``steady_mean_ms`` is the number the note tabulates. Set
``--xla_dump_to=<dir> --xla_dump_hlo_as_text`` in ``XLA_FLAGS`` to have the
submit script's HLO pass report whether the dot lowered to a Triton
``__triton_nested_gemm_fusion`` or to the ``__cublas$lt$matmul`` custom call.
"""

import os
import socket
import subprocess
import sys
import time

os.environ["JAX_ENABLE_X64"] = "True"

ARM = sys.argv[1] if len(sys.argv) > 1 else "?"

print(f"=== ARM {ARM} ===", flush=True)
print("hostname:", socket.gethostname(), flush=True)
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"), flush=True)
cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
print("JAX_COMPILATION_CACHE_DIR:", cache_dir, flush=True)
try:
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv"],
        capture_output=True, text=True, timeout=120).stdout, flush=True)
except Exception as e:
    print("nvidia-smi failed:", e, flush=True)

import numpy as np
import jax
import jax.numpy as jnp

print("jax.__version__:", jax.__version__, flush=True)

# Enable the persistent compilation cache the way autonerves does.
if cache_dir:
    jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
print("jax_persistent_cache_enable_xla_caches:",
      jax.config.jax_persistent_cache_enable_xla_caches, flush=True)
print("jax.devices():", jax.devices(), flush=True)

rng = np.random.default_rng(0)
M_np = rng.standard_normal((15361, 1560)).astype(np.float64)
w_np = (1.0 / (1.0 + rng.random(15361))).astype(np.float64)
M = jnp.asarray(M_np)
w = jnp.asarray(w_np)
jax.block_until_ready(M)
jax.block_until_ready(w)
print("dtypes:", M.dtype, w.dtype, "shapes:", M.shape, w.shape, flush=True)


def f(M, w):
    A = M * w[:, None]
    return jnp.dot(A.T, A)


jitted = jax.jit(f)

t0 = time.perf_counter()
lowered = jitted.lower(M, w)
t1 = time.perf_counter()
compiled = lowered.compile()
t2 = time.perf_counter()
lower_s = t1 - t0
compile_s = t2 - t1
print(f"lower_s={lower_s:.4f} compile_s={compile_s:.4f}", flush=True)

t3 = time.perf_counter()
out = compiled(M, w)
out.block_until_ready()
t4 = time.perf_counter()
first_ms = (t4 - t3) * 1e3
print(f"first_ms={first_ms:.4f}", flush=True)

times = []
for _ in range(10):
    ta = time.perf_counter()
    o = compiled(M, w)
    o.block_until_ready()
    tb = time.perf_counter()
    times.append((tb - ta) * 1e3)
steady_mean = float(np.mean(times))
steady_min = float(np.min(times))
print("per-call ms:", ["%.3f" % t for t in times], flush=True)

val = np.float64(np.asarray(compiled(M, w))[0, 0])
print("F[0,0] = %.17g" % val, flush=True)

level = "?"
for tok in (os.environ.get("XLA_FLAGS") or "").split():
    if tok.startswith("--xla_gpu_autotune_level="):
        level = tok.split("=", 1)[1]
print(
    f"RESULT arm={ARM} level={level} compile_s={compile_s:.4f} "
    f"first_ms={first_ms:.4f} steady_mean_ms={steady_mean:.4f} "
    f"steady_min_ms={steady_min:.4f} cache_dir={cache_dir}",
    flush=True,
)
