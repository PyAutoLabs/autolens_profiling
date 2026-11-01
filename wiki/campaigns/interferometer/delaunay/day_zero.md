<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: interferometer/delaunay

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 65 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): Any interferometer dataset with a Delaunay pixelized source on a many-core CPU node with no GPU uses numba_cpu_sparse_fp64_wtilde, one serial likelihood per core (workers capped by node memory over peak RSS), falling back to jax_cpu_sparse_fp64_wtilde. Untested: CPU streaming and the MGE DFT route.
  Optimal `numba_cpu_sparse_fp64_wtilde`; secondary `jax_cpu_sparse_fp64_wtilde`. Pulse record [`interferometer-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-delaunay-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): Any interferometer dataset with a Delaunay pixelized source on an A100 node uses jax_gpu_sparse_fp64_wtilde, falling back to jax_gpu_dense_fp64_nufft. Untested: the GPU streaming route (measured up to jvla, 2.5e7 visibilities, on W~ sparse).
  Optimal `jax_gpu_sparse_fp64_wtilde`; secondary `jax_gpu_dense_fp64_nufft`. Pulse record [`interferometer-delaunay-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-delaunay-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| alma | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 20.83 ms | 192 | [`lifecycle/8dada4df3a4ed30d1721`](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| alma_high | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 63.71 ms | 62.8 | [`lifecycle/ba12584bdfe8c2e16f30`](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| jvla | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 465.2 ms | 8.6 | [`lifecycle/3c33efbabefe647c901b`](../../../../results/interferometer/delaunay/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| sma | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 11.36 ms | 352 | [`lifecycle/742eaf1292ed737e591e`](../../../../results/interferometer/delaunay/sma/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| sma | `hpc_a100` | `jax_gpu_dense_fp64_nufft` | secondary | 57.35 ms | 69.8 | [`lifecycle/ff5c53947f845ee5f2d8`](../../../../results/interferometer/delaunay/sma/jax_gpu_dense_fp64_nufft/hpc_a100/runtime.json) |
| alma | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 764.9 ms | 308 | [`lifecycle/5e9d64046821795b4651`](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 2.121 s | 13.7 | [`lifecycle/60747d6f7a1867da50c9`](../../../../results/interferometer/delaunay/alma/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma_high | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 9.329 s | 25.3 | [`lifecycle/bfe464d85eab437c56be`](../../../../results/interferometer/delaunay/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma_high | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 7.601 s | 3.81 | [`lifecycle/d54c6814c51c381564cf`](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| jvla | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 63.8 s | 1.22 | [`lifecycle/e63a97d451e1633c91aa`](../../../../results/interferometer/delaunay/jvla/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| jvla | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 51.44 s | 0.564 | [`lifecycle/757d2c2d7167cdef0457`](../../../../results/interferometer/delaunay/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| sma | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 279.3 ms | 845 | [`lifecycle/b2d01ef74bf71e49d774`](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| sma | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 784.6 ms | 37 | [`lifecycle/00838694b2c3c955a644`](../../../../results/interferometer/delaunay/sma/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |

## Untested branches

- Named in the `cpu_many_core` rule: CPU streaming and the MGE DFT route.
- Named in the `gpu_a100` rule: the GPU streaming route (measured up to jvla, 2.5e7 visibilities, on W~ sparse).
- Instruments: measured on `alma`, `alma_high`, `jvla`, `sma` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; measured but not designated: `source_pixels_1500` (rows inherit the role as secondary, no decision); other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Measured on the designated hosts but not named by the rule (class `experimental`): `jax_cpu_dense_fp64_nufft`.
- Mixed precision: measured (`jax_gpu_sparse_mixed_wtilde`) and experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`interferometer-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-delaunay-standard-cpu-many-core.md#revisit-conditions), [`interferometer-delaunay-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-delaunay-standard-gpu-a100.md#revisit-conditions).
