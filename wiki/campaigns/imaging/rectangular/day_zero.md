<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: imaging/rectangular

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 60 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): Any imaging dataset with a rectangular pixelized source on a many-core CPU node with no GPU uses numba_cpu_sparse_fp64, one serial likelihood per core (workers capped by node memory over peak RSS).
  Optimal `numba_cpu_sparse_fp64`; secondary `jax_cpu_sparse_fp64`. Pulse record [`imaging-rectangular-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-rectangular-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): Any imaging dataset with a rectangular pixelized source on an A100 node uses jax_gpu_dense_fp64, falling back to jax_gpu_sparse_fp64.
  Optimal `jax_gpu_dense_fp64`; secondary `jax_gpu_sparse_fp64`. Pulse record [`imaging-rectangular-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-rectangular-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| ao | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 317.1 ms | 12.6 | [`lifecycle/d570569f32c953038fde`](../../../../results/imaging/rectangular/ao/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| ao | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 937 ms | 4.27 | [`lifecycle/46db4990a22e25aee123`](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| euclid | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 7.684 ms | 521 | [`lifecycle/456f7d65babc5524183a`](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| euclid | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 12.93 ms | 309 | [`lifecycle/bc5ed323618c58fc7682`](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| hst | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 24.68 ms | 162 | [`lifecycle/42f6a707360ce63b8150`](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| hst | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 40.52 ms | 98.7 | [`lifecycle/cc3f3365b6ac65388a93`](../../../../results/imaging/rectangular/hst/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| jwst | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 60.71 ms | 65.9 | [`lifecycle/ec69a200557ea8212e5e`](../../../../results/imaging/rectangular/jwst/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| jwst | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 107.3 ms | 37.3 | [`lifecycle/bdffd9d6d157b3b66435`](../../../../results/imaging/rectangular/jwst/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| ao | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 9.982 s | 3.21 | [`lifecycle/403420f8debaaa38956c`](../../../../results/imaging/rectangular/ao/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| ao | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 76.82 s | 0.378 | [`lifecycle/d7020017fca2249bf32b`](../../../../results/imaging/rectangular/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| euclid | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 282.5 ms | — | [`lifecycle/6060fdcfa2e5a4a0c168`](../../../../results/imaging/rectangular/euclid/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| euclid | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 910.6 ms | 31.9 | [`lifecycle/4c53e409a27040f5d6a4`](../../../../results/imaging/rectangular/euclid/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| hst | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 696.7 ms | 339 | [`lifecycle/b62534053c00fb102bf4`](../../../../results/imaging/rectangular/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| hst | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 2.963 s | 9.79 | [`lifecycle/2b5e34f509175fd4d030`](../../../../results/imaging/rectangular/hst/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| jwst | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 1.688 s | 140 | [`lifecycle/304ac75d242dcbc2c22f`](../../../../results/imaging/rectangular/jwst/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| jwst | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 5.754 s | 5.04 | [`lifecycle/9bb2e30e94491253641b`](../../../../results/imaging/rectangular/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |

## Untested branches

- Instruments: measured on `ao`, `euclid`, `hst`, `jwst` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; measured but not designated: `source_pixels_1521`, `source_pixels_784` (rows inherit the role as secondary, no decision); other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Measured on the designated hosts but not named by the rule (class `experimental`): `jax_cpu_sparse_fp64`.
- Mixed precision: measured (`jax_gpu_dense_mixed`, `jax_gpu_sparse_mixed`) and experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`imaging-rectangular-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-rectangular-standard-cpu-many-core.md#revisit-conditions), [`imaging-rectangular-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-rectangular-standard-gpu-a100.md#revisit-conditions).
