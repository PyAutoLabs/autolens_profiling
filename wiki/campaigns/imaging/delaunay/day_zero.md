<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: imaging/delaunay

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 60 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): Any imaging dataset with a Delaunay pixelized source on a many-core CPU node with no GPU uses numba_cpu_sparse_fp64, one serial likelihood per core (workers capped by node memory over peak RSS).
  Optimal `numba_cpu_sparse_fp64`; secondary `jax_cpu_sparse_fp64`. Pulse record [`imaging-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-delaunay-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): Any imaging dataset with a Delaunay pixelized source on an A100 node uses jax_gpu_dense_fp64, falling back to jax_gpu_sparse_fp64.
  Optimal `jax_gpu_dense_fp64`; secondary `jax_gpu_sparse_fp64`. Pulse record [`imaging-delaunay-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-delaunay-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| ao | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 347.9 ms | 11.5 | [`lifecycle/fa028d54604cd08d3555`](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| ao | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 821.5 ms | 4.87 | [`lifecycle/99d7c3c7623e3d43d059`](../../../../results/imaging/delaunay/ao/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| euclid | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 14.5 ms | 276 | [`lifecycle/c07af59a8d393975dd28`](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| euclid | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 17.96 ms | 223 | [`lifecycle/371f67d50254479493ee`](../../../../results/imaging/delaunay/euclid/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| hst | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 34.19 ms | 117 | [`lifecycle/73c19173fcc983318b61`](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| hst | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 47.16 ms | 84.8 | [`lifecycle/12c678df16e04f4f9011`](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| jwst | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 69.7 ms | 57.4 | [`lifecycle/12348bea7be2ca70cffb`](../../../../results/imaging/delaunay/jwst/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| jwst | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 107.6 ms | 37.2 | [`lifecycle/d34ebb3d52e02dfb8b16`](../../../../results/imaging/delaunay/jwst/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| ao | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 8.671 s | 4.61 | [`lifecycle/1f22a4a5fcd73276ec53`](../../../../results/imaging/delaunay/ao/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| ao | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 81.31 s | 0.357 | [`lifecycle/e08b7a6ff7b9daea8790`](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| euclid | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 820 ms | 288 | [`lifecycle/5556ca430bc180c2d536`](../../../../results/imaging/delaunay/euclid/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| euclid | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 1.247 s | 23.3 | [`lifecycle/b202eb6f7da58a38882f`](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| hst | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 951.7 ms | 248 | [`lifecycle/321ec1f6f2a5955a9a20`](../../../../results/imaging/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| hst | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 2.991 s | 9.7 | [`lifecycle/06a9dad80c65ecd81c6c`](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| jwst | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 1.745 s | 135 | [`lifecycle/d93b4dad29e8a0f77987`](../../../../results/imaging/delaunay/jwst/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| jwst | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 6.562 s | 4.42 | [`lifecycle/b5b87c504713b9831857`](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |

## Untested branches

- Instruments: measured on `ao`, `euclid`, `hst`, `jwst` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; measured but not designated: `source_pixels_1500` (rows inherit the role as secondary, no decision); other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Measured on the designated hosts but not named by the rule (class `experimental`): `jax_cpu_sparse_fp64`.
- Mixed precision: measured (`jax_gpu_dense_mixed`, `jax_gpu_sparse_mixed`) and experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`imaging-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-delaunay-standard-cpu-many-core.md#revisit-conditions), [`imaging-delaunay-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-delaunay-standard-gpu-a100.md#revisit-conditions).
