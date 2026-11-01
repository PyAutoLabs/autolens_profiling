<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: group/delaunay

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 15 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): A group-scale imaging dataset with a Delaunay pixelized source on a many-core CPU node with no GPU uses numba_cpu_sparse_fp64, one serial likelihood per core (workers capped by node memory over peak RSS), falling back to jax_cpu_sparse_fp64.
  Optimal `numba_cpu_sparse_fp64`; secondary `jax_cpu_sparse_fp64`. Pulse record [`group-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/group-delaunay-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): A group-scale imaging dataset with a Delaunay pixelized source on an A100 node uses jax_gpu_dense_fp64, falling back to jax_gpu_sparse_fp64.
  Optimal `jax_gpu_dense_fp64`; secondary `jax_gpu_sparse_fp64`. Pulse record [`group-delaunay-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/group-delaunay-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| euclid | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 36.56 ms | 109 | [`lifecycle/826e517e5786c2398767`](../../../../results/group/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| euclid | `hpc_a100` | `jax_gpu_sparse_fp64` | secondary | 55.59 ms | 72 | [`lifecycle/af408d3bf87b7b904f2d`](../../../../results/group/delaunay/euclid/jax_gpu_sparse_fp64/hpc_a100/runtime.json) |
| euclid | `hpc_ral_cpu` | `numba_cpu_sparse_fp64` | optimal | 803.7 ms | 238 | [`lifecycle/22696238c7226a6971dd`](../../../../results/group/delaunay/euclid/numba_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |
| euclid | `hpc_ral_cpu` | `jax_cpu_sparse_fp64` | secondary | 6.054 s | 4.79 | [`lifecycle/f4474b23fc3ee5d4b43f`](../../../../results/group/delaunay/euclid/jax_cpu_sparse_fp64/hpc_ral_cpu/runtime.json) |

## Untested branches

- Instruments: measured on `euclid` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Measured on the designated hosts but not named by the rule (class `experimental`): `jax_cpu_sparse_fp64`.
- Mixed precision: unmeasured; experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`group-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/group-delaunay-standard-cpu-many-core.md#revisit-conditions), [`group-delaunay-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/group-delaunay-standard-gpu-a100.md#revisit-conditions).
