<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: datacube/delaunay

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 50 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): A datacube (interferometer channels fitted together) with a Delaunay pixelized source on a many-core CPU node with no GPU uses numba_cpu_sparse_fp64_wtilde, one serial likelihood per core (workers capped by node memory over peak RSS), falling back to jax_cpu_sparse_fp64_wtilde.
  Optimal `numba_cpu_sparse_fp64_wtilde`; secondary `jax_cpu_sparse_fp64_wtilde`. Pulse record [`datacube-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/datacube-delaunay-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): **held, not designated.** Provisional (lifecycle report section 3.3): `jax_gpu_sparse_fp64_streaming`, `jax_gpu_sparse_fp64_wtilde`; fallback `jax_gpu_dense_fp64_dft`, `jax_gpu_dense_fp64_nufft`. Human 2026-10-31 (AskUserQuestion, 'Hold, keep provisional'): does not separate at sma, jax_gpu_sparse_fp64_wtilde 3.46 vs jax_gpu_dense_fp64_nufft 3.37 calls/s/node (1.03x, inside the 9 % tolerance); alma and alma_high measured only W~ sparse.

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| alma | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | secondary (held) | 1.252 s | 3.2 | [`lifecycle/c3ed1c5a8847d28eb792`](../../../../results/datacube/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| alma_high | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | secondary (held) | 1.437 s | 2.78 | [`lifecycle/5ddc372d7a91e6f979a8`](../../../../results/datacube/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| sma | `hpc_a100` | `jax_gpu_dense_fp64_nufft` | secondary (held) | 1.185 s | 3.37 | [`lifecycle/b62555fd25c854bfdf3d`](../../../../results/datacube/delaunay/sma/jax_gpu_dense_fp64_nufft/hpc_a100/runtime.json) |
| sma | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | secondary (held) | 1.154 s | 3.46 | [`lifecycle/5a76ad3529beab044b72`](../../../../results/datacube/delaunay/sma/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| alma | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 9.356 s | 25.2 | [`lifecycle/5800a501664b981824d4`](../../../../results/datacube/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 14.29 s | 2.03 | [`lifecycle/8e374ab509e7626573da`](../../../../results/datacube/delaunay/alma/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma_high | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 17.64 s | 3.8 | [`lifecycle/3ea56488a996d6f0ffd1`](../../../../results/datacube/delaunay/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma_high | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 22.05 s | 1.31 | [`lifecycle/6669e2835516e93361f1`](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| sma | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 8.444 s | 27.9 | [`lifecycle/5b411896dcf5b1cc5853`](../../../../results/datacube/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| sma | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 12.91 s | 2.25 | [`lifecycle/649dfad640b1afce5631`](../../../../results/datacube/delaunay/sma/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |

## Untested branches

- `gpu_a100`: the setups do not separate; the cell stays provisional until a designation is recorded.
- Instruments: measured on `alma`, `alma_high`, `sma` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Measured on the designated hosts but not named by the rule (class `experimental`): `jax_cpu_dense_fp64_nufft`.
- Mixed precision: unmeasured; experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`datacube-delaunay-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/datacube-delaunay-standard-cpu-many-core.md#revisit-conditions).
