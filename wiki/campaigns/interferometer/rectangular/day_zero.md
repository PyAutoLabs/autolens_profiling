<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: interferometer/rectangular

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 65 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): Any interferometer dataset with a rectangular pixelized source on a many-core CPU node with no GPU uses numba_cpu_sparse_fp64_wtilde, one serial likelihood per core (workers capped by node memory over peak RSS), falling back to jax_cpu_sparse_fp64_wtilde. Untested: CPU streaming and the MGE DFT route.
  Optimal `numba_cpu_sparse_fp64_wtilde`; secondary `jax_cpu_sparse_fp64_wtilde`. Pulse record [`interferometer-rectangular-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-rectangular-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): Any interferometer dataset with a rectangular pixelized source on an A100 node uses jax_gpu_sparse_fp64_wtilde, falling back to jax_gpu_dense_fp64_nufft. Untested: the GPU streaming route (measured up to jvla, 2.5e7 visibilities, on W~ sparse).
  Optimal `jax_gpu_sparse_fp64_wtilde`; secondary `jax_gpu_dense_fp64_nufft`. Pulse record [`interferometer-rectangular-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-rectangular-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| alma | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 48.97 ms | 81.7 | [`lifecycle/fe32c3ee13ad9afc455c`](../../../../results/interferometer/rectangular/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| alma_high | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 81.06 ms | 49.4 | [`lifecycle/2ab9504c46bb8170ba54`](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| jvla | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 456.3 ms | 8.77 | [`lifecycle/220d076d8421130e83cf`](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| sma | `hpc_a100` | `jax_gpu_sparse_fp64_wtilde` | optimal | 6.748 ms | 593 | [`lifecycle/a39d6114339d546f6bd6`](../../../../results/interferometer/rectangular/sma/jax_gpu_sparse_fp64_wtilde/hpc_a100/runtime.json) |
| sma | `hpc_a100` | `jax_gpu_dense_fp64_nufft` | secondary | 51.98 ms | 77 | [`lifecycle/e44acbf15a2387706fa1`](../../../../results/interferometer/rectangular/sma/jax_gpu_dense_fp64_nufft/hpc_a100/runtime.json) |
| alma | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 888.8 ms | 266 | [`lifecycle/4663f221c8c3ce49b76f`](../../../../results/interferometer/rectangular/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 1.929 s | 15 | [`lifecycle/41fd14e1f19bad947f24`](../../../../results/interferometer/rectangular/alma/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma_high | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 8.919 s | 26.5 | [`lifecycle/d79f1b94f4b03452d745`](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| alma_high | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 7.43 s | 3.9 | [`lifecycle/fe0f252f68dcc26d7d13`](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| jvla | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 61.44 s | 1.37 | [`lifecycle/15aded078686cdf03ff5`](../../../../results/interferometer/rectangular/jvla/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| jvla | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 48.48 s | 0.598 | [`lifecycle/99c70939e335d08c4cff`](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| sma | `hpc_ral_cpu` | `numba_cpu_sparse_fp64_wtilde` | optimal | 230.9 ms | 1.02e+03 | [`lifecycle/0e27b1175ce084dff51e`](../../../../results/interferometer/rectangular/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |
| sma | `hpc_ral_cpu` | `jax_cpu_sparse_fp64_wtilde` | secondary | 643.9 ms | 45 | [`lifecycle/29d350ddd936276d4121`](../../../../results/interferometer/rectangular/sma/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) |

## Untested branches

- Named in the `cpu_many_core` rule: CPU streaming and the MGE DFT route.
- Named in the `gpu_a100` rule: the GPU streaming route (measured up to jvla, 2.5e7 visibilities, on W~ sparse).
- Instruments: measured on `alma`, `alma_high`, `jvla`, `sma` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; measured but not designated: `source_pixels_1521` (rows inherit the role as secondary, no decision); other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Measured on the designated hosts but not named by the rule (class `experimental`): `jax_cpu_dense_fp64_nufft`.
- Mixed precision: measured (`jax_gpu_sparse_mixed_wtilde`) and experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`interferometer-rectangular-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-rectangular-standard-cpu-many-core.md#revisit-conditions), [`interferometer-rectangular-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/interferometer-rectangular-standard-gpu-a100.md#revisit-conditions).
