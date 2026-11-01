<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: imaging/mge

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 24 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): Any imaging dataset fitted with a parametric (MGE) source on a many-core CPU node with no GPU uses jax_cpu_dense_fp64.
  Optimal `jax_cpu_dense_fp64`; secondary none. Pulse record [`imaging-mge-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-mge-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): Any imaging dataset fitted with a parametric (MGE) source on an A100 node uses jax_gpu_dense_fp64.
  Optimal `jax_gpu_dense_fp64`; secondary none. Pulse record [`imaging-mge-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-mge-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| ao | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 18.78 ms | 213 | [`lifecycle/4b514c4262172f610fee`](../../../../results/imaging/mge/ao/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| euclid | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 0.1985 ms | 2.02e+04 | [`lifecycle/770dce7d7db7a1fc3b53`](../../../../results/imaging/mge/euclid/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| hst | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 0.3915 ms | 1.02e+04 | [`lifecycle/8a475e9abd97a8c4a1c0`](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| jwst | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 0.918 ms | 4.36e+03 | [`lifecycle/0756e600e86d7ae6a462`](../../../../results/imaging/mge/jwst/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| ao | `hpc_ral_cpu` | `jax_cpu_dense_fp64` | optimal | 511.8 ms | 56.7 | [`lifecycle/3216ffc143645fb46c72`](../../../../results/imaging/mge/ao/jax_cpu_dense_fp64/hpc_ral_cpu/runtime.json) |
| euclid | `hpc_ral_cpu` | `jax_cpu_dense_fp64` | optimal | 7.068 ms | 4.1e+03 | [`lifecycle/190ebc582d9aba9aaa9c`](../../../../results/imaging/mge/euclid/jax_cpu_dense_fp64/hpc_ral_cpu/runtime.json) |
| hst | `hpc_ral_cpu` | `jax_cpu_dense_fp64` | optimal | 16.29 ms | 1.78e+03 | [`lifecycle/54979849c807d452c510`](../../../../results/imaging/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/runtime.json) |
| jwst | `hpc_ral_cpu` | `jax_cpu_dense_fp64` | optimal | 50.55 ms | 574 | [`lifecycle/4baa9bc04c8d24a713b6`](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/runtime.json) |

## Untested branches

- Instruments: measured on `ao`, `euclid`, `hst`, `jwst` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Mixed precision: unmeasured; experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`imaging-mge-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-mge-standard-cpu-many-core.md#revisit-conditions), [`imaging-mge-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/imaging-mge-standard-gpu-a100.md#revisit-conditions).
