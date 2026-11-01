<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: point_source_image/image_plane

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 8 stamped records, frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): A galaxy-scale point-source dataset fitted in the image plane on a many-core CPU node with no GPU uses jax_cpu_na_fp64.
  Optimal `jax_cpu_na_fp64`; secondary none. Pulse record [`point-source-image-image-plane-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/point-source-image-image-plane-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): A galaxy-scale point-source dataset fitted in the image plane on an A100 node uses jax_gpu_na_fp64.
  Optimal `jax_gpu_na_fp64`; secondary none. Pulse record [`point-source-image-image-plane-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/point-source-image-image-plane-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| simple | `hpc_a100` | `jax_gpu_na_fp64` | optimal | 0.3951 ms | 1.01e+04 | [`lifecycle/6282556b81be530f7bce`](../../../../results/point_source_image/image_plane/simple/jax_gpu_na_fp64/hpc_a100/runtime.json) |
| simple | `hpc_ral_cpu` | `jax_cpu_na_fp64` | optimal | 0.954 ms | 3.04e+04 | [`lifecycle/6cbf08c848ae6cdb820f`](../../../../results/point_source_image/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/runtime.json) |

## Untested branches

- Instruments: measured on `simple` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Mixed precision: unmeasured; experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`point-source-image-image-plane-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/point-source-image-image-plane-standard-cpu-many-core.md#revisit-conditions), [`point-source-image-image-plane-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/point-source-image-image-plane-standard-gpu-a100.md#revisit-conditions).
