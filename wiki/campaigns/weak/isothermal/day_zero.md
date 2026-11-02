<!-- generated: scripts/misc/tooling/build_day_zero_pages.py; do not edit -->
# Day zero: weak/isothermal

Baseline tag `2026.10.25.1` ([day_zero.json](../../../../catalogue/day_zero.json)); 10 stamped records, 8 frozen in the [PyAutoPulse day_zero snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero/lens.json) and 2 gradient record(s) in the [day_zero_gradient snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero_gradient/lens.json). [All likelihoods](../../../day_zero.md).

## Rule

- **Many-core CPU node, no GPU** (`cpu_many_core`): A weak-lensing shear dataset with an isothermal lens on a many-core CPU node with no GPU uses jax_cpu_dense_fp64.
  Optimal `jax_cpu_dense_fp64`; secondary none. Pulse record [`weak-isothermal-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/weak-isothermal-standard-cpu-many-core.md) (2026-10-31).
- **A100 GPU node** (`gpu_a100`): A weak-lensing shear dataset with an isothermal lens on an A100 node uses jax_gpu_dense_fp64.
  Optimal `jax_gpu_dense_fp64`; secondary none. Pulse record [`weak-isothermal-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/weak-isothermal-standard-gpu-a100.md) (2026-10-31).

## Day-zero evidence

Stamped current runtime records (each cell's parity witness is a `match`; the cell's breakdown, compile, memory and parity records carry the same stamp). Per call is the batched JAX `vmap` per-call time or the numba warm median; calls/s is projected per node with parallel efficiency 1.0 (declared, unmeasured).

| instrument | host | setup | class | per call | projected calls/s per node | record |
|---|---|---|---|---|---|---|
| simple | `hpc_a100` | `jax_gpu_dense_fp64` | optimal | 0.008935 ms | 4.48e+05 | [`lifecycle/fbd315b1a0e03f308c76`](../../../../results/weak/isothermal/simple/jax_gpu_dense_fp64/hpc_a100/runtime.json) |
| simple | `hpc_ral_cpu` | `jax_cpu_dense_fp64` | optimal | 0.04041 ms | 7.18e+05 | [`lifecycle/03601d3a29aca4f5c389`](../../../../results/weak/isothermal/simple/jax_cpu_dense_fp64/hpc_ral_cpu/runtime.json) |

## Untested branches

- Instruments: measured on `simple` only; the rule extrapolates to any other instrument, data size or visibility/pixel count.
- Sizes: only the `standard` regime is designated; other sizes are unmeasured.
- Hosts: laptop and local GPUs have no designation (local rows inherit the role, no decision); GPUs other than the A100 and CPU nodes of other shapes are unmeasured.
- Measured on the designated hosts but not named by the rule (class `experimental`): `jax_cpu_dense_fp64`.
- Mixed precision: unmeasured; experimental at day zero.
- Parallel efficiency: projected throughput assumes 1.0 (declared, unmeasured); phase 4 measures it, and a measured value that reorders the setups supersedes the designation. The sampler's batch ceiling (Nautilus `n_batch`) is not modelled.

## Revisit

Each designation's revisit conditions live in its Pulse record: [`weak-isothermal-standard-cpu-many-core`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/weak-isothermal-standard-cpu-many-core.md#revisit-conditions), [`weak-isothermal-standard-gpu-a100`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/weak-isothermal-standard-gpu-a100.md#revisit-conditions).
