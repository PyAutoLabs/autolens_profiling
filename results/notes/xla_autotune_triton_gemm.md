# XLA autotuning, Triton GEMM and the persistent autotune cache (A100, 2026-09-08)

PyAutoNerves issue #161, task `xla-triton-gemm-off`. Controlled probe:
`scripts/misc/jax_compile/gemm_probe.py` via
`hpc/batch_gpu/submit_xla_autotune_gemm_probe`.

## What was measured and why

The 2026-09-05 A/B in [`delaunay_nn_breakdown.md`](./delaunay_nn_breakdown.md)
("XLA GPU autotuning A/B") measured the Curvature matrix (F) step at **25.63 ms**
with the shipped `--xla_gpu_autotune_level=0` against **4.90 ms** at XLA's default
level 4 — a 5.2× penalty on one dense fp64 GEMM. But four later jobs (342315,
342316, 342321, 342322), all with autotuning still at level 0, measured F at
**4.80–4.83 ms** with `curvature_matrix_jit_compile` at 0.066 s. Both cannot be
explained by the flag alone, so the flag was not the whole story. This note is the
controlled experiment that separates the flag from the state it leaves behind.

## Job

| Job | Node | GPU | Start (2026-09-08) | Elapsed | State |
|---|---|---|---|---:|---|
| 342333 | `euclid-ral-gpu-2` | NVIDIA A100 80GB PCIe | 13:32:26 | 0:14 | COMPLETED |

`jax 0.10.2`, `jax_persistent_cache_min_compile_time_secs=0`,
`jax_persistent_cache_enable_xla_caches = xla_gpu_per_fusion_autotune_cache_dir`. The
timed program is `F = (M*w[:,None]).T @ (M*w[:,None])` with `M` a (15361, 1560) float64
array from `default_rng(0)` — the masked-pixel × mesh-vertex mapping-matrix shape of the
HST / Delaunay-1500 cell, likelihood stripped away. All five arms in one job, one node.

## Arms

Every arm carries `--xla_disable_hlo_passes=constant_folding` plus the HLO dump
flags; the arms differ only in the autotune/Triton flags and in which
`JAX_COMPILATION_CACHE_DIR` they are pointed at. **A** is level 0 on a fresh cache
(the shipped default on a node nothing has seeded); **B** is level 4 on a fresh
cache (XLA's own default); **C** is level 0 pointed at the cache **B** just wrote;
**D** is level 0 pointed at its own cache **A** wrote; **E** is level 0 plus
`--xla_gpu_enable_triton_gemm=false` on a fresh cache.

## Results

| arm | flags beyond the base | cache | compile s | first ms | steady mean ms | steady min ms | autotune entries after | lowering |
|---|---|---|---:|---:|---:|---:|---:|---|
| A | level 0 | fresh `cache_A` | 0.461 | 25.99 | 25.523 | 25.338 | 0 | Triton `__triton_nested_gemm_fusion`, num_warps=2, output_tiles [16,8], num_stages=1 |
| B | level 4 | fresh `cache_B` | 2.080 | 5.50 | 4.918 | 4.815 | 3 | cuBLASLt `__cublas$lt$matmul`, selected_algorithm 1 |
| C | level 0 | `cache_B` (seeded by B) | 0.072 | 43.8 | 4.832 | 4.711 | 3 | cuBLASLt algorithm 1 |
| D | level 0 | `cache_A` (own) | 0.005 | 25.76 | 25.510 | 25.326 | 0 | Triton (executable-cache hit) |
| E | level 0 + `--xla_gpu_enable_triton_gemm=false` | fresh `cache_E` | 0.074 | 43.4 | 4.789 | 4.696 | 0 | cuBLASLt, `autotune_workspace_size` 0 |

## HLO lowering

Arm A (and D) lower the dot to a Triton fusion carrying XLA's *default* tile
configuration — never measured, because nothing autotuned it:

```
%gemm_fusion_dot_general.1 = f64[1560,1560]{1,0} fusion(...), kind=kCustom,
  backend_config={..."fusion_backend_config":{"kind":"__triton_nested_gemm_fusion",
  "block_level_fusion_config":{"num_warps":"2","output_tiles":[{"sizes":["16","8"]}],
  "num_ctas":1,"num_stages":1,...}}...}
```

Arms B, C and E lower it to the cuBLASLt custom call instead:

```
%custom-call.1 = (f64[1560,1560]{1,0}, s8[4194304]{0}) custom-call(...),
  custom_call_target="__cublas$lt$matmul",
  backend_config={..."gemm_backend_config":{"selected_algorithm":"1","alpha_real":1,
  "beta":0,...,"autotune_workspace_size":"4194304",...}...}
```

In arm E the same call carries `"autotune_workspace_size":"0"` and no
`selected_algorithm` — cuBLAS is reached without the autotuner running at all.

## Mechanism

JAX persists XLA's **per-fusion autotune cache** under `JAX_COMPILATION_CACHE_DIR` by
default, in `xla_gpu_per_fusion_autotune_cache_dir/`, keyed by fusion fingerprint —
*not* by the autotune flag. The flag **is** part of the executable-cache key, so arm C
was a genuine recompile (0.072 s, no executable hit) that nevertheless read arm B's
autotune entry and emitted B's cuBLAS kernel. Arm D hit the executable cache outright
(0.005 s) and stayed on Triton.

RAL `$HOME` — hence the default cache root `~/.cache/pyauto_jax` — is **node-local**. The
2026-09-05 level-4 jobs 342282 and 342283 therefore seeded `euclid-ral-gpu-2`, and every
later level-0 run on that node (342315, 342316, 342321, 342322) read that seeded entry:
F 4.80–4.83 ms, `curvature_matrix_jit_compile` 0.066 s. The two 2026-09-05 level-0 rows
(342277 on `gpu-1`, 342278 on `gpu-2`) predate any level-4 run on their nodes, which is
why they measured 25.6 ms. A level-0 run never writes autotune entries itself, so a
fresh node — or a new fusion shape on a seeded node — always pays the un-tuned Triton
default.

## Numerics

`F[0,0]` is `7685.1171726898065` in arms A, D and E (bit-identical) and
`7685.1171726897901` in arms B and C — about 2 ULP apart. The cure (arm E)
reproduces the Triton arms bit-for-bit, so adopting it changes no pinned value; it
is the autotuned cuBLAS kernel that sits 2 ULP away.

## Consequences

- **The A/B trap.** A valid autotune A/B needs a **fresh `JAX_COMPILATION_CACHE_DIR` per
  arm**, or a node no level-4 run has ever touched; otherwise the level-0 arm quietly
  inherits the level-4 arm's kernel and the comparison collapses — exactly what happened
  to the four post-09-05 `gpu-2` rows.
- **Read the F compile time.** ~2 s means the autotuner ran; ~0.4 s means a real compile
  without it (and so the slow Triton default); ~0.07 s or less means a cache hit and
  tells you nothing about which kernel is underneath — check the HLO.
- **Dashboard rows recorded 2026-07-17 → 2026-09-08** (the window in which
  `--xla_gpu_autotune_level=0` shipped without the Triton flag) are slow on GEMM-bound
  steps *unless* the node's cache happened to be seeded. The `_autotune4` rows, and the
  post-09-05 `gpu-2` rows, are the fast reference.

## Cure

Adopted in PyAutoNerves#161: keep `--xla_gpu_autotune_level=0` and also append
`--xla_gpu_enable_triton_gemm=false` by default, so dense fp64 GEMMs reach cuBLAS
without paying autotune compile time (an explicit autotune level or an explicit
Triton flag in `XLA_FLAGS` is still respected).

## Files

- `scripts/misc/jax_compile/gemm_probe.py` — the probe.
- `hpc/batch_gpu/submit_xla_autotune_gemm_probe` — the five-arm submit script.
- `results/notes/xla_autotune_triton_gemm.md` — this note.

The raw job output lives on RAL under `/mnt/ral/jnightin/scratch_xla_autotune/` and is
not committed.
