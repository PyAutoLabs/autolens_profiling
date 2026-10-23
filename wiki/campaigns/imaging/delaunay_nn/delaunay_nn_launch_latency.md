# DelaunayNN Sibson — launch-latency cuts + query-chunk sweep (A100, 2026-09-08)

PyAutoArray issue #532 / PR #533, branch `feature/delaunay-nn-launch-latency`, against its
merge base `d7c96762` (PyAutoArray `main`, #531 merged). Two commits are measured here:
`9e9a7d50` (Phase A at the old chunk 256) and `e5d31d74` (Phase A **as shipped**, with the
new `SIBSON_QUERY_CHUNK = 4096` default this A/B chose).

## What changed

Three cuts to `autoarray/inversion/mesh/interpolator/sibson.py`, none of which change a
number:

1. **The 3-trip candidate-edge `fori_loop` inside the cavity walk is unrolled at trace
   time** when `_sibson_unroll_candidates()` is true — `PYAUTO_SIBSON_UNROLL_CANDIDATES`
   wins if set, otherwise `jax.default_backend() != "cpu"`. Same `add_candidate` calls for
   edges 0, 1, 2 in that order, so the same cavity insertion order, the same stencil column
   order and the same floating-point summation order. It removes ~28 % of the kernel
   launches per chunk (1,244 → 892). CPU stays rolled (measured ~8-9 % faster there).
2. **`jax_delaunay_nn` locates and interpolates the data grid and the 6,000 `ConstantSplit`
   cross points in one concatenated Sibson pass**, computing the circumcircles once — the
   #531 pattern, now applied to the Sibson path.
3. **`SIBSON_QUERY_CHUNK` is a memory guard again**, with `PYAUTO_SIBSON_QUERY_CHUNK` as an
   import-time override so it can be swept without editing source. The sweep below set the
   new default.

## Verdict

On the A100, the HST / Hilbert-1500 / MGE-60 / `ConstantSplit` **DelaunayNN** imaging
likelihood's **params→H prefix — the aggregate invariant to the attribution shift below —
falls 143.90 → 28.21 ms unbatched (−115.7 ms, 5.10×)** and 24.32 → 16.44 ms per call at
`vmap` 16 (−7.9 ms, 1.48×). Whole-likelihood single-JIT runtime goes **201.32 → 76.32 ms
(2.64×)** and the step-by-step sum 196.36 → 80.72 ms; at `vmap` 16 the whole likelihood
goes 58.02 → 52.62 ms per call (1.10×). The `29144.581944` pin held **exactly on every leg
and every chunk**.

**The witness is half met.** Its unbatched half (params→H ≤ 60 ms) is met at every chunk
≥ 512 — 28.21 ms at the shipped default, 24.66 ms on the sweep's own 4096 row. Its
batched half (≤ 11 ms per call at `vmap` 16) is **not** met: 16.44 ms. The new
split-Sibson stage says why, and it is not the Sibson loops — see
"Where the remaining per-call cost is".

## Provenance

| Job | Cell | Leg | Node | Start (2026-09-08) | Elapsed | State |
|---|---|---|---|---|---:|---|
| 342321 | `likelihood_breakdown/imaging/delaunay_nn` | control `d7c96762` | `euclid-ral-gpu-2` | 02:16:25 | 3:00 | COMPLETED |
| 342322 | `likelihood_breakdown/imaging/delaunay_nn` | feature `9e9a7d50`, chunk 256 | `euclid-ral-gpu-2` | 02:19:27 | 2:49 | COMPLETED |
| 342323_0 | `likelihood_breakdown/imaging/delaunay_nn` | feature `9e9a7d50`, chunk 512 | `euclid-ral-gpu-2` | 02:22:17 | 2:36 | COMPLETED |
| 342323_1 | `likelihood_breakdown/imaging/delaunay_nn` | feature `9e9a7d50`, chunk 1024 | `euclid-ral-gpu-2` | 02:24:54 | 2:26 | COMPLETED |
| 342323_2 | `likelihood_breakdown/imaging/delaunay_nn` | feature `9e9a7d50`, chunk 2048 | `euclid-ral-gpu-2` | 02:27:20 | 2:37 | COMPLETED |
| 342323_3 | `likelihood_breakdown/imaging/delaunay_nn` | feature `9e9a7d50`, chunk 4096 | `euclid-ral-gpu-2` | 02:29:58 | 2:33 | COMPLETED |
| 342324 | `likelihood_runtime/imaging/delaunay_nn` | control `d7c96762` | `euclid-ral-gpu-2` | 02:32:32 | 1:22 | COMPLETED |
| 342325 | `likelihood_runtime/imaging/delaunay_nn` | feature `9e9a7d50`, chunk 256 | `euclid-ral-gpu-2` | 02:33:55 | 1:18 | COMPLETED |
| **342329** | `likelihood_breakdown/imaging/delaunay_nn` | **shipped `e5d31d74`, chunk 4096** | `euclid-ral-gpu-2` | 02:40:12 | 2:41 | COMPLETED |
| **342330** | `likelihood_runtime/imaging/delaunay_nn` | **shipped `e5d31d74`, chunk 4096** | `euclid-ral-gpu-2` | 02:42:54 | 1:25 | COMPLETED |

- Breakdown flags `--split-setup --vmap-batch 16`; runtime cell full timing only (no
  `--vmap-probe`), `vmap` batch 16 on both legs. `PyAutoLens 2026.8.17.1`, fp64
  (`JAX_ENABLE_X64=True`), dense inversion path, NVIDIA A100 80GB PCIe.
- Recorded `xla_flags` on every row: `--xla_disable_hlo_passes=constant_folding
  --xla_gpu_autotune_level=0` — identical across all ten job units, so these rows are
  comparable with the 2026-09-07 rows in `delaunay_walk_early_exit.md` and **not** with the
  2026-07-10 rows in `preopt_breakdown_baseline.md`.
- **All ten job units ran on the same node inside one 28-minute window**, so this is a
  same-session A/B, not a comparison against a recorded baseline.
- `backend = gpu`, `device = cuda:0` on every leg, so the feature legs took the *unrolled*
  candidate branch (the gate is `jax.default_backend() != "cpu"`).

### The A/B differs in PyAutoArray and nothing else

| Repo | Revision |
|---|---|
| PyAutoArray — control leg | `d7c96762801237526d383ce3c53da6cf847875c1` |
| PyAutoArray — feature leg (chunk 256, sweep) | `9e9a7d5020f3467a113af596d9df5edb4ecf0ff2` |
| PyAutoArray — feature leg (shipped) | `e5d31d74f938b2707e11b971683104c441789647` |
| PyAutoNerves | `efe7c04a79f8389bdcccb88b4d6f8781b9b27142` |
| PyAutoFit | `2680b32d6633cc9c1818e86d881d34bff7dc293b` |
| PyAutoGalaxy | `6d216c151c914b2ce79af18fbc0348a6fc5425d7` |
| PyAutoLens | `9468e3e1bc10405c622e7408c04c0760dcb26882` |
| autolens_profiling (`AP_ROOT`) | `99f4b533b52cda974f62c59e8b2995ccb941474b` + the uncommitted split-Sibson stage and these five submits |

The **shared** `/mnt/ral/jnightin/PyAuto` install was deliberately not touched — the
subhalo-validation and Euclid DR1 CPU runs (`342299`, `342311`, `342314`) were live on it
throughout. Every leg instead prepends a private PyAutoArray checkout to the `PYTHONPATH`
that `activate.sh` exports:
`/mnt/ral/jnightin/PyAuto_wt/delaunay-nn-launch-latency/PyAutoArray_{control,feature}`, and
each job prints `autoarray.__file__` and the checkout's `git rev-parse HEAD` into its SLURM
log. Submits run from `/mnt/ral/jnightin/autolens_profiling_wt/delaunay-nn-launch-latency`
(detached at `99f4b53`):
`hpc/batch_gpu/submit_breakdown_imaging_delaunay_nn_a100_hst_fp64_{launch_latency_control,launch_latency,chunk_sweep}`
and `submit_runtime_imaging_delaunay_nn_a100_hst_fp64_{control,launch_latency}`.

### Pins

`EXPECTED_LOG_EVIDENCE_HST = 29144.581944` passed **unchanged on every leg** — both
breakdown A/B legs, all four sweep chunks, the shipped re-measure, and `pinned_drift: []`
on all four runtime rows. The chunk is bit-neutral by construction and the sweep proves it
end-to-end: five different `lax.map` block sizes, one log evidence.

## Control vs the shipped feature

All values ms per likelihood call. `vmap/16` is `jax.jit(jax.vmap(fn))` over a params
pytree broadcast to batch 16, reported as batch time / 16; only the combined
inversion-setup block, the `--split-setup` prefixes and the params→H prefix are re-timed
under `vmap`, so the remaining rows read `—` there.

| Step | control | shipped | Δ | control vmap/16 | shipped vmap/16 |
|---|---:|---:|---:|---:|---:|
| Ray-trace data grid | 0.169 | 0.169 | −0.001 | — | — |
| Ray-trace mesh grid | 0.161 | 0.160 | −0.001 | — | — |
| Lens light images (pre-PSF) | 0.138 | 0.133 | −0.006 | — | — |
| Blurred image (PSF convolution) | 0.854 | 0.807 | −0.047 | — | — |
| Profile-subtracted image | 0.138 | 0.123 | −0.016 | — | — |
| **Inversion setup (steps 5–8 combined)** | **111.552** | **27.482** | **−84.071** | **20.364** | **17.674** |
| Data vector (D) | 0.348 | 0.336 | −0.012 | — | — |
| Curvature matrix (F) | 4.804 | 4.812 | +0.008 | — | — |
| **Regularization matrix (H)** | **45.103** | **13.768** | **−31.335** | **12.622** | **7.491** |
| Regularized reconstruction | 30.820 | 30.736 | −0.084 | — | — |
| Mapped recon + log evidence | 2.269 | 2.191 | −0.078 | — | — |
| **Total step-by-step (unbatched)** | **196.357** | **80.716** | **−115.641** | — | — |

### Four-way setup split and the prefixes

| Piece | control | shipped | Δ | control vmap/16 | shipped vmap/16 |
|---|---:|---:|---:|---:|---:|
| Border relocation | 1.526 | 1.079 | −0.447 | 0.092 | 0.064 |
| **Triangulation + interpolation** | **97.274** | **13.365** | **−83.909** | **11.603** | **8.879** |
| Mapping matrix | −0.909 | 0.102 | +1.011 | 0.074 | −2.403 |
| Blurred mapping matrix (PSF) | 8.943 | 8.353 | −0.590 | 10.705 | 10.808 |
| *Interpolator prefix (params→step 6)* | *98.800* | *14.444* | *−84.356* | *11.695* | *8.944* |
| *Split-Sibson prefix (params→split mappings)* | *134.159* | *14.242* | *−119.917* | *14.432* | *6.434* |
| **params→H prefix (the witness)** | **143.903** | **28.212** | **−115.691 (5.10×)** | **24.317** | **16.435 (1.48×)** |

The ±2 ms sign flips in the "Mapping matrix" rows are the by-difference artifact documented
in `delaunay_nn_breakdown.md`; they are below the run-to-run noise of two independently
compiled prefixes. The shipped leg's interpolator-prefix `vmap` number (8.944 ms) is the
*first* prefix measured in that run and carries warm-up: the split-Sibson prefix in the
same run — a strict superset — reads 6.434 ms, and the sweep's own 4096 row reads 6.398 ms.
Read the Sibson share at `vmap` 16 as **≈ 6.4 ms per call**.

### Split-point Sibson vs ConstantSplit assembly (the new stage)

| Piece | control | feature @256 | shipped @4096 | control vmap/16 | feature @256 vmap/16 | shipped vmap/16 |
|---|---:|---:|---:|---:|---:|---:|
| Split-point Sibson | 35.358 | 0.996 | −0.201 | 2.737 | 2.432 | −2.510 |
| H, ConstantSplit assembly | 9.745 | 13.730 | 13.970 | 9.885 | 10.152 | 10.001 |

**The 35.4 → ~0 ms collapse of "Split-point Sibson" is the attribution shift, not a
saving.** Before the change a prefix stopping at step 6 never asked for the split points,
so XLA dead-code-eliminated the second Sibson pass out of it and its whole cost surfaced in
this row's subtraction. The feature runs both query sets in *one* concatenated pass inside
the step-6 prefix, so the row empties (and can go slightly negative — a prefix that must
*materialise* the split mappings/sizes/weights as outputs against one that fuses them
straight into `reg_split_from`). Only `regularization_matrix_prefix_s` — the params→H
prefix — says whether work went away, and it fell 5.10×. This is exactly what #531 did to
the Delaunay H row, one boundary further in.

## The chunk sweep

`PYAUTO_SIBSON_QUERY_CHUNK` swept on the feature checkout, `--split-setup --vmap-batch 16`,
peak VRAM from a 2 s `nvidia-smi --query-gpu=memory.used` sampler running for the life of
the Python process (a floor on the true peak: a sub-2 s spike can be missed).

| chunk | params→H unbatched | params→H per call @vmap 16 | Sibson share @vmap 16 | ConstantSplit assembly @vmap 16 | peak sampled VRAM | pin |
|---:|---:|---:|---:|---:|---:|---|
| 256 (`9e9a7d50`, job 342322) | 88.287 | 23.072 | 10.488 | 10.152 | 41,495 MiB | PASS |
| 512 (342323_0) | 52.205 | 18.243 | 8.309 | 10.014 | 41,503 MiB | PASS |
| 1024 (342323_1) | 35.324 | 19.567 | 7.062 | 12.557 | 41,503 MiB | PASS |
| 2048 (342323_2) | 27.218 | 19.127 | 6.602 | 10.058 | 41,503 MiB | PASS |
| **4096 (342323_3)** | **24.664** | **16.451** | **6.398** | 10.028 | 41,503 MiB | PASS |
| 4096 shipped (`e5d31d74`, 342329) | 28.212 | 16.435 | 6.434 | 10.001 | not sampled | PASS |
| *control 256 (342321)* | *143.903* | *24.317* | *11.695* | *9.885* | *41,495 MiB* | *PASS* |

"Sibson share @vmap 16" is the interpolator prefix (or, where the run's first prefix carries
warm-up, the split-Sibson prefix, which is a strict superset of it and reads the same).

The **256 row is the same-chunk A/B**: it isolates the unroll + single concatenated pass
from the chunk change (143.903 → 88.287 ms unbatched, 1.63×). Everything below it is the
chunk doing the rest of the work.

### The default: 4096

The pre-registered rule (plan A.3) was "the largest of 512/1024/2048/4096 whose peak VRAM at
`vmap` 16 stays under ~50 % of the 80 GB and which is fastest per call; if 4096 wins on
both, ship 2048 unless the margin is > 15 %".

4096 is fastest on both readings. **The VRAM clause turned out uninformative and was
therefore not used to break the tie.** The ~41.5 GiB plateau (50.7 % of 81,920 MiB) is
identical at every chunk *and* on the control leg, and the GPU read 0 MiB with no other
process at the start of every job, so the plateau is this cell's `vmap`-16 dense inversion
block — not the cavity intermediates. A 16× change of chunk moves it by 8 MiB. The guard
arithmetic says why there is room: `(C, 3, 2)` fp64 intermediates at ~25 kB per query per
lane over 4096 queries × 16 lanes is ~1.6 GB, ~2 % of an 80 GB card.

With the VRAM leg vacuous, the decision rests on speed alone: 4096 is 14.0 % faster per
call at `vmap` 16 than 2048 (16.451 vs 19.127) and 9.4 % faster unbatched — a margin the
rule's "ship 2048 unless > 15 %" clause would have sent to 2048 had the VRAM clause been
live. It is not, so **4096 ships** (`e5d31d74`), with `PYAUTO_SIBSON_QUERY_CHUNK` kept as
the documented escape hatch for a smaller GPU or a much larger cell. The constant's comment
now carries this table, the rule and the arithmetic.

## Whole-likelihood runtime cell (`likelihood_runtime/imaging/delaunay_nn`)

| | control | feature @256 | shipped @4096 | Δ (control → shipped) |
|---|---:|---:|---:|---:|
| Full pipeline (single JIT) | 201.320 | 136.082 | 76.322 | **−125.0 (2.64×)** |
| `vmap` batch 16, per call | 58.022 | 54.209 | 52.618 | −5.4 (1.10×) |

This is what a production evaluation pays. The unbatched saving (−125.0 ms) exceeds the
params→H prefix drop (−115.7 ms) by the fusion slack expected when the whole pipeline is
one program. The batched saving is small for the same reason as #531: at batch 16 the dense
linear algebra dominates.

## Where the remaining per-call cost is

At `vmap` 16 and the shipped chunk, the 16.44 ms params→H prefix decomposes as:

| piece | per call @vmap 16 | comparison |
|---|---:|---|
| Sibson (locate + both interpolation passes) | ~6.4 ms | Delaunay's whole interpolator prefix: 5.07 ms |
| split-point Sibson residual after concatenation | ~0 ms | (folded into the row above) |
| **H, ConstantSplit assembly** | **~10.0 ms** | **Delaunay's H row: 0.07 ms** |

The assembly row reads 9.9–10.2 ms at *every* chunk and on the control — it is untouched by
Phase A and cannot be touched by any change to the Sibson chunking. **After Phase A roughly
60 % of the per-call params→H prefix is the 33-wide split-stencil regularization assembly,
not the Sibson loops.** That is why the ≤ 11 ms batched witness could not be met: 10 ms of
it was never Sibson's to give.

Unbatched the split is the other way round — 14.44 ms interpolator prefix against 13.97 ms
assembly out of 28.21 ms — because the assembly's 33-wide block build amortises under
`vmap` while the Sibson loops do not.

## Phase B recommendation — the numbers, not the decision

Phase B as planned is the cavity `fori_loop` → `lax.while_loop` early exit. It targets the
Sibson share only:

- **Its headroom at `vmap` 16 is ~6.4 ms per call.** The observed cavity occupancy on the
  production audit is 11 of the 32 fixed trips (`max_cavity=11`, `max_neighbors=13` in the
  workspace_test gate), so the early exit removes at most the tail of a 32-trip loop that
  is already latency-amortised at chunk 4096. The plan's own estimate was ~20 % of the
  remaining Sibson cost, i.e. **≈ 1.3 ms per call at `vmap` 16** and ≈ 3 ms unbatched.
- **The ConstantSplit assembly is ~10.0 ms per call at `vmap` 16** — 1.6× the entire Sibson
  share and 143× Delaunay's equivalent H row (0.07 ms). It is a 33-wide split-stencil
  block-diagonal build (`reg_split_from`), and nothing in Phase A or Phase B touches it.
- For scale: the whole likelihood at `vmap` 16 is 52.6 ms per call, so Phase B's ~1.3 ms is
  ~2.5 % of a production evaluation and the assembly's 10.0 ms is ~19 %.

So the bigger per-call lever is a **ConstantSplit-assembly rework**, not the cavity early
exit — by roughly 8× on the batched number that production pays. Phase B remains the
cheaper, lower-risk change (one loop, `stop_gradient` boundary already scoped, no numerical
reordering) and is the only one of the two with a written plan; the assembly rework has no
prompt yet and would need its own investigation of where the 33-wide build spends its time.
Both are worth filing; which goes first is a call about risk appetite, not about which is
larger.

Phase C (loop-free k-ring cavity, changes fp summation order) is now the smallest of the
three levers on the batched number and should be re-costed against these figures before it
is planned.

## Artifacts

- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_launch_latency_control.{json,png}`
- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_launch_latency.{json,png}` (shipped `e5d31d74`, chunk 4096)
- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_launch_latency_chunk256.{json,png}` (`9e9a7d50`, the same-chunk A/B)
- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk{512,1024,2048,4096}.{json,png}`
- `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64_launch_latency_control.json`
- `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64_launch_latency.json` (shipped)
- `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64_launch_latency_chunk256.json`
- `hpc/batch_gpu/submit_breakdown_imaging_delaunay_nn_a100_hst_fp64_{launch_latency_control,launch_latency,chunk_sweep}`
- `hpc/batch_gpu/submit_runtime_imaging_delaunay_nn_a100_hst_fp64_{control,launch_latency}`

The 2026-09-07 `..._walk_early_exit` rows in `delaunay_walk_early_exit.md` are left
untouched; the control leg here supersedes them for any comparison against this branch (its
merge base moved from `bcd15cd9` to `d7c96762`, and this profiling checkout adds the
split-point Sibson cut, so the row sets are not interchangeable).
