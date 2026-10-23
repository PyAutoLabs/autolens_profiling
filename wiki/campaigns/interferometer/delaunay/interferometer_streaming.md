# Interferometer streaming scaling

**Status:** open
**Question:** How do the array-free `Interferometer.from_stream` path and the in-memory `apply_sparse_operator()` path scale in memory and time as the visibility count goes towards the 2e8 of a real ALMA cube?
**Pre-registered rule:** Mind contract (2026-09-30), written before the versioned runs. The epic's phases 3–5 were GO if the accumulator is linear in N_vis and fast enough to reach 2e8, and if streaming memory stays flat where the in-memory path fails under a 10 GB cap.
**Verdict:** The rule's conditions hold. The in-memory path fails at 1e6 under the cap on both arms. Streaming is linear at about 10 s per 1e6 vis, with peak RSS 1.5–1.7 GB up to 5e7. The go/no-go itself is **overtaken**, because phases 3–5 shipped while this campaign waited (PyAutoArray #597/#599/#601, PyAutoLens #761/#762; released). This page therefore records the measurement as release evidence for 2026.10.4.1, not as a decision. Phase 2 (2026-10-17, release 2026.10.17.1, RAL) adds the 1e8 row on CPU and the A100 rows. Streaming stays linear and flat in memory to 1e8 on both. Uncapped, the in-memory path is faster per visibility on CPU, but it needs about 7.5 kB per visibility of host RSS (119 GB at 1.6e7). On the A100 it fails at 4e6.
**Headline:** phase 2 (2026.10.17.1): `from_stream` 1e8 vis in 707.5 s at 1.78 GB peak RSS on one RAL CPU node (Xeon 8490H, 8 threads). On an A100 it takes 157.3 s at 1.12 GB host RSS and 1.48 GB device peak. Both use chunk 65536 and fp64; compile is about 3 s of either. In memory, the A100 fails at 4e6 (46.2 GB device peak against a 63.8 GB JAX limit). Uncapped RAL CPU builds 1.6e7 in 68.3 s at 119.3 GB peak RSS. Phase 1: `from_stream` 5e7 vis in 504 s at 1.69 GB peak RSS, chunk 65536 (10.1 s per 1e6 vis). In-memory `apply_sparse_operator()` fails at 1e6 under a 10 GB `RLIMIT_AS` cap. Local laptop CPU (i9-10885H, WSL2, 8 threads), no RAL job.
**Library PRs:** none in this campaign. It measures PyAutoArray #589/#593 (phases 1–2), #597/#599/#601 (phases 3–5), PyAutoGalaxy #639 and PyAutoLens #758/#761/#762, all at PyAutoArray `2026.10.4.1+2` (`9a6237f0`).
**Profiling PRs:** phase 1 #375 (issue #368, merged 2026-10-04); phase 2 (issue #368 reopened, PR pending)
**Ledger:** this page plus the result JSONs in [`results/streaming_scaling/`](../../../../results/misc/streaming_scaling/). There is no separate `results/notes/` ledger.
**Mind contract:** epic `streaming-visibilities`; Pulse task `tasks/interferometer_streaming_scaling.md` (migrated from Mind `draft/research/autolens_profiling/interferometer_streaming_scaling.md`); phase 1 is Mind `active/interferometer_streaming_scaling.md`
**Next:** none owed by the scaling question. A 2e8 row is within reach (predicted about 24 min CPU, about 5 min A100) if a reviewer wants it measured rather than extrapolated. A library candidate, not filed: chunk the dirty-image NUFFT (`transformer.image_from`) inside `apply_sparse_operator`, which is the in-memory path's second memory wall.

## Why this campaign

PyAutoLabs Discussion #13 asked about a 2e8-visibility ALMA cube. In-house interferometer
datasets are at most about 1.1e5 visibilities. The array-free dataset adds a second dataset
kind that every interferometer code path branches on, and it earns that cost only if it makes
a real difference at the visibility counts real data reach. A scratchpad benchmark
(`bench_stream.py`, 2026-09-30) gave a GO, but it was never versioned. This campaign ports it
into re-runnable cells (`scripts/interferometer/streaming_scaling/`) so the evidence can be
regenerated per release.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 0 (scratchpad) | 2026-09-30 | Is the accumulator linear and flat in memory, and where does in-memory fail? | GO if linear, fast enough for 2e8 and flat RSS | GO: 11 s/1e6 vis, 1.5–1.8 GB; in-memory OOM at 1e6 | laptop, unversioned | none |
| 1 (CPU, versioned) | 2026-10-04 | The same, versioned, plus parity | the same (now release evidence) | reproduced: 10 s/1e6 vis at chunk 65536, linear to 5e7; in-memory fails at 1e6 on both arms; parity 1.2e-9 nats at 5e5 | laptop, no RAL job | #375 (issue #368) |
| 2 (RAL CPU 1e8 + A100) | 2026-10-17 | The 1e8 row and the A100 rows, with streaming and in-memory side by side on the same hardware, memory and seconds in separate columns, and compile split from execution | none new: the phase-0 conditions, read per host | streaming linear and flat to 1e8 on both: CPU 7.04 s/1e6 vis at 1.6–1.8 GB, A100 1.53 s/1e6 vis at 1.1 GB host and 1.48 GB device; in memory, A100 fails at 4e6, CPU fails at 2.5e5 under the 10 GB cap and builds 1.6e7 uncapped at 119 GB; parity 6.1e-8 nats at 4e6 (CPU, uncapped) | RAL 412522 (ral CPU), 412523 (A100) | phase 2 PR (issue #368) |

## Results (phase 1, PyAutoArray 2026.10.4.1+2)

Timing, memory, correctness and hardware are kept as separate axes below. No speed score
is formed.

### Hardware and settings

Intel i9-10885H laptop under WSL2. The OS sees 8 CPUs (`nproc_all` 8) and 15.6 GB RAM. JAX
runs on CPU (`JAX_PLATFORMS=cpu`) with `OMP/OPENBLAS/MKL_NUM_THREADS=8`. The session shell
had pinned these to 1, and they were overridden for this run. Load average was 1.6 at the
first import and 3.9 at the last write, because the laptop was shared with an IDE server and
other agent sessions. Read the timings as indicative, not quiet-host numbers.

Every row ran in a fresh child process with a 10 GB `RLIMIT_AS` (virtual address space) cap
and a per-child timeout. The dataset is synthetic and seeded per chunk. It uses a 400×400
circular mask at 0.05″/pix (125 676 pixels) and `TransformerNUFFT`. The full provenance
block, with library SHAs and dependency versions, is in each JSON.

### Timing and memory: streamed (`from_stream`)

Wall time excludes imports. Peak RSS is the child's `ru_maxrss`.

| N_vis | chunk | wall [s] | s per 1e6 vis | peak RSS [GB] |
|---|---|---|---|---|
| 1e6 | 65536 | 17.5 | 17.5 | 1.47 |
| 4e6 | 65536 | 43.8 | 10.9 | 1.48 |
| 1.6e7 | 65536 | 180.7 | 11.3 | 1.48 |
| 5e7 | 65536 | 504.5 | 10.1 | 1.69 |
| 1e8 | 65536 | skipped (wall budget: predicted 1005 s after 1552 s elapsed) | — | — |
| 1e6 | 4096 | 43.6 | 43.6 | 0.78 |
| 4e6 | 4096 | 149.2 | 37.3 | 0.84 |
| 1.6e7 | 4096 | 608.6 | 38.0 | 0.89 |

Least-squares fits of `wall = a + b·N`:

- **Chunk 65536:** a = 9.9 s, b = 9.96 s per 1e6 vis, largest relative residual 14 % (the 1e6 row, where the fixed cost dominates).
- **Chunk 4096:** a = 2.2 s, b = 37.8 s per 1e6 vis, residual 8 %.

Chunk 4096 is 3.8× slower per visibility and holds about 0.6 GB less.

### Timing and memory: in memory (`apply_sparse_operator()`)

The grid ascends and stops at the first failure.

| N_vis | arm | wall [s] | peak RSS [GB] | outcome |
|---|---|---|---|---|
| 1e5 | defaults | 2.9 | 2.06 | OK |
| 2.5e5 | defaults | 3.4 | 4.55 | OK |
| 5e5 | defaults | 5.3 | 5.81 | OK |
| **1e6** | defaults | 3.3 | 5.53 | **FAILED_MEM**: 1.568 GB allocation in `nufft_precision_operator_via_nufft_from` |
| 1e5 | `nufft_chunk_size=65536` | 6.4 | 2.15 | OK |
| 2.5e5 | `nufft_chunk_size=65536` | 7.6 | 4.50 | OK |
| 5e5 | `nufft_chunk_size=65536` | 8.6 | 5.79 | OK |
| **1e6** | `nufft_chunk_size=65536` | 11.8 | 5.63 | **FAILED_MEM**: 1.568 GB allocation in `transformer.image_from` (dirty image) |

The second arm is new in this phase. Chunking the precision-operator builder removes its
gather buffer, and the in-memory path then fails at the same N_vis one step later, in the
un-chunked dirty-image NUFFT that `apply_sparse_operator` computes over every visibility.
That transient is 1568 B per visibility (1.568 GB at 1e6, 6.272 GB at 4e6). The in-memory
path therefore has two memory walls, and only one of them has a user-facing knob today.

### Correctness: log_evidence parity

The test inversion is a 20×20 `RectangularUniform` sparse inversion with `Constant(1.0)`
regularization and `InversionInterferometerSparse` on both arms.

| N_vis | streamed log_evidence | in-memory log_evidence | \|Δ\| [nats] | relative |
|---|---|---|---|---|
| 5e5 | -922920.6794568285 | -922920.6794568297 | 1.2e-9 | 1.3e-15 |
| 4e6 | -7355872.521471734 (3.65 GB) | FAILED_MEM (6.272 GB allocation in `transformer.image_from`) | not measurable under the cap | — |

The contract asked for parity at 4e6. The in-memory arm cannot be built there under the
10 GB cap, which is the result of the memory table above. Parity is therefore recorded at
5e5, the largest N_vis where both arms build. The 4e6 streamed evidence is kept as the
reference for a future capped-higher run.

### 2e8 extrapolation

- **Streamed time.** Chunk 65536 predicts 9.9 + 9.96 × 200 ≈ 2000 s, about **33 min**. Chunk 4096 predicts about 2.1 h. The scratchpad estimate was 37 min.
- **Streamed memory.** Peak RSS is 1.47–1.48 GB from 1e6 to 1.6e7 and 1.69 GB at 5e7. If the 1.6e7 → 5e7 growth (about 6 B/vis) is real and continues, 2e8 reaches about 2.6 GB. If it is allocator settling, it stays near 1.7 GB. **1.7–2.6 GB** in either case. The skipped 1e8 row is the cheapest way to tell these apart.
- **In-memory memory.** The visibility arrays alone are 48 B/vis (`uv` 16 B, data 16 B, noise 16 B), which is about 9.6 GB at 2e8. The dirty-image transient is 1568 B/vis, about 314 GB, and the default operator builder makes a transient of the same size. Peak RSS measured 5.8 GB at 5e5 already. The contract's estimate of about 96 B/vis plus temporaries understated the temporaries by more than an order of magnitude. Without chunking the transformer, 2e8 in memory is out of reach on any host.

### cProfile (chunk 4096, 1.6384e5 vis = 40 chunks, 10.4 s)

The profile was triggered because every OK row exceeds 5 s per 1e6 vis. The table lists the
top 10 by own time.

| function | ncalls | tottime [s] | share |
|---|---|---|---|
| `numpy.asarray` (device→host copies) | 8822 | 3.69 | 35.6 % |
| `compiler.py:backend_compile_and_load` (XLA compiles) | 73 | 1.59 | 15.4 % |
| `grid_2d_util.grid_2d_slim_via_mask_from` | 80 | 0.37 | 3.6 % |
| `jax dispatch.apply_primitive` | 4880 | 0.21 | 2.0 % |
| `mask_2d_util.native_index_for_slim_index_2d_from` | 240 | 0.19 | 1.9 % |
| `mlir.make_ir_context` | 73 | 0.18 | 1.7 % |
| `numpy stack` | 360 | 0.16 | 1.5 % |
| `jax ufunc_api.__call__` | 4317 | 0.15 | 1.5 % |
| `array_2d_util.array_2d_via_indexes_from` | 160 | 0.14 | 1.4 % |
| `builtins.isinstance` | 327247 | 0.09 | 0.8 % |

By cumulative time, the per-chunk `nufft_precision_operator_from` (the type-1 NUFFT,
`nufft2d1` ×120) is 4.76 s, or 46 %. The cost per chunk is fixed: about 73 XLA recompiles
across 40 chunks, the same as the scratchpad, plus the device→host copies of each chunk's
result. That fixed cost is why the large chunk is 3.8× cheaper per visibility.

## Results (phase 2, release 2026.10.17.1, RAL)

### Hardware, settings and provenance

- **CPU:** RAL job 412522, partition `ral`, node `euclid-ral-compute-10-5` (Intel Xeon Platinum 8490H), `--cpus-per-task=8` with `OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=8` and `JAX_PLATFORMS=cpu`. The node was otherwise idle at submit (load 0.0). The memo after-measurement (job 412526, one thread) ran beside it from 15:00, and load was 3.0 during the in-memory and parity cells.
- **A100:** RAL job 412523, partition `gpu`, `--gres=gpu:1 --exclusive`, node `euclid-ral-gpu-1` (NVIDIA A100 80GB PCIe; host AMD EPYC 7702, 8 host threads). The release-sweep reference node `euclid-ral-gpu-2` was held by another A100 job. `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so the device column is the build's own `peak_bytes_in_use`. The JAX device limit was 63.8 GB, the default 75 % of 80 GB.
- **Both:** fp64 (`jax_enable_x64` true in every child). The five libraries come from a private checkout at the `2026.10.17.1` tag (PyAutoArray `e59d5fa9`, PyAutoLens `c90eceb0`); the shared mirror was not used. The JAX persistent compile cache was disabled. The dataset is the phase-1 synthetic set (seed 1234, 400×400 mask, `TransformerNUFFT`). The venv carries jax 0.10.2, which the release's own pins exclude (`!=0.10.*`); see Caveats.
- **Compile vs execution.** Each child sums JAX's monitoring compile events (trace, lowering, XLA backend compile) inside the timed build; `exec = wall − compile`. These are new columns in phase 2.
- **Memory.** Host peak RSS (`ru_maxrss`) and device peak are separate columns. The CPU in-memory cell ran twice: under the phase-1 10 GB `RLIMIT_AS` cap (`*_hpc_ral_cpu_fp64_*`), and with no cap, bounded only by the SLURM `--mem=128gb` (`*_hpc_ral_cpu_fp64_nocap_*`). CUDA cannot initialise under `RLIMIT_AS`, so the A100 cells run uncapped.

### Streamed (`from_stream`), chunk 65536

| N_vis | CPU wall [s] | CPU compile / exec [s] | CPU s per 1e6 | CPU peak RSS [GB] | A100 wall [s] | A100 compile / exec [s] | A100 s per 1e6 | A100 host RSS [GB] | A100 device peak [GB] |
|---|---|---|---|---|---|---|---|---|---|
| 1e6 | 15.43 | 3.27 / 12.16 | 15.4 | 1.62 | 6.62 | 3.65 / 2.97 | 6.6 | 1.10 | 1.48 |
| 4e6 | 32.83 | 3.06 / 29.77 | 8.2 | 1.61 | 10.51 | 3.59 / 6.92 | 2.6 | 1.10 | 1.48 |
| 1.6e7 | 117.64 | 3.17 / 114.47 | 7.4 | 1.65 | 27.47 | 3.54 / 23.93 | 1.7 | 1.10 | 1.48 |
| 5e7 | 368.08 | 3.27 / 364.81 | 7.4 | 1.81 | 79.20 | 3.81 / 75.39 | 1.6 | 1.12 | 1.48 |
| **1e8** | **707.50** | 3.13 / 704.37 | 7.1 | **1.78** | **157.32** | 3.60 / 153.72 | 1.6 | 1.12 | 1.48 |

Chunk 4096 (1e6 / 4e6 / 1.6e7): CPU 29.8 / 105.4 / 410.5 s at 0.88–1.06 GB; A100 23.9 / 84.9 / 395.0 s at 1.06–1.07 GB host and 0.33 GB device.

Least-squares `wall = a + b·N`:

| host | chunk | a [s] | b [s per 1e6 vis] | largest residual | 2e8 extrapolation |
|---|---|---|---|---|---|
| RAL CPU | 65536 | 7.7 | 7.04 | 9 % | 1415 s (24 min) |
| RAL CPU | 4096 | 4.2 | 25.4 | 1 % | 5083 s |
| A100 | 65536 | 4.0 | 1.53 | 16 % | 309 s (5 min) |
| A100 | 4096 | −7.4 | 25.1 | 26 % | 5003 s |

Compile is flat: about 3 s and **119 XLA compiles in every streamed child**, whatever N_vis or the chunk. It does not grow with the number of chunks. The phase-1 reading of the chunk-4096 penalty as per-chunk recompilation therefore does not hold on this stack. The penalty is execution: about 25 s per 1e6 vis on both devices, against 7.0 (CPU) and 1.5 (A100) at chunk 65536. On the A100 the small chunk loses all of the device's advantage.

The 1e8 row settles the phase-1 memory question. CPU peak RSS is 1.81 GB at 5e7 and 1.78 GB at 1e8, so it is flat, not growing about 6 B per visibility. The 2e8 streamed RSS is therefore about 1.8 GB on CPU and 1.1 GB host plus 1.5 GB device on the A100.

### In memory (`apply_sparse_operator()`), side by side

| N_vis | RAL CPU, 10 GB cap | RAL CPU uncapped: wall [s] (compile / exec) | RAL CPU uncapped: peak RSS [GB] | A100: wall [s] (compile / exec) | A100: host RSS / device peak [GB] |
|---|---|---|---|---|---|
| 1e5 | OK, 2.19 GB | 2.58 (2.06 / 0.52) | 2.16 | 3.78 (2.41 / 1.37) | 1.09 / 1.90 |
| 2.5e5 | **FAILED_MEM** (784 MB allocation in `_type1_real_grid_from`, 3.52 GB RSS) | 2.91 (2.25 / 0.66) | 4.49 | 3.87 (2.43 / 1.44) | 1.23 / 4.21 |
| 5e5 | — | 3.71 (2.32 / 1.39) | 6.69 | 3.89 (2.40 / 1.49) | 1.52 / 8.07 |
| 1e6 | — | 5.60 (2.54 / 3.06) | 8.45 | 4.07 (2.39 / 1.68) | 2.10 / 15.76 |
| 4e6 | — | 17.32 (2.56 / 14.76) | 30.61 | **FAILED_MEM** (5.84 GiB allocation in `_type1_real_grid_from`) | 3.30 / 46.15 |
| 1.6e7 | — | 68.34 (2.65 / 65.69) | **119.27** | — | — |

These are the default-arm rows. The `nufft_chunk_size=65536` arm moves the failure site to `transformer.image_from` (the dirty image, as in phase 1) at the same N_vis on both devices. Uncapped on CPU it costs 8.67 s and 8.30 GB at 1e6 and 76.25 s and 119.53 GB at 1.6e7, about the same memory as the default arm.

Matched on hardware, precision, dataset and SHAs:

- **CPU, uncapped.** In memory is faster per visibility than streaming: 68.3 s against 117.6 s at 1.6e7. It holds 119.3 GB against 1.65 GB, about 72× the RSS and about 7.5 kB per visibility. At 2e8 that is about 1.5 TB, beyond any RAL node (0.9–1.0 TB). Streaming is the only route there, and it does not need a large node.
- **CPU, capped.** Under the same 10 GB `RLIMIT_AS` the 8490H node fails at 2.5e5, where the laptop failed at 1e6. Virtual address space is host-dependent (JAX/XLA thread pools and arenas), so a cap-based failure point is a property of the host, not of the library. The uncapped RSS column is the portable figure.
- **A100.** In memory fails at 4e6 at a 46.2 GB device peak, asking for 5.84 GiB more under the 63.8 GB JAX limit. Streaming reaches 1e8 at a 1.48 GB device peak.

### Correctness: log_evidence parity

| host | N_vis | streamed | in memory | \|Δ\| [nats] | build streamed / in memory [s] | peak RSS streamed / in memory [GB] |
|---|---|---|---|---|---|---|
| RAL CPU (uncapped) | 5e5 | -922920.6794568284 | -922920.6794568296 | 1.2e-9 | 7.08 / 6.55 | 3.44 / 7.08 |
| RAL CPU (uncapped) | **4e6** | -7355872.521471734 | -7355872.521471673 | **6.1e-8** (relative 8.4e-15) | 31.72 / 21.45 | 3.41 / 30.80 |
| A100 | 5e5 | -922920.6794568285 | -922920.6794568297 | 1.2e-9 | 5.89 / 6.76 | 3.70 host / 1.48 device; 4.24 host / 7.79 device |
| A100 | 4e6 | -7355872.521471734 | FAILED_MEM (`transformer.image_from`) | not measurable | 10.6 / — | — |

The contract's 4e6 parity is measured for the first time, on uncapped CPU. The streamed 4e6 evidence matches the phase-1 laptop value to every printed digit.

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| autolens_profiling #375 (phase 1, issue #368) | `scripts/interferometer/delaunay/streaming_*.py` cells and the `results/streaming_scaling/` rows | 2026-10-04 | n/a (profiling repo) |
| autolens_profiling phase 2 PR (issue #368) | `--hardware` result tag, compile / exec split and device memory in the harness; `hpc/batch_cpu/submit_streaming_scaling_ral_cpu_fp64`, `hpc/batch_gpu/submit_streaming_scaling_a100_fp64`; the `*_hpc_ral_cpu_fp64*` and `*_hpc_a100_fp64*` rows | pending | n/a (profiling repo) |

## Open / parked / drafts

- 2e8 rows (optional): predicted 24 min on CPU and 5 min on the A100. Measuring them would replace the extrapolation.
- Library candidate, not filed: chunk `transformer.image_from` in `apply_sparse_operator` (the dirty-image wall) so that `nufft_chunk_size` bounds the whole in-memory build.
- Small-chunk cost (phase 2): execution of about 25 s per 1e6 vis at chunk 4096 on both devices, with compile flat at about 3 s. The lever is per-chunk host overhead (device→host copies, per-chunk NumPy), not recompilation.

## Caveats

- Phase 2 venv: RAL's shared venv carries jax/jaxlib 0.10.2. The 2026.10.17.1 PyAutoNerves pins exclude `0.10.*` (`!=0.10.*,!=0.11.0`), so these rows are the release's source on a JAX the release would not install. The library does not refuse at import. A fresh `pip install` would resolve another JAX.
- Phase 2 CPU in-memory and parity cells shared the node with a one-thread job from 15:00 (load about 3 on 236 cores). The streamed rows to 1e8 ran first, at load 0.0–3.0.
- Phase 2 A100 ran on `euclid-ral-gpu-1`, not the release-sweep reference `euclid-ral-gpu-2`, which was held by another job. Read these as campaign rows, not dashboard trend points.

- The host is a shared laptop (load average 1.6 → 3.9 over the run). Timings are indicative.
  The 2026-09-30 scratchpad agrees within about 20 % on every chunk-65536 row.
- `RLIMIT_AS` caps virtual address space, so a failure "under 10 GB" happens at a peak RSS of
  5.5–5.6 GB. A host with 10 GB of free RSS might build 1e6 in memory. The 1568 B/vis
  transient still rules out anything near 2e8.
- The data are synthetic, with uniform uv coverage and constant noise. The cost depends on
  N_vis, the mask and the chunk, not on the data values.
- The release suffix `+2` means two non-source commits (a hook propagation and a merge) past
  the `2026.10.4.1` tag on every library.

## Journal

### 2026-10-17 — phase 2, RAL CPU 1e8 and A100 rows (2026.10.17.1)

The human authorised RAL compute. The phase-1 cells gained a `--hardware` tag, a compile / exec split and device memory columns, and ran from a private 2026.10.17.1 checkout on one `ral` node (job 412522) and one exclusive A100 (job 412523). Streaming is linear and flat to 1e8: 707.5 s at 1.78 GB on CPU and 157.3 s on the A100. That retires the phase-1 2e8 memory range (1.7–2.6 GB) in favour of about 1.8 GB. Compile is about 3 s and 119 compiles, flat in N and chunk, so the small-chunk penalty is execution, not recompilation. Uncapped, CPU in-memory builds 1.6e7 at 119 GB (faster than streaming, at 72× the memory). The A100 in-memory build fails at 4e6. The 10 GB cap fails at 2.5e5 on this host, against 1e6 on the laptop. Parity at 4e6 is now measured: 6.1e-8 nats. Issue #368 reopened.

### 2026-09-30 — scratchpad GO

`bench_stream.py` was run in a CLI session. The results were 16 / 52 / 174 / 539 s at 1e6 /
4e6 / 1.6e7 / 5e7 (chunk 65536) and 1.5–1.8 GB, with in-memory OOM at 1e6. The verdict was GO
on phases 3–5, and the table went on the epic ledger.

### 2026-10-04 — phase 1, versioned CPU rows

The benchmark was ported into `scripts/interferometer/streaming_scaling/` (accumulate,
in_memory, parity, plot_scaling) and run in about 30 min on the laptop at 8 threads. The
scratchpad scaling was reproduced. Two things are new. First, a `nufft_chunk_size` in-memory
arm, which exposes the dirty-image NUFFT as a second memory wall. Second, a parity check:
1.2e-9 nats at 5e5, because 4e6 in-memory does not fit under the cap. Phases 3–5 had already
shipped, so the result is filed as release evidence for 2026.10.4.1. The 1e8 row was skipped
on the 40-min wall budget.
