# Interferometer streaming scaling

**Status:** open
**Question:** How do the array-free `Interferometer.from_stream` path and the in-memory `apply_sparse_operator()` path scale in memory and time as the visibility count goes towards the 2e8 of a real ALMA cube?
**Pre-registered rule:** Mind contract (2026-09-30), written before the versioned runs. The epic's phases 3–5 were GO if the accumulator is linear in N_vis and fast enough to reach 2e8, and if streaming memory stays flat where the in-memory path fails under a 10 GB cap.
**Verdict:** The rule's conditions hold. The in-memory path fails at 1e6 under the cap on both arms. Streaming is linear at about 10 s per 1e6 vis, with peak RSS 1.5–1.7 GB up to 5e7. The go/no-go itself is **overtaken**, because phases 3–5 shipped while this campaign waited (PyAutoArray #597/#599/#601, PyAutoLens #761/#762; released). This page therefore records the measurement as release evidence for 2026.10.4.1, not as a decision.
**Headline:** `from_stream` 5e7 vis in 504 s at 1.69 GB peak RSS, chunk 65536 (10.1 s per 1e6 vis). In-memory `apply_sparse_operator()` fails at 1e6 under a 10 GB `RLIMIT_AS` cap. Local laptop CPU (i9-10885H, WSL2, 8 threads), no RAL job.
**Library PRs:** none in this campaign. It measures PyAutoArray #589/#593 (phases 1–2), #597/#599/#601 (phases 3–5), PyAutoGalaxy #639 and PyAutoLens #758/#761/#762, all at PyAutoArray `2026.10.4.1+2` (`9a6237f0`).
**Profiling PRs:** phase 1 (issue #368, PR pending)
**Ledger:** this page plus the result JSONs in [`results/streaming_scaling/`](../../results/streaming_scaling/). There is no separate `results/notes/` ledger.
**Mind contract:** epic `streaming-visibilities`; Pulse task `tasks/interferometer_streaming_scaling.md` (migrated from Mind `draft/research/autolens_profiling/interferometer_streaming_scaling.md`); phase 1 is Mind `active/interferometer_streaming_scaling.md`
**Next:** optional A100 rows (RAL) and a 1e8 CPU row. A library candidate, not filed: chunk the dirty-image NUFFT (`transformer.image_from`) inside `apply_sparse_operator`, which is the in-memory path's second memory wall.

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
| 1 (CPU, versioned) | 2026-10-04 | The same, versioned, plus parity | the same (now release evidence) | reproduced: 10 s/1e6 vis at chunk 65536, linear to 5e7; in-memory fails at 1e6 on both arms; parity 1.2e-9 nats at 5e5 | laptop, no RAL job | #368 (PR pending) |

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

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| autolens_profiling (phase 1, issue #368) | `scripts/interferometer/streaming_scaling/` cells and the `results/streaming_scaling/` rows | pending | n/a (profiling repo) |

## Open / parked / drafts

- 1e8 CPU row (budget-skipped) and A100 rows on RAL. Both are optional.
- Library candidate, not filed: chunk `transformer.image_from` in `apply_sparse_operator` (the dirty-image wall) so that `nufft_chunk_size` bounds the whole in-memory build.
- Per-chunk recompiles (73 per 40 chunks) as a lever for small chunks.

## Caveats

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
