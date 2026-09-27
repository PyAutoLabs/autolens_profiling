# DelaunayNN A100 series

**Status:** complete
**Question:** What does Sibson natural-neighbour DelaunayNN cost against barycentric Delaunay on the A100, and which kernel cuts help?
**Pre-registered rule:** per phase (see the table); the common contract is a same-node, same-session control (private merge-base checkout) vs feature A/B, `EXPECTED_LOG_EVIDENCE_HST` unchanged (a shift is a bug, not a re-pin), judged on the params→H prefix (`regularization_matrix_prefix_s`), never on a single prefix-difference row.
**Verdict:** PyAutoArray#531 walk early exit passed its witness by 2.1x; PyAutoArray#533 launch-latency Phase A met its unbatched half and missed its batched half (the ConstantSplit assembly, not Sibson, held ~10 ms/call), and the human pointed the next prompt at the assembly; PyAutoArray#537 assembly compaction met both halves; autolens_profiling#232 re-based the cells on AdaptSplit with the compaction intact. Phase 2 (static seed), Phase B (cavity early exit) and Phase C (k-ring cavity) deliberately not filed. Absolute rows superseded by the [A100 pixelized baseline](a100_pixelized_baseline.md).
**Headline:** DelaunayNN whole likelihood 50.04 → 40.86 ms per call at `vmap` 16 (1.22x, PyAutoArray#537 same-session A/B), A100 `euclid-ral-gpu-2` jobs 342336 → 342337; 1.42x from 58.02 ms (job 342324) across PyAutoArray#533 + PyAutoArray#537, a cross-session figure.
**Library PRs:** PyAutoArray#531 (released 2026.9.8.1), PyAutoArray#533 (released 2026.9.8.1), PyAutoArray#537 (released 2026.9.11.1).
**Profiling PRs:** #106, #221, #224, #227, #231, #233.
**Ledger:** [delaunay_nn_breakdown.md](../../results/notes/delaunay_nn_breakdown.md), [delaunay_nn_cap_audit.md](../../results/notes/delaunay_nn_cap_audit.md), [delaunay_nn_constant_split_assembly.md](../../results/notes/delaunay_nn_constant_split_assembly.md), [delaunay_nn_launch_latency.md](../../results/notes/delaunay_nn_launch_latency.md), [delaunay_walk_early_exit.md](../../results/notes/delaunay_walk_early_exit.md), [delaunay_adapt_split_regularization.md](../../results/notes/delaunay_adapt_split_regularization.md)
**Mind contract:** epic not recorded; records `complete/2026/08/delaunay-nn-laptop-gpu-profile.md`, `complete/2026/09/delaunay-nn-breakdown.md`, `delaunay-walk-early-exit.md`, `sibson-single-concatenated-walk.md`, `delaunay-nn-launch-latency.md`, `delaunay-nn-constant-split-assembly.md`, `delaunay-adapt-split-regularization.md`.
**Next:** none (superseded by [A100 pixelized baseline](a100_pixelized_baseline.md)); Phase B / Phase C and the NNLS-iteration lever are described but unfiled.

## Why this campaign

DelaunayNN (Sibson, cap 32) landed in August; the cap audit (autolens_profiling#105 / #106)
measured it at 4.30x barycentric Delaunay on a mapper-only A100 benchmark (157 vs 37 ms), while
Nautilus saw ~61 ms per evaluation for both meshes. autolens_profiling#219 asked for a
like-for-like per-step breakdown, unbatched and under `vmap`, to reconcile the two, and to fix
the Delaunay "Regularization matrix (H)" row, which timed a 19.5 MB host-to-device copy (14.4 ms
of PCIe) rather than a JIT step.

The breakdown showed the unbatched cost is latency-bound point location and the split-point
walk, which the human then attacked one lever at a time in PyAutoArray: the Delaunay walk
(PyAutoArray#530), the Sibson launch count (PyAutoArray#532, folding in the
`sibson_single_concatenated_walk` draft), and the ConstantSplit assembly (PyAutoArray#536). Each
phase's witness was written into its prompt before the A100 run (`## Original prompt` of each
record). At the PyAutoArray#537 close-out the human ruled that the cells should profile AdaptSplit, as
production does (autolens_profiling#232).

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| Cap audit | 2026-08-13 | Which static cap is safe, and does the DelaunayNN overhead grow on GPU? | not recorded (decision: smallest tested cap with no overflow in local and CI audits) | keep cap 32 (16 overflows; 24 on the platform boundary); A100 cap-32 0.1573 s vs Delaunay 0.0366 s = 4.30x; RTX 2060's 5.44x an unstable-clock artifact | 334949, laptop | #106 |
| Breakdown + H-row fix | 2026-09-05 | Per-step DelaunayNN vs Delaunay, unbatched and `vmap` 16; reconcile benchmark vs Nautilus | not recorded (witness: JSONs with in-JIT H row, four-way split, `vmap` timings) | step-sum 256.4 vs 117.4 ms (2.18x); hybrid `vmap` 16 step-sum 98.4 vs 78.2 ms (1.26x, ~20 ms/call, 3/4 of it the split-point H walk); H row 14.4 ms PCIe → 10.4 ms JIT | 342277–342281 (342280 FAILED) | #221 |
| Autotune A/B | 2026-09-05 | Is the F row's 4.8 → 25.6 ms move the autotuner? | not recorded | F 25.63 → 4.90 ms at level 4; later traced to an un-tuned Triton GEMM ([A100 pixelized baseline](a100_pixelized_baseline.md)) | 342282, 342283 | #221 |
| Walk early exit | 2026-09-07 | Early-exit `while_loop` walk, unchunked, one concatenated locate | Delaunay params→H drops ≥ 18 ms unbatched, pin unchanged, walk parity and FD pass (amended mid-task from "Tri+interp 26.6 → < 8 ms") | PASSED: Delaunay params→H 45.15 → 7.23 ms (6.25x), full pipeline 93.46 → 60.36 ms; DelaunayNN 178.97 → 144.79 ms (1.24x); Phase 2 not filed | 342315–342320 | PyAutoArray#531, #224 |
| Sibson single walk | 2026-09-08 | Concatenate Sibson's two walk calls | params→H materially below 144.789 / 24.424 ms, pin unchanged | folded into PyAutoArray#533 as cut A.2; record retired at that close-out | see next row | PyAutoArray#533 |
| Launch latency (Phase A) | 2026-09-08 | Unroll the candidate loop, one concatenated Sibson pass, sweep `SIBSON_QUERY_CHUNK` | params→H ≤ 60 ms unbatched and ≤ 11 ms/call at `vmap` 16; chunk rule A.3 (largest chunk under ~50 % VRAM and fastest; 2048 unless 4096 wins by > 15 %) | HALF MET: 143.90 → 28.21 ms unbatched (5.10x); 24.32 → 16.44 ms/call at `vmap` 16 (assembly ~10.0 ms untouched); whole likelihood 201.32 → 76.32 ms single JIT; chunk 4096 shipped (VRAM clause vacuous, 14.0 % margin) | 342321–342325, 342329, 342330 | PyAutoArray#533, #227 |
| ConstantSplit assembly | 2026-09-08 | Can the 33-wide split-stencil assembly be cut? | assembly < 3 ms/call at `vmap` 16 (from 10.0), params→H < 11 ms/call (from 16.4), pin unchanged | MET: assembly 10.03 → 0.84 ms/call (12.0x), params→H 16.42 → 7.26 ms (2.26x), whole likelihood 50.04 → 40.86 ms/call; pin bit-identical | 342334–342338 (priced by 342331, 342332) | PyAutoArray#537, #231 |
| AdaptSplit re-base | 2026-09-08 | Profile the regularization production uses | every re-based pin passes on both legs; AdaptSplit params→H within ~1 ms/call of the 7.26 ms ConstantSplit row | MET: 7.277 vs 7.250 ms/call; whole likelihood +11.8 % DelaunayNN (40.92 → 45.76 ms/call), +7.0 % Delaunay (39.68 → 42.47), all in the NNLS reconstruction | 342340–342347 | #233 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#531 | early-exit `while_loop` Delaunay walk, chunked seed only, one concatenated locate (issue PyAutoArray#530) | `d7c96762` | 2026.9.8.1 |
| PyAutoArray#533 | GPU-gated candidate unroll, one concatenated Sibson pass, `SIBSON_QUERY_CHUNK` 4096 (issue PyAutoArray#532) | `bc113fb5` | 2026.9.8.1 |
| PyAutoArray#537 | split-stencil compaction to width 12 + 256-row wide supplement + NaN on overflow (issue PyAutoArray#536) | `47a00e8c` | 2026.9.11.1 |

Test-workspace companions (not library PRs): autolens_workspace_test#306, #307, #309.

## Open / parked / drafts

- Phase B (cavity `fori_loop` → `while_loop` early exit): unfiled; re-costed at ≈ 1.3 ms/call at `vmap` 16 (≈ 3 % of an evaluation) after PyAutoArray#537.
- Phase C (loop-free k-ring cavity, changes fp summation order): unfiled; must be re-costed against the post-PyAutoArray#537 6.46 ms Sibson share.
- NNLS iteration count under AdaptSplit (warm starts, preconditioning, per-lane exit): described in the autolens_profiling#232 record, not filed there; see [Certified positive solver](certified_positive_solver.md).
- Probe-recommendation cap for `resolve_vmap_batch` (the cuFFT trap): recommended in the breakdown ledger, no draft found.
- No Mind draft is open for this series.

## Caveats

- **Absolute rows are superseded.** Every row here predates the 2026-09-10 fresh-cache set (jobs 342617–342628); the canonical DelaunayNN cost is 46.44 ms/call at `vmap` 16 dense, AdaptSplit, on [A100 pixelized baseline](a100_pixelized_baseline.md). The rows here stand for their same-session A/B ratios.
- **Autotune state differs across the series.** All rows ran `--xla_gpu_autotune_level=0`. The 2026-09-05 rows (342277/342278/342281) carry the un-tuned Triton F at 25.6 ms, which inflates both meshes' step-sums, so the 2.18x and 1.26x ratios were not re-measured under the fixed flag. Jobs 342315, 342316, 342321 and 342322 read a `gpu-2` autotune cache seeded by 342282/342283 (F 4.80–4.83 ms); the later rows' 4.8 ms F is consistent with the same state but not verified per job; only autolens_profiling#232's jobs 342340–342347 carry `--xla_gpu_enable_triton_gemm=false`. Compare within a session, not across ledgers.
- **Prefix-difference rows are attribution, not work.** "Triangulation + interpolation", "H", "Split-point Sibson", the differenced assembly cell and "Mapping matrix" shift meaning when a change moves work across a compile boundary (H 13.555 → 0.225 ms on PyAutoArray#531 and Split-point Sibson 35.4 → ~0 on PyAutoArray#533 are re-attribution). Negative cells (−0.605, −1.599, −2.510 ms) are artifacts. Only the params→H prefix is quotable across versions.
- **Launch latency is inferred, not measured.** PyAutoArray#533's prompt derived ~1.1 µs per launch from a counted 1,244 launches per chunk against the measured prefix time; no profiler trace separates launch latency from kernel time. The shipped leg's first-measured interpolator prefix (8.944 ms at `vmap` 16) carries warm-up; read the Sibson share as ≈ 6.4 ms.
- **Step-sums are upper bounds.** The breakdown's hybrid `vmap` 16 step-sum (98.4 ms) sits 16 % above the fused runtime (82.2 ms); treat step-sums as attribution only.
- **Cap values.** Cap 32 rests on a local ensemble maximum of 27 main natural neighbours and a platform-sensitive cap-24 result; the HST production cell occupies at most 11 cavity / 13 neighbours. Until autolens_workspace_test#285 (2026-08-28, `complete/2026/08/delaunay-nn-env-header.md`) CI ran `delaunay_nn_caps.py` with JAX off and small datasets; whether the cap audit's CI reruns were affected is not verified. Raising a cap costs the assembly quadratically.
- **Laptop rows are not quotable.** The RTX 2060 Max-Q baseline spanned 1.58x across warm calls; its 5.44x overhead is an artifact. Laptop CPU rows are local-only.
- **AdaptSplit unbatched offset.** The AdaptSplit leg ran first in every autolens_profiling#232 pair and shows +4.9 ms on a regularization-free block unbatched; no reversed-order repeat was run. Quote the `vmap` 16 columns.
- **Batch 16 is the ceiling on the dense path.** `vmap` 64 OOMs (46.86 GiB allocation, job 342279); the probe-recommended batch 64 fails cuFFT plan creation (job 342280).
- **PyAutoArray#537's wide-row supplement costs ~0.26 ms/call** on geometries that never use it.
- **The shared RAL install moved between sessions** (PyAutoFit `2680b32d` → `6331b800` → `68ff9bd5`); within-session A/Bs are clean, cross-session absolutes carry that caveat.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the eight phases from the August cap audit to the AdaptSplit re-base, with each
witness as written in its prompt and the job ids from the six ledgers. Resolved: library PRs and
releases from the release sheet (PyAutoArray#531 / PyAutoArray#533 at 2026.9.8.1, PyAutoArray#537 at 2026.9.11.1), profiling PRs
from the records, the headline re-pointed from the superseded 2.18x to the PyAutoArray#537 same-session `vmap` 16
whole-likelihood change. Open: Phase B, Phase C and the NNLS lever remain unfiled.
