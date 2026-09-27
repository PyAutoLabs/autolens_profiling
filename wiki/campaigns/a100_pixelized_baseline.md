# A100 pixelized baseline

**Status:** complete
**Question:** What is the A100 fp64 pixelized likelihood reference cost, dense vs sparse, for rectangular, Delaunay and DelaunayNN?
**Pre-registered rule:** autolens_profiling#241 witness: fresh-cache, same-node A100 fp64 rows dated after PyAutoNerves#162 with `--xla_gpu_enable_triton_gemm=false` in `device.xla_flags`, dense "Curvature matrix (F)" under 6 ms on every dense row, w-tilde-native sparse breakdown rows, per-call-at-`vmap`-16 numbers for every mesh. Parent PyAutoNerves#161 witness: a fresh-cache Delaunay breakdown with F < 6 ms and `curvature_matrix_jit_compile` < 0.2 s.
**Verdict:** every #241 clause met, on 12 legs rather than the 8 the witness named (DelaunayNN added); dense F 4.824–4.830 ms on all three meshes; sparse is the memory lever, not the speed lever (+6–14 % per call at `vmap` 16, ~7x less device memory); the ~37 ms reconstruction is the target, refined by #243 to 21–22 NNLS PDIP iterations at ~1.7 ms each. XLA cure: the human chose arm E (keep autotune level 0, add Triton GEMM off) over the prompt's proposed "drop level 0".
**Headline:** dense per call at `vmap` 16: rectangular 32.08 / Delaunay 42.53 / DelaunayNN 46.44 ms, A100 `euclid-ral-gpu-2` jobs 342617 / 342618 / 342619 (fresh per-job cache, Triton GEMM off).
**Library PRs:** PyAutoNerves#162 (merge `0e7163bc`, released 2026.9.11.1); consumes PyAutoArray#531 / #533 / #537 (see [DelaunayNN A100 series](delaunay_nn_a100_series.md)).
**Profiling PRs:** #230 (merge `fb532c01`), #242 (merge `c8b6058`), #244 (merge `2a4216a`).
**Ledger:** [a100_pixelized_baseline_2026_09.md](../../results/notes/a100_pixelized_baseline_2026_09.md), [xla_autotune_triton_gemm.md](../../results/notes/xla_autotune_triton_gemm.md)
**Mind contract:** epic not recorded; `complete/2026/09/xla-triton-gemm-off.md` (parent), `complete/2026/09/a100-pixelized-baseline.md`, `complete/2026/09/reconstruction-row-split.md`.
**Next:** none; the matrix-free comparison this set was built for is answered no-go on [Matrix-free pixelized](matrix_free_pixelized.md).

## Why this campaign

On 2026-09-05 the DelaunayNN breakdown's autotune A/B found the fp64 curvature GEMM 5.2x slower
under PyAutoNerves' `--xla_gpu_autotune_level=0` default (25.63 vs 4.90 ms), but four later
level-0 jobs measured F at 4.8 ms. The human asked for reconfirmation before any library edit
(PyAutoNerves#161); a five-arm probe separated the flag from the node's persistent autotune
cache, and the cheaper cure (Triton GEMM off) shipped as PyAutoNerves#162
(`complete/2026/09/xla-triton-gemm-off.md`).

With the flag and the Delaunay speed-ups merged, the human asked for "the exact numbers we need
to then compare against the matrix free approach" (autolens_profiling#241): one same-node,
fresh-cache A100 grid, dense and sparse, with the sparse breakdown timing the real w-tilde
steps. Its follow-up autolens_profiling#243 split the 37 ms reconstruction row the grid could
only measure as one block and emitted the exact log-det terms an SLQ estimator is checked
against.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| Autotune A/B (origin) | 2026-09-05 | Is the F row's 4.8 → 25.6 ms move the autotuner? | not recorded | F 25.63 → 4.90 ms at level 4; DelaunayNN runtime 82.20 → 62.87 ms/call at `vmap` 16; +1.4 s F compile (ledger: [delaunay_nn_breakdown.md](../../results/notes/delaunay_nn_breakdown.md)) | 342277 vs 342282, 342281 vs 342283 | #221 |
| XLA probe + Triton GEMM off | 2026-09-08 | Why did later level-0 rows read 4.8 ms, and what is the cure? | PyAutoNerves#161 witness (fresh-cache F < 6 ms, F compile < 0.2 s, flag in `device.xla_flags`) | five arms on a bare (15361 × 1560) GEMM: level 0 fresh 25.52 ms (Triton), level 4 4.92, level 0 on level-4 cache 4.83, level 0 own cache 25.51, level 0 + Triton off 4.79 ms at 0.074 s compile; cure adopted; witness run deferred to the baseline, which met it (dense F compile 0.06 s) | 342333 | PyAutoNerves#162, #230 |
| Pixelized baseline grid | 2026-09-10 | Reference cost, {rect, Delaunay, DelaunayNN} × {dense, sparse} × {runtime, breakdown} | #241 witness (header) | MET: dense F 4.826 / 4.824 / 4.830 ms; `vmap` 16 per call dense/sparse 32.08/36.65, 42.53/48.04, 46.44/49.25 ms; sparse F/D equal dense to 3.9e-15; reconstruction 36.7–37.9 ms = 61–71 % of the "must beat" total | 342617–342628 | #242 |
| Reconstruction split | 2026-09-11 | What is the 37 ms reconstruction made of? | #243 witness: sub-rows, `nnls` block matching the library to 1e-8, non-null `log_evidence_terms`, rect runtime pin = breakdown pin | MET: 21 / 22 / 22 PDIP iterations × 1.70–1.77 ms; one Cholesky 1.19–1.26 ms, exact solve 1.52–1.58 ms; `vmap` 16 identical lanes only 1.70x; log-dets emitted; rect runtime pin re-measured to 28621.128714 | 342643–342646 | #244 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoNerves#162 | `jax_wrapper` appends `--xla_gpu_enable_triton_gemm=false` to the level-0 default (issue PyAutoNerves#161) | `0e7163bc` | 2026.9.11.1 |

The baseline and reconstruction split are profiling-only (#242, #244); no library PR.

## Open / parked / drafts

- none. The four #241 follow-ups resolved: log-dets and the reconstruction split (#243), the rectangular runtime pin (#243), the matrix-free prompt ([Matrix-free pixelized](matrix_free_pixelized.md), no-go).
- The NNLS iteration count, which #243 names as the lever, continues on [Certified positive solver](certified_positive_solver.md).
- Not filed: rewiring `aggregate.py` so A100 runtime rows reach the README headline table (out of scope in #241).

## Caveats

- **Fiducial tier, not the production preset.** 1500 Delaunay vertices / 1521 rectangular pixels, pixelization over-sampling 1; the #235 production preset (decision 3 open) is heavier. Do not mix the two.
- **Sparse rows are a comparator, not a production GPU path.** Production GPU runs never apply the sparse operator; these legs call `apply_sparse_operator(batch_size=128)` explicitly.
- **Autotune setting.** Every leg ran `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false` with a fresh per-job `JAX_COMPILATION_CACHE_DIR` and `AUTOTUNE_ENTRIES count=0`. A100 rows recorded 2026-07-17 → 2026-09-08 are slow on GEMM-bound steps unless the node's per-fusion cache happened to be seeded, so they are inconsistent with each other; jobs 342315, 342316, 342321, 342322 read the cache seeded by 342282/342283.
- **`device.xla_flags` proves the flag reached the environment, never which kernel ran.** The F compile time is the tell (~2 s autotuned, ~0.4 s Triton default, ~0.07 s cuBLAS or a cache hit); check the HLO when in doubt.
- **Arm E is bit-identical to the old Triton output; the autotuned cuBLAS kernel sits ~2 ULP away**, so the cure changed no pin. The probe's raw output lives on RAL scratch and is not committed.
- **Negative differenced cells are artifacts.** Rectangular H (−0.568 / −0.123 ms), DelaunayNN Mapping matrix (−0.194, −2.457 at `vmap` 16), rectangular sparse triplets (−0.098); judge H on the params→H prefix and triplets on the direct `steps_sparse_setup_rows`.
- **Post-setup steps have no batched row in the 2026-09-10 grid**; the runtime table is the only batched measurement for D, F, solve and log-evidence. The `nvidia-smi` memory column is indicative only.
- **Reconstruction sub-rows overlap the reconstruction row**; they are comparators, not addends. The one-iteration row is an upper bound; the `vmap` 16 NNLS row uses identical lanes and is the best case for batching.
- **Cross-run comparability.** The shared RAL install moved between 2026-09-10 and the #243 re-run (PyAutoNerves unchanged at `0e7163bc`); every step row landed within 0.7 %.
- **Pins.** The three Delaunay-family breakdown JSONs carry no `pinned_drift` key; their pins were audited from SLURM stdout. Rectangular uses `Constant(1.0)`, the Delaunay family `adapt_split` (0.1 / 10.0 / 0.1), so compare log-det terms within a mesh, not across.
- **DelaunayNN `sparse_nnz` is 491,552** against 46,083 (Delaunay) and 61,444 (rectangular); do not extrapolate DelaunayNN sparse costs from Delaunay by vertex count.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records four phases: the 2026-09-05 autotune A/B that opened the question, the five-arm
XLA probe and PyAutoNerves#162, the twelve-job baseline (#241 / #242) and the reconstruction
split (#243 / #244), which lives in the same ledger. Resolved: the Mind contract (three records),
the profiling PRs and merges, and the witness verdicts. PyAutoNerves#162 verified released 2026.9.11.1 (first tag containing `0e7163bc`).
Nothing left open on this page.
