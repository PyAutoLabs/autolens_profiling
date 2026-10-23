# DelaunayNN ConstantSplit assembly — split-stencil compaction (A100, 2026-09-08)

PyAutoArray issue #536, branch `feature/delaunay-nn-constant-split-assembly`, commit
`04152bb7`, against its merge base `180c8a40` (PyAutoArray `main`, #534 merged). This is the
follow-up to `delaunay_nn_launch_latency.md` (#532/#533), whose closing section identified
the `ConstantSplit` assembly — not the Sibson loops — as the larger remaining per-call lever
on this cell, at ~10.0 ms per call out of a 16.4 ms params→H prefix at `vmap` 16.

## What changed

One function, `pixel_splitted_regularization_matrix_from`'s JAX path in
`autoarray/inversion/regularization/regularization_util.py`:

1. **The main scatter is compacted to the first `SPLIT_REG_COMPACT_WIDTH = 12` columns.**
   The split stencil tables are fixed-shape `(4P, K)` with `K = 33` for `DelaunayNN`
   (`SIBSON_MAX_NEIGHBORS` 32 + 1 spare column), while the real occupied width on this cell
   is min 1 / mean 5.45 / max 11 over its 6,000 rows. The old code scattered the full padded
   `(4P, K, K)` outer product — 6.53 M entries for 187 k real contributions, ~97 % padding,
   at a cost quadratic in the padded width.
2. **The `SPLIT_REG_WIDE_ROW_BUDGET = 256` widest rows get a full-width supplement.** The
   head × head block the compact pass already scattered is masked to zero and the remaining
   head × tail / tail × head / tail × tail blocks go in as one masked `(W, K, K)` scatter —
   one kernel launch rather than three block scatters. Rows wider than 12 are therefore
   reproduced exactly, not dropped. The cap audit over 101 ensemble geometries
   (`delaunay_nn_cap_audit.md`) saw at most 50 above-width rows in one geometry, so 256 is a
   ~5× margin.
3. **Beyond the budget the matrix is poisoned with NaN**, matching the Sibson cap convention
   (NaN weights → NaN likelihood → the sample is discarded), rather than returning a
   silently truncated `H`. When `K <= compact_width` — the `Delaunay` mesh's `K = 4`, and the
   adapt-split family — the compaction is a no-op: today's single scatter, no supplement, no
   guard.

## Verdict

On the A100, the HST / Hilbert-1500 / MGE-60 / `ConstantSplit` **DelaunayNN** imaging
likelihood's **params→H prefix falls 24.34 → 15.19 ms unbatched (−9.15 ms, 1.60×)** and
**16.42 → 7.26 ms per call at `vmap` 16 (−9.16 ms, 2.26×)**. The assembly itself, measured
directly on the real split tables, goes **10.03 → 0.84 ms per call at `vmap` 16 (12.0×)** and
10.49 → 1.29 ms unbatched (8.2×). Whole-likelihood single-JIT runtime goes 75.61 → 66.12 ms
(1.14×) and at `vmap` 16 **50.04 → 40.86 ms per call (1.22×)**. The `29144.581944` pin held
**bit-identically on every leg**.

**The witness is met on both halves.**

| Witness | Target | Measured | |
|---|---|---|---|
| "H, ConstantSplit assembly" per call @ `vmap` 16 | < 3 ms (from 10.0) | **0.80** (breakdown H row) / **0.84** (direct bench) | met |
| `regularization_matrix_prefix_s` per call @ `vmap` 16 | < 11 ms (from 16.4) | **7.26** | met |
| `EXPECTED_LOG_EVIDENCE_HST = 29144.581944` at rtol 1e-4 | unchanged | identical to the last printed digit on both legs, `pinned_drift: []` on both | met |

One honest caveat on the first row, discussed in full below: the `--split-setup` table's
"H, ConstantSplit assembly" cell is a *difference of two independently compiled prefixes*
and on the feature leg it reads **−1.60 ms** per call — negative. That number is an
attribution artifact, not a saving, and is not what the witness is judged on. The two
readings that do carry it — the breakdown's own `Regularization matrix (H)` step row (0.80 ms)
and the direct assembly bench (0.84 ms) — agree with each other to 5 %.

## Provenance

| Job | Cell | Leg | Node | Start (2026-09-08 BST) | Elapsed | State |
|---|---|---|---|---|---:|---|
| 342334 | `likelihood_breakdown/imaging/delaunay_nn` | control `180c8a40` | `euclid-ral-gpu-2` | 14:16:38 | 2:36 | COMPLETED |
| 342335 | `likelihood_breakdown/imaging/delaunay_nn` | feature `04152bb7` | `euclid-ral-gpu-2` | 14:19:15 | 2:29 | COMPLETED |
| 342336 | `likelihood_runtime/imaging/delaunay_nn` | control `180c8a40` | `euclid-ral-gpu-2` | 14:21:45 | 1:17 | COMPLETED |
| 342337 | `likelihood_runtime/imaging/delaunay_nn` | feature `04152bb7` | `euclid-ral-gpu-2` | 14:23:03 | 1:22 | COMPLETED |
| 342338 | `misc/delaunay_nn/assembly_bench.py` | feature `04152bb7` | `euclid-ral-gpu-2` | 14:24:26 | 0:33 | COMPLETED |

- **All five jobs ran back-to-back on the same node inside one 8.4-minute window**
  (14:16:38 → 14:24:59), so this is a same-session A/B, not a comparison against a recorded
  baseline.
- Breakdown flags `--split-setup --vmap-batch 16`; runtime cell full timing only (no
  `--vmap-probe`), `vmap` batch 16 on both legs. `PyAutoLens 2026.8.17.1`, fp64
  (`JAX_ENABLE_X64=True`), dense inversion path, NVIDIA A100 80GB PCIe, `backend = gpu`,
  `device = cuda:0` on every leg.
- Recorded `xla_flags` identical on all five job units:
  `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0`. These rows are
  therefore comparable with the 2026-09-08 rows in `delaunay_nn_launch_latency.md` and
  **not** with the 2026-07-10 rows in `preopt_breakdown_baseline.md`.
- `PYAUTO_SIBSON_QUERY_CHUNK` deliberately unset on every leg; the library default
  `query_chunk = 4096` (chosen by #533's sweep) is what both legs ran, recorded under
  `sibson` in each result JSON. Caps `max_cavity_triangles = 32`, `max_neighbors = 32`.

### The A/B differs in PyAutoArray and nothing else

| Repo | Revision |
|---|---|
| PyAutoArray — control leg | `180c8a40ceaffe1252cc4db86e85461bfe4d036a` |
| PyAutoArray — feature leg | `04152bb776350de589ed4349d07e1a7e2cb1608b` |
| PyAutoNerves | `efe7c04a79f8389bdcccb88b4d6f8781b9b27142` |
| PyAutoFit | `6331b80031b1dedae3ce277a8c2b116b52ddd261` |
| PyAutoGalaxy | `6d216c151c914b2ce79af18fbc0348a6fc5425d7` |
| PyAutoLens | `9468e3e1bc10405c622e7408c04c0760dcb26882` |
| autolens_profiling (`AP_ROOT`) | `3739481432d1ee3322d87158d2881896eb89ba6a` |

The **shared** `/mnt/ral/jnightin/PyAuto` install was deliberately not touched — the
subhalo-validation runs (`342299`, `342311`) were live on it throughout, on the `ral`
partition. Every leg instead prepends a private PyAutoArray checkout to the `PYTHONPATH`
that `activate.sh` exports:
`/mnt/ral/jnightin/PyAuto_wt/delaunay-nn-constant-split-assembly/PyAutoArray_{control,feature}`,
and each job prints `autoarray.__file__` and the checkout's `git rev-parse HEAD` into its
SLURM log (verified: control leg imported `PyAutoArray_control`, feature leg imported
`PyAutoArray_feature`, on every job). Submits ran from
`/mnt/ral/jnightin/autolens_profiling_wt/delaunay-nn-constant-split-assembly` (detached at
`3739481`).

`PyAutoFit` on the shared install moved between #533's session and this one
(`2680b32d` → `6331b800`). It is identical across *these* two legs, so this A/B is clean;
the caveat is only for reading absolute rows across the two notes.

Every `error/*.err` was clean of `Traceback` on the four A/B legs (269 bytes each, the
`nvidia-smi`/preflight banner), and `grep -c "truncated to dtype float32"` was 0 on every
`.out` and `.err`. The bench job's `.err` carries one traceback: the diagnostic variant
`V2b_bcoo_bcoo` fails inside JAX with `UNIMPLEMENTED: Stable sorting of more than 2^31-1
elements is not implemented`. That is a bench-only variant that never enters the library,
the bench catches it and records it as `FAIL`, and all other rows completed.

### Pins

`EXPECTED_LOG_EVIDENCE_HST = 29144.581944` passed on all four A/B legs. The two breakdown
legs printed **identical values to the last digit**:

| | control (342334) | feature (342335) |
|---|---|---|
| `log_evidence (reference)` — eager `FitImaging` | 29144.581943564488 | 29144.581943564488 |
| `log_evidence (step-by-step)` — the JIT chain | 29144.581943577425 | 29144.581943577425 |
| `log_evidence (inv matrices)` | 29144.5819435774 | 29144.5819435774 |

Both runtime legs: `pinned_expected = 29144.581943885652`, `pinned_drift: []`.

So on the production geometry the compacted assembly is not merely within `rtol 1e-4` — the
JIT log evidence is bit-identical to the uncompacted one. That is the expected result: no
row on this cell exceeds the compact width (max size 11 vs `kc = 12`), so the compact pass
alone reproduces every block and the supplement adds exact zeros. The bench's V0-vs-V6
`max abs diff` of 1.78e-15 (`max rel` 4.77e-13) on the `H` *entries* is GPU scatter
reassociation between two differently-shaped `.at[].add`, not a semantic difference, and it
does not reach the log evidence.

## Control vs feature — step table

All values ms per likelihood call. `vmap/16` is `jax.jit(jax.vmap(fn))` over a params pytree
broadcast to batch 16, reported as batch time / 16; only the combined inversion-setup block,
the `--split-setup` prefixes and the params→H prefix are re-timed under `vmap`, so the
remaining rows read `—` there.

| Step | control | feature | Δ | control vmap/16 | feature vmap/16 |
|---|---:|---:|---:|---:|---:|
| Ray-trace data grid | 0.142 | 0.196 | +0.054 | — | — |
| Ray-trace mesh grid | 0.188 | 0.186 | −0.001 | — | — |
| Lens light images (pre-PSF) | 0.136 | 0.167 | +0.032 | — | — |
| Blurred image (PSF convolution) | 1.227 | 0.822 | −0.405 | — | — |
| Profile-subtracted image | 0.151 | 0.120 | −0.031 | — | — |
| **Inversion setup (steps 5–8 combined)** | **32.359** | **27.845** | **−4.514** | **17.658** | **17.416** |
| Data vector (D) | 0.366 | 0.370 | +0.004 | — | — |
| Curvature matrix (F) | 4.817 | 4.851 | +0.034 | — | — |
| **Regularization matrix (H)** | **9.917** | **0.800** | **−9.117** | **10.047** | **0.802** |
| Regularized reconstruction | 30.681 | 30.846 | +0.166 | — | — |
| Mapped recon + log evidence | 2.224 | 2.260 | +0.036 | — | — |
| **Total step-by-step (unbatched)** | **82.207** | **68.464** | **−13.743** | — | — |

The `H` row is the whole story: 12.4× unbatched, 12.5× per call at `vmap` 16, and it is the
only row that moves by more than the ±0.4 ms run-to-run noise of the small steps. The
inversion-setup block moves by −4.5 ms unbatched but is flat under `vmap` (17.66 → 17.42),
because the assembly does not live inside it — it lives in the `H` step downstream.

### Four-way setup split and the prefixes

| Piece | control | feature | Δ | control vmap/16 | feature vmap/16 |
|---|---:|---:|---:|---:|---:|
| Border relocation | 1.634 | 1.086 | −0.548 | 0.065 | 0.064 |
| Triangulation + interpolation | 12.790 | 13.302 | +0.512 | 6.311 | 6.395 |
| Mapping matrix | −0.096 | −0.031 | +0.065 | 0.230 | 0.086 |
| Blurred mapping matrix (PSF) | 12.247 | 8.401 | −3.846 | 10.760 | 8.322 |
| *Interpolator prefix (params→step 6)* | *14.424* | *14.389* | *−0.035* | *6.377* | *6.459* |
| *Split-Sibson prefix (params→split mappings)* | *18.092* | *18.426* | *+0.334* | *6.406* | *8.860* |
| **params→H prefix (the witness)** | **24.341** | **15.189** | **−9.152 (1.60×)** | **16.423** | **7.260 (2.26×)** |

The interpolator prefix is flat to 0.2 % on both readings, exactly as it should be: this
change touches nothing upstream of `reg_split_from`. The whole −9.15 ms comes out of the
last interval, the assembly. The "Blurred mapping matrix (PSF)" unbatched sign is the
by-difference artifact documented in `delaunay_nn_breakdown.md` and its 3.8 ms swing is
between two rows that neither leg touches.

Two sanity anchors against #533's shipped leg, measured twelve hours earlier on the same
node with the same flags: this run's **control** reads 16.423 ms params→H per call at
`vmap` 16 against #533's 16.435 (0.07 % apart) and 10.018 ms assembly against 10.001
(0.2 % apart). The control leg reproduces the predecessor to within a fifth of a percent,
which is the strongest available statement that the two notes' rows are on one scale.

### Split-point Sibson vs ConstantSplit assembly — and why the feature cell goes negative

| Piece | control | feature | control vmap/16 | feature vmap/16 |
|---|---:|---:|---:|---:|
| Split-point Sibson | 3.668 | 4.037 | 0.029 | 2.401 |
| H, ConstantSplit assembly | 6.249 | −3.237 | **10.018** | **−1.599** |

**The feature leg's negative assembly cell is an attribution artifact, not a saving of more
than 100 %.** This row is `regularization_matrix_prefix − split_sibson_prefix`, a difference
of two *independently compiled* programs. The split-Sibson prefix must **materialise** the
split mappings, sizes and weights as its outputs; the params→H prefix consumes them straight
into `reg_split_from` and never writes them out. Once the assembly costs only ~0.8 ms, that
materialisation overhead (~1.6 ms per call at `vmap` 16 here) is larger than the interval
being measured, and the subtraction goes below zero. #533's note recorded the same
phenomenon one boundary earlier, where its "Split-point Sibson" row read −2.510; the caveat
is the one in `delaunay_walk_early_exit.md`, that a prefix difference is only meaningful when
it is large against the compile-boundary slack.

Three independent readings of the same quantity at `vmap` 16 — and only the first is a
prefix difference:

| reading | control | feature |
|---|---:|---:|
| `--split-setup` differenced cell | 10.018 | −1.599 (artifact) |
| breakdown `Regularization matrix (H)` step row | 10.047 | 0.802 |
| direct assembly bench, real tables (below) | 10.031 | 0.837 |

The last two agree to 4 %, on two different machines-worth of code paths, and they are what
the verdict rests on. The differenced cell is retained here only because dropping a row that
misbehaves would hide the mechanism.

## Direct assembly bench (job 342338, `scripts/misc/delaunay_nn/assembly_bench.py`)

The assembly function timed on its own, on the real `(4P, K) = (6000, 33)` HST split stencil
tables committed at `results/delaunay_nn/tables_hst_1500.npz` (`P = 1500` mesh pixels;
occupied sizes min 1 / mean 5.45 / max 11; 6,534,000 padded scatter entries for 187,242 real
contributions into 29,020 unique `(i, j)` pairs, mean multiplicity 6.45). `V0` is the
pre-#536 full padded scatter, forced from the same source by `compact_width = K`; every
other variant is checked against it. fp64, 3 warm-ups + median of 10, `block_until_ready`.

| variant | unbatched ms | per call @ `vmap` 16 | max abs diff vs V0 | max rel diff |
|---|---:|---:|---:|---:|
| **V0** (pre-#536, full `(4P, 33, 33)` scatter) | **10.495** | **10.031** | — | — |
| V0a_4wide (the `Delaunay` mesh's `K = 4`, for scale) | 0.184 | 0.038 | — | — |
| V1_dense_gemm | 9.837 | 10.107 | 1.78e-15 | 1.38e-13 |
| V1_highest | 9.818 | 10.108 | 1.78e-15 | 1.38e-13 |
| V2_bcoo_dense | 19.935 | 20.957 | 1.78e-15 | 2.38e-13 |
| V2b_bcoo_bcoo | FAIL | FAIL | — | — |
| V3_dedup_segsum | 31.788 | 39.135 | 1.78e-15 | 1.29e-13 |
| V5_compact_k12 (compaction only, no supplement) | 0.846 | 0.579 | 1.78e-15 | 4.77e-13 |
| V5_compact_k16 (compaction only, no supplement) | 1.783 | 1.459 | 1.78e-15 | 2.38e-13 |
| **V6_library** (shipped: `kc = 12` + wide-row supplement) | **1.285** | **0.837** | **1.78e-15** | **4.77e-13** |

**V0 → V6_library is 8.2× unbatched and 12.0× per call at `vmap` 16**, and it lands within
5 % of the breakdown's own `H` row on both readings. Three things this table settles:

- **The width, not the algorithm, was the cost.** Every clever reformulation loses: a dense
  GEMM (V1) is flat at 9.8 ms because it is still `K`-wide; a BCOO route (V2) is 2× worse;
  deduplicate-then-segment-sum (V3) is 3× worse and 4× worse batched. Only narrowing the
  scatter helps — and it helps *faster* than the `O(K²)` entry count predicts. 33 → 12 is
  7.6× by entry count and 17.3× measured (10.031 → 0.579); 33 → 4, the `Delaunay` mesh's own
  width, is 68× by entry count and 264× measured (10.031 → 0.038). The extra factor is the
  scatter's own behaviour, not arithmetic: a narrower block means fewer atomic conflicts per
  destination and a smaller working set. `O(K²)` is therefore a floor on the saving, not an
  estimate of it.
- **The safety guarantee costs ~0.26 ms per call.** V5_compact_k12 is the compaction with
  no supplement and no guard, at 0.579 ms; V6_library adds the 256-row full-width supplement
  and the overflow check for 0.837 ms. On this geometry the supplement contributes exactly
  zero to the matrix (no row exceeds 12), so that 0.26 ms buys only correctness on the tail
  geometries the cap audit found — which is the right trade, but it should be recorded as a
  price rather than assumed free.
- **Width 12 is the right pick.** V5 at 16 costs 1.459 ms per call against 0.579 at 12 —
  2.5× — for margin the supplement already provides exactly.

`peak MiB` in the bench JSON is a process high-water mark that only ratchets upward across
variants, so it does not attribute memory per variant and is not quoted here.

## Whole-likelihood runtime cell (`likelihood_runtime/imaging/delaunay_nn`)

| | control | feature | Δ |
|---|---:|---:|---:|
| Full pipeline (single JIT) | 75.614 | 66.117 | **−9.50 (1.14×)** |
| `vmap` batch 16, per call | 50.044 | 40.858 | **−9.19 (1.22×)** |

This is what a production evaluation pays. Both savings track the params→H prefix drop
(−9.15 / −9.16 ms) essentially exactly — unlike #533, where the unbatched saving exceeded the
prefix drop by fusion slack. That is the signature of a change local to one late step: the
work removed is removed whole, and nothing upstream re-fuses around it.

At `vmap` 16 the whole likelihood is now **40.86 ms per call**, down from 52.62 ms at the end
of #533 and 58.02 ms before it — 1.42× for the two changes together on the batched production
number. Those two earlier figures are from #533's session rather than this one; the
within-session statement is this note's own 50.04 → 40.86.

## Where the remaining per-call cost is

At `vmap` 16 on the feature leg, the 7.26 ms params→H prefix decomposes as:

| piece | per call @ `vmap` 16 | share of params→H | comparison |
|---|---:|---:|---|
| Sibson (locate + both interpolation passes) | 6.459 | 89 % | Delaunay's whole interpolator prefix: 5.07 ms |
| **H, ConstantSplit assembly** | **0.80–0.84** | **11 %** | Delaunay's H row: 0.07 ms |

(6.459 + 0.84 = 7.30 against the measured 7.26, so the decomposition closes to within 0.6 %.)

**This inverts #533's closing finding.** There, after Phase A, ~60 % of the per-call params→H
prefix was the 33-wide assembly and ~40 % was Sibson, and the note concluded that "the bigger
per-call lever is a ConstantSplit-assembly rework, not the cavity early exit — by roughly 8×".
That rework is now done, and the ratio is 11 : 89 the other way. The assembly is 12× closer
to Delaunay's 0.07 ms H row than it was, and there is no second order of magnitude left in
it: 0.80 ms is 2.0 % of the 40.86 ms whole likelihood, so even taking it to zero would be
worth less than half of what this change was worth.

### Phase B re-costed against these numbers

Phase B as planned is the cavity `fori_loop` → `lax.while_loop` early exit. It targets the
Sibson share only, which is now the whole of the remaining prefix:

- **Its headroom at `vmap` 16 is 6.46 ms per call** (up from a 6.4 ms headroom out of a
  16.4 ms prefix — the headroom did not grow, the prefix around it shrank). The observed
  cavity occupancy on the production audit is 11 of the 32 fixed trips (`max_cavity = 11`,
  `max_neighbors = 13` in the workspace_test gate), so the early exit removes the tail of a
  32-trip loop that is already latency-amortised at chunk 4096. The plan's own estimate was
  ~20 % of the remaining Sibson cost, i.e. **≈ 1.3 ms per call at `vmap` 16** and ≈ 3 ms
  unbatched.
- Against the new whole-likelihood number that ≈ 1.3 ms is **≈ 3.2 % of a production
  evaluation**, up from ≈ 2.5 % when #533 costed it, purely because the denominator fell
  from 52.6 to 40.9 ms.
- **Phase B is now the largest remaining lever on the params→H prefix**, by ~8× over what is
  left of the assembly — the exact reverse of the ordering #533 recorded, and the reason its
  recommendation should not be read forward without this note.

Phase C (loop-free k-ring cavity, which changes fp summation order) also targets the Sibson
share and should be re-costed against the same 6.46 ms rather than against #533's figures.

Outside the params→H prefix, the standing rock is unchanged and is now much the largest
single row: **Regularized reconstruction at 30.85 ms unbatched**, the dense solve, untouched
by any of this work.

## Artifacts

- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_assembly_control.{json,png}`
- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_assembly.{json,png}`
- `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64_assembly_control.json`
- `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64_assembly.json`
- `results/delaunay_nn/assembly_bench_a100_fp64.json`
- `results/delaunay_nn/{tables_hst_1500.npz,mesh_points_hst.npy}` (the bench's inputs)
- `hpc/batch_gpu/submit_breakdown_imaging_delaunay_nn_a100_hst_fp64_assembly{,_control}`
- `hpc/batch_gpu/submit_runtime_imaging_delaunay_nn_a100_hst_fp64_assembly{,_control}`
- `hpc/batch_gpu/submit_delaunay_nn_assembly_bench_a100`

The 2026-09-08 `..._launch_latency` rows in `delaunay_nn_launch_latency.md` are left
untouched; this note's control leg supersedes that note's rows for any comparison against
this branch (its merge base moved from `d7c96762` to `180c8a40`), and the two control legs
agree to 0.2 %, so the row sets are in practice on one scale.

**Follow-up (2026-09-08, autolens_profiling#232):** the Delaunay cells now default to
`AdaptSplit(0.1, 10.0, 0.1)`, and the same-node A/B in
`results/notes/delaunay_adapt_split_regularization.md` shows this note's compaction carries
over unchanged — the DelaunayNN params→H prefix at `vmap` 16 reads 7.28 ms on AdaptSplit
against 7.25 on its ConstantSplit control (7.26 here). The canonical
`delaunay{,_nn}_hpc_a100_fp64` rows are AdaptSplit from that date.
