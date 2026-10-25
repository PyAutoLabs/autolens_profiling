# Likelihood lifecycle phase 2a: imaging at the standard size

**Status:** open
**Question:** What does every planned imaging/{mge, rectangular, delaunay, delaunay_nn} cell cost at the standard source size (Delaunay 1250, rectangular 35x35) on the pinned release 2026.10.23.1, measured from the production pipeline, and does each cell's log-likelihood match an independent reference?
**Pre-registered rule:** every planned cell lands `current`, with runtime and breakdown, provenance, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy dense-mapping reference (tolerance 1e-4 x |reference|, fp64). Any `diverged` witness blocks that setup until explained. Recorded in the Mind prompt before batch 1 ran (2026-10-24).
**Verdict:** 104 of 106 planned imaging cells current, 53/53 parity witnesses `match`; no setup blocked. The 2 missing cells are the known limitation below (ao `jax_gpu_dense_fp64@hpc_a100` breakdowns out of A100 memory). Class designations (optimal / secondary) are phase 3's decision, not this page's.
**Headline:** fastest per call on every likelihood x instrument is `jax_gpu_dense_fp64@hpc_a100` (e.g. hst Delaunay 35.05 ms per call at vmap 16, job 412944). Per call is per device, not per node: numba runs one serial call per core, so per node a ral node is projected at ≈248 calls/s on hst Delaunay against ≈114 for a 4×A100 node, and on ao Delaunay ≈4.6 against ≈10.9 (upper bounds, parallel efficiency unmeasured; [Comparing CPU and GPU setups](#comparing-cpu-and-gpu-setups)).
**Library PRs:** none (measurement only, tag 2026.10.23.1)
**Profiling PRs:** PR A (`--standard-size`, parity writer, submits) #457, batch-1 walls #460, this PR B (rows, catalogue, coverage, this page)
**Ledger:** this page; rows under `results/imaging/{mge,rectangular,delaunay,delaunay_nn}/`; work list [`scripts/misc/campaigns/imaging_p2a.py`](../../../scripts/misc/campaigns/imaging_p2a.py) and its [README](../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2a; `active/likelihood_lifecycle_p2a_imaging_campaign.md`; issue autolens_profiling#456
**Next:** refetch the Pulse receipt after merge and adopt Pulse `hst_mge_cpu_pilot`; phase 2b–2g re-measure the other likelihoods on the same pin; phase 3 designates classes from this evidence
**Fit figures:** opened hst delaunay `jax_gpu_dense_fp64@hpc_a100`, ao rectangular `numba_cpu_sparse_fp64@hpc_ral_cpu`, euclid mge `jax_cpu_dense_fp64@hpc_ral_cpu`: ring reconstructed in the source plane, residuals confined to the lens core / a low-level pattern expected of iid draw 0 (a prior draw, not a best fit). None looked wrong.

## Why this campaign

The lifecycle catalogue's coverage matrix listed every imaging cell as missing or stale: the
2026-09 rows sat at Delaunay 1500 / rectangular 39x39, not the standard size the human ruled
on 2026-10-23, and on older releases. Phase 2a re-measures the 106 planned imaging cells on
one pinned tag so phase 3 can choose setups from like-for-like evidence.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| PR A | 2026-10-24 | Can every setup measure the same production likelihood at the standard size, with a parity witness? | JAX and numba cells of one likelihood x instrument build the same production preset | `--standard-size`, parity writer, campaign submits | — | #457 |
| Batch 0 | 2026-10-24 | How long and how large are the bounding legs? | `--time` >= 1.25 x the slowest batch-0 leg | walls and MaxRSS set batch 1's `--time` / `--mem` | 412877, 412884, 412911, 412915 | #460 |
| Batch 1 | 2026-10-25 | All 106 cells | every cell current with parity `match` | 104/106 legs complete, 53/53 `match`; 2 known OOMs | 412923–412926 (RAL CPU), 412927, 412943, 412944, 412945 (A100) | this PR |

## Coverage after this PR

`dashboard/coverage.md`, all dataset types (246 planned cells):

| | current | stale | missing |
|---|---|---|---|
| before | 2 | 6 | 238 |
| after | 104 | 7 | 135 |

Imaging (106 planned): 104 current, 2 missing, 0 stale. The catalogue pin moves from
2026.10.17.1 to 2026.10.23.1, so the two point-source rows that were current on 2026.10.17.1 now
count as stale; the 2b–2g campaigns re-measure them, as the plan says.

**Release pin.** A library tag checkout still reports the last built `__version__`, so every
batch-1 row stamps `autolens_version: 2026.8.17.1`. Their `library_revisions` equal tag
2026.10.23.1's SHAs, and [`catalogue/release_revisions.json`](../../../catalogue/release_revisions.json)
lets the catalogue and the dashboard publish them as 2026.10.23.1
(`identity.software.release_pin_basis` says so). Without it every new row would read as stale.

## Per-call evidence (batch 1, tag 2026.10.23.1)

JAX: `single-jit median` is the median of 200 individually timed calls after 5 warm calls; †
marks rows whose leg recorded only the one-block mean (`full_pipeline_single_jit`, a 10-call
block after the first call). `vmap per call` is the batch time / batch size at the batch the
script chose. numba: `warm iid median` is the median over 16 seeded iid draws after the
numba compile; `cold median` repeats the first instance 3 times (PyAutoFit's logged
evaluation time). numba runs one process per likelihood call; a sampler's parallel workers
divide its wall time, so a per-call comparison against a vmap batch is per device, not per
node. Loadavg at write was 8.6–37 on the RAL CPU nodes and 9–13 on the A100 nodes, above the
dashboard's cap of 8, so the run-time dashboard shows these points as refused for trends; the
catalogue and coverage do not use that cap.

### imaging/mge

| instrument | setup @ host | single-jit median (ms) | vmap per call (ms) [batch] |
|---|---|---|---|
| euclid | `jax_cpu_dense_fp64@hpc_ral_cpu` | 7.90 | 6.80 [3] |
| euclid | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 5.92 | 0.20 [64] |
| hst | `jax_cpu_dense_fp64@hpc_ral_cpu` | 21.22 | 14.39 [3] |
| hst | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 6.29 | 0.41 [64] |
| jwst | `jax_cpu_dense_fp64@hpc_ral_cpu` | 50.25 | 41.55 [3] |
| jwst | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 7.00 | 1.00 [64] |
| ao | `jax_cpu_dense_fp64@hpc_ral_cpu` | 388.90 | 329.65 [3] |
| ao | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 15.73 | 18.98 [3] |

### imaging/rectangular

| instrument | setup @ host | single-jit median (ms) | vmap per call (ms) [batch] | numba warm iid median (ms) | numba cold median (ms) |
|---|---|---|---|---|---|
| euclid | `jax_cpu_sparse_fp64@hpc_a100` | 827.35 | 633.75 [3] | — | — |
| euclid | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 858.50 | 683.02 [3] | — | — |
| euclid | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 29.83 | 7.93 [64] | — | — |
| euclid | `jax_gpu_sparse_fp64@hpc_a100` | 34.68 | 13.19 [64] | — | — |
| euclid | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 282.5 | 266.5 |
| hst | `jax_cpu_sparse_fp64@hpc_a100` | 1,968 | 1,709 [3] | — | — |
| hst | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 2,155† | 2,029 [3] | — | — |
| hst | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 40.66 | 25.50 [16] | — | — |
| hst | `jax_gpu_sparse_fp64@hpc_a100` | 61.55 | 41.19 [16] | — | — |
| hst | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 696.7 | 664.8 |
| jwst | `jax_cpu_sparse_fp64@hpc_a100` | 4,141† | 3,940 [3] | — | — |
| jwst | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 4,453† | 4,210 [3] | — | — |
| jwst | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 66.30 | 62.84 [8] | — | — |
| jwst | `jax_gpu_sparse_fp64@hpc_a100` | 116.84 | 107.46 [8] | — | — |
| jwst | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 1,688 | 1,513 |
| ao | `jax_cpu_sparse_fp64@hpc_a100` | 50,291† | 50,843 [1] | — | — |
| ao | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 50,877† | 48,794 [1] | — | — |
| ao | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 309.62 | 310.83 [1] | — | — |
| ao | `jax_gpu_sparse_fp64@hpc_a100` | 968.37 | 1,053 [1] | — | — |
| ao | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 9,982 | 9,988 |

### imaging/delaunay

| instrument | setup @ host | single-jit median (ms) | vmap per call (ms) [batch] | numba warm iid median (ms) | numba cold median (ms) |
|---|---|---|---|---|---|
| euclid | `jax_cpu_sparse_fp64@hpc_a100` | 1,146 | 857.44 [3] | — | — |
| euclid | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 1,122 | 826.60 [3] | — | — |
| euclid | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 44.73 | 14.80 [64] | — | — |
| euclid | `jax_gpu_sparse_fp64@hpc_a100` | 46.85 | 18.16 [64] | — | — |
| euclid | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 820.0 | 706.2 |
| hst | `jax_cpu_sparse_fp64@hpc_a100` | 2,562† | 2,110 [3] | — | — |
| hst | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 2,221† | 1,895 [3] | — | — |
| hst | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 59.58 | 35.05 [16] | — | — |
| hst | `jax_gpu_sparse_fp64@hpc_a100` | 76.23 | 47.93 [16] | — | — |
| hst | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 951.7 | 591.0 |
| jwst | `jax_cpu_sparse_fp64@hpc_a100` | 5,086† | 4,686 [3] | — | — |
| jwst | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 4,759† | 4,361 [3] | — | — |
| jwst | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 88.63 | 72.29 [8] | — | — |
| jwst | `jax_gpu_sparse_fp64@hpc_a100` | 124.21 | 108.11 [8] | — | — |
| jwst | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 1,745 | 1,350 |
| ao | `jax_cpu_sparse_fp64@hpc_a100` | 56,742† | 57,179 [1] | — | — |
| ao | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 56,746† | 55,292 [1] | — | — |
| ao | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 368.16 | 366.93 [1] | — | — |
| ao | `jax_gpu_sparse_fp64@hpc_a100` | 886.71 | 935.34 [1] | — | — |
| ao | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 8,671 | 8,133 |

### imaging/delaunay_nn

| instrument | setup @ host | single-jit median (ms) | vmap per call (ms) [batch] | numba warm iid median (ms) | numba cold median (ms) |
|---|---|---|---|---|---|
| hst | `jax_cpu_sparse_fp64@hpc_a100` | 6,250† | 6,235 [3] | — | — |
| hst | `jax_cpu_sparse_fp64@hpc_ral_cpu` | 6,831† | 6,427 [3] | — | — |
| hst | `jax_gpu_dense_fp64@hpc_a100` **fastest** | 89.59 | 42.74 [16] | — | — |
| hst | `jax_gpu_sparse_fp64@hpc_a100` | 1,201 | 1,149 [16] | — | — |
| hst | `numba_cpu_sparse_fp64@hpc_ral_cpu` | — | — | 1,295 | 850.2 |

### Fastest per call, by likelihood x instrument

Fastest = the lowest vmap per call (JAX) or warm iid median (numba). This is evidence for
phase 3, not a designation, and it is per call: per node, see
[Comparing CPU and GPU setups](#comparing-cpu-and-gpu-setups).

| likelihood | instrument | fastest setup (per call, ms) | fastest CPU-only setup (per call, ms) |
|---|---|---|---|
| mge | euclid | `jax_gpu_dense_fp64@hpc_a100` (0.20) | `jax_cpu_dense_fp64@hpc_ral_cpu` (6.80) |
| mge | hst | `jax_gpu_dense_fp64@hpc_a100` (0.41) | `jax_cpu_dense_fp64@hpc_ral_cpu` (14.39) |
| mge | jwst | `jax_gpu_dense_fp64@hpc_a100` (1.00) | `jax_cpu_dense_fp64@hpc_ral_cpu` (41.55) |
| mge | ao | `jax_gpu_dense_fp64@hpc_a100` (18.98) | `jax_cpu_dense_fp64@hpc_ral_cpu` (329.65) |
| rectangular | euclid | `jax_gpu_dense_fp64@hpc_a100` (7.93) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (282.5) |
| rectangular | hst | `jax_gpu_dense_fp64@hpc_a100` (25.50) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (696.7) |
| rectangular | jwst | `jax_gpu_dense_fp64@hpc_a100` (62.84) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (1,688) |
| rectangular | ao | `jax_gpu_dense_fp64@hpc_a100` (310.83) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (9,982) |
| delaunay | euclid | `jax_gpu_dense_fp64@hpc_a100` (14.80) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (820.0) |
| delaunay | hst | `jax_gpu_dense_fp64@hpc_a100` (35.05) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (951.7) |
| delaunay | jwst | `jax_gpu_dense_fp64@hpc_a100` (72.29) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (1,745) |
| delaunay | ao | `jax_gpu_dense_fp64@hpc_a100` (366.93) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (8,671) |
| delaunay_nn | hst | `jax_gpu_dense_fp64@hpc_a100` (42.74) | `numba_cpu_sparse_fp64@hpc_ral_cpu` (1,295) |

### Comparing CPU and GPU setups

The tables above are per call. They do not say which setup does more work per node: under
Nautilus multiprocessing, numba runs one serial likelihood per core, so a CPU node runs many
calls side by side while an A100 runs one process per GPU. Compare setups by projected node
throughput (`projected_node_throughput` on every runtime record of
[`dashboard/catalogue.json`](../../../dashboard/catalogue.json), autolens_profiling#462). Each
row is projected only onto the node class it was measured on (#464): a ral-node row onto a ral
node, an A100 row onto a gpu node, never across machines. Workers per node = min(cores / cores per call, node memory / peak memory per worker), and one
per GPU on a GPU node, times the parallel efficiency over the per-call time. The parallel
efficiency is declared 1.0, an upper bound, unmeasured (phase 4 measures it), and the
sampler's batch ceiling (Nautilus `n_batch`) is not modelled, so every number here is a
projection, not a measurement, and the CPU numbers (hundreds of workers sharing memory
bandwidth) are the most optimistic. Peak memory is the leg's sacct MaxRSS
([`catalogue/peak_rss_backfill.json`](../../../catalogue/peak_rss_backfill.json)): exact per leg on
RAL CPU, the multi-leg A100 job's peak (an upper bound) on A100. Hosts are
[`catalogue/reference_hosts.json`](../../../catalogue/reference_hosts.json): a ral node with 236
cores and 921 GB, a gpu node with 124 cores, 885 GB and 4 A100s.

| cell | setup @ node | per call | cores per call | peak memory per worker | workers (bound by) | projected calls/s per node |
|---|---|---|---|---|---|---|
| hst Delaunay | `numba_cpu_sparse_fp64` @ ral node | 951.7 ms | 1 | 0.85 GB | 236 (cores) | ≈248 |
| hst Delaunay | `jax_cpu_sparse_fp64` @ ral node | 1,895 ms | 8 | 3.2 GB | 29 (cores) | ≈15 |
| hst Delaunay | `jax_gpu_dense_fp64` @ gpu node | 35.05 ms | 1 GPU | ≤10.7 GB | 4 (GPUs) | ≈114 |
| hst Delaunay | `jax_gpu_sparse_fp64` @ gpu node | 47.93 ms | 1 GPU | ≤10.7 GB | 4 (GPUs) | ≈83 |
| ao Delaunay | `numba_cpu_sparse_fp64` @ ral node | 8,671 ms | 1 | 23.0 GB | 40 (memory) | ≈4.6 |
| ao Delaunay | `jax_cpu_sparse_fp64` @ ral node | 55,292 ms | 8 | 19.5 GB | 29 (cores) | ≈0.52 |
| ao Delaunay | `jax_gpu_dense_fp64` @ gpu node | 366.93 ms | 1 GPU | ≤39.8 GB | 4 (GPUs) | ≈10.9 |
| ao Delaunay | `jax_gpu_sparse_fp64` @ gpu node | 935.34 ms | 1 GPU | ≤39.8 GB | 4 (GPUs) | ≈4.3 |

On hst Delaunay a ral node is projected to do more calls per second than a 4×A100 node; on ao
Delaunay, where each numba worker needs 23 GB and memory caps the node at 40 workers, the
A100 node is projected ahead. Which hardware a fit should use is phase 3's decision, made per
node and with the parallel efficiency phase 4 measures.

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| #457 | `--standard-size`, parity writer, campaign submits | merged | measurement code, no release |
| #460 | batch 1 `--time` / `--mem` from batch 0 | merged | — |
| PR B | 104 rows, 13 superseded rows to `history/`, catalogue/dashboard/coverage, release pin from revisions | open | — |

## Open / parked / drafts

- **Known limitation (human ruling): ao `jax_gpu_dense_fp64@hpc_a100` breakdown, rectangular
  and delaunay.** The 384,753 x 1,280 fp64 blurred mapping matrix (10.76 / 10.29 GiB
  allocations) does not fit on the A100 in the breakdown's step-by-step decomposition. Both
  legs failed in under two minutes and the job carried on; their runtime legs completed and
  are current. Coverage shows these 2 cells as missing; that is expected and not a re-run
  target unless the breakdown learns to stream the matrix.
- `delaunay_nn` `jax_gpu_sparse_fp64@hpc_a100`: 1,149 ms per call against 42.74 ms dense, and
  against 80 ms for the 2026-09 1500-pixel row (history/). The standard-size cell is the
  production preset (MGE lens light, adapt split regularisation, uncapped over-sampling cut),
  so the rows are not like-for-like, but a 14x gap is worth a look in phase 3.

## Fit figures

See the header. Each figure is a seeded iid draw of the production preset's priors, not a
fit: lens-core residuals and a low-level ring pattern are expected.

## Caveats

- **Euclid positions penalty.** The euclid positions penalty is part of every euclid runtime
  row (JAX and numba time `log_likelihood_function`), but the JAX breakdowns decompose
  `FitImaging`, which has no positions term: a euclid JAX breakdown does not time the penalty.
- **Legacy presets differ from production.** Without `--standard-size`, the numba production
  presets still build their data the historic way: the adapt image is the noise-free
  lensed-source flux, capped at 3 before it drives the over-sampling cut (so on hst, jwst and
  ao the cut never fires and every pixel gets sub-size 2), and the rectangular presets use
  `RectangularBilinearAdaptDensity`. Production does neither; only the standard-size cells
  match production. A follow-up fixes the legacy preset.
- jwst and ao borrow the hst production preset (`size_override` in each row).
- The 13 superseded hst A100 rows (2026-09 baseline grid at 1500 / 39x39, and the 2026-07 mge
  breakdown) moved to `<cell>/history/<release_pin>_<date>/`; the date is the last commit that
  changed each file before the day-zero move.

## Journal

### 2026-10-25 — cores per call, peak memory and node throughput (#462)

Phase 2x attached `cpus_per_eval` and `peak_rss_gb` to the 104 committed rows through
`catalogue/peak_rss_backfill.json` (sacct MaxRSS: 37 exact per leg, 60 A100 legs at their
job's peak as an upper bound, 7 short RAL tasks sacct never sampled, left null) and added
`projected_node_throughput` to the catalogue's runtime records, each projected only onto the
node class it was measured on (#464). Rows were not edited and nothing was re-run. See [Comparing CPU and GPU setups](#comparing-cpu-and-gpu-setups).

### 2026-10-25 — batch 1 committed

Batch 1 (RAL 412923–412926, A100 412927 / 412943 / 412944 / 412945) ran from the private
checkout at tag 2026.10.23.1, autolens_profiling 86c287d1. 104 of 106 legs completed; no leg
was killed at `--time`. All 53 cells had final rows, `fit.png` and a valid `parity.json`, and
every row's `library_revisions` equal the tag's SHAs. 53/53 witnesses `match`. Imaging
coverage: 104 current, 2 missing (the known A100 OOMs). Next: Pulse receipt refetch, adopt
`hst_mge_cpu_pilot`, then 2b.
