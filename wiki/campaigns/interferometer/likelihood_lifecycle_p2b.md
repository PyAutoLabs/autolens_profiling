# Likelihood lifecycle phase 2b: interferometer at the standard size

**Status:** open
**Question:** What does every planned interferometer/{mge, rectangular, delaunay} cell cost at the standard source size (Delaunay 1250 + 30 edge, rectangular 35x35) on the pinned release 2026.10.25.1, on the sparse W~ and dense NUFFT routes, and does each cell's log evidence match an independent dense reference?
**Pre-registered rule:** every one of the 60 planned cells lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the visibility-chunked eager NumPy dense reference (tolerance 1e-8 x |reference| in fp64, 1e-3 mixed precision; rulings R6, R7). Any `diverged` witness blocks that setup until explained. Recorded here with PR A, before batch 0 runs (2026-10-27).
**Verdict:** 60 of 60 planned interferometer cells current, 30/30 parity witnesses `match` (largest |delta| 2.7% of the 1e-8 bound); no `diverged`, no setup blocked. The rule is met. Against the expectation written before the runs: W~ beats the dense NUFFT route on both hosts for every pixelized cell where both were measured (sma), as expected; the packaged cost model picks the NumPy FFT route at jvla on RAL CPU, as expected (and at alma_high too), and at jvla it is right by 4-6x. It is not right everywhere: at delaunay alma_high its NumPy FFT pick (11.3 s) is slower than the forced `direct_conv` arm (8.8 s), flagged for phase 3. The parity expectation was off in its detail: the cells closest to the bound are rectangular at sma (2.7e-10 relative, A100 dense), not Delaunay (at most 1.9e-11). One experiment leg is missing by human ruling: the matrix-free alma 192x192 scan (RAL 413191_6) hit TIMEOUT at 12:00:29 and wrote no row; it is not a coverage cell. Class designations are phase 3's decision, not this page's.
**Headline:** fastest per call on every likelihood x instrument is the A100: `jax_gpu_sparse_fp64_wtilde@hpc_a100` for the pixelized cells (rectangular 22.95 / 36.01 / 73.70 / 437.8 ms single-jit median at sma / alma / alma_high / jvla; Delaunay 27.49 / 40.26 / 82.48 / 440.2 ms; jobs 413162-413165) and `jax_gpu_dense_fp64_nufft@hpc_a100` for MGE (3.72 ms, 0.57 ms per call at vmap 64). Per node it splits by instrument: a ral node is projected ahead of a 4xA100 node at sma and alma (rectangular numba ≈822 against ≈556 calls/s at sma, ≈263 against ≈78 at alma), behind it at alma_high and jvla (≈13 against ≈46, ≈1.2 against ≈8.9); upper bounds, parallel efficiency unmeasured, and the RAL CPU rows were measured on a loaded node ([Per-call evidence](#per-call-evidence-batch-1-tag-20261025-1)).
**Library PRs:** none (measurement only, tag 2026.10.25.1); the matrix-free experiment reads PyAutoArray#658 from a main checkout
**Profiling PRs:** PR A #471 (standard-size interferometer cells, numba runtime scripts, `--dense` breakdowns, chunked parity reference, campaign submits, matrix-free harness, and the batch-1 walls commit), then PR B (rows, catalogue, coverage, this page's result)
**Ledger:** this page; rows under `results/interferometer/{mge,rectangular,delaunay}/`, the matrix-free rows under `results/interferometer/rectangular/*/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/{experiments,scans}/`; work list [`scripts/misc/campaigns/interferometer_p2b.py`](../../../scripts/misc/campaigns/interferometer_p2b.py) and its [README](../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2b; `active/likelihood_lifecycle_p2b_interferometer_campaign.md`; issue autolens_profiling#469
**Next:** human review of PR B; optionally rerun the RAL CPU legs on a quiet node (trend points only) and widen the jvla RAL and reference `--time` before any rerun ([Walls](#walls-against---time)); phase 3 designates classes from this evidence

## Why this campaign

The coverage matrix listed 84 interferometer cells as missing or stale. The old rows ran
different models per script (Delaunay 1500 with `ConstantSplit`, rectangular 32x32 or 39x39 with
`Constant`), on older releases, and the pixelized breakdowns wrote no resources, figure or parity
witness. No interferometer numba runtime script existed, and the numba breakdown forced the
`direct_conv` route rather than measuring the route the library picks.

## Rulings (human, 2026-10-25 and 2026-10-26)

| Ruling | What it says |
|---|---|
| R1(a) | Dense pixelized NUFFT cells are planned at sma only (84 -> 60 cells). The library forms the dense transformed mapping matrix in one transform: 20.5 GB at alma, 102 GB at alma_high, 512 GB at jvla. Recorded as `instrument_scoped_setups` in `catalogue/provisional_designations.json`. |
| R2 | No DFT cells. |
| R3 | The hst `source_pix[2]` mesh and regularization at the standard size: Delaunay 1250 Hilbert + 30 zeroed edge points with `AdaptSplit`; rectangular 35x35 `RectangularBilinearAdaptImage` with `Adapt`. Adapt image: the true lensed source, no S/N cap. No lens light. MGE-20 unchanged. |
| R4 | The numba headline is the route the packaged cost model picks; the forced `direct_conv` and NumPy FFT routes are sub-rows. `--n-instances` 20 / 20 / 10 / 5 on sma / alma / alma_high / jvla. |
| R5 (changed 2026-10-26) | Pin 2026.10.25.1, from a private RAL checkout. Its NumPy NUFFT uses finufft, now installed in the shared venv. |
| R6 | The parity reference is a visibility-chunked eager NumPy dense evaluation; above sma it runs once per mesh as its own RAL task. |
| R7 | Parity tolerance 1e-8 x |reference| in fp64 (1e-4 would be about 30,000 nats at jvla), 1e-3 for mixed precision. |
| R8 | Batch 0 runs the bounding cells: RAL 14 (jvla numba x4, jvla JAX CPU sparse x4, sma JAX CPU dense x4, mge sma x2) and A100 10 (jvla GPU sparse x4, sma GPU dense x4, mge sma x2), plus the optional jvla reference tasks. |
| R9 | Planned instruments come from the registry: no sdp81, MGE at sma only. |

The human also asked for a matrix-free experiment (2026-10-26): `numpy_cpu_matrixfree`
rectangular runtime at 35x35 on all four instruments and a mesh scan at alma (64x64, 128x128,
192x192), written to `experiments/` and `scans/` only, never `current`, with the log-evidence
agreement against the dense reference recorded.

## The cells

60 planned cells (38 on RAL CPU, 22 on the A100), each with runtime and breakdown:

| Likelihood | Instruments | RAL CPU setups | A100 setups |
|---|---|---|---|
| interferometer/mge | sma | `jax_cpu_dense_fp64_nufft` | `jax_gpu_dense_fp64_nufft` |
| interferometer/rectangular, /delaunay | sma, alma, alma_high, jvla | `numba_cpu_sparse_fp64_wtilde`, `jax_cpu_sparse_fp64_wtilde`, `jax_cpu_dense_fp64_nufft` (sma only) | `jax_gpu_sparse_fp64_wtilde`, `jax_gpu_dense_fp64_nufft` (sma only) |

Every setup of one likelihood and instrument evaluates the same likelihood, built once in
`_production_config.standard_interferometer_cell` and evaluated at the preset's seeded iid
draw 0. The JAX runtime and the numba runtime share one driver
(`scripts/misc/likelihood_breakdown/interferometer_standard.py`); the breakdown harnesses build
the same cell under `--standard-size`, and `--dense` makes the library's dense route the
breakdown's primary arm.

## The parity reference

The reference runs the library's dense route (`InversionInterferometerMapping`) for everything
except the transformed mapping matrix, which it builds one chunk of visibilities at a time and
accumulates into the curvature matrix and data vector. It is independent of the W~ operator, of
numba and of JAX. On the sma dataset the chunked result equals the library's one-shot dense
figure of merit to the last bit for both meshes (checked on the laptop, 2026-10-27). At sma each
cell computes it in-process; above sma the reference task
(`scripts/interferometer/pixelized/parity_reference.py`) caches it per mesh and rewrites each
cell's `parity.json` from the row's `parity_candidate`.

## Batch 0 (2026-10-27/28)

| Array | Job | Legs | Result |
|-------|-----|------|--------|
| RAL CPU timing arm | 413115 | 14 | all exit 0; slowest leg the rectangular jvla numba breakdown, 6461 s |
| jvla dense references | 413116 | 2 | all exit 0; 10093 s (rectangular), 10727 s (delaunay) |
| A100 timing arm | 413117 | 10 | all exit 0; slowest leg the delaunay jvla GPU sparse breakdown, 714 s |

Every jvla parity witness is `match`. The walls set batch 1's `--time` (RAL 2:30:00, A100
1:30:00, references 4:00:00) and the MaxRSS its `--mem` (64 GB everywhere); the generator's
`BATCH0_WALLS` and `BATCH0_PEAK_RSS_KIB` hold the numbers. These are timing-arm rows only;
the campaign's rows come from batch 1.

## Batch 1 (2026-10-28/29)

| Array | Job | Legs | Result |
|-------|-----|------|--------|
| RAL CPU sma | 413146 | 14 | all exit 0, `revisions OK` |
| RAL CPU alma | 413158 | 8 | all exit 0, `revisions OK` |
| RAL CPU alma_high | 413159 | 8 | all exit 0, `revisions OK` |
| RAL CPU jvla | 413160 | 8 | all exit 0, `revisions OK`; slowest leg the rectangular jvla numba breakdown, 2:19:27 of 2:30:00 |
| dense references (alma, alma_high, jvla) | 413161 | 6 | all exit 0; slowest the delaunay jvla reference, 3:41:47 of 4:00:00 |
| A100 sma / alma / alma_high / jvla | 413162-413165 | 10 / 4 / 4 / 4 | all exit 0, `revisions OK`; slowest job jvla, 29:41 of 1:30:00 |
| matrix-free (unreleased main 7eba0c07) | 413191 | 7 | 6 exit 0; **413191_6 (alma 192x192 scan) TIMEOUT at 12:00:29**, no row |

All ran from the private RAL checkout of autolens_profiling 80bbb028 (the walls commit) over the
five libraries at tag 2026.10.25.1, and every cell row's `library_revisions` equal the tag's SHAs
(the matrix-free rows equal `MF_LIBRARY_SHAS`). Every row is a final write (no `PENDING` vmap
block). 72 of 73 legs landed: the 60 cells (each with `fit.png` and `parity.json`), the 6
references and 6 of the 7 matrix-free legs. The pull is byte-identical to RAL. 21 superseded
rows moved to `<cell>/history/2026.8.17.1_<date>/` first: the 20 `size_fallback` breakdowns of
the 2026-09 grids (Delaunay 1500 `ConstantSplit`, rectangular 32x32 / 39x39) and the stale MGE
sma A100 breakdown.

**Known gap (human ruling 2026-10-29, "go ahead with cell missing").** The alma 192x192
matrix-free scan leg ran its 20 matrix-free instances (about 12-15 s each), then spent the
rest of its 12 h `--time` in the `dense_fft` comparison arm at 36,864 source pixels and was
killed at 12:00:29 before writing its row. It is recorded as missing
(`BATCH1_LIMITATIONS` in the work list) and was not resubmitted. It is an experiment leg, not
a coverage cell, so the verdict does not depend on it. No other leg failed or timed out.

## Coverage after this PR

`dashboard/coverage.md`, all dataset types, on the 222-cell plan (this branch does not carry
the planner fix #481, which drops two cells from the plan):

| | current | stale | missing |
|---|---|---|---|
| before | 106 | 7 | 109 |
| after | 166 | 6 | 50 |

Interferometer (60 planned): 60 current, 0 stale, 0 missing. Missing parity witnesses across
the plan fell from 58 to 28 of 111 planned setups.

## Parity

30 witnesses, one per cell directory (the runtime and breakdown legs of a setup share it), all
`match`. At sma each cell computed the chunked dense reference in-process; above sma the
reference task (413161) rewrote each witness from the row's `parity_candidate`. The MGE cells
use the eager NumPy `FitInterferometer` reference.

| | largest relative delta | fraction of the bound |
|---|---|---|
| rectangular sma (A100 dense / sparse; RAL dense / sparse) | 2.7e-10 / 2.4e-10; 1.2e-10 / 1.2e-10 | 2.7% / 2.4%; 1.2% / 1.2% |
| delaunay sma | 1.9e-11 (A100 dense) | 0.2% |
| alma, alma_high, jvla (both meshes, every setup) | 6.1e-14 or less | below 0.01% |
| numba sma (both meshes) | 0 (bit-identical) | 0 |
| mge sma | 1.4e-12 | below 0.01% |

The expectation that Delaunay would sit closest to the bound at sma did not hold: the largest
deltas are the rectangular sma cells, and every one is two orders of magnitude inside the bound.

## Per-call evidence (batch 1, tag 2026.10.25.1)

JAX: `single-jit median` is the median of individually timed calls after the warm calls; †
marks rows whose leg recorded only the one-block mean (`full_pipeline_single_jit`). `vmap per
call` is the batch time over the batch size at the batch the script chose. numba: `warm iid
median` is the median over seeded iid draws after the numba compile (16 at sma and alma, 6 at
alma_high, 1 at jvla, per R4's `--n-instances`); `cold median` repeats the first instance 3
times. The numba headline is the route the packaged cost model picks (R4). numba runs one
serial call per core; JAX CPU uses 8 threads (`cpus_per_eval` 8). Per-call times are per
device, not per node.

**Load.** The four RAL CPU arrays shared euclid-ral-compute-1, and every RAL CPU row started at
a 1-minute load average of 45-72 (the A100 rows at 1-7). The dashboard's cap is 8, so all 38
RAL CPU rows (and the two sma sweep comparisons built from them) are refused as trend points
(refused 58 -> 100). They stay `current` in the catalogue and coverage; the cap governs trend
points only. The RAL CPU numbers are upper bounds on a quiet node's.

### interferometer/mge (sma)

| setup @ host | single-jit median (ms) | vmap per call (ms) [batch] |
|---|---|---|
| `jax_cpu_dense_fp64_nufft@hpc_ral_cpu` | 90.63 | 112.47 [3] |
| `jax_gpu_dense_fp64_nufft@hpc_a100` **fastest** | 3.72 | 0.57 [64] |

### interferometer/rectangular

| instrument | setup @ host | single-jit median (ms) | vmap per call (ms) [batch] | numba warm iid median (ms) | numba cold median (ms) | numba route |
|---|---|---|---|---|---|---|
| sma | `jax_cpu_dense_fp64_nufft@hpc_ral_cpu` | 3,781 † | 3,488 [3] | | | |
| sma | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 856.0 | 757.7 [3] | | | |
| sma | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 287.3 | 234.8 | numba 20/20 |
| sma | `jax_gpu_dense_fp64_nufft@hpc_a100` | 50.06 | 50.13 [1] | | | |
| sma | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 22.95 | 7.19 [64] | | | |
| alma | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 2,907 † | 2,796 [3] | | | |
| alma | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 897.9 | 906.7 | numba 20/20 |
| alma | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 36.01 | 51.22 [3] | | | |
| alma_high | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 8,731 † | 9,809 [3] | | | |
| alma_high | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 17,723 | 12,983 | numpy_fft 10/10 |
| alma_high | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 73.70 | 86.62 [3] | | | |
| jvla | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 58,366 † | 58,633 [3] | | | |
| jvla | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 71,933 | 71,571 | numpy_fft 5/5 |
| jvla | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 437.8 | 450.6 [3] | | | |

### interferometer/delaunay

| instrument | setup @ host | single-jit median (ms) | vmap per call (ms) [batch] | numba warm iid median (ms) | numba cold median (ms) | numba route |
|---|---|---|---|---|---|---|
| sma | `jax_cpu_dense_fp64_nufft@hpc_ral_cpu` | 6,411 † | 5,702 [3] | | | |
| sma | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 1,168.7 | 859.2 [3] | | | |
| sma | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 335.1 | 339.7 | numba 20/20 |
| sma | `jax_gpu_dense_fp64_nufft@hpc_a100` | 55.86 | 55.77 [1] | | | |
| sma | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 27.49 | 12.09 [64] | | | |
| alma | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 2,845 † | 2,712 [3] | | | |
| alma | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 823.6 | 877.7 | numba 20/20 |
| alma | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 40.26 | 22.65 [64] | | | |
| alma_high | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 9,692 † | 8,500 [3] | | | |
| alma_high | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 11,301 | 10,320 | numpy_fft 10/10 |
| alma_high | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 82.48 | 68.81 [16] | | | |
| jvla | `jax_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | 34,832 † | 34,209 [3] | | | |
| jvla | `numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu` | | | 73,492 | 81,439 | numpy_fft 5/5 |
| jvla | `jax_gpu_sparse_fp64_wtilde@hpc_a100` **fastest** | 440.2 | 459.4 [3] | | | |

### The numba route (R4)

The numba runtime rows also time the forced `direct_conv` (`numba`) and NumPy FFT arms
(warm iid medians, ms):

| cell | cost-model pick (headline) | forced `direct_conv` | forced NumPy FFT |
|---|---|---|---|
| rectangular sma | numba, 287.3 | 301.3 | 1,040.1 |
| rectangular alma | numba, 897.9 | 887.8 | 1,959.6 |
| rectangular alma_high | numpy_fft, 17,723 | 16,789 | 16,329 |
| rectangular jvla | numpy_fft, 71,933 | 437,013 | 70,490 |
| delaunay sma | numba, 335.1 | 338.4 | 1,165.6 |
| delaunay alma | numba, 823.6 | 817.9 | 3,052.2 |
| delaunay alma_high | numpy_fft, 11,301 | **8,777** | 11,268 |
| delaunay jvla | numpy_fft, 73,492 | 315,196 | 77,134 |

The cost model picks `direct_conv` at sma and alma and NumPy FFT at alma_high and jvla. At
jvla it is right by 4-6x. At delaunay alma_high it is not: `direct_conv` was 22% faster than
the NumPy FFT route it picked, on 6 warm draws on a loaded node. Flagged for phase 3.

### Comparing CPU and GPU setups

Projected calls per second per node (`projected_node_throughput`, upper bounds, parallel
efficiency 1.0 unmeasured): a ral node (236 cores) against a 4xA100 node.

| cell | numba @ ral node | JAX CPU sparse @ ral node | JAX GPU sparse @ 4xA100 node |
|---|---|---|---|
| rectangular sma | ≈822 (236 workers) | ≈38 (29) | ≈556 |
| rectangular alma | ≈263 | ≈10 | ≈78 |
| rectangular alma_high | ≈13 | ≈3.0 | ≈46 |
| rectangular jvla | ≈1.2 (84, memory-bound) | ≈0.49 | ≈8.9 |
| delaunay sma | ≈704 | ≈34 | ≈331 |
| delaunay alma | ≈287 | ≈11 | ≈177 |
| delaunay alma_high | ≈21 | ≈3.4 | ≈58 |
| delaunay jvla | ≈1.1 (78, memory-bound) | ≈0.85 | ≈8.7 |

Per call, JAX CPU sparse is faster than numba at alma_high and jvla, but it holds 8 threads per
call, so per node numba stays ahead of it on every instrument.

## Walls against `--time`

Batch 1's `--time` was 1.25x the batch-0 leg plus slack (the walls commit). Measured (sacct
Elapsed):

| leg | batch 0 | batch 1 | `--time` | used |
|---|---|---|---|---|
| rectangular jvla numba breakdown (413160_6) | 6,461 s | 8,367 s (2:19:27) | 2:30:00 | 93% |
| delaunay jvla numba breakdown (413160_2) | 5,541 s | 5,720 s (1:35:20) | 2:30:00 | 64% |
| delaunay jvla reference (413161_2) | 10,727 s | 13,307 s (3:41:47) | 4:00:00 | 92% |
| rectangular jvla reference (413161_5) | 10,093 s | 12,762 s (3:32:42) | 4:00:00 | 89% |
| A100 jvla job (413165, 4 legs in turn) | 1,776 s (sum of the 4 legs' WALL) | 1,781 s (29:41) | 1:30:00 | 33% |
| matrix-free alma 192x192 (413191_6) | not measured | killed at 12:00:29 | 12:00:00 | TIMEOUT |

The rectangular jvla numba breakdown and both jvla references ran 24-30% longer than in
batch 0. The batch-1 RAL legs ran on a node at load 45-72, which is the likely cause; the
1.25x headroom left 7-11% margin. A rerun of these legs should take batch 1's walls as its
basis, not batch 0's.

## The matrix-free experiment (unreleased main 7eba0c07, never current)

`numpy_cpu_matrixfree` rectangular runtime (`RectangularUniform`, `MaternSparse`), RAL 413191:

| instrument, mesh | matrix-free warm median | dense_fft route warm median | log evidence vs dense_fft | log det (SLQ vs Cholesky) |
|---|---|---|---|---|
| sma 35x35 | 0.48 s | 0.88 s | 0.23 nats (7.3e-5 relative; chunked dense reference the same) | 1.0e-4 |
| alma 35x35 | 1.13 s | 2.16 s | 1.29 nats (1.1e-7) | 3.1e-4 |
| alma_high 35x35 | 8.34 s | 11.54 s | 0.50 nats (8.3e-9) | 7.9e-5 |
| jvla 35x35 | 67.6 s | 56.3 s | 51.8 nats (1.7e-7) | 5.7e-3 |
| alma 64x64 | 1.69 s | 16.4 s | 0.68 nats (5.7e-8) | 5.4e-5 |
| alma 128x128 | 5.09 s | 935 s | 4.53 nats (3.8e-7) | 8.4e-5 |
| alma 192x192 | **missing (TIMEOUT, see Batch 1)** | | | |

Above sma the chunked dense reference was skipped (more than 1e5 visibilities without
`--chunked-reference`), so the agreement there is against the library's dense_fft route, not
against the R6 reference. The matrix-free route is not a parity cell: its stochastic log
determinant (SLQ, 10 probes) puts it 0.2-52 nats from the dense routes, far outside the 1e-8
bound. It is faster than the dense_fft route at every size measured except jvla 35x35, and
the gap grows with the mesh (184x at 128x128).

## Fit figures

Opened all 30 (18 distinct images; setups of one likelihood x instrument often write
byte-identical figures). Each is a seeded iid draw of the preset's priors, not a fit. The ring is
reconstructed on every instrument and the fits agree across setups. At sma the data are
noise-dominated (sigma 1000 per visibility), as expected. At alma_high and jvla the residuals
show ring-shaped structure: an iid draw at high S/N, not a fault. Nothing looked wrong in the
numbers. Presentation points, none affecting a measurement:

- Rectangular jvla (all three setups) and, faintly, alma_high: a few bright edge pixels in the
  source-plane corners. These are weakly constrained edge pixels of the adaptive rectangular
  mesh at high S/N.
- Every alma+ figure: the dirty S/N and chi-squared maps are dominated by one bright central
  pixel.
- Rectangular sma: the source-plane colour bar labels every tick 0.00.
- Delaunay jvla: the zoomed source plane frames the caustic, with the source off-centre in it.

## Journal

### 2026-10-31: phase 3, the A100 JAX legs on jax 0.11.2 (autolens_profiling#495)

All 11 A100 cells were re-measured on jax/jaxlib 0.11.2, the day-zero environment (human Q7;
A100 413622-413625). The alma, alma_high and jvla witnesses were finalised by the reference
array 413669. Every witness is `match`. Only the jax version changed on these legs (GPU, no
eigen flag, same node class), so the ratios isolate it. Prior rows are in
`history/2026.10.25.1_2026-10-28/`.

| cell (`hpc_a100`) | `vmap.per_call` before (ms) | after (ms) | after/before |
|---|---|---|---|
| delaunay alma `jax_gpu_sparse_fp64_wtilde` | 22.65 | 20.83 | 0.92 |
| delaunay alma_high `jax_gpu_sparse_fp64_wtilde` | 68.81 | 63.71 | 0.93 |
| delaunay jvla `jax_gpu_sparse_fp64_wtilde` | 459.42 | 465.18 | 1.01 |
| delaunay sma `jax_gpu_dense_fp64_nufft` | 55.77 | 57.35 | 1.03 |
| delaunay sma `jax_gpu_sparse_fp64_wtilde` | 12.09 | 11.36 | 0.94 |
| mge sma `jax_gpu_dense_fp64_nufft` | 0.57 | 0.60 | 1.05 |
| rectangular alma `jax_gpu_sparse_fp64_wtilde` | 51.22 | 48.97 | 0.96 |
| rectangular alma_high `jax_gpu_sparse_fp64_wtilde` | 86.62 | 81.06 | 0.94 |
| rectangular jvla `jax_gpu_sparse_fp64_wtilde` | 450.56 | 456.28 | 1.01 |
| rectangular sma `jax_gpu_dense_fp64_nufft` | 50.13 | 51.98 | 1.04 |
| rectangular sma `jax_gpu_sparse_fp64_wtilde` | 7.19 | 6.75 | 0.94 |

The jax upgrade is neutral to slightly faster on the A100 (0.92-1.05). No verdict on this page
moves.

### 2026-10-31: quiet-node rerun of the RAL CPU legs (autolens_profiling#493)

The 38 batch-1 RAL CPU legs ran on loaded nodes (load 45-59), so they were rerun one per
`--exclusive` node behind the quiet gate (RAL 413320-413323; 38/38 COMPLETED 0:0, every leg
`QUIET:`, all on euclid-ral-compute-10-3 at load 0.9-1.7). The rerun also moved to jax/jaxlib
0.11.2 with `--xla_cpu_multi_thread_eigen=false`, the day-zero environment (human ruling Q7,
2026-10-30), so each before/after ratio mixes load with the jax version and the flag. The new
rows replace the old ones; each prior row, its figures and witness are in
`history/2026.10.25.1_2026-10-28/`.

| cell (sma, `hpc_ral_cpu`) | statistic | before (ms) | after (ms) | after/before |
|---|---|---|---|---|
| delaunay `jax_cpu_dense_fp64_nufft` | `vmap.per_call` | 5701.81 | 5822.22 | 1.02 |
| delaunay `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 859.23 | 784.58 | 0.91 |
| delaunay `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 335.06 | 279.34 | 0.83 |
| mge `jax_cpu_dense_fp64_nufft` | `vmap.per_call` | 112.47 | 83.29 | 0.74 |
| rectangular `jax_cpu_dense_fp64_nufft` | `vmap.per_call` | 3488.11 | 5380.33 | **1.54** |
| rectangular `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 757.71 | 643.92 | 0.85 |
| rectangular `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 287.30 | 230.86 | 0.80 |

Every row but the two JAX CPU dense NUFFT rows got faster. The rectangular dense row is 1.54x
slower on a quiet node. That is consistent with the eigen flag (it turns off XLA:CPU's
multi-threaded Eigen kernels, which dense NUFFT leans on), but this rerun cannot separate the
flag from the jax version. A flag-off/on pair on one quiet node would. The dense routes were
never a CPU contender at sma (the W~ sparse rows are 7-8x faster), so no verdict on this page
moves.

The 24 rows above sma (alma, alma_high, jvla) followed once their witnesses were finalised
against references computed in the rerun checkout: RAL 413675, then finalise 413682, which
replaced 413676 (an `afterok` that could never be satisfied after a false "no row" on a missing
`output/p2b_batch1/`). 12/12 cells are `match`.

| cell (`hpc_ral_cpu`) | statistic | before (ms) | after (ms) | after/before |
|---|---|---|---|---|
| delaunay alma `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 2711.55 | 2120.71 | 0.78 |
| delaunay alma `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 823.60 | 764.95 | 0.93 |
| delaunay alma_high `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 8500.40 | 7601.44 | 0.89 |
| delaunay alma_high `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 11301.31 | 9328.69 | 0.83 |
| delaunay jvla `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 34209.21 | 51435.14 | 1.50 |
| delaunay jvla `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 73491.57 | 63799.15 | 0.87 |
| rectangular alma `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 2796.37 | 1928.96 | 0.69 |
| rectangular alma `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 897.85 | 888.77 | 0.99 |
| rectangular alma_high `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 9808.54 | 7429.59 | 0.76 |
| rectangular alma_high `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 17723.04 | 8919.04 | 0.50 |
| rectangular jvla `jax_cpu_sparse_fp64_wtilde` | `vmap.per_call` | 58633.22 | 48481.20 | 0.83 |
| rectangular jvla `numba_cpu_sparse_fp64_wtilde` | `warm_iid_median_s` | 71933.46 | 61436.05 | 0.85 |

Above sma, numba got faster everywhere on a quiet node (0.50-0.99), and most JAX CPU W~ rows did
too. The exception is delaunay jvla JAX CPU, which is **1.50x slower** (34.2 s -> 51.4 s). That
is the pattern the eigen flag left on the RAL imaging JAX CPU rows (phase 3, #495), and it
narrows JAX's per-call lead over numba on jvla Delaunay from 2.1x (34.2 s against 73.5 s) to
1.24x (51.4 s against 63.8 s). Per call the ordering holds; the per-node projection is
re-derived from these rows in the catalogue. The phase-3 designation draft's interferometer/delaunay jvla `cpu_many_core`
hold and its alma_high per-call call are re-checked on these rows.

### 2026-10-29: batch 1 committed (PR B)

72 of 73 legs landed; 60/60 cells current with 30/30 `match` witnesses; the matrix-free alma
192x192 scan timed out and is recorded as missing by human ruling. The historical
decision-matrix finding (`historical/interferometer-matrix/extent`) cited four alma rows that
moved to `history/`. Setup and record ids hash the row's path, so its setup ids, record ids and
evidence paths in `catalogue/registry.json` now name the history rows. Otherwise it would have
silently cited the new standard-size rows. The two sma sweep comparisons were re-aggregated
because the sweep layout reads the placed `hpc_*` rows.

## Caveats

- The numba fork segfault (multi-worker Nautilus, Aris campaign) does not touch these
  single-process legs. It matters in phase 4.
- Before batch 0, jvla had never been measured on CPU; batch 0's jvla CPU walls are the first.
- At sma the noise is large (sigma 1000 per visibility), so the curvature matrix is tiny and the
  numba, NumPy FFT and dense routes agree to the last printed digit there; the parity bound bites
  at alma and above.
- Per-call times are per device. Compare CPU and GPU setups as projected node throughput
  (`projected_node_throughput` in the catalogue), never as a per-call "GPU is N x faster".
