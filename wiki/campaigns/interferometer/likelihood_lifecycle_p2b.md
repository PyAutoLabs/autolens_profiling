# Likelihood lifecycle phase 2b: interferometer at the standard size

**Status:** open
**Question:** What does every planned interferometer/{mge, rectangular, delaunay} cell cost at the standard source size (Delaunay 1250 + 30 edge, rectangular 35x35) on the pinned release 2026.10.25.1, on the sparse W~ and dense NUFFT routes, and does each cell's log evidence match an independent dense reference?
**Pre-registered rule:** every one of the 60 planned cells lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the visibility-chunked eager NumPy dense reference (tolerance 1e-8 x |reference| in fp64, 1e-3 mixed precision; rulings R6, R7). Any `diverged` witness blocks that setup until explained. Recorded here with PR A, before batch 0 runs (2026-10-27).
**Verdict:** pending (no runs yet). Expected, written before the runs: W~ is the optimal route on both hosts for every pixelized cell, with the packaged cost model picking the NumPy FFT route over numba `direct_conv` at jvla on RAL CPU (survey 2026-10-25); every parity witness is `match`, with the Delaunay cells closest to the bound at sma (about 1e-8 relative jitter between the scipy and JAX Delaunay walks). Class designations are phase 3's decision, not this page's.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1); the matrix-free experiment reads PyAutoArray#658 from a main checkout
**Profiling PRs:** PR A (standard-size interferometer cells, numba runtime scripts, `--dense` breakdowns, chunked parity reference, campaign submits, matrix-free harness), then a walls PR after batch 0, then PR B (rows, catalogue, coverage, this page's result)
**Ledger:** this page; rows will land under `results/interferometer/{mge,rectangular,delaunay}/`; work list [`scripts/misc/campaigns/interferometer_p2b.py`](../../../scripts/misc/campaigns/interferometer_p2b.py) and its [README](../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2b; `active/likelihood_lifecycle_p2b_interferometer_campaign.md`; issue autolens_profiling#469
**Next:** human go for batch 0 (RAL `ral` array of 14, A100 array of 10, optional jvla reference tasks); then set batch 1's `--time` and `--mem` from batch 0

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

## Caveats

- The numba fork segfault (multi-worker Nautilus, Aris campaign) does not touch these
  single-process legs. It matters in phase 4.
- jvla has never been measured on CPU; every CPU jvla wall before batch 0 is a projection.
- At sma the noise is large (sigma 1000 per visibility), so the curvature matrix is tiny and the
  numba, NumPy FFT and dense routes agree to the last printed digit there; the parity bound bites
  at alma and above.
- Per-call times are per device. Compare CPU and GPU setups as projected node throughput
  (`projected_node_throughput` in the catalogue), never as a per-call "GPU is N x faster".
