# Likelihood lifecycle phase 2c: point source, image plane

**Status:** open
**Question:** What do the four planned `point_source_image/image_plane` cells (instrument `simple`) cost on the pinned release 2026.10.25.1 on RAL CPU and the A100, and does each cell's log-likelihood match an independent eager NumPy evaluation of the same instance?
**Pre-registered rule:** the four cells (`jax_cpu_na_fp64@hpc_ral_cpu`, `jax_gpu_na_fp64@hpc_a100`, each runtime and breakdown) land `current` with provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy `AnalysisPoint(use_jax=False)` value (1e-4 x |reference| in fp64); a `diverged` witness blocks the setup until explained. Recorded before the batch runs (2026-10-28). The full rule, decisions D1-D3 and the parity reference are on the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md).
**Verdict:** the rule is met: the four cells are `current` on 2026.10.25.1 with provenance, `resources`, `fit.png` and a `match` witness (all six image-plane legs logged `match`, \|JIT − eager\| ≤ 1.2e-14 against 7.2e-4), so neither setup is blocked. As expected, the A100 is the faster host: 0.864 ms per call against 1.527 ms on RAL CPU (plain runtime), and the RAL CPU fused solved call dropped from the stale 24.7 ms to 2.01 ms without passing the A100's 1.06 ms. Expected, as written before the runs: the A100 optimal (1.09 ms against 24.7 ms in the stale 2026.8.17.1 breakdowns; the RAL CPU number should drop, but not below the A100's); every parity witness `match`.
**Headline:** `jax_gpu_na_fp64@hpc_a100` 0.864 ms per call (single-jit median, free-centre `FitPositionsImagePairAll`; solved library default 0.847 ms) against `jax_cpu_na_fp64@hpc_ral_cpu` 1.527 ms (solved 1.556 ms); A100 job 413193, RAL 413203. Which likelihood publishes as the runtime cell is open for the human (the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md#which-likelihood-the-runtime-cell-times-open-question-for-the-human)).
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the campaign branch `feature/likelihood-lifecycle-p2c-point-source`, then PR B on the same branch (rows, catalogue, coverage, these pages)
**Ledger:** the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md); work list [`scripts/misc/campaigns/point_source_p2c.py`](../../../../scripts/misc/campaigns/point_source_p2c.py)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2c; issue autolens_profiling#472
**Next:** the human's ruling on the runtime cell (plain or solved); then merge PR B after #471
**Fit figures:** opened all four image-plane `fit.png` (cells and solved experiments): the model positions land on the four observed images; nothing looked wrong.

## Why this campaign

The coverage matrix lists the two image-plane breakdowns as `stale` (2026.8.17.1) and the two
runtimes as `missing`. This page holds the image-plane half of 2c; the source-plane page holds the
shared material and the two mis-designated source-plane cells.

## The cells

| Setup @ host class | Runtime | Breakdown | Runs in |
|---|---|---|---|
| `jax_cpu_na_fp64@hpc_ral_cpu` (optimal) | `likelihood_runtime.py` | `likelihood_breakdown.py` | the `ral` array |
| `jax_gpu_na_fp64@hpc_a100` (optimal) | `likelihood_runtime.py` | `likelihood_breakdown.py` | the A100 job (`JAX_PLATFORMS=cuda`) |

`likelihood_runtime.py` times the free-centre `FitPositionsImagePairAll`; the breakdown and the
library default are the solved `FitPositionsImagePairAllSolved`. The batch also runs
`likelihood_runtime_solved.py` on both setups into `experiments/solved_runtime/`, so the human
can choose which publishes as the runtime cell (see the source-plane page).

## Fit figures

Open each new row's `fit.png` before you accept it. Note here any cell whose figure looked wrong.

Opened 2026-10-29. Both cell folders and both solved experiments show the four observed images
of `simple` on the critical curve with the model positions on them: −2 ln L = 2.22 for the
plain likelihood (A100 cell, written by its runtime leg) and 1.13 for the solved one (RAL CPU
cell, written by its breakdown leg, and both experiments). Nothing looked wrong.

## Results (RAL 413202/413203, A100 413193)

Single-jit median of 200 calls (p10–p90), ms; breakdown is the fused 5-call steady mean:

| Setup @ host class | plain runtime (published cell) | solved runtime (`experiments/solved_runtime/`) | breakdown fused solved / plain | stale breakdown superseded |
|---|---|---|---|---|
| `jax_cpu_na_fp64@hpc_ral_cpu` | 1.527 (1.452–1.920) | 1.556 (1.466–1.910) | 2.014 / 1.770 | 24.69 (2026.8.17.1) |
| `jax_gpu_na_fp64@hpc_a100` | 0.864 (0.830–0.896) | 0.847 (0.833–0.906) | 1.060 / 1.110 | 1.086 (2026.8.17.1) |

The plain A100 row's vmap batch of 64 runs at 0.025 ms per call; the solved A100 row's script
chose batch 3 (0.41 ms per call), so the vmap numbers are not like-for-like. The RAL CPU solved
experiment and breakdown wrote at load average 54–59 on a node shared with the other RAL legs,
the plain runtime at 6.1. Parity, per leg: runtime 1.8e-15 / 5.3e-15, breakdown 5.3e-15 /
1.2e-14, solved 5.3e-15 / 1.2e-14 (RAL CPU / A100), tolerance 7.2e-4 (plain) and 7.7e-4
(solved). The two stale breakdowns moved to `history/2026.8.17.1_2026-09-23/` (RAL CPU) and
`history/2026.8.17.1_2026-10-17/` (A100). Coverage: 4/4 image-plane cells current (2 stale and
2 missing before).

## Journal

### 2026-10-29 — batch committed (PR B)

All six image-plane legs exit 0 with `revisions OK` on tag 2026.10.25.1; four cells and two
solved experiments copied, the two stale breakdowns moved to `history/`. Rule met. The runtime
cell's likelihood (plain or solved) waits for the human.

### 2026-10-28 — branch ready for the batch

Rule and expected verdict recorded before any RAL run.
