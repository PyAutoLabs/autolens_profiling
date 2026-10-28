# Likelihood lifecycle phase 2c: point source, image plane

**Status:** open
**Question:** What do the four planned `point_source_image/image_plane` cells (instrument `simple`) cost on the pinned release 2026.10.25.1 on RAL CPU and the A100, and does each cell's log-likelihood match an independent eager NumPy evaluation of the same instance?
**Pre-registered rule:** the four cells (`jax_cpu_na_fp64@hpc_ral_cpu`, `jax_gpu_na_fp64@hpc_a100`, each runtime and breakdown) land `current` with provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy `AnalysisPoint(use_jax=False)` value (1e-4 x |reference| in fp64); a `diverged` witness blocks the setup until explained. Recorded before the batch runs (2026-10-28). The full rule, decisions D1-D3 and the parity reference are on the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md).
**Verdict:** pending (batch not run). Expected, written before the runs: the A100 is the optimal host for the image-plane likelihood (fused solved call 1.09 ms on the A100 against 24.7 ms on RAL CPU in the stale 2026.8.17.1 breakdown rows; the RAL CPU number predates the IP-2..IP-4c levers and should drop, but not below the A100's); every parity witness is `match`.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the campaign branch `feature/likelihood-lifecycle-p2c-point-source`, then PR B
**Ledger:** the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md); work list [`scripts/misc/campaigns/point_source_p2c.py`](../../../../scripts/misc/campaigns/point_source_p2c.py)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2c; issue autolens_profiling#472
**Next:** the batch, then PR B
**Fit figures:** none yet.

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

## Journal

### 2026-10-28 — branch ready for the batch

Rule and expected verdict recorded before any RAL run.
