# Likelihood lifecycle phase 2e: cluster image plane

**Status:** draft
**Question:** What does each planned cluster/image_plane cell cost on the pinned release 2026.10.25.1 with the decided workspace PointSolver (100x100 @ 1.0", precision 0.001, `residual_threshold` 10), and does each cell's JIT log-likelihood match the eager NumPy one?
**Pre-registered rule:** all 4 planned cluster/image_plane cells (`jax_cpu_na_fp64@hpc_ral_cpu`, `jax_gpu_na_fp64@hpc_a100`, each runtime and breakdown) land `current` with provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` (JIT `AnalysisPoint` vs the eager NumPy `AnalysisPoint` of the same model, instance and solver, tolerance 1e-4 x |reference| in fp64). A `diverged` witness blocks that setup until explained. The prior rows on these keys become `superseded`. Recorded here before any RAL run (2026-10-28).
**Verdict:** pending (draft; no RAL batch has run). Expected, written before the runs: the forward solve dominates every image-plane call, as in the historic breakdown (job 350580: the per-system eager fit is ~2 s, the fused JIT call 0.13 s on RAL CPU); both parity witnesses are `match`, since the solver finds every image of the `simple` truth at 1.0" (306/306 prior roots, cluster PointSolver campaign); per-call latency favours the A100, but CPU and GPU are compared by projected node throughput, not per call. Class designations are phase 3's decision.
**Headline:** none yet (laptop smokes only, below).
**Library PRs:** none (measurement only, tag 2026.10.25.1, which carries PyAutoLens#800 `residual_threshold` and bdb44a87c)
**Profiling PRs:** the 2e setup draft PR (autolens_profiling#474), then the rows PR after the human's batch go
**Ledger:** this page; the source-plane half is [likelihood_lifecycle_p2e.md](../source_plane/likelihood_lifecycle_p2e.md); work list [`scripts/misc/campaigns/cluster_p2e.py`](../../../../scripts/misc/campaigns/cluster_p2e.py) and its [README](../../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2e; issue autolens_profiling#474; solver settings from epic `cluster-pointsolver-speed` ([cluster PointSolver](cluster_pointsolver.md))
**Next:** the human reviews decisions D1-D6, then gives a go for batch 0
**Fit figures:** laptop smoke `fit.png` opened (2026-10-28): every observed image of both systems has a model image on it, χ² = 2.6; none from RAL yet.

## Why this campaign

Coverage plans 4 cluster/image_plane cells, all missing. The only rows on the likelihood are
breakdowns from 2026.8.17.1 (job 350580) with a tutorial-scale solver (200x200 @ 0.7", precision
0.01), no runtime script, no `parity.json`, and no standard cell.

## Decisions awaiting the human

| # | Decision (drafted) | Alternative |
|---|---|---|
| D1 | The standard image-plane cell uses the workspace cluster PointSolver the `cluster-pointsolver-speed` epic kept (human 2026-10-23): 100x100 @ 1.0", `pixel_scale_precision` 0.001, `magnification_threshold` 0.1, `residual_threshold` 10 (on), every setting passed explicitly (`cluster_standard.STANDARD_SOLVER`), under `--standard-size`. Not re-tuned here. Without the flag the breakdown keeps its historic solver. | Keep the historic 200x200 @ 0.7" solver for continuity with the 2026.8.17.1 rows. |
| D2 | cluster/source_plane is planned by analogy with point_source_source (see the [source-plane page](../source_plane/likelihood_lifecycle_p2e.md)). | Leave source_plane unplanned and re-measure it as unplanned rows only. |
| D3 | The headline of each runtime row is the plain fit class the cluster breakdowns decompose: `FitPositionsImagePairRepeat` (free `Point` centres); `FitPositionsImagePairRepeatSolved` (`PointSolved`) is a sub-row. | The library default since PyAutoLens d838ca596 is `FitPositionsImagePairAllSolved`; the workspace cluster search uses the source-plane `FitPositionsSourceSolved`. Either could be the headline instead. |
| D5 | Batch plan: batch 0 = the 4 image-plane legs (they bound every cluster leg), then batch 1 = all 10 runnable legs with walls from batch 0. | One batch: run batch 1 directly under its conservative provisional walls (`P2E_ALLOW_PROVISIONAL_WALL=1`). |
| D6 | The cell's `parity.json` is written by the runtime (as for imaging); breakdowns write none. Tolerance is the default 1e-4 x |reference| (no cluster override). The vmap batch is declared (CPU 3, GPU 16; the A100 vram table has no cluster entry). | A tighter tolerance; a `--vmap-probe` run first. |

D4 (the unrunnable `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells) is on the source-plane page.

## The cells and the batch plan

| Cell | Host | Batch 0 | Batch 1 |
|---|---|---|---|
| image_plane `jax_cpu_na_fp64` runtime, breakdown | `ral` array | yes (2) | yes |
| image_plane `jax_gpu_na_fp64` runtime, breakdown | A100 | yes (2) | yes |
| source_plane: 6 runnable legs | `ral` array (2), A100 job (4) | no | yes |

Batch 0: `submit_campaign_p2e_batch0_ral_cpu_fp64` (array of 2) and
`submit_campaign_p2e_batch0_a100_fp64` (array of 2, `%2`). Batch 1:
`submit_campaign_p2e_cluster_ral_cpu_fp64` (array of 4) and `submit_campaign_p2e_cluster_a100_fp64`
(one job, 6 legs in turn; the 2 `jax_cpu` legs run on the A100 node's CPU). Batch 1 refuses to run
while `WALL_BASIS_STATUS` is `provisional`. 12 coverage cells in all (image plane 4, source plane
8), 10 submitted legs, 2 not submitted (D4).

## Laptop smokes (2026-10-28, WSL2, 8 threads, local PyAutoLens at 2026.10.26.1)

Not evidence for any claim; they show the scripts run end to end and size the walls.

| Leg | Process wall | Peak RSS | Result |
|---|---|---|---|
| runtime (`--standard-size`) | 1167 s | 2.0 GB | each pipeline ~380-425 s lower + compile (plain, solved), vmap 3 first call 318 s; JIT block 37.3 ms, steady median 32.7 ms, vmap 3 37.4 ms per call; eager NumPy 5.9 s per call; parity `match` (|Δ log L| = 4e-13); solved sub-row JIT = eager to 7e-15 |
| breakdown (`--standard-size`) | 2230 s (the solved compile ran beside the test suite: 1046 s lower + 698 s compile) | 2.1 GB | per-system solve 14-16 ms; fused plain 40.7 ms, fused solved 75.0 ms; fused eager = JIT for both arms; the runtime's `parity.json` left untouched |

Both legs' `fit.png` show every observed image of both systems with a model image on it
(χ² = 2.6). The compile, not the call, sets the wall: batch 0's declared guess is 2400 s per
leg (`cluster_p2e.BATCH0_WALL_GUESS_S`), `--time` 2:30:00.

## Caveats

- The 1.0" start loses close high-|μ| image pairs on some stress draws of `simple` and the
  A2744 pilot (cluster PointSolver campaign). The profiled instance is the truth lens, whose
  images the solver finds; the timing does not test robustness.
- Per-call times are per device. Compare CPU and GPU setups as projected node throughput, never
  as a per-call "GPU is N x faster".
