# Likelihood lifecycle phase 2e: cluster image plane

**Status:** open
**Question:** What does each planned cluster/image_plane cell cost, for the solved image-plane likelihood (`FitPositionsImagePairRepeatSolved`, D3), on the pinned release 2026.10.25.1 with the decided workspace PointSolver (100x100 @ 1.0", precision 0.001, `residual_threshold` 10), and does each cell's JIT log-likelihood match the eager NumPy one?
**Pre-registered rule:** all 4 planned cluster/image_plane cells (`jax_cpu_na_fp64@hpc_ral_cpu`, `jax_gpu_na_fp64@hpc_a100`, each runtime and breakdown) land `current` with provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` (JIT `AnalysisPoint` vs the eager NumPy `AnalysisPoint` of the same model, instance and solver, tolerance 1e-4 x |reference| in fp64). A `diverged` witness blocks that setup until explained. The prior rows on these keys become `superseded`. Recorded here before any RAL run (2026-10-28). Amended 2026-10-29, still before any run: the witness and `fit.png` evaluate the solved headline class (D3).
**Verdict:** pending (batch 0 run 2026-10-29; batch 1 not yet). Expected, written before the runs: the forward solve dominates every image-plane call, as in the historic breakdown (job 350580: the per-system eager fit is ~2 s, the fused JIT call 0.13 s on RAL CPU for the plain class and 0.135 s for the solved one); the solved headline costs the same order per call as the plain sub-row (laptop smokes disagree on which is faster: runtime 10.7 vs 18.1 ms, breakdown fused 39.8 vs 16.5 ms, both on a loaded single CPU); both parity witnesses are `match`, since the solver finds every image of the `simple` truth at 1.0" (306/306 prior roots, cluster PointSolver campaign); per-call latency favours the A100, but CPU and GPU are compared by projected node throughput, not per call. Class designations are phase 3's decision.
**Headline:** none yet (laptop smokes only, below).
**Library PRs:** none (measurement only, tag 2026.10.25.1, which carries PyAutoLens#800 `residual_threshold` and bdb44a87c)
**Profiling PRs:** the 2e setup draft PR (autolens_profiling#474), then the rows PR after the human's batch go
**Ledger:** this page; the source-plane half is [likelihood_lifecycle_p2e.md](../source_plane/likelihood_lifecycle_p2e.md); work list [`scripts/misc/campaigns/cluster_p2e.py`](../../../../scripts/misc/campaigns/cluster_p2e.py) and its [README](../../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2e; issue autolens_profiling#474; solver settings from epic `cluster-pointsolver-speed` ([cluster PointSolver](cluster_pointsolver.md))
**Next:** batch 1 (human go; RAL CPU legs on shared nodes, no quiet gate, human 2026-10-30), from the walls commit; then the rows PR
**Fit figures:** laptop smoke `fit.png` of the solved headline opened (2026-10-29, runtime and breakdown): every observed image of both systems has a model image from β* on it, χ² = 0.065 (the plain fit's was 2.6, 2026-10-28); none from RAL yet.

## Why this campaign

Coverage plans 4 cluster/image_plane cells, all missing. The only rows on the likelihood are
breakdowns from 2026.8.17.1 (job 350580) with a tutorial-scale solver (200x200 @ 0.7", precision
0.01), no runtime script, no `parity.json`, and no standard cell.

## Decisions

All six are decided. D3 (human, 2026-10-29) is below, D4 on the source-plane page. D1, D2, D5
and D6 were drafted; the human adopted each drafted option on 2026-10-29 ("ok can we do all
that now then", in reply to the list of open decisions and the batch-0 go). The code already
implemented the drafts, so the adoption changed no code or submit. The human can still change
any of them before batch 1.

| # | Decision | Alternative (not taken) |
|---|---|---|
| D1 (decided, human 2026-10-29: adopted the recommended option) | The standard image-plane cell uses the workspace cluster PointSolver the `cluster-pointsolver-speed` epic kept (human 2026-10-23): 100x100 @ 1.0", `pixel_scale_precision` 0.001, `magnification_threshold` 0.1, `residual_threshold` 10 (on), every setting passed explicitly (`cluster_standard.STANDARD_SOLVER`), under `--standard-size`. Not re-tuned here. Without the flag the breakdown keeps its historic solver. | Keep the historic 200x200 @ 0.7" solver for continuity with the 2026.8.17.1 rows. |
| D2 (decided, human 2026-10-29: adopted the recommended option) | cluster/source_plane is planned by analogy with point_source_source (see the [source-plane page](../source_plane/likelihood_lifecycle_p2e.md)). | Leave source_plane unplanned and re-measure it as unplanned rows only. |
| D3 (decided, human 2026-10-29) | The headline of each runtime row is the solved fit class: `FitPositionsImagePairRepeatSolved` (`PointSolved`; each source centre is β*, the Jacobian-tensor-weighted mean of its back-traced images), the solved counterpart of the class the cluster breakdowns decomposed. Plain `FitPositionsImagePairRepeat` (free `Point` centres) is a sub-row. The runtime times, vmaps and witnesses the solved class and draws its `fit.png`; the breakdown decomposes it (β*, the solve at β*, the solved fit total; fused solved first). The source plane headlines `FitPositionsSourceSolved`, the fit the workspace cluster search uses (`autolens_workspace/scripts/cluster/modeling.py`). Ruling: "the cluster default setup should use solved as I think it does on the workspace". | Drafted: plain headline, solved sub-row. The library default since PyAutoLens d838ca596 is `FitPositionsImagePairAllSolved`. |
| D5 (decided, human 2026-10-29: adopted the recommended option) | Batch plan: batch 0 = the 4 image-plane legs (they bound every cluster leg), then batch 1 = all 10 runnable legs with walls from batch 0. | One batch: run batch 1 directly under its conservative provisional walls (`P2E_ALLOW_PROVISIONAL_WALL=1`). |
| D6 (decided, human 2026-10-29: adopted the recommended option) | The cell's `parity.json` is written by the runtime (as for imaging); breakdowns write none. Tolerance is the default 1e-4 x |reference| (no cluster override). The vmap batch is declared (CPU 3, GPU 16; the A100 vram table has no cluster entry). | A tighter tolerance; a `--vmap-probe` run first. |

D4 (the `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells, no longer planned) is on the source-plane page.

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
while `WALL_BASIS_STATUS` is `provisional`. 10 coverage cells in all (image plane 4, source plane
6), all 10 submitted. The planner plans a GPU setup only on a GPU host (autolens_profiling#481,
D4), so the 2 `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells the draft held back are no longer
planned.

## Laptop smokes (2026-10-29, WSL2 sandbox, 1 CPU, local PyAutoLens at 2026.10.26.1; solved headline, D3)

Not evidence for any claim. The breakdown ran beside the test suite.

| Leg | Process wall | Peak RSS | Result |
|---|---|---|---|
| runtime (`--standard-size`) | 756 s | 2.1 GB | lower + compile 122 s (solved), 109 s (plain); vmap 3 first call 473 s. Headline `FitPositionsImagePairRepeatSolved`: JIT block 10.7 ms, steady median 19.2 ms, vmap 3 9.0 ms per call; eager NumPy 2.0 s per call; parity `match` (\|Δ log L\| = 7e-15). Plain sub-row: JIT block 18.1 ms, JIT = eager to 4e-13 |
| breakdown (`--standard-size`) | 660 s | 1.9 GB | β* 0.19 s per system (eager), solve at β* 10-14 ms; solved fit total 3.2-3.6 s, plain 2.3-2.6 s (eager); fused solved 39.8 ms, fused plain 16.5 ms; fused eager = JIT for both arms; 3/3 finite solved positions per system |

## Laptop smokes (2026-10-28, WSL2, 8 threads, local PyAutoLens at 2026.10.26.1; plain headline, before D3)

Not evidence for any claim; they show the scripts run end to end and size the walls.

| Leg | Process wall | Peak RSS | Result |
|---|---|---|---|
| runtime (`--standard-size`) | 1167 s | 2.0 GB | each pipeline ~380-425 s lower + compile (plain, solved), vmap 3 first call 318 s; JIT block 37.3 ms, steady median 32.7 ms, vmap 3 37.4 ms per call; eager NumPy 5.9 s per call; parity `match` (|Δ log L| = 4e-13); solved sub-row JIT = eager to 7e-15 |
| breakdown (`--standard-size`) | 2230 s (the solved compile ran beside the test suite: 1046 s lower + 698 s compile) | 2.1 GB | per-system solve 14-16 ms; fused plain 40.7 ms, fused solved 75.0 ms; fused eager = JIT for both arms; the runtime's `parity.json` left untouched |

Both legs' `fit.png` show every observed image of both systems with a model image on it
(χ² = 2.6). The compile, not the call, sets the wall: batch 0's declared guess is 2400 s per
leg (`cluster_p2e.BATCH0_WALL_GUESS_S`), `--time` 2:30:00.

## Batch 0 (2026-10-29)

| Array | Job | Legs | Result |
|-------|-----|------|--------|
| RAL CPU timing arm | 413273 | 2 | both exit 0; runtime 360 s, breakdown 281 s (sacct Elapsed) |
| A100 timing arm | 413295 | 2 | both exit 0; runtime 403 s, breakdown 309 s |

Both parity witnesses are `match`. Peak `.batch` MaxRSS: 2.3 GiB (RAL), 2.6 GiB (A100). Every
leg took about 60 s more than its WALL for the cluster dataset preflight. The walls set batch 1's
`--time` (RAL 0:30:00, A100 job 1:00:00) and `--mem` (32 GB, the floor); the generator holds the
numbers. The RAL CPU rows were measured on a loaded node (`loadavg_at_import` 106, shared with
euclid SLAM tasks), so they would be refused as trend points; the A100 rows are quiet.

## Caveats

- The 1.0" start loses close high-|μ| image pairs on some stress draws of `simple` and the
  A2744 pilot (cluster PointSolver campaign). The profiled instance is the truth lens, whose
  images the solver finds; the timing does not test robustness.
- Per-call times are per device. Compare CPU and GPU setups as projected node throughput, never
  as a per-call "GPU is N x faster".

## Journal

### 2026-10-29 — D1, D2, D5, D6 adopted; batch 0 held

The human adopted the drafted options of D1 (the workspace cluster PointSolver), D2 (plan
source_plane by analogy with point_source_source), D5 (batch 0 = the 4 image-plane legs, then
batch 1) and D6 (runtime writes `parity.json`, default tolerance, declared vmap batch): "ok can
we do all that now then". D3 and D4 were already decided. No code, plan or submit changed.

The batch-0 go was given with the decisions, but batch 0 was not submitted. The shared RAL venv (`/mnt/ral/jnightin/PyAuto/PyAuto`, which every submit activates) carries jax/jaxlib 0.10.2 (`site-packages/jax-0.10.2.dist-info`, `jaxlib-0.10.2.dist-info`), while the pin's PyAutoNerves excludes 0.10.* ("CPU batched LAPACK can deadlock in 0.10.x / 0.11.0", Heart#274). Rows from that environment would not match the pin's dependency floor, so the environment is the human's call. Later the same evening the human also said not to start anything new on RAL. A detached checkout of main 32c0becd was being prepared at `/mnt/ral/jnightin/profiling/autolens_profiling_p2dg`, but nothing has been submitted from it.

### 2026-10-30 — batch 0 in; the walls commit

Batch 0 ran 2026-10-29 (RAL 413273, A100 413295; every leg exit 0, both witnesses `match`). The
walls commit fills `BATCH0_WALLS` / `BATCH0_PEAK_RSS_KIB` and flips `WALL_BASIS_STATUS` to
`measured`: batch 1 runs 0:30:00 (RAL) / 1:00:00 (A100) at 32 GB.

### 2026-10-30 — loaded RAL nodes accepted

Human ruling 2026-10-30 (chat): "ignore loaded node stuff for now I dont think that massive impacts run times and we cant ever expect to be unloaded nodes for everything". Batch 1 runs on shared `ral` nodes as rendered, with no quiet gate; a loaded row keeps its `loadavg_at_import`.
