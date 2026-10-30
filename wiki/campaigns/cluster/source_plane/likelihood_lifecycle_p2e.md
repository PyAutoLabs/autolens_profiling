# Likelihood lifecycle phase 2e: cluster source plane

**Status:** open
**Question:** What does the cluster source-plane likelihood (the solved, Jacobian-weighted `FitPositionsSourceSolved` the workspace cluster search fits, D3; PyAutoLens bdb44a87c) cost on the pinned release 2026.10.25.1, on the drafted planned cells, and does each cell's JIT log-likelihood match the eager NumPy one?
**Pre-registered rule:** if the human keeps D2 (kept 2026-10-29), the 6 runnable planned cluster/source_plane cells (`jax_cpu_na_fp64` on `hpc_ral_cpu` and `hpc_a100`, `jax_gpu_na_fp64` on `hpc_a100`, each runtime and breakdown) land `current` with provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` (JIT vs eager NumPy `AnalysisPoint`, 1e-4 x |reference| in fp64). A `diverged` witness blocks that setup until explained. Recorded here before any RAL run (2026-10-28). Amended 2026-10-29, still before any run: the witness and `fit.png` evaluate the solved headline class (D3), and the 2 `jax_gpu_na_fp64@hpc_ral_cpu` cells the draft left to D4 are no longer planned (autolens_profiling#481).
**Verdict:** pending (no RAL batch has run; the source plane is in batch 1, not batch 0). Expected, written before the runs: the solved headline call costs milliseconds (laptop smoke 2026-10-29: 1.41 ms JIT block, 0.46 ms per call at vmap 3; the plain sub-row 1.57 ms), one to two orders below the image plane; on the A100 node the `jax_cpu` setup beats `jax_gpu` per call because launch latency dominates (the §3.3 point_source_source reasoning); every parity witness `match` (laptop, solved headline: |Δ log L| = 2e-10). Class designations are phase 3's decision.
**Headline:** none yet (laptop smokes only).
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the 2e setup draft PR (autolens_profiling#474), then the rows PR after the human's batch go
**Ledger:** this page and the [image-plane page](../image_plane/likelihood_lifecycle_p2e.md), which holds the shared decisions and the batch plan; work list [`scripts/misc/campaigns/cluster_p2e.py`](../../../../scripts/misc/campaigns/cluster_p2e.py)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2e; issue autolens_profiling#474
**Next:** batch 1 (human go), which runs these 6 legs under the image-plane batch 0's walls (in since 2026-10-30, see the image-plane page)
**Fit figures:** laptop smoke `fit.png` of the solved headline opened (2026-10-29): both systems' images, critical curves per source redshift and mass centres drawn, χ² = 0.033 (the plain fit's was 0.064, 2026-10-28); none from RAL yet.

## Why this campaign

The committed cluster/source_plane rows are archived breakdowns from 2026.7.x, before PyAutoLens
bdb44a87c (2026-10-21) changed `FitPositionsSource` to the Jacobian precision tensor and a
model-independent noise term. The breakdown script was re-derived in autolens_profiling#455. There
was no runtime script, and coverage plans no source_plane cell.

## Decisions

D2, D3 and D4 are decided (human, 2026-10-29). D2 was drafted; the human adopted the drafted
option on 2026-10-29 ("ok can we do all that now then"), which changed no code or submit.

| # | Decision | Alternative (not taken) |
|---|---|---|
| D2 (decided, human 2026-10-29: adopted the recommended option) | Plan cluster/source_plane by adding `cluster/source_plane/unspecified` to `catalogue/registry.json`. The catalogue already routes it to the point_source_source §3.3 table, so coverage plans what point_source_source plans: `jax_cpu_na_fp64` optimal on `hpc_ral_cpu` and `hpc_a100`, `jax_gpu_na_fp64` secondary on both (8 cells; 6 since the GPU-host planner fix, see D4). | Drop the registry cell: re-measure source_plane as unplanned rows only, and coverage plans the 4 image-plane cells. |
| D4 (decided, human 2026-10-29) | The 2 `jax_gpu_na_fp64@hpc_ral_cpu` cells (a GPU setup on the CPU-only `ral` host; the §3.3 table puts its secondary on every host) are not planned. Ruling: "fix the planner gap" (seen in 2c, 2e and 2g). autolens_profiling#481, merged into this branch, adds `gpu_hosts` (`hpc_a100`) to `catalogue/provisional_designations.json`; `catalogue_lifecycle.planned_setups` plans a GPU setup only on a GPU host. The setup stays designated. It also drops the 2 point_source_source cells on the same key. | Drafted: leave them `missing` and unsubmitted until a host-scoped rule. |
| D3 (decided, human 2026-10-29) | Headline `FitPositionsSourceSolved` (`PointSolved`: β* solved analytically and marginalised over), the fit the workspace cluster search uses (`autolens_workspace/scripts/cluster/modeling.py`, `fit_positions_cls=al.FitPositionsSourceSolved`); plain `FitPositionsSource` (free `Point` centres) is a sub-row. The runtime times, vmaps and witnesses the solved class and draws its `fit.png`; the breakdown decomposes it (trace, Jacobian, β*, χ² about β*, assembly with the marginalisation term). Ruling: "the cluster default setup should use solved as I think it does on the workspace". | Drafted: plain headline, solved sub-row. |

D1, D5 and D6 are on the image-plane page.

## Laptop smokes (2026-10-29, WSL2 sandbox, 1 CPU, local PyAutoLens at 2026.10.26.1; solved headline, D3)

| Leg | Process wall | Peak RSS | Result |
|---|---|---|---|
| runtime (`--standard-size`) | 42 s | 0.54 GB | headline `FitPositionsSourceSolved`: JIT block 1.41 ms, steady median 1.42 ms, vmap 3 0.46 ms per call; parity `match` (26.063158936 both sides, \|Δ log L\| = 2e-10). Plain sub-row: JIT block 1.57 ms, JIT = eager to 6e-11 |
| breakdown | 25 s | 0.49 GB | solved step total 6.5 ms (trace, Jacobian, β*, χ² about β*, assembly); decomposition matches `FitPositionsSourceSolved` and β* the fit's `source_plane_coordinate`; plain sub-row matches `FitPositionsSource`, both at rtol 1e-4 |

## Laptop smokes (2026-10-28, WSL2, 8 threads, local PyAutoLens at 2026.10.26.1; plain headline, before D3)

| Leg | Process wall | Peak RSS | Result |
|---|---|---|---|
| runtime (`--standard-size`) | 152 s (compile of plain + solved dominates) | 0.64 GB | JIT block 3.91 ms, steady median 5.77 ms, vmap 3 1.40 ms per call; parity `match` (52.520324224 both sides); solved sub-row JIT = eager to 2e-10 |
| breakdown | 33 s | 0.46 GB | component total 6.7 ms; decomposition matches `FitPositionsSource` at rtol 1e-4 |

## Caveats

- Per-call times are per device. Compare CPU and GPU setups as projected node throughput.
