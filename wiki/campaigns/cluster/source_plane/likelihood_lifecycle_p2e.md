# Likelihood lifecycle phase 2e: cluster source plane

**Status:** draft
**Question:** What does the cluster source-plane likelihood (Jacobian-weighted `FitPositionsSource`, PyAutoLens bdb44a87c) cost on the pinned release 2026.10.25.1, on the drafted planned cells, and does each cell's JIT log-likelihood match the eager NumPy one?
**Pre-registered rule:** if the human keeps D2, the 6 runnable planned cluster/source_plane cells (`jax_cpu_na_fp64` on `hpc_ral_cpu` and `hpc_a100`, `jax_gpu_na_fp64` on `hpc_a100`, each runtime and breakdown) land `current` with provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` (JIT vs eager NumPy `AnalysisPoint`, 1e-4 x |reference| in fp64). A `diverged` witness blocks that setup until explained. The 2 `jax_gpu_na_fp64@hpc_ral_cpu` cells are resolved by the human's D4, not by a run. Recorded here before any RAL run (2026-10-28).
**Verdict:** pending (draft; no RAL batch has run). Expected, written before the runs: the call costs milliseconds (laptop smoke 3.9 ms JIT block, 1.4 ms per call at vmap 3), two to three orders below the image plane; on the A100 node the `jax_cpu` setup beats `jax_gpu` per call because launch latency dominates (the §3.3 point_source_source reasoning); every parity witness `match` (laptop: |Δ log L| = 6e-11). Class designations are phase 3's decision.
**Headline:** none yet (laptop smokes only).
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the 2e setup draft PR (autolens_profiling#474), then the rows PR after the human's batch go
**Ledger:** this page and the [image-plane page](../image_plane/likelihood_lifecycle_p2e.md), which holds the shared decisions and the batch plan; work list [`scripts/misc/campaigns/cluster_p2e.py`](../../../../scripts/misc/campaigns/cluster_p2e.py)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2e; issue autolens_profiling#474
**Next:** the human decides D2 and D4, then gives a go for batch 1 (the source plane is not in batch 0)
**Fit figures:** laptop smoke `fit.png` opened (2026-10-28): both systems' images, critical curves per source redshift and mass centres drawn, χ² = 0.064; none from RAL yet.

## Why this campaign

The committed cluster/source_plane rows are archived breakdowns from 2026.7.x, before PyAutoLens
bdb44a87c (2026-10-21) changed `FitPositionsSource` to the Jacobian precision tensor and a
model-independent noise term. The breakdown script was re-derived in autolens_profiling#455. There
was no runtime script, and coverage plans no source_plane cell.

## Decisions awaiting the human

| # | Decision (drafted) | Alternative |
|---|---|---|
| D2 | Plan cluster/source_plane by adding `cluster/source_plane/unspecified` to `catalogue/registry.json`. The catalogue already routes it to the point_source_source §3.3 table, so coverage plans what point_source_source plans: `jax_cpu_na_fp64` optimal on `hpc_ral_cpu` and `hpc_a100`, `jax_gpu_na_fp64` secondary on both (8 cells). | Drop the registry cell: re-measure source_plane as unplanned rows only, and coverage plans the 4 image-plane cells. |
| D4 | The 2 `jax_gpu_na_fp64@hpc_ral_cpu` cells (a GPU setup on the CPU-only `ral` host; the §3.3 table puts its secondary on every host) are not submitted. They stay `missing` until the human scopes them out, for example with a host-scoped rule beside `instrument_scoped_setups` in `catalogue/provisional_designations.json`. Such a rule would also apply to point_source_source (phase 2c), so it is left to the human. | Run the `jax_gpu` setup's script on `ral` with `JAX_PLATFORMS=cpu` and file it under that key. That would mislabel a CPU run as a GPU setup. |
| D3 | Headline `FitPositionsSource`; `FitPositionsSourceSolved` (the workspace cluster search's fit) is a sub-row. | Make the solved fit the headline. |

D1, D5 and D6 are on the image-plane page.

## Laptop smokes (2026-10-28, WSL2, 8 threads, local PyAutoLens at 2026.10.26.1)

| Leg | Process wall | Peak RSS | Result |
|---|---|---|---|
| runtime (`--standard-size`) | 152 s (compile of plain + solved dominates) | 0.64 GB | JIT block 3.91 ms, steady median 5.77 ms, vmap 3 1.40 ms per call; parity `match` (52.520324224 both sides); solved sub-row JIT = eager to 2e-10 |
| breakdown | 33 s | 0.46 GB | component total 6.7 ms; decomposition matches `FitPositionsSource` at rtol 1e-4 |

## Caveats

- Per-call times are per device. Compare CPU and GPU setups as projected node throughput.
