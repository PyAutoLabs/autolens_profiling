# Likelihood lifecycle phase 2f: multi_dataset at the standard size

**Status:** rows committed (batch 1, PR B); closes on merge
**Question:** What does every planned multi_dataset/delaunay cell cost at the standard source size (Delaunay 1250 + 30 edge points) on the pinned release 2026.10.25.1, for the production multi-dataset likelihood (one `AnalysisImaging` per dataset summed in a `FactorGraphModel`), and does each cell's summed log likelihood match an independent reference?
**Pre-registered rule:** every one of the 10 planned multi_dataset cells lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the sum of 2a's eager NumPy dense-mapping reference per dataset (D8; tolerance 1e-4 x |reference| in fp64, the imaging rule). Any `diverged` witness blocks that setup until explained. Recorded here with the setup PR, before any run (2026-10-28).
**Verdict:** rule met: all 10 cells `current` with `fit.png` and a `match` witness (5/5, worst 1.9e-12 relative). Against the expectation: the setup ranking is 2a's (`jax_gpu_dense_fp64@hpc_a100` fastest per call, 110 ms; numba the fastest CPU setup, 2.85 s against JAX CPU 6.34 s), as predicted; "about twice 2a's hst Delaunay cell" holds only roughly: numba 3.0x (2.85 s vs 0.95 s), JAX CPU 2.9x RAL / 1.9x A100, the GPU setups 1.8x (110 vs 59.6 ms dense, 137 vs 76.2 ms sparse); every witness `match`. Per-call figures, not node throughput. Expected, written before the runs: every witness is `match`; each setup costs about twice 2a's hst Delaunay cell of the same setup (nothing is shared between the datasets, D6), so the setup ranking is 2a's: `jax_gpu_dense_fp64@hpc_a100` fastest per call, numba the fastest CPU setup per core. Class designations are phase 3's decision, not this page's.
**Headline:** multi_dataset/delaunay/hst, two datasets, steady median per call: `numba_cpu_sparse_fp64@hpc_ral_cpu` 2.85 s; `jax_cpu_sparse_fp64@hpc_ral_cpu` 6.34 s; `jax_cpu_sparse_fp64@hpc_a100` 4.92 s; `jax_gpu_dense_fp64@hpc_a100` 110 ms; `jax_gpu_sparse_fp64@hpc_a100` 137 ms (RAL 413448, A100 413456).
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** setup PR (draft, shared with the datacube half), then a walls PR after batch 0, then PR B
**Ledger:** this page; rows will land under `results/multi_dataset/delaunay/hst/`; work list [`scripts/misc/campaigns/multi_dataset_datacube_p2f.py`](../../../scripts/misc/campaigns/multi_dataset_datacube_p2f.py); the decision table D1-D9 is on the [datacube page](../datacube/likelihood_lifecycle_p2f.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2f; issue autolens_profiling#475
**Next:** merge PR B (human /prm); class designations are phase 3's
**Fit figures:** batch-1 `numba_cpu_sparse_fp64@hpc_ral_cpu` opened 2026-10-30: ring reconstructed, residuals confined to the lens core, as 2a's draw 0. Earlier: Laptop smoke (hst, dataset 0, `numba_cpu_sparse_fp64`) opened: ring reconstructed, residuals confined to the lens core as in 2a's iid draw 0, nothing wrong.

## Why this campaign

The coverage matrix listed the 10 multi_dataset/delaunay cells as missing. The only script was
a `shared_preloads` speed-up comparison (4 identical exposures, Hilbert 1500, `ConstantSplit`)
that wrote no row. There was no runtime, numba or breakdown cell.

## Decisions (multi_dataset)

All four were drafted; the human adopted every drafted option on 2026-10-29 ("ok can we do all
that now then"). The code already implemented them, so nothing changed.

| Decision | What it says | Why |
|---|---|---|
| D6 (**decided**, human 2026-10-29: adopted the recommended option) | The cell: 2a's hst Delaunay standard cell (`_production_config.standard_jax_cell("delaunay", "hst")`: the hst `source_pix[2]` preset at 1250 + 30 edge points, MGE lens light, free `AdaptSplit`, production over-sampling and adapt image, iid draw 0) fitted to 2 datasets whose data and noise map are hst's x 1 and x 0.5. They share the mask, PSF, over-sampling, adapt image, mesh and lens model. The likelihood is `FactorGraphModel` of `AnalysisImaging` with the default `shared_preloads=False` (the workspace `multi_dataset/modeling.py` configuration). | Same S/N keeps every data-derived production input identical (scaling the data alone would change the lens-subtracted S/N map, the over-sampling and the mesh). Each dataset's curvature, solve and evidence differ, so XLA cannot merge them. Dataset 0 is exactly 2a's hst cell. Two datasets mirror the workspace's two-band example. |
| D7 (**decided**, human 2026-10-29: adopted the recommended option) | Breakdown: 2a's imaging breakdown (`scripts/imaging/delaunay/likelihood_breakdown{,_numba}.py --standard-size`) of dataset 0, its `total_step_by_step` x 2 (dataset 0's own total kept as `total_step_by_step_lead_dataset`), then an eager evaluation of the sum on the cell's route for the witness. | With `shared_preloads=False` every step runs once per dataset and both datasets share one cost structure, so the sum is exactly 2 x one dataset's steps. |
| D8 (**decided**, human 2026-10-29: adopted the recommended option) | Parity: the sum over datasets of 2a's reference (`numpy_dense_reference`: eager NumPy dense-mapping `AnalysisImaging(use_jax=False)` without a sparse operator), computed in-process; tolerance 1e-4 x \|reference\|. | 2a's rule and reference, per dataset. hst is small enough to compute it in every leg. |
| D9 (**decided**, human 2026-10-29: adopted the recommended option) | Batch 0 runs all 10 multi_dataset legs (4 RAL, 6 A100) beside the datacube's bounding legs. | hst is the only instrument, and 2a's batch 0 measured ao, not hst, for Delaunay. The legs are cheap (about 2 x 2a's hst legs), so batch 0 measures them directly rather than borrowing an ao bound. |

## The cells

| Instrument | RAL CPU setups | A100 setups |
|---|---|---|
| hst | `numba_cpu_sparse_fp64`, `jax_cpu_sparse_fp64` | `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64`, `jax_cpu_sparse_fp64` (on the GPU node's CPU, inside the GPU job) |

Each with runtime and breakdown: 4 legs in `submit_campaign_p2f_multi_dataset_ral_cpu_hst_fp64`
and 6 in the A100 job `submit_campaign_p2f_multi_dataset_a100_hst_fp64` (batch 1, provisional
walls), all 10 also in batch 0. The drivers are in
`scripts/misc/likelihood_breakdown/multi_dataset_standard.py`; the cell scripts are
`scripts/multi_dataset/delaunay/likelihood_{runtime,breakdown}{,_numba}.py` (new).

## Batch 0 (2026-10-29)

| Array | Job | Legs | Result |
|-------|-----|------|--------|
| RAL CPU timing arm | 413274 (tasks 6-9) | 4 | all exit 0; slowest leg the `jax_cpu_sparse_fp64` runtime, 336 s |
| A100 timing arm | 413296 (tasks 4-9) | 6 | all exit 0; slowest leg the `jax_cpu_sparse_fp64@hpc_a100` runtime, 284 s |

Every parity witness is `match`; peak `.batch` MaxRSS 3.0 GiB (413274_7). The walls set batch
1's `--time` (RAL 0:30:00, A100 job 1:00:00) and `--mem` (32 GB, the floor). The datacube page
has the rest of the arrays and the loaded-node caveat on the RAL CPU rows.

## Caveats

- The two datasets share every input but the flux calibration. A real multi-band fit has
  different PSFs, masks or resolutions per band; these cells measure the cost of summing two
  same-shape imaging likelihoods, not a heterogeneous joint fit.
- `shared_preloads=True` (mesh sharing) is not measured here; the old
  `likelihood_runtime_shared_preloads.py` comparison script remains for that question.

## Journal

### 2026-10-28 — setup (draft for review)

Driver, four cell scripts, routes, submits and this page. Laptop smokes at hst (1 CPU): see the
setup PR for the list and any failures. Nothing submitted to RAL.

### 2026-10-29 — D6-D9 adopted; batch 0 held

D6-D9 adopted as drafted (the datacube page's journal has the details and why batch 0 was held).

### 2026-10-30 — batch 0 in; the walls commit

All 10 legs exit 0, every witness `match`; batch 1's walls and memory are measured (see Batch 0).

### 2026-10-30 — loaded RAL nodes accepted

Human ruling 2026-10-30 (chat): "ignore loaded node stuff for now I dont think that massive impacts run times and we cant ever expect to be unloaded nodes for everything". Batch 1 runs on shared `ral` nodes as rendered, with no quiet gate; a loaded row keeps its `loadavg_at_import`.

### 2026-10-30 — batch 1 in; PR B

Batch 1 (RAL 413448-413452, A100 413456-413459, 2026-10-30, from autolens_profiling_p2dg at
830314e9) all exit 0; the alma and alma_high references (413452_0/_1) finalised the datacube
witnesses, so every witness is `match` (16/16). PR B copies the 32 rows (22 datacube, 10
multi_dataset) into `results/`, re-renders `results/datacube/delaunay/sma/comparison.{json,png}`
(the legacy sweep aggregator reads the placed hpc rows, as 2b), and regenerates catalogue,
dashboard (committed stamp), coverage, setup wiki, series map and READMEs: coverage 240 -> 272
current of 272, 0 stale, 0 missing; no planned parity witness missing.
