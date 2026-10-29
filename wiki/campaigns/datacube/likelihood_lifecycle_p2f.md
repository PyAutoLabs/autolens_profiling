# Likelihood lifecycle phase 2f: datacube at the standard size

**Status:** open
**Question:** What does every planned datacube/delaunay cell cost at the standard source size (Delaunay 1250 + 30 edge points) on the pinned release 2026.10.25.1, for the production datacube likelihood (one channel per `AnalysisInterferometer`, `shared_preloads=True`, summed in a `FactorGraphModel`), on the W~ and dense routes, and does each cell's cube log likelihood match an independent reference?
**Pre-registered rule:** every planned datacube cell (22 under D1) lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the cube reference derived from 2b's visibility-chunked eager NumPy dense reference (D3; tolerance 1e-8 x |reference| in fp64, 2b's R7). Any `diverged` witness blocks that setup until explained. Recorded here with the setup PR, before any run (2026-10-28).
**Verdict:** pending (decisions D1-D9 adopted 2026-10-29; batch 0 not yet submitted). Expected, written before the runs: every witness is `match` (the sma smokes match to 7e-8 nats against a 1.2e-4 tolerance); W~ is the cheaper route on both hosts at every instrument; the cube costs much less than 34 single channels, because the lead channel's curvature matrix is shared and only the per-channel solve, log determinants and data vector repeat; on RAL CPU the packaged cost model picks the NumPy FFT route over numba `direct_conv` at alma_high as it did for the interferometer. Class designations are phase 3's decision, not this page's.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** setup PR (draft; datacube and multi_dataset standard-size drivers, the 2f submits, this page), then a walls PR after batch 0, then PR B (rows, catalogue, coverage, this page's result)
**Ledger:** this page; rows will land under `results/datacube/delaunay/`; work list [`scripts/misc/campaigns/multi_dataset_datacube_p2f.py`](../../../scripts/misc/campaigns/multi_dataset_datacube_p2f.py) and its [README](../../../scripts/misc/campaigns/README.md); the multi_dataset half is [its own page](../multi_dataset/likelihood_lifecycle_p2f.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2f; issue autolens_profiling#475
**Next:** batch 0 with the alma_high reference (go given 2026-10-29; held: the RAL venv carries jax/jaxlib 0.10.2, which the pin's PyAutoNerves excludes; the human decides the environment); then the walls commit, batch 1 (human go), PR B
**Fit figures:** none from RAL yet. Laptop smoke (sma, lead channel, `jax_cpu_sparse_fp64_wtilde`) opened: noise-dominated sma dirty maps as in 2b, the source reconstructed inside the caustic, nothing wrong.

## Why this campaign

The coverage matrix listed 30 datacube/delaunay cells (and 10 multi_dataset/delaunay) as
missing. The old datacube scripts ran 34 identical copies of a Hilbert-1500 `ConstantSplit`
channel, timed only the eager fit (the cube JIT was opt-in), had no numba script, no
per-setup route control, no resources, no `fit.png` and no parity witness. The human
(2026-10-28): set the other dataset types up and leave them for review before any RAL run.

## Decisions (D1-D9)

D1-D5 and D9 are the datacube's; D6-D8 are the multi_dataset page's, listed here so the set is
in one place. All nine were drafted; the human adopted every drafted option on 2026-10-29 ("ok
can we do all that now then", in reply to the list of open decisions and the batch-0 go). The
code already implemented the drafts, so the adoption changed no code or submit. The human can
still change any of them before batch 1.

| Decision | What it says | Why |
|---|---|---|
| D1 (**decided**, human 2026-10-29: adopted the recommended option) | Dense datacube NUFFT cells (`jax_cpu_dense_fp64_nufft`, `jax_gpu_dense_fp64_nufft`) are planned at sma only, 2b's R1(a) carried over: 30 datacube cells become 22, the campaign 40 becomes 32. Recorded as a second `instrument_scoped_setups` rule in `catalogue/provisional_designations.json`. | Every channel is the instrument's interferometer dataset and the dense route forms each channel's transformed mapping matrix as the interferometer does (20.5 GB at alma, 102 GB at alma_high). Reverting D1 is one commit (the rule) plus a regeneration. |
| D2 (**decided**, human 2026-10-29: adopted the recommended option) | The cube: 34 channels (the earlier datacube cells' fiducial). Every channel is 2b's delaunay standard cell (R3) of the instrument and shares its uv coverage, noise map, mask, adapt image and lens model; channel c's visibilities are scaled by a_c = exp(-0.5 (c / 17)^2) (a Gaussian emission line, a_0 = 1, a_33 = 0.15). The likelihood is the production one: `FactorGraphModel` of `AnalysisInterferometer(shared_preloads=True)` (the workspace `interferometer/features/datacube/modeling.py` configuration). | Identical uv and noise across channels is the narrow-line regime the library's `shared_preloads` exists for. Distinct data per channel keeps XLA from merging channels that are copies (the old cells used identical copies). a_0 = 1 makes channel 0 exactly 2b's cell, which D3 and D4 rely on. |
| D3 (**decided**, human 2026-10-29: adopted the recommended option) | Parity reference: the cube reference follows from 2b's chunked dense reference (R6) of the same (delaunay, instrument, instance): sum over channels of -0.5 (a_c^2 (chi^2 + s^T H s) + log det(F + H) - log det H + noise normalisation). Computed in-process at sma; above sma read from 2b's reference cache (`output/p2b_references/`), else a `not_assessed` witness that `scripts/datacube/delaunay/parity_reference.py` finalises. Tolerance 1e-8 x \|reference\| (R7). | Scaling a channel's data by a with H and the noise fixed scales the positive-only solution by a, so the terms scale exactly. The reference stays independent of W~, numba and JAX and costs one 2b reference per instrument, not 34. Checked at sma: the derived per-channel reference equals a direct chunked reference of the scaled channel to 1.5e-16 relative (a = 1, 0.61, 0.15). |
| D4 (**decided**, human 2026-10-29: adopted the recommended option) | Breakdown cells: 2b's standard breakdown harness on the lead channel (2b's cell), projected to the cube: steps the shared path builds once (ray-tracing, mapper, mapping matrix L, its sparse triplets, curvature matrix F) count once, every other step (H, D, the dense route's T, the solve, the log determinants, the figure of merit) 34 times, written as `total_step_by_step_cube` (the catalogue's `cube_component_total`); then an eager cube evaluation on the cell's route gives the cube witness. | A step-by-step decomposition of a 34-factor graph would repeat 2b's harness 34 times for identical-cost steps. The runtime cells measure the real cube, so the projection can be checked against them. |
| D5 (**decided**, human 2026-10-29: adopted the recommended option) | The numba runtime's headline is the packaged cost-model route (2b's R4) over the iid stream (`--n-instances` 20 / 20 / 10 on sma / alma / alma_high); the forced `numba` and `numpy_fft` routes are off by default (`--arms` adds them). | Each forced arm costs a full cube per instance; 2b already measured the forced routes per channel. |
| D6 (**decided**, human 2026-10-29: adopted the recommended option) | (multi_dataset) 2a's hst Delaunay standard cell fitted to 2 datasets: data and noise map x 1 and x 0.5 (same S/N), `FactorGraphModel` of `AnalysisImaging`, `shared_preloads=False`. | See the [multi_dataset page](../multi_dataset/likelihood_lifecycle_p2f.md). |
| D7 (**decided**, human 2026-10-29: adopted the recommended option) | (multi_dataset) Breakdown: 2a's imaging breakdown of dataset 0, total x 2. | As above. |
| D8 (**decided**, human 2026-10-29: adopted the recommended option) | (multi_dataset) Parity: the sum of 2a's eager NumPy dense-mapping reference per dataset, in-process, tolerance 1e-4 x \|reference\|. | As above. |
| D9 (**decided**, human 2026-10-29: adopted the recommended option) | Batch 0 (2b's R8 pattern): RAL array of 10 (datacube alma_high numba x2 and JAX CPU W~ x2, sma JAX CPU dense x2; all 4 multi_dataset RAL legs), A100 array of 10 `%2` (datacube alma_high GPU W~ x2, sma GPU dense x2; all 6 multi_dataset A100 legs), optional alma_high reference (1). Batch 1: 18 RAL legs in 4 arrays, 14 A100 legs in 4 jobs, 2 references. | alma_high bounds every datacube W~ and numba leg (most visibilities); the dense legs exist at sma only. hst is multi_dataset's only instrument and 2a measured no hst Delaunay leg in its batch 0, so all 10 multi_dataset legs are their own bound. |

Beside D2 (adopted with it, 2026-10-29: 34 channels): 34 channels is the old fiducial, not a production
survey; the per-channel cost is reported as `per_channel_equivalent_s`, and a different count
is one constant (`N_CHANNELS`) and a re-render.

## The cells

22 planned datacube cells under D1 (14 on RAL CPU, 8 on the A100), each with runtime and
breakdown:

| Instruments | RAL CPU setups | A100 setups |
|---|---|---|
| sma | `numba_cpu_sparse_fp64_wtilde`, `jax_cpu_sparse_fp64_wtilde`, `jax_cpu_dense_fp64_nufft` | `jax_gpu_sparse_fp64_wtilde`, `jax_gpu_dense_fp64_nufft` |
| alma, alma_high | `numba_cpu_sparse_fp64_wtilde`, `jax_cpu_sparse_fp64_wtilde` | `jax_gpu_sparse_fp64_wtilde` |

The drivers are in `scripts/misc/likelihood_breakdown/datacube_standard.py`; the cell scripts
are `scripts/datacube/delaunay/likelihood_{runtime,breakdown}{,_numba}.py` under
`--standard-size` (the historic fiducial paths of the two existing scripts are unchanged) and
`parity_reference.py`. W~ is built once per cube (it depends only on the shared uv coverage and
noise map) and handed to every channel; each channel computes its own dirty image.

## Batch plan (D9) and cost

| Submit | Partition | Tasks | `--time` | `--mem` |
|---|---|---|---|---|
| `batch_cpu/submit_campaign_p2f_batch0_ral_cpu_fp64` | `ral`, array `%8` | 10 | 12:00:00 (guess) | 128 GB (estimate) |
| `batch_gpu/submit_campaign_p2f_batch0_a100_fp64` | `gpu`, `--gres=gpu:1`, array `%2` | 10 | 6:00:00 (guess) | 64 GB (estimate) |
| `batch_cpu/submit_campaign_p2f_batch0_references_ral_cpu_fp64` | `ral`, array | 1 (optional) | 12:00:00 (guess) | 64 GB |
| `batch_cpu/submit_campaign_p2f_datacube_ral_cpu_<instrument>_fp64` | `ral`, array `%16` | sma 6, alma 4, alma_high 4 | provisional | provisional |
| `batch_gpu/submit_campaign_p2f_datacube_a100_<instrument>_fp64` | `gpu`, `--gres=gpu:1` | sma 4, alma 2, alma_high 2 in turn | provisional | provisional |
| `batch_cpu/submit_campaign_p2f_references_ral_cpu_fp64` | `ral`, array | 2 (alma, alma_high) | provisional | 64 GB |

Batch 1 refuses to run until batch 0's walls are in (`WALL_BASIS_STATUS = "provisional"`).
No wall is guessed for batch 1: batch 0 measures them, as in 2a and 2b.

Cost estimate (not a measurement). Laptop, 1 CPU, sma, 4 channels: the JAX W~ cube took
6.5 s per call (2b's single channel 1.9 s), the numba cube 4.4 s warm median. On RAL the 2026-09
single-channel rows (Delaunay 1500) give alma_high 8.3 s per call on JAX CPU and 13.8 s on
numba; with F shared, a 34-channel alma_high cube should take one F plus 34 solves, of the
order of a minute or two per call on CPU and well under a second on the A100. The batch-0 legs
should therefore finish inside the 12 h / 6 h guesses with a wide margin; the guesses carry the
gate's 3x floor because nothing is measured.

The alma and alma_high witnesses need 2b's delaunay references at those instruments
(`output/p2b_references/` of the submitting checkout). If 2b's batch 1 has written them there
they are reused; otherwise the 2f reference tasks compute them (2b's jvla reference took
10,727 s for 2.5e7 visibilities; alma_high has 5e6).

## Caveats

- XLA may merge per-channel work that is identical across channels (H and the two log
  determinants do not depend on the data). The runtime cells measure what production pays
  under XLA; the breakdown projection (D4) counts every per-channel step 34 times, so the two
  can differ. The laptop smoke at sma, 4 channels, 1 CPU: the JAX W~ projection was 6.24 s
  (0.91 s once + 4 x 1.33 s) against 6.53 s measured by the runtime cell; numba 3.33 s
  (0.07 s once + 4 x 0.82 s) against a 4.4 s warm median (a noisy laptop, 4 instances).
- The cube's channels share the uv coverage exactly; a real cube's uv scales with frequency,
  which breaks `shared_preloads`' contract. These cells measure the narrow-line regime only.
- Per-call times are per device. Compare CPU and GPU setups as projected node throughput, never
  as a per-call ratio.

## Journal

### 2026-10-28 — setup (draft for review)

Drivers, cell scripts, routes, D1 rule, campaign generator and submits, this page. Laptop
smokes at sma with 4 channels (1 CPU), each writing a row, `fit.png` and a `parity.json`
`match`: JAX W~ runtime (6.53 s per cube call; 7.2e-8 nats from the reference), numba runtime
(4.4 s warm median; 0 nats), JAX W~ breakdown and numba breakdown (eager cube 0 nats), JAX dense
runtime (7.2e-8 nats). `parity_reference.py --instrument sma --results-root` finalised all five
rows `match`. The JAX dense breakdown was not completed on the laptop: 2b's harness forms the
library's one-shot dense T (about 6 GB at sma), which swaps on a 15 GB laptop (2b ran the same
leg on RAL in 176 s); the eager dense cube its witness uses was run separately (4 channels, 0 nats
from the reference). D3 was checked against direct chunked references of scaled channels at
sma (1.5e-16 relative). Nothing submitted to RAL.

### 2026-10-29 — D1-D9 adopted; batch 0 held

The human adopted every drafted decision, D1-D9, including 34 channels (D2) ("ok can we do all
that now then"). No code, plan or submit changed. Under D9, batch 0 includes the optional
alma_high reference (`submit_campaign_p2f_batch0_references_ral_cpu_fp64`). A new checkout has no
2b reference cache, so without it the alma_high witnesses would stay `not_assessed`, and batch 1's
reference submit needs its wall.

The batch-0 go was given with the decisions, but batch 0 was not submitted. The shared RAL venv (`/mnt/ral/jnightin/PyAuto/PyAuto`, which every submit activates) carries jax/jaxlib 0.10.2 (`site-packages/jax-0.10.2.dist-info`, `jaxlib-0.10.2.dist-info`), while the pin's PyAutoNerves excludes 0.10.* ("CPU batched LAPACK can deadlock in 0.10.x / 0.11.0", Heart#274). Rows from that environment would not match the pin's dependency floor, so the environment is the human's call. Later the same evening the human also said not to start anything new on RAL. A detached checkout of main 32c0becd was being prepared at `/mnt/ral/jnightin/profiling/autolens_profiling_p2dg`, but nothing has been submitted from it.
