# Likelihood lifecycle phase 2d: multi_galaxy and group at the standard size

**Status:** draft
**Question:** What does every planned multi_galaxy/{mge, rectangular, delaunay} and group/{mge, rectangular, delaunay} cell cost at the standard size on the pinned release 2026.10.25.1, on the §3.3 "as imaging" setups, and does each cell's log evidence match an eager NumPy dense reference?
**Pre-registered rule:** every one of the 48 planned cells (20 RAL CPU, 28 A100) lands `current`, with runtime and breakdown rows, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy dense-mapping reference in the same process (tolerance 1e-4 x |reference| in fp64, the imaging rule). Any `diverged` witness blocks that setup until explained. Written 2026-10-28, before any run, and before the human has ruled on decisions D1-D11 below; a ruling that changes the cell set re-bases the count, not the rule.
**Verdict:** pending (nothing run). Expected, written before the runs: as imaging, `jax_gpu_dense_fp64` is the fastest per call on the A100 and `numba_cpu_sparse_fp64` is the RAL CPU per-worker route for the pixelized cells; group is slower than multi_galaxy for every setup (1.6x the masked pixels, two dPIE extra galaxies); the extra lens galaxies cost little against the inversion (the light is linear, the mass adds deflections only); every parity witness is `match`. Class designations are phase 3's decision, not this page's.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the 2d draft PR (new cell scripts, shared driver, cell builder, registry cells, campaign submits, this page), stacked on #471; then a walls commit after batch 0; then PR B (rows, catalogue, coverage, result)
**Ledger:** this page and the group page [`../group/likelihood_lifecycle_p2d.md`](../group/likelihood_lifecycle_p2d.md); rows will land under `results/{multi_galaxy,group}/{mge,rectangular,delaunay}/`; work list [`scripts/misc/campaigns/multi_galaxy_group_p2d.py`](../../../scripts/misc/campaigns/multi_galaxy_group_p2d.py) and its [README](../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2d; issue autolens_profiling#473
**Next:** the human rules on D1-D11; then batch 0 (human go)
**Fit figures:** none yet (laptop smoke figures only, not reviewed as campaign rows)

## Why this campaign

The epic's report §3.3 designates "multi_galaxy/\*, group/\* — as imaging rules" with
`n_galaxies` deferred (one `standard` bin), and §4 Phase 2 pairs the two types in one sub-issue.
No profiling script existed for either type and the registry declared no cell, so the coverage
matrix listed none. This PR adds the scripts, declares the cells (so the matrix lists 48
missing cells), and renders the submits. Nothing is run on RAL until the human has reviewed the
design below and given a batch go.

## Decisions awaiting the human

Each is a draft by analogy with imaging (and point_source where it could apply). The code
implements the draft; each alternative is a small change in `catalogue/registry.json`,
`scripts/misc/cell_builders/galaxy_group.py` or the campaign generator.

| # | Decision (draft) | Why | Alternatives |
|---|---|---|---|
| D1 | Variants: `mge`, `rectangular`, `delaunay` for both types. No `delaunay_nn`, no `sersic`. | Imaging's planned variants less `delaunay_nn`, which imaging plans at hst only as a mesh-interpolation variant; the mesh cost does not depend on the number of lens galaxies, so imaging's row answers it. `sersic` has no imaging cell either. | Add `delaunay_nn` (+10 cells per type). |
| D2 | Instruments: one per type. `multi_galaxy` at `hst`, `group` at `euclid`: the pixel scales of the workspace simulators (0.05", 0.1"). The PSF is the instrument preset's (21x21, sigma = pixel scale), not the workspace's 11x11, so the cells compare with imaging at the same instrument. | §3.3 names no instrument axis for these types; the workspace datasets set the scale. Instrument names keep the catalogue's vocabulary and leave room for more instruments as new cells, not new grammar. | `simple` as the instrument for both (the point_source / cluster convention, keyed to the workspace dataset name); or every imaging instrument (x4 cells). |
| D3 | Datasets: the workspace simulators' galaxies, grid (200x200, 250x250), exposure (900 s, 300 s), sky and over-sampling; masks 3.0" and 7.5" (workspace `modeling.py`); the multi_galaxy extra galaxy noise-scaled inside a 0.9" circle. | Mirrors `autolens_workspace/scripts/{multi_galaxy,group}/simulator.py` and the autolens_visualization true models. | Use the visualization producers' smaller cut-outs (140x140, 160x160): the masked pixels, and so the cost, are the same. |
| D4 | Model: the workspace modeling MGE model. Each main lens: 20 linear Gaussians + `Isothermal`; external shear; each group extra galaxy: 10 Gaussians (centre fixed) + `dPIEMassSph` (sigma free, `r_core` 0, `r_cut` 10); multi_galaxy mass centres fixed at the light centres (as the workspace). Priors: narrow Gaussians on the truth (mass, and the MGE centres and ellipticities). | The cost model a user runs. Narrow priors keep the seeded iid draws near the truth: with the workspace's generic MGE priors the multi_galaxy lens light sat ~0.1 off in ellipticity (chi-squared ~30 per pixel), not the regime a sampler spends its time in. | Workspace priors verbatim; or Sersic lens light (the workspace's first model). |
| D5 | Pixelized source: the hst production preset at the standard size, exactly as the imaging cells: Delaunay 1250 Hilbert + 30 zeroed edge points with a free `AdaptSplit`; rectangular `RectangularBilinearAdaptImage` 35x35 with a free `Adapt`. Adapt image: the subtracted S/N with every lens-plane galaxy of the truth tracer removed, capped at S/N 3 for the mesh; pixelization over-sampling 4 / 2 cut at S/N 3. No positions penalty, also at `euclid`. MGE source: 20 Gaussians, one basis. | The standard size (human ruling 2026-10-23) and the one construction the imaging cells share. The Euclid preset's positions penalty needs solved image positions, which no group dataset has. | The Euclid `vis_pix` preset for group (500 Hilbert, positions penalty), which is not the standard size. |
| D6 | Setups and hosts: §3.3 "as imaging", verbatim. MGE: `jax_cpu_dense_fp64`@`hpc_ral_cpu`, `jax_gpu_dense_fp64`@`hpc_a100`. Pixelized: `numba_cpu_sparse_fp64` (optimal) + `jax_cpu_sparse_fp64`@`hpc_ral_cpu`; `jax_gpu_dense_fp64` (optimal) + `jax_gpu_sparse_fp64` + `jax_cpu_sparse_fp64`@`hpc_a100`. | The provisional table already routes both types to the imaging tables; no change to `provisional_designations.json`. The point_source analogy does not apply: these are extended-source `AnalysisImaging` fits. A group point-source likelihood would follow point_source rules and is out of this scope. | Drop `jax_cpu_sparse_fp64`@`hpc_a100` (the GPU node's CPU; -4 cells per type). |
| D7 | Instance and parity: iid draw 0 of the seeded sequence (seed 235, central 20 % of every prior) for every setup; the reference is the eager NumPy dense-mapping fit of the same instance; tolerance 1e-4 x |reference| (imaging). The runtime row writes `parity.json` and `fit.png`; the breakdown row leaves both to it (they share the cell folder), the imaging rule. | Matches 2a. | A tighter tolerance (2b used 1e-8 for interferometer only because \|log L\| reaches 3e8). |
| D8 | A100 vmap batch: 16 (pixelized) and 64 (MGE), halving on out-of-memory; CPU capped at 3. | The imaging hst rows of `vram/config.py` (15k masked pixels; these datasets have 11k and 18k). | Probe first (`--vmap-probe` is not wired into the new driver). |
| D9 | Batch 0 = every group leg (24: RAL 10, A100 14), as the timing arm; batch 1 = all 48 cells, walls from batch 0's group leg of the same script, setup and host class. | Group bounds multi_galaxy (1.6x the masked pixels, more mass components), and every leg is cheap enough to run them all (imaging hst legs took 1-6 min). | Run both types in batch 0 (48 legs) and promote them, with no batch 1; or the slowest setups only. |
| D10 | Breakdowns: a new generic stage walk of `AnalysisImaging.fit_from`. JAX: seven jitted prefixes (traced grids, blurred image, data vector, curvature, F + H, reconstruction, figure of merit) split by successive differences. numba: each lazily evaluated stage timed on first access, median over the warm iid instances. | One harness for both types and every variant, built from the library's own stages. The imaging breakdowns' hand-written per-step kernels assume one lens and one source. | Port the imaging per-step breakdowns (larger change; the step names would then match imaging's). |
| D11 | Walls and memory, provisional: batch 0 guesses 3600 s a leg (`source: unmeasured`, 3x floor), `--time=4:00:00`, `--mem=32gb`; batch 1 renders `WALL_BASIS_STATUS=provisional` and refuses to run until batch 0's walls and MaxRSS replace the placeholders (RAL 4:00:00, A100 job 2-00:00:00). | About 10x the imaging hst legs of the nearest size, and about 3x their 10.2 GiB peak. | Tighter guesses. |

### The cells (draft)

48 planned cells, each with runtime and breakdown; 24 per dataset type.

| Likelihood | Instrument | Masked pixels | RAL CPU setups | A100 setups |
|---|---|---|---|---|
| multi_galaxy/mge | hst | 11,304 | `jax_cpu_dense_fp64` | `jax_gpu_dense_fp64` |
| multi_galaxy/rectangular, /delaunay | hst | 11,304 | `numba_cpu_sparse_fp64`, `jax_cpu_sparse_fp64` | `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64`, `jax_cpu_sparse_fp64` |
| group/mge | euclid | 17,692 | `jax_cpu_dense_fp64` | `jax_gpu_dense_fp64` |
| group/rectangular, /delaunay | euclid | 17,692 | `numba_cpu_sparse_fp64`, `jax_cpu_sparse_fp64` | `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64`, `jax_cpu_sparse_fp64` |

Per type: MGE 4 cells, rectangular 10, Delaunay 10. Hosts: 20 `hpc_ral_cpu`, 28 `hpc_a100`.

## The batch plan (proposed)

| Batch | Submit | Legs | `--time` | `--mem` |
|---|---|---|---|---|
| 0 | `hpc/batch_cpu/submit_campaign_p2d_batch0_ral_cpu_fp64` (`ral`, array `%8`) | 10 group | 4:00:00 per task | 32 GB |
| 0 | `hpc/batch_gpu/submit_campaign_p2d_batch0_a100_fp64` (`--gres=gpu:1`, array `%2`) | 14 group | 4:00:00 per task | 32 GB |
| 1 | `hpc/batch_cpu/submit_campaign_p2d_{multi_galaxy,group}_ral_cpu_{hst,euclid}_fp64` (`ral`, array `%16`) | 10 + 10 | provisional | provisional |
| 1 | `hpc/batch_gpu/submit_campaign_p2d_{multi_galaxy,group}_a100_{hst,euclid}_fp64` (one GPU job each, legs in turn) | 14 + 14 | provisional | provisional |

Batch 0 writes into the gitignored `output/p2d_batch0/`, batch 1 into `output/p2d_batch1/`;
complete rows are copied into `results/` by PR B. Every submit imports the libraries only from the
private checkout `/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1` at tag 2026.10.25.1 and
checks each row's library revisions against it.

## A finding on the workspace dataset

`autolens_workspace/scripts/multi_galaxy/simulator.py` writes `mask_extra_galaxies.fits` with
`al.Mask2D.circular(..., invert=True)`, commented "True inside the circle, i.e. the region whose
noise is scaled". `Imaging.apply_noise_scaling` scales the pixels where the mask is **False**, so
under the current library that mask scales everything except the extra galaxy (38,980 of 40,000
pixels on the laptop). The profiling cell scales the 0.9" circle itself (1,020 pixels) and does
not mirror the inversion. Worth a workspace bug report; not fixed here.

## Fit figures

Open each new row's `fit.png` before you accept it (`skills/profile_likelihood/reference.md`,
"Inspect the fit figure"). Note here any cell whose figure looked wrong and what was done.

## Caveats

- The laptop smokes ran on the local library main checkouts, not tag 2026.10.25.1; the RAL
  submits pin the tag.
- The JAX breakdown rows are prefix differences: XLA can move work across a prefix boundary, so a
  small negative row is noise and the comparable aggregate is the absolute prefix
  (`prefix_per_call_s` in the row).

## Journal

### 2026-10-28 — draft for review

Scripts, cell builder, shared driver, registry cells, campaign generator and submits written;
local smokes on the laptop (see the PR). Decisions D1-D11 await the human; batch 0 is not
submitted.
