# Likelihood lifecycle phase 2d: multi_galaxy and group at the standard size

**Status:** open
**Question:** What does every planned multi_galaxy/{mge, rectangular, delaunay} and group/{mge, rectangular, delaunay} cell cost at the standard size on the pinned release 2026.10.25.1, on the §3.3 "as imaging" setups, and does each cell's log evidence match an eager NumPy dense reference?
**Pre-registered rule:** every one of the 48 planned cells (20 RAL CPU, 28 A100) lands `current`, with runtime and breakdown rows, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy dense-mapping reference in the same process (tolerance 1e-4 x |reference| in fp64, the imaging rule). Any `diverged` witness blocks that setup until explained. Written 2026-10-28, before any run, and before the human has ruled on decisions D1-D12 below; a ruling that changes the cell set re-bases the count, not the rule. Ruled 2026-10-29, before any run: D1-D11 adopted as drafted and D12 kept, so the 48 cells and the rule stand unchanged.
**Verdict:** pending (nothing run). Expected, written before the runs: as imaging, `jax_gpu_dense_fp64` is the fastest per call on the A100 and `numba_cpu_sparse_fp64` is the RAL CPU per-worker route for the pixelized cells; group is slower than multi_galaxy for every setup (1.6x the masked pixels, two dPIE extra galaxies); the extra lens galaxies cost little against the inversion (the light is linear, the mass adds deflections only); every parity witness is `match`. Class designations are phase 3's decision, not this page's.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the 2d draft PR (new cell scripts, shared driver, cell builder, registry cells, campaign submits, this page), stacked on #471; then a walls commit after batch 0; then PR B (rows, catalogue, coverage, result)
**Ledger:** this page and the group page [`../group/likelihood_lifecycle_p2d.md`](../group/likelihood_lifecycle_p2d.md); rows will land under `results/{multi_galaxy,group}/{mge,rectangular,delaunay}/`; work list [`scripts/misc/campaigns/multi_galaxy_group_p2d.py`](../../../scripts/misc/campaigns/multi_galaxy_group_p2d.py) and its [README](../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2d; issue autolens_profiling#473
**Next:** batch 0, every group leg (go given 2026-10-29; held: the RAL venv carries jax/jaxlib 0.10.2, which the pin's PyAutoNerves excludes; the human decides the environment); then the walls commit, batch 1 (human go), PR B
**Fit figures:** none yet (laptop smoke figures only, not reviewed as campaign rows)

## Why this campaign

The epic's report §3.3 designates "multi_galaxy/\*, group/\* — as imaging rules" with
`n_galaxies` deferred (one `standard` bin), and §4 Phase 2 pairs the two types in one sub-issue.
No profiling script existed for either type and the registry declared no cell, so the coverage
matrix listed none. This PR adds the scripts, declares the cells (so the matrix lists 48
missing cells), and renders the submits. Nothing is run on RAL until the human has reviewed the
design below and given a batch go.

## Decisions

All twelve are decided. D1-D11 were drafted by analogy with imaging (and point_source where it
could apply); the human adopted every drafted option on 2026-10-29 ("ok can we do all that now
then", in reply to the list of open decisions and the batch-0 go), and D12 was ruled separately
the same day. The code already implemented the drafts, so the adoption changed no code or
submit. The human can still change any decision before batch 1; each alternative is a small
change in `catalogue/registry.json`, `scripts/misc/cell_builders/galaxy_group.py` or the
campaign generator.

| # | Decision | Why | Alternatives (not taken) |
|---|---|---|---|
| D1 (**decided**, human 2026-10-29: adopted the recommended option) | Variants: `mge`, `rectangular`, `delaunay` for both types. No `delaunay_nn`, no `sersic`. | Imaging's planned variants less `delaunay_nn`, which imaging plans at hst only as a mesh-interpolation variant; the mesh cost does not depend on the number of lens galaxies, so imaging's row answers it. `sersic` has no imaging cell either. | Add `delaunay_nn` (+10 cells per type). |
| D2 (**decided**, human 2026-10-29: adopted the recommended option) | Instruments: one per type. `multi_galaxy` at `hst`, `group` at `euclid`: the pixel scales of the workspace simulators (0.05", 0.1"). The PSF is the instrument preset's (21x21, sigma = pixel scale), not the workspace's 11x11, so the cells compare with imaging at the same instrument. | §3.3 names no instrument axis for these types; the workspace datasets set the scale. Instrument names keep the catalogue's vocabulary and leave room for more instruments as new cells, not new grammar. | `simple` as the instrument for both (the point_source / cluster convention, keyed to the workspace dataset name); or every imaging instrument (x4 cells). |
| D3 (**decided**, human 2026-10-29: adopted the recommended option) | Datasets: the workspace simulators' galaxies, grid (200x200, 250x250), exposure (900 s, 300 s), sky and over-sampling; masks 3.0" and 7.5" (workspace `modeling.py`); the multi_galaxy extra galaxy noise-scaled inside a 0.9" circle. | Mirrors `autolens_workspace/scripts/{multi_galaxy,group}/simulator.py` and the autolens_visualization true models. | Use the visualization producers' smaller cut-outs (140x140, 160x160): the masked pixels, and so the cost, are the same. |
| D4 (**decided**, human 2026-10-29: adopted the recommended option) | Model: the workspace modeling MGE model. Each main lens: 20 linear Gaussians + `Isothermal`; external shear; each group extra galaxy: 10 Gaussians (centre fixed) + `dPIEMassSph` (sigma free, `r_core` 0, `r_cut` 10); multi_galaxy mass centres fixed at the light centres (as the workspace). Priors: narrow Gaussians on the truth (mass, and the MGE centres and ellipticities). | The cost model a user runs. Narrow priors keep the seeded iid draws near the truth: with the workspace's generic MGE priors the multi_galaxy lens light sat ~0.1 off in ellipticity (chi-squared ~30 per pixel), not the regime a sampler spends its time in. | Workspace priors verbatim; or Sersic lens light (the workspace's first model). |
| D5 (**decided**, human 2026-10-29: adopted the recommended option) | Pixelized source: the hst production preset at the standard size, exactly as the imaging cells: Delaunay 1250 Hilbert + 30 zeroed edge points with a free `AdaptSplit`; rectangular `RectangularBilinearAdaptImage` 35x35 with a free `Adapt`. Adapt image: the subtracted S/N with every lens-plane galaxy of the truth tracer removed, capped at S/N 3 for the mesh; pixelization over-sampling 4 / 2 cut at S/N 3. No positions penalty, also at `euclid`. MGE source: 20 Gaussians, one basis. | The standard size (human ruling 2026-10-23) and the one construction the imaging cells share. The Euclid preset's positions penalty needs solved image positions, which no group dataset has. | The Euclid `vis_pix` preset for group (500 Hilbert, positions penalty), which is not the standard size. |
| D6 (**decided**, human 2026-10-29: adopted the recommended option) | Setups and hosts: §3.3 "as imaging", verbatim. MGE: `jax_cpu_dense_fp64`@`hpc_ral_cpu`, `jax_gpu_dense_fp64`@`hpc_a100`. Pixelized: `numba_cpu_sparse_fp64` (optimal) + `jax_cpu_sparse_fp64`@`hpc_ral_cpu`; `jax_gpu_dense_fp64` (optimal) + `jax_gpu_sparse_fp64` + `jax_cpu_sparse_fp64`@`hpc_a100`. | The provisional table already routes both types to the imaging tables; no change to `provisional_designations.json`. The point_source analogy does not apply: these are extended-source `AnalysisImaging` fits. A group point-source likelihood would follow point_source rules and is out of this scope. | Drop `jax_cpu_sparse_fp64`@`hpc_a100` (the GPU node's CPU; -4 cells per type). |
| D7 (**decided**, human 2026-10-29: adopted the recommended option) | Instance and parity: iid draw 0 of the seeded sequence (seed 235, central 20 % of every prior) for every setup; the reference is the eager NumPy dense-mapping fit of the same instance; tolerance 1e-4 x |reference| (imaging). The runtime row writes `parity.json` and `fit.png`; the breakdown row leaves both to it (they share the cell folder), the imaging rule. | Matches 2a. | A tighter tolerance (2b used 1e-8 for interferometer only because \|log L\| reaches 3e8). |
| D8 (**decided**, human 2026-10-29: adopted the recommended option) | A100 vmap batch: 16 (pixelized) and 64 (MGE), halving on out-of-memory; CPU capped at 3. | The imaging hst rows of `vram/config.py` (15k masked pixels; these datasets have 11k and 18k). | Probe first (`--vmap-probe` is not wired into the new driver). |
| D9 (**decided**, human 2026-10-29: adopted the recommended option) | Batch 0 = every group leg (24: RAL 10, A100 14), as the timing arm; batch 1 = all 48 cells, walls from batch 0's group leg of the same script, setup and host class. | Group bounds multi_galaxy (1.6x the masked pixels, more mass components), and every leg is cheap enough to run them all (imaging hst legs took 1-6 min). | Run both types in batch 0 (48 legs) and promote them, with no batch 1; or the slowest setups only. |
| D10 (**decided**, human 2026-10-29: adopted the recommended option) | Breakdowns: a new generic stage walk of `AnalysisImaging.fit_from`. JAX: seven jitted prefixes (traced grids, blurred image, data vector, curvature, F + H, reconstruction, figure of merit) split by successive differences. numba: each lazily evaluated stage timed on first access, median over the warm iid instances. | One harness for both types and every variant, built from the library's own stages. The imaging breakdowns' hand-written per-step kernels assume one lens and one source. | Port the imaging per-step breakdowns (larger change; the step names would then match imaging's). |
| D11 (**decided**, human 2026-10-29: adopted the recommended option) | Walls and memory, provisional: batch 0 guesses 3600 s a leg (`source: unmeasured`, 3x floor), `--time=4:00:00`, `--mem=32gb`; batch 1 renders `WALL_BASIS_STATUS=provisional` and refuses to run until batch 0's walls and MaxRSS replace the placeholders (RAL 4:00:00, A100 job 2-00:00:00). | About 10x the imaging hst legs of the nearest size, and about 3x their 10.2 GiB peak. | Tighter guesses. |
| D12 (**decided**, human 2026-10-29: keep) | Every JAX leg compiles with `--xla_cpu_multi_thread_eigen=false` appended to `XLA_FLAGS` (the driver does it; the row records it in `device.xla_flags` and `configuration.xla_cpu_workaround`). | Without it the 2d executables deadlock on jaxlib 0.11.2's XLA:CPU after a few calls (see "XLA:CPU deadlock" below). The flag is the cheaper of the two that avoid it on the laptop and touches only CPU-compiled code; it applies to the RAL CPU legs and the A100 node's `jax_cpu_sparse_fp64` leg, and is inert for a GPU executable. It is a compile option the 2a/2b imaging rows did not use, so a 2d JAX CPU row is not like-for-like with an imaging row on that axis. | Not taken (decided): `--xla_cpu_enable_concurrency_optimized_scheduler=false` instead, for every cell or only the pixelized ones (also avoids it; about 2x slower per call than the eigen flag on the laptop MGE cells, within ~15 % and slightly faster on the pixelized cells); or no flag and a jaxlib that fixes it (none known yet); or drop the JAX CPU setups from 2d. |

**D12 decided (human, 2026-10-29), verbatim:** "keep the workaround which ensures we retain the fastest run times and make sure this knowledge is retained in Pulse for going forward". Kept as drafted: the JAX legs compile
with `--xla_cpu_multi_thread_eigen=false`, which the driver already applies
(`XLA_CPU_WORKAROUND_FLAG` in `scripts/misc/likelihood_breakdown/galaxy_group_standard.py`), so no
code changed. The knowledge is kept in Pulse:
[XLA:CPU deadlock on repeated calls (jaxlib 0.11.2)](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/knowledge/xla_cpu_deadlock_jaxlib_0_11_2.md) and the decision record
[xla-cpu-deadlock-eigen-workaround](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/xla-cpu-deadlock-eigen-workaround.md) (PyAutoPulse#81).
One flag for every cell (human, 2026-10-29: "keep one option for every cell, as now"): a
per-variant choice, such as the scheduler flag on the pixelized cells where it measured slightly
faster, is not taken. Every 2d JAX CPU leg, MGE and pixelized, compiles with the eigen flag.

### The cells

48 planned cells, each with runtime and breakdown; 24 per dataset type.

| Likelihood | Instrument | Masked pixels | RAL CPU setups | A100 setups |
|---|---|---|---|---|
| multi_galaxy/mge | hst | 11,304 | `jax_cpu_dense_fp64` | `jax_gpu_dense_fp64` |
| multi_galaxy/rectangular, /delaunay | hst | 11,304 | `numba_cpu_sparse_fp64`, `jax_cpu_sparse_fp64` | `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64`, `jax_cpu_sparse_fp64` |
| group/mge | euclid | 17,692 | `jax_cpu_dense_fp64` | `jax_gpu_dense_fp64` |
| group/rectangular, /delaunay | euclid | 17,692 | `numba_cpu_sparse_fp64`, `jax_cpu_sparse_fp64` | `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64`, `jax_cpu_sparse_fp64` |

Per type: MGE 4 cells, rectangular 10, Delaunay 10. Hosts: 20 `hpc_ral_cpu`, 28 `hpc_a100`.

## The batch plan (D9)

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

## XLA:CPU deadlock (found in the laptop smokes, avoided by D12)

**Symptom.** Every 2d JAX cell on the CPU backend (dense and sparse, MGE, rectangular and
Delaunay, multi_galaxy and group) hung after its compiled likelihood had run a few times: lower,
compile and the first call return, then a later call of the same compiled executable never does.
The process sits at 0 % CPU; `faulthandler` puts the Python thread in `stages.py` `__call__`, and
native stacks (`py-spy record --native --idle`, started as the parent) show it in
`CpuPjRtRawLoadedExecutable::Execute` -> `tsl::BlockUntilReady` -> `absl::Mutex` futex wait,
with no thread doing work. Which call hangs varies run to run (call 1 to call ~35); a cell's
`--fast` smoke (2 steady calls) could pass by luck.

**What it is not.** Not the persistent compilation cache (hangs with
`JAX_ENABLE_COMPILATION_CACHE=false`); not BLAS threads (`OPENBLAS/OMP/MKL_NUM_THREADS=1`); not
the Delaunay `pure_callback` (the MGE cells have no host callback and hang too); not the
flat-leaves change or `dPIEMassSph` (multi_galaxy has no dPIE and hangs); not
`--xla_disable_hlo_passes=constant_folding` (hangs without it). A plain jitted `A.T @ A`
(17,692 x 1,300), with and without a scipy `pure_callback` beside it, ran 30 calls fine. Prefix
bisection (the breakdown stages): group MGE hangs from prefix 3 (data vector) on, group Delaunay
from prefix 4 (curvature matrix); prefixes 1-2 ran 300 calls.

**Minimal reproduction (no PyAuto library in the process).** Export the jitted group MGE prefix-3
function with `jax.export` and run it in a fresh `python -I` process that imports only jax:

```python
exp = jax.export.deserialize(blob)  # serialized in the cell's process
c = jax.jit(exp.call).lower(params).compile()
for i in range(300):
    c(params).block_until_ready()  # deadlocks at call ~14 with XLA_FLAGS=""
```

It deadlocks with the default flags and runs 300 calls with either
`--xla_cpu_multi_thread_eigen=false` or `--xla_cpu_enable_concurrency_optimized_scheduler=false`,
so the fault is in XLA:CPU (jax / jaxlib 0.11.2), in the interplay of multi-threaded Eigen ops
and the concurrency-optimized thunk schedule. Not patched here (it is not PyAuto code); worth a
JAX issue with the exported module. The lasting record is the Pulse
[knowledge page](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/knowledge/xla_cpu_deadlock_jaxlib_0_11_2.md).

**Full pipelines, laptop, 8 cores, calls before the hang / steady median per call:**

| Cell (dense JAX CPU) | default flags | `multi_thread_eigen=false` | `concurrency_optimized_scheduler=false` |
|---|---|---|---|
| group/mge | hung after 35 (0.064 s) | 150 of 150, 0.080 s | 150 of 150, 0.154 s |
| multi_galaxy/mge | hung after 1 | 40 of 40, 0.053 s | 40 of 40, 0.104 s |
| group/delaunay | hung after 1 | 15 of 15, 4.57 s | 15 of 15, 4.42 s |
| multi_galaxy/rectangular | hung after 1 | 15 of 15, 2.19 s | 15 of 15, 1.90 s |

**RAL on the pin.** The PyAuto code involved is identical at tag 2026.10.25.1 (`Tracer.traced_grid_2d_list_from`,
`dPIEMassSph`), and the pin's dependency floor excludes jax 0.10.*, so the RAL venv runs jaxlib
0.11.x: its 2026.10.17.1 `hpc_ral_cpu` point_source_source rows record jaxlib 0.11.2. Whether the
deadlock fires on a 236-core RAL node was not tested (no RAL runs before review); D12 applies the
flag there regardless, so a leg cannot sit on its `--time` doing nothing. The GPU executables are
not affected by the flag; whether the GPU runtime has an analogue was not tested.

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

- Every JAX leg compiles with `--xla_cpu_multi_thread_eigen=false` (D12): CPU-compiled Eigen ops
  run single-threaded inside their thunk, which the 2a/2b JAX CPU rows did not do.
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

### 2026-10-29 — the JAX smoke hangs: an XLA:CPU deadlock, avoided

Two driver fixes for the JAX legs: the jitted functions take the instance's flat pytree leaves
and unflatten inside the trace (`dPIEMassSph` computes `b0` from `sigma` in its constructor, so
`jax.jit(f).lower(tree)` cannot rebuild it from abstract leaves), and the breakdown's ray-trace
prefix passes `xp=jnp` (the NumPy default raised `TracerArrayConversionError`). The remaining
hangs were a jaxlib 0.11.2 XLA:CPU deadlock (section above), reproduced without any PyAuto code
and avoided by D12. Every JAX smoke then completed; the table is in the 2d draft PR.

### 2026-10-29 — D12 decided: keep the eigen flag

The human ruled "keep the workaround which ensures we retain the fastest run times and make sure this knowledge is retained in Pulse for going forward". D12 stands as drafted (`--xla_cpu_multi_thread_eigen=false`, already
in the driver; no code change). The deadlock, its repro, both flags' costs and the open
questions (RAL, the 2a/2b JAX CPU cells, when a jaxlib change should trigger a re-test) are kept
in Pulse: [knowledge page](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/knowledge/xla_cpu_deadlock_jaxlib_0_11_2.md), [decision record](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/xla-cpu-deadlock-eigen-workaround.md) (PyAutoPulse#81). D1-D11 still
await the human.

### 2026-10-29 — the GPU-host planner fix merged in

#481 (a GPU setup is planned only on a `gpu_hosts` host) is merged into this branch. It does not
change 2d's plan: the imaging tables list no GPU setup at `hpc_ral_cpu` for these cells, so 2d
stays 48 cells (20 `hpc_ral_cpu`, 28 `hpc_a100`); the catalogue total goes 270 -> 268 from the
two point_source_source cells #481 drops.

### 2026-10-29 — D1-D11 adopted; batch 0 held

The human adopted every drafted decision, D1-D11 ("ok can we do all that now then", in reply to
the list of open decisions and the batch-0 go); D12 was already decided. The code implemented
the drafts, so no code, plan or submit changed: 48 cells, batch 0 = the 24 group legs, batch 1 =
all 48 with batch 0's walls. The human can still change any decision before batch 1.

The batch-0 go was given with the decisions, but batch 0 was not submitted. The shared RAL venv (`/mnt/ral/jnightin/PyAuto/PyAuto`, which every submit activates) carries jax/jaxlib 0.10.2 (`site-packages/jax-0.10.2.dist-info`, `jaxlib-0.10.2.dist-info`), while the pin's PyAutoNerves excludes 0.10.* ("CPU batched LAPACK can deadlock in 0.10.x / 0.11.0", Heart#274). Rows from that environment would not match the pin's dependency floor, so the environment is the human's call. Later the same evening the human also said not to start anything new on RAL. A detached checkout of main 32c0becd was being prepared at `/mnt/ral/jnightin/profiling/autolens_profiling_p2dg`, but nothing has been submitted from it.

### 2026-10-29 — the eigen flag moves into the shared submit preamble, for every campaign

RAL's shared venv was upgraded to jax/jaxlib 0.11.2 (human, 2026-10-29), so every campaign's JAX
CPU legs now run on the jaxlib that deadlocks. The human picked "All JAX CPU legs": the shared
campaign preamble (`scripts/misc/campaigns/_common.py`, `run_leg`) appends
`--xla_cpu_multi_thread_eigen=false` to `XLA_FLAGS` on every `JAX_PLATFORMS=cpu` leg of 2a-2g
(the A100 nodes' CPU legs included; numba legs also run with `JAX_PLATFORMS=cpu` and so carry the
inert flag), and never on a GPU leg (`cuda` / `cuda,cpu`). Each row records it in
`device.xla_flags`. The 2d driver still appends the flag only when absent, so a 2d leg carries it
once and keeps `configuration.xla_cpu_workaround`. D12's scope widens from 2d to every campaign;
the 2a-2c rows already landed were measured on jax 0.10.2 without the flag (phase-3 Q7 covers
that comparison).
