# Likelihood lifecycle phase 2g: weak lensing (new cells)

**Status:** open (every decision decided 2026-10-29; batch 0 has a go but is held, see Next)
**Question:** What does one weak-lensing likelihood evaluation cost at the standard catalogue size on the pinned release 2026.10.25.1, on JAX CPU and JAX GPU, on both reference hosts, and does each cell's log-likelihood match an independent eager NumPy evaluation?
**Pre-registered rule:** every runnable planned weak cell (6 of 8; D5) lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the in-process eager NumPy `FitWeak` (tolerance 1e-4 x |reference| in fp64, 1e-3 mixed precision; D7). Any `diverged` witness blocks that setup until explained. The two `jax_gpu_dense_fp64@hpc_ral_cpu` cells stay `missing` until the human rules on D5. Recorded here with the setup PR, before any RAL run (2026-10-28). Note 2026-10-29: D5 is settled by the planner rule of autolens_profiling#481 (human ruling "fix the planner gap"); those two cells are no longer planned, so the rule's 6 runnable cells are all 6 planned cells. The rule is otherwise unchanged. Note 2026-10-29, still before any run: the human adopted every drafted decision (D1-D4, D6-D10), so the rule stands as written.
**Verdict:** pending (no batch run). Expected, written before the runs: every parity witness is `match`, about 1e-12 relative (the laptop gave |delta log L| = 2.3e-10 at |log L| = 84.87). `jax_cpu_dense_fp64` costs about 1 ms per single JIT call (steady median) and 0.3-0.5 ms per point in a vmap batch of 32 on 4-8 CPU threads (laptop: 1.04 ms and 0.47 ms at NPROC=4). On the A100 host, the GPU single call is launch-bound and no faster than the CPU one, which is the §3.3 reading that auto may decline the GPU; the GPU may still win per point at batch 32. Class designations are phase 3's decision, not this page's.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1; `autolens/weak/` is identical at the tag and on main)
**Profiling PRs:** the setup PR (draft, stacked on 2b's PR #471), then a walls PR after batch 0, then the rows PR
**Ledger:** this page; rows will land under `results/weak/isothermal/simple/`; work list [`scripts/misc/campaigns/weak_p2g.py`](../../../scripts/misc/campaigns/weak_p2g.py) and its [README](../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2g; issue autolens_profiling#476
**Next:** batch 0, all 6 legs (go given 2026-10-29; held: the RAL venv carries jax/jaxlib 0.10.2, which the pin's PyAutoNerves excludes; the human decides the environment); under D9 its rows are the campaign's rows (no batch 1), then the rows PR
**Fit figures:** the laptop `fit.png` (scratch tree, not committed) was opened: data, model, data-vs-model and chi-squared panels look right; the chi-squared colour bar is labelled `e- s^-1`, a library plot-unit quirk (D10)

## Why this campaign

Weak lensing is the one dataset type with no image-plane fit, and it had no profiling script, no
simulator in this repo and no planned cell. The epic (report §4 Phase 2) names it among the
types whose scripts are new. §3.3 already carries a weak row: `jax_cpu_dense_fp64` on every
host ("cheap"), `jax_gpu_dense_fp64` secondary, and §3.4 notes that for weak lensing
`backend=jax, device=cpu`, so auto may decline the GPU. What one "likelihood evaluation" is,
and at what size, had to be designed first. That design is the judgment fork reserved for the
human; the version below is drafted, implemented and tested locally, but not run on RAL.

## The likelihood (the library)

`al.AnalysisWeak(dataset, use_jax=...)` (default `use_jax=False`, unlike the other analyses)
fits a `WeakDataset`: a shear catalogue `ShearYX2DIrregular` (N galaxies, columns gamma_2,
gamma_1) with a per-galaxy noise sigma. `FitWeak` evaluates the model shear at the catalogue
positions with `LensCalc.shear_yx_2d_via_hessian_from`. On JAX that is an exact `jacfwd`
Hessian of the deflections per position; on NumPy it is adaptive Richardson finite
differences. The residuals are taken over 2N components, then chi-squared, a concrete noise
normalization, and `log L = -0.5 (chi^2 + norm)`. There is no inversion and no linear algebra.
All of it is present at 2026.10.25.1, unchanged since.

## Decisions

All ten are decided. D5 was settled by #481. The others were drafted; the human adopted every
drafted choice on 2026-10-29 ("ok can we do all that now then", in reply to the list of open
decisions and the batch-0 go), including D9's proposal to accept batch 0's rows as the
campaign's rows and skip batch 1. The code already implemented the drafts, so the adoption
changed no code or submit (batch 1 stays rendered and unused). The human can still change any
of them before rows land.

| # | Decision | Adopted choice | Alternatives (not taken) / why it matters |
|---|---|---|---|
| D1 (**decided**, human 2026-10-29: adopted the recommended option) | What one evaluation is | `AnalysisWeak.log_likelihood_function(instance)`: tracer from instance, model shear at N positions via the Hessian, chi-squared over 2N, noise normalization. Plain shear, no per-galaxy redshifts. | Reduced shear (`is_reduced`) adds a convergence pass from the same Hessian. Redshift scaling is eager NumPy outside the trace. Either would be a model variant later, not the standard cell. |
| D2 (**decided**, human 2026-10-29: adopted the recommended option) | Cell naming | likelihood `weak/isothermal` (the lens mass profile, as an imaging variant names its source model); instrument `simple` (the catalogue preset, as point sources use `simple`). Registry cell `weak/isothermal/simple`. | `weak/shear` or `weak/simple`. An NFW cell (the a2744 real-data model) would be `weak/nfw`. |
| D3 (**decided**, human 2026-10-29: adopted the recommended option) | Standard size | 1500 galaxies (3000 shear components), the autolens_workspace and autolens_visualization `simple` catalogue: isothermal Einstein radius 25", q = 0.8 at 45 deg, galaxies uniform in area in the 50"-200" annulus (about 46 per arcmin^2), sigma 0.25, seed 1. | Cost is linear in N. A survey-scale catalogue (1e4-1e5 galaxies) could be a declared size variant later, as rectangular 28x28 is for imaging. |
| D4 (**decided**, human 2026-10-29: adopted the recommended option) | Setups and hosts | The §3.3 weak row as transcribed: `jax_cpu_dense_fp64` (provisional optimal) and `jax_gpu_dense_fp64` (secondary) on `hpc_ral_cpu` and `hpc_a100`. `jax_cpu_dense_fp64@hpc_a100` runs on the A100 node's CPU (`JAX_PLATFORMS=cpu`) inside the GPU job, before the GPU legs, so the CPU-vs-GPU choice §3.4 describes is measured on one node. | Drop `jax_cpu@hpc_a100` and compare across hosts instead (projected node throughput already does that). |
| D5 (**settled** by #481, human 2026-10-29: "fix the planner gap") | `jax_gpu_dense_fp64@hpc_ral_cpu` | The recommended planner rule landed: a `gpu` setup is planned only on a `gpu_hosts` host, so these 2 cells are no longer planned and `UNRUNNABLE` is empty. As drafted: the §3.3 machine reading plans it (2 cells), but the `ral` partition has no GPU. It is left planned and `missing`, and the manifest skips it (`UNRUNNABLE`). | Recommended: a planner rule that a `gpu` setup is planned only on `hpc_a100`. That is a shared change (`catalogue_lifecycle.planned_setups`): it would also remove `point_source_source`'s two `jax_gpu_na_fp64@hpc_ral_cpu` cells, which are missing for the same reason. Until then the witness "zero missing" cannot be met for weak (2 of 8). |
| D6 (**decided**, human 2026-10-29: adopted the recommended option) | Setup spelling | `_dense_`, as the §3.3 weak row writes, so no designation file changes. | Weak has no linear operator, like point sources, whose setups were re-spelled `_na_` (R1). Re-spelling weak `jax_*_na_fp64` is a one-table change in `provisional_designations.json` before any row lands; after rows land it is a migration. |
| D7 (**decided**, human 2026-10-29: adopted the recommended option) | Parity reference | Eager NumPy `AnalysisWeak(use_jax=False)` at the same instance in the same process: a different Hessian algorithm (finite differences) from the cell's `jacfwd`. Tolerance: the default fp64 1e-4 x \|ref\| (no weak override). | The laptop agreement is 2.7e-12 relative, so 1e-8 (2b's R7) would also hold and say more; \|log L\| is only about 85 here. An analytic isothermal shear would be a third, library-independent reference. |
| D8 (**decided**, human 2026-10-29: adopted the recommended option) | What is timed | Runtime: JIT full pipeline (lower, compile, first call, 10-call block mean, the steady median headline) and vmap per call at batch 32 (`VMAP_BATCH`, what a sampler pays per point). Breakdown: two JIT steps, `model_shear` (tracer + Hessian at every position) and `likelihood_from_shear` (chi-squared and normalization), steady medians summed into `total_step_by_step`; one deflection evaluation is recorded for scale, not as a step. Both write `parity.json` and `fit.png` (the same cell folder). | A vmap batch sized by a VRAM probe (the memory is tiny, so a fixed batch was chosen); a NumPy eager breakdown arm. |
| D9 (**decided**, human 2026-10-29: adopted the recommended option) | Batch plan | Batch 0 runs all 6 runnable legs: one `ral` array of 2 and one A100 job of 4, wall guess 1200 s per row (3x floor), `--time 1:30:00`, 16 GB / 32 GB. Since every leg is its own bound, proposal: accept batch 0's rows as the campaign's rows and skip batch 1. Batch 1 (the same 6 legs) is rendered and refuses to run while the walls are provisional. | Run batch 0 as a timing arm only and batch 1 for the rows, as 2a and 2b did (one extra short submission). |
| D10 (**decided**, human 2026-10-29: adopted the recommended option) | Fit figure | `_profile_cli._draw_fit` gains a `weak` branch: `PlotterWeak.fit_weak(quick_update=True)` writes `subplot_fit_weak.png`, renamed `fit.png`, as the autolens_visualization weak producer does. | The chi-squared panel's colour bar reads `e- s^-1`, a unit label inherited from imaging plots; a library visualization fix, for Eyes, not this campaign. |

## The cells

6 planned cells (registry `weak/isothermal/simple`; `dashboard/coverage.md`; 8 before the #481
planner rule), each with runtime and breakdown; all runnable:

| Setup | hpc_ral_cpu | hpc_a100 |
|---|---|---|
| `jax_cpu_dense_fp64` (provisional optimal) | RAL array, `JAX_PLATFORMS=cpu` | A100 job, `JAX_PLATFORMS=cpu` |
| `jax_gpu_dense_fp64` (secondary) | not planned (#481, D5) | A100 job, `JAX_PLATFORMS=cuda,cpu` |

Scripts: `scripts/weak/isothermal/likelihood_runtime.py` and `likelihood_breakdown.py`, on the
shared cell in `scripts/misc/likelihood_breakdown/weak_standard.py`; the catalogue simulator is
`scripts/misc/simulators/weak.py` (`dataset/weak/simple/`, gitignored, simulated on demand).

## Laptop smoke (2026-10-28, not evidence)

On the laptop (WSL2, NPROC=4, library main, which equals the tag in `autolens/weak/`), into the
gitignored `output/weak_local/` tree: the runtime leg ran in 48 s (dataset simulated in it), and
the breakdown leg in 38 s. Each wrote its row with `device.provenance` and `resources`, plus
`parity.json` (`match`, |delta| = 2.3e-10) and `fit.png`. JIT steady median 1.04 ms, vmap
per point 0.47 ms at batch 32, eager NumPy 44 ms per call. The breakdown steps' medians sum to
1.75 ms against a 0.97 ms full pipeline, because splitting at the model shear costs a
materialised (N, 2) array and a second dispatch. These numbers only size the walls; the rows
come from RAL.

## Caveats

- §3.3 is provisional; which setup is optimal on which host is phase 3's call.
- The eager NumPy Hessian is adaptive and costs about 40x the JIT call; it runs once per leg,
  outside every timed region.
- Per-call times are per device. Compare CPU and GPU setups as projected node throughput, never
  as a per-call "GPU is N x faster".

## Journal

### 2026-10-28 — setup PR (draft)

Drafted D1-D10; added the simulator, the shared cell, the runtime and breakdown scripts, the
`weak` fit-figure branch, the registry cell (with the `weak` family), the campaign module and
its submits, and tests. Laptop smokes as above. Stopped at a draft PR for the human's review;
nothing was submitted to RAL.

### 2026-10-29 — D5 settled by the planner fix (#481)

Merged main, which carries autolens_profiling#481 (human ruling "fix the planner gap"): the planner
plans a GPU setup only on a GPU host. The two `jax_gpu_dense_fp64@hpc_ral_cpu` cells are no longer
planned; 2g plans 6 cells, all runnable, and `weak_p2g.py` follows (`SETUPS`, empty `UNRUNNABLE`).
The submits did not change.

### 2026-10-29 — D1-D4, D6-D10 adopted; batch 0 held

The human adopted every drafted decision ("ok can we do all that now then"), including D9's
proposal: batch 0's rows are the campaign's rows, and batch 1 is not run. No code, plan or submit
changed (batch 1 stays rendered and unused). A known trade-off: batch 0 legs run on shared `ral`
nodes, so a row whose `loadavg_at_import` exceeds 8 is refused as a dashboard trend point. It
still counts as `current`.

The batch-0 go was given with the decisions, but batch 0 was not submitted. The shared RAL venv (`/mnt/ral/jnightin/PyAuto/PyAuto`, which every submit activates) carries jax/jaxlib 0.10.2 (`site-packages/jax-0.10.2.dist-info`, `jaxlib-0.10.2.dist-info`), while the pin's PyAutoNerves excludes 0.10.* ("CPU batched LAPACK can deadlock in 0.10.x / 0.11.0", Heart#274). Rows from that environment would not match the pin's dependency floor, so the environment is the human's call. Later the same evening the human also said not to start anything new on RAL. A detached checkout of main 32c0becd was being prepared at `/mnt/ral/jnightin/profiling/autolens_profiling_p2dg`, but nothing has been submitted from it.
