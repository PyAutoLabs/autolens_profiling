# Likelihood lifecycle phase 2c: point source

**Status:** open
**Question:** What does every planned point-source cell (`point_source_image/image_plane` and `point_source_source/source_plane`, instrument `simple`) cost on the pinned release 2026.10.25.1 on RAL CPU and the A100, and does each cell's log-likelihood match an independent eager NumPy evaluation of the same instance?
**Pre-registered rule:** every one of the 10 measurable planned cells lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy `AnalysisPoint(use_jax=False)` value of the same dataset, solver, model and instance (tolerance 1e-4 x |reference| in fp64, the default `parity_rtol`; see "The parity reference"). Any `diverged` witness blocks that setup until explained. The two `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells stay `missing` (decision D2). Recorded here with the campaign branch, before the batch runs (2026-10-28).
**Verdict:** pending (batch not run). Expected, written before the runs from the stale rows these cells supersede: the source-plane likelihood is fastest on CPU (fused solved call 0.147 ms on RAL CPU against 0.233 ms on the A100, runtime rows on 2026.10.17.1; the A100 call sits at its dispatch floor), so `jax_cpu_na_fp64` stays the optimal setup on both host classes; the image-plane likelihood is fastest on the A100 (fused solved 1.09 ms against 24.7 ms on RAL CPU, breakdown rows on 2026.8.17.1, before the IP-2..IP-4c CPU levers shipped). Every parity witness is `match`. Class designations are phase 3's decision, not this page's.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the campaign branch `feature/likelihood-lifecycle-p2c-point-source` (manifest, submits, harness gaps, this page), stacked on #471; then PR B (rows, catalogue, coverage, this page's result)
**Ledger:** this page and its image-plane twin [likelihood_lifecycle_p2c](../../point_source_image/image_plane/likelihood_lifecycle_p2c.md); work list [`scripts/misc/campaigns/point_source_p2c.py`](../../../../scripts/misc/campaigns/point_source_p2c.py) and its [README](../../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2c; issue autolens_profiling#472
**Next:** the batch (one RAL CPU array, one A100 job; human go given 2026-10-28), then PR B
**Fit figures:** none yet.

## Why this campaign

The coverage matrix lists 12 point-source cells: 6 `missing` and 6 `stale` (rows on 2026.8.17.1
and 2026.10.17.1, behind the 2026.10.25.1 pin). The plain runtime scripts wrote no `device`
block (so no provenance or `resources`) and no parity witness, and the breakdowns wrote neither
`fit.png` nor `parity.json`.

## Decisions (taken under --auto 2026-10-28, flagged for the human on #472)

| Decision | What it says |
|---|---|
| D1 | Pin 2026.10.25.1 and the private checkout `/mnt/ral/jnightin/profiling/PyAuto_2026.10.25.1` over the shared venv `/mnt/ral/jnightin/PyAuto`, the constants of 2b. |
| D2 | `source_plane jax_cpu_na_fp64@hpc_a100` is JAX on the A100 node's CPU (`JAX_PLATFORMS=cpu`, `NPROC=8`) inside the A100 job, as 2a's `jax_cpu_sparse@hpc_a100` legs. `source_plane jax_gpu_na_fp64@hpc_ral_cpu` is impossible (`ral` has no GPU): no task, both cells left `missing`, a designation/planner bug awaiting a human ruling (below). |
| D3 | No timing batch: every evaluation is milliseconds, so one batch runs every cell under 1:00:00 (`--partition=ral` array, one `--gres=gpu:1` job). |

## The cells

| Likelihood | RAL CPU (`hpc_ral_cpu`) | A100 (`hpc_a100`) |
|---|---|---|
| point_source_image/image_plane | `jax_cpu_na_fp64` (optimal) | `jax_gpu_na_fp64` (optimal) |
| point_source_source/source_plane | `jax_cpu_na_fp64` (optimal); `jax_gpu_na_fp64` (secondary, **no task**, D2) | `jax_cpu_na_fp64` (optimal, on the node's CPU); `jax_gpu_na_fp64` (secondary) |

Each with runtime (`likelihood_runtime.py`) and breakdown (`likelihood_breakdown.py`): 10
measurable cells, 4 on RAL CPU and 6 in the A100 job.

## The mis-designated cells (D2)

`catalogue/provisional_designations.json` (#442 PR 6, commit 1f3d336d) transcribes the
lifecycle report's §3.3 row for `point_source_source/source_plane`: GPU A100 `jax_cpu_dense_fp64`
("cheap; GPU warm-up dominates"), CPU many-core `jax_cpu_dense_fp64`, laptop "same", and a
host-independent Secondary column `jax_gpu_dense_fp64`. The machine reading copies that one
Secondary value into every host table of `tables.point_source_source` (`hpc_a100`,
`hpc_ral_cpu` and `laptop`), so `coverage_matrix.py` plans `jax_gpu_na_fp64@hpc_ral_cpu`, a GPU
setup on a host class with no GPU. Neither `coverage_matrix.designated_class` nor
`catalogue_lifecycle` checks a setup's device against its host class. The `hpc_a100` optimal
`jax_cpu_na_fp64` is a faithful reading of the report (CPU on the GPU node), not a bug. Left
unchanged here for the human: the likely fix is to drop `jax_gpu_na_fp64` from the
`hpc_ral_cpu` (and `laptop`) secondary lists, or to have the planner skip a GPU setup on a
GPU-less host.

## Which likelihood the runtime cell times (open question for the human)

The cell scripts the brief names time different likelihoods. `likelihood_runtime.py` times the
free-centre likelihoods (`PointFlux` with `FitPositionsImagePairAll` / `FitPositionsSource`,
8 free parameters); `likelihood_breakdown.py` decomposes the solved ones (`PointSolved` with
`FitPositionsImagePairAllSolved` / `FitPositionsSourceSolved`, 5 parameters), which are also the
library default (`AnalysisPoint`'s `fit_positions_cls`) and what the stale source-plane runtime
rows measured (`likelihood_runtime_solved.py`, routed to the same cell by the day-zero map). The
batch runs the brief's `likelihood_runtime.py` as the cell and, beside it on every setup,
`likelihood_runtime_solved.py` into the cell's `experiments/solved_runtime/` folder (5 extra
legs, never current), so PR B can publish either without a second batch.

## The parity reference

Every cell row gets a `parity.json` comparing the JIT value with the eager NumPy
`AnalysisPoint(use_jax=False)` value of the same dataset, solver, model and instance, computed in
the same process: the non-JIT `xp=np` code path, so no eager-JAX fallback is needed. The runtime
cells compare the JIT `log_likelihood_function` with the eager fit's `figure_of_merit`; the
breakdowns compare the fused JIT solved likelihood (their authoritative runtime) with the eager
solved likelihood. The tolerance is the default fp64 `parity_rtol`, 1e-4 x |reference|, not 2b's
1e-8: the eager source-plane solved lane takes a finite-difference Hessian where the JIT lane
takes `jacfwd`, so the two agree to about 1e-4 relative, not bit for bit (the breakdown's own
`methodology.eager_hessian`). The laptop smokes below show the margins.

## Laptop smokes (2026-10-28, WSL2, NPROC=4, JAX CPU, local library main)

Every leg of both rendered submits was run through its manifest arrays (script, `--instrument
simple`, `FLAGS`, `--output-path` into a scratch tree) against local worktrees of the five
libraries at tag 2026.10.25.1; the A100 job's legs ran on the laptop CPU, so they check wiring
only (the A100 image-plane breakdown was skipped there: it is the RAL leg's script). All 14
legs run exit 0, write a row with `device.provenance` and `resources`, a `fit.png` and a
`match` witness:

| Cell script | reference log L | \|JIT - eager\| | tolerance | laptop wall (largest) | peak RSS |
|---|---|---|---|---|---|
| image_plane `likelihood_runtime.py` (`FitPositionsImagePairAll`) | 7.196577 | 1.8e-15 | 7.2e-4 | 102 s | 0.71 GB |
| image_plane `likelihood_breakdown.py` (solved lane) | 7.743201 | 5.3e-15 | 7.7e-4 | 1051 s | 1.10 GB |
| image_plane `likelihood_runtime_solved.py` | 7.743201 | 5.3e-15 | 7.7e-4 | 192 s | 0.88 GB |
| source_plane `likelihood_runtime.py` (`FitPositionsSource`) | 12.752810 | 1.1e-9 | 1.3e-3 | 31 s | 0.43 GB |
| source_plane `likelihood_breakdown.py` (solved lane) | 0.598650 | 1.6e-9 | 6.0e-5 | 267 s | 0.78 GB |
| source_plane `likelihood_runtime_solved.py` | 0.598650 | 1.6e-9 | 6.0e-5 | 35 s | 0.46 GB |

The walls are the `LAPTOP_WALLS` the submits' WALL-BASIS rows cite; several smokes ran side by
side, and one image-plane solved run that shared the laptop with a test suite and a dashboard
build took 1192 s (its compile alone 530 s), so laptop compile times are noisy. The smallest
margin is the source-plane solved lane, 1.6e-9 against 6.0e-5 (the finite-difference Hessian).

The smokes caught one bug that would have failed three batch legs: the source-plane breakdown
asserts its plain-lane literal at 1e-8, and PyAutoLens bdb44a87c (in every tag from 2026.10.22.1,
so in the pin) changed `FitPositionsSource` to jacobian weighting, moving the value from
-33788.355 to 12.752810. The literal (and the runtime script's pinned value, which only records
drift) are re-captured on the pin.

## Fit figures

Open each new row's `fit.png` before you accept it (`skills/profile_likelihood/reference.md`,
"Inspect the fit figure"). Note here any cell whose figure looked wrong and what was done.

## Caveats

- The image-plane stale rows predate the IP-2..IP-4c CPU levers (2026.9.26.1 and 2026.9.27.2),
  so the RAL CPU image-plane number should drop well below 24.7 ms; the A100-optimal
  expectation rests on the 2026-09 IP-4b cumulative 2.095 ms (loaded node) against 1.09 ms.
- The `simple` dataset is single-source; cluster point-source cells are 2e, not this page.

## Journal

### 2026-10-28 — branch ready for the batch

Manifest, submits and harness gaps on `feature/likelihood-lifecycle-p2c-point-source`; every
script smoke-run on the laptop. Rule and expected verdict recorded above before any RAL run.
