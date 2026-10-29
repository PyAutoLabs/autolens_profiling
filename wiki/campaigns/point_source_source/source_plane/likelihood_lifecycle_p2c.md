# Likelihood lifecycle phase 2c: point source

**Status:** open
**Question:** What does every planned point-source cell (`point_source_image/image_plane` and `point_source_source/source_plane`, instrument `simple`) cost on the pinned release 2026.10.25.1 on RAL CPU and the A100, and does each cell's log-likelihood match an independent eager NumPy evaluation of the same instance?
**Pre-registered rule:** every one of the 10 measurable planned cells lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy `AnalysisPoint(use_jax=False)` value of the same dataset, solver, model and instance (tolerance 1e-4 x |reference| in fp64, the default `parity_rtol`; see "The parity reference"). Any `diverged` witness blocks that setup until explained. The two `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells stay `missing` (decision D2). Recorded here with the campaign branch, before the batch runs (2026-10-28).
**Verdict:** the rule is met. 10 of 10 measurable planned cells are `current` on 2026.10.25.1, each with provenance, `resources`, `fit.png` and a `match` parity witness (15/15 legs, cells and solved experiments, logged `match`; 0 `diverged`), so no setup is blocked. The two `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells stay `missing` (D2). Against the expectation written before the runs: the image-plane likelihood is fastest on the A100 (0.86 ms per call against 1.53 ms on RAL CPU), as expected, and the RAL CPU number did drop well below the stale 24.7 ms (fused solved 2.01 ms) without passing the A100; the source-plane likelihood is fastest on RAL CPU (0.156 ms against 0.266 ms for the A100 GPU), as expected, but on the `hpc_a100` host class `jax_cpu_na_fp64` (0.255 ms) and `jax_gpu_na_fp64` (0.266 ms) are within 5 % and swap order between scripts (breakdown 0.261 against 0.250 ms), so "`jax_cpu_na_fp64` stays optimal on both host classes" is not separated by this evidence on the A100 node. Expected, as written before the runs: source plane fastest on CPU (0.147 ms RAL CPU against 0.233 ms A100, 2026.10.17.1), `jax_cpu_na_fp64` optimal on both host classes; image plane fastest on the A100 (1.09 ms against 24.7 ms, 2026.8.17.1); every witness `match`. Class designations are phase 3's decision, not this page's.
**Headline:** per call (single-jit median of 200, the cell script's free-centre likelihood): image plane `jax_gpu_na_fp64@hpc_a100` 0.864 ms against `jax_cpu_na_fp64@hpc_ral_cpu` 1.527 ms; source plane `jax_cpu_na_fp64@hpc_ral_cpu` 0.156 ms against 0.255 ms (`jax_cpu`) and 0.266 ms (`jax_gpu`) on the A100 node (RAL 413192, A100 413193). The solved library default measures within 4 % of each (below). **Which of the two publishes as the runtime cell is open for the human** ([below](#which-likelihood-the-runtime-cell-times-open-question-for-the-human)).
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the campaign branch `feature/likelihood-lifecycle-p2c-point-source` (manifest, submits, harness gaps, this page), stacked on #471; PR B (rows, catalogue, coverage, this page's result), the same branch
**Ledger:** this page and its image-plane twin [likelihood_lifecycle_p2c](../../point_source_image/image_plane/likelihood_lifecycle_p2c.md); work list [`scripts/misc/campaigns/point_source_p2c.py`](../../../../scripts/misc/campaigns/point_source_p2c.py) and its [README](../../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2c; issue autolens_profiling#472
**Next:** the human rules on the runtime cell (plain or solved) and on the two mis-designated `hpc_ral_cpu` cells; then merge PR B after #471
**Fit figures:** opened every new `fit.png` (10 files, 5 distinct images): the four observed images sit on the critical curve, the image-plane model positions land on them, and nothing looked wrong. Two presentation points, [below](#fit-figures).

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

**Effect on 2c (batch result).** These are the only 2 of the 12 planned point-source cells
still `missing` after PR B; every other point-source cell is `current`. The same planner gap
(a GPU setup planned on the GPU-less `ral` host) has turned up in 2e (cluster) and 2g as well,
so one ruling on the planner, rather than one per campaign, would settle all three.

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

**The numbers, for the ruling (batch 413192 / 413193).** PR B publishes the cell script's
plain row as `runtime.json`, as the work list defines the cell, and commits the solved row
beside it under `experiments/solved_runtime/`; nothing is chosen yet. Single-jit median of 200
individually timed calls after 5 warm calls, p10–p90 in brackets, ms:

| Cell | plain `likelihood_runtime.py` (published) | solved `likelihood_runtime_solved.py` (experiment) | breakdown fused solved / plain control | stale row superseded |
|---|---|---|---|---|
| image plane `jax_cpu_na_fp64@hpc_ral_cpu` | 1.527 (1.452–1.920) | 1.556 (1.466–1.910) | 2.014 / 1.770 | breakdown 24.69 (2026.8.17.1) |
| image plane `jax_gpu_na_fp64@hpc_a100` | 0.864 (0.830–0.896) | 0.847 (0.833–0.906) | 1.060 / 1.110 | breakdown 1.086 (2026.8.17.1) |
| source plane `jax_cpu_na_fp64@hpc_ral_cpu` | 0.156 (0.137–0.212) | 0.156 (0.136–0.208) | 0.139 / 0.149 | runtime 0.147 solved (2026.10.17.1) |
| source plane `jax_cpu_na_fp64@hpc_a100` (node CPU) | 0.255 (0.226–0.282) | 0.265 (0.238–0.315) | 0.261 / 0.264 | none (was missing) |
| source plane `jax_gpu_na_fp64@hpc_a100` | 0.266 (0.229–0.285) | 0.262 (0.253–0.283) | 0.250 / 0.250 | runtime 0.233 solved (2026.10.17.1) |

The breakdown column is the mean of a 5-call steady block, not a median. Plain and solved
differ by at most 4 % on every setup, so the choice changes no ordering between setups on a
host class except the A100-node source-plane tie, which is inside the p10–p90 spread either
way. What differs is what the cell means: the plain likelihood frees the source centre
(8 parameters), the solved one is the library default (5 parameters) and what the stale
source-plane runtime rows and every breakdown measure. Two further differences:

- **vmap.** The image-plane plain A100 row records a vmap batch of 64 at 0.025 ms per call;
  the solved A100 row's script chose batch 3 (0.41 ms per call). The source-plane plain rows
  record no vmap block (`full_pipeline_jits`, no batch); the solved rows do (batch 3: 0.061 ms
  RAL CPU, 0.111 ms A100 CPU, 0.109 ms A100 GPU). The vmap columns are not like-for-like.
- **Load.** On RAL the six legs shared one node (euclid-ral-compute-10-5, 236 cores). The two
  solved legs and the image-plane breakdown wrote at load average 44–59; the plain runtime legs
  at 6.1–6.5. The solved RAL CPU medians are therefore, if anything, biased high.

If the ruling is "solved", the swap is a file move per setup (`experiments/solved_runtime/*` to
the cell, the plain row to an experiment folder) and a re-render; no second batch.

## Results (batch 413192 RAL CPU, 413193 A100)

All 15 legs exited 0 (`exit status: 0` in every SLURM log; the RAL array's six tasks ran as
jobs 413202–413207 and 413192_5), every row is the final write, and the submit's revision
check printed `revisions OK` for all 15 (the five library SHAs of tag 2026.10.25.1;
autolens_profiling 2f98e01a). No leg came near `--time` (1:00:00): the A100 job's nine legs summed to
442 s and the slowest RAL leg took 141 s (image-plane breakdown), below the laptop walls the
WALL-BASIS rows cite (1051 s for that script).

| Cell | job | wall (s), runtime / breakdown | peak RSS (GB), runtime / breakdown | parity witness committed | parity (all legs, from the logs) |
|---|---|---|---|---|---|
| image plane `jax_cpu_na_fp64@hpc_ral_cpu` runtime / breakdown | 413203 / 413202 | 25 / 141 | 0.88 / 2.34 | breakdown (solved lane), 5.3e-15 vs 7.7e-4 | runtime 1.8e-15, breakdown 5.3e-15, solved 5.3e-15 |
| image plane `jax_gpu_na_fp64@hpc_a100` | 413193 | 28 / 239 | 0.98 / 2.67 | runtime (plain), 5.3e-15 vs 7.2e-4 | runtime 5.3e-15, breakdown 1.2e-14, solved 1.2e-14 |
| source plane `jax_cpu_na_fp64@hpc_ral_cpu` | 413205 / 413204 | 4 / 43 | 0.55 / 1.40 | breakdown (solved lane), 1.6e-9 vs 6.0e-5 | runtime 1.1e-9, breakdown 1.6e-9, solved 1.6e-9 |
| source plane `jax_cpu_na_fp64@hpc_a100` | 413193 | 5 / 46 | 0.54 / 1.41 | runtime (plain), 1.1e-9 vs 1.3e-3 | runtime 1.1e-9, breakdown 1.6e-9, solved 1.6e-9 |
| source plane `jax_gpu_na_fp64@hpc_a100` | 413193 | 6 / 57 | 0.67 / 1.51 | runtime (plain), 1.1e-9 vs 1.3e-3 | runtime 1.1e-9, breakdown 1.6e-9, solved 1.6e-9 |

|JIT − eager| against the tolerance. Every witness, committed or logged, is `match`; the
margins are the laptop smokes' (above). A cell folder holds one `parity.json` and one
`fit.png`, shared by its runtime and breakdown rows, so the committed witness is the one the
leg that finished last wrote: the solved lane on RAL CPU (breakdown last), the plain lane in
the A100 job (runtime after breakdown). The other lane's witness is in the leg's log only. The
`experiments/solved_runtime/` folders carry their own `parity.json` (5/5 `match`).

**Compared with the rows superseded.** Image plane on RAL CPU: 24.69 → 2.01 ms (fused solved),
the IP-2..IP-4c levers, as the caveat below expected. Image plane on the A100: 1.086 → 1.060 ms.
Source plane on RAL CPU: breakdown 0.146 → 0.139 ms; runtime (solved) 0.147 → 0.156 ms, measured
at load 44 on a shared node. Source plane on the A100 GPU: runtime (solved) 0.233 → 0.262 ms
(+12 %), breakdown 0.222 → 0.250 ms (+13 %); both of this batch's A100 numbers were taken on an
otherwise idle node (load ≤2.5). Not investigated here; flagged for phase 3.

## Coverage after this PR

`dashboard/coverage.md`, all dataset types (222 planned cells):

| | current | stale | missing |
|---|---|---|---|
| before | 106 | 7 | 109 |
| after | 116 | 1 | 105 |

Point source (12 planned): 6 stale and 6 missing before; 10 current and 2 missing after (the
two D2 cells). Parity witnesses: 53 → 58 `match` of 111 planned setup cells, 0 `diverged`.
The one remaining stale cell is `interferometer/mge/sma jax_gpu_dense_fp64_nufft@hpc_a100
[breakdown]`, 2b's, not this campaign's.

**Superseded rows.** The six stale rows moved, with their figures, to
`<cell>/history/<release_pin>_<last content commit>/`: the four breakdowns at 2026.8.17.1
(image plane RAL CPU `2026-09-23`, A100 `2026-10-17`; source plane RAL CPU and A100
`2026-09-26`) and the two source-plane runtimes at 2026.10.17.1 (`2026-10-17`).

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

Opened 2026-10-29, all ten new `fit.png` (five distinct images; same-likelihood legs draw the
same figure). Every figure shows the four observed images of `simple` on the critical curve
around the mass centre. Image plane: the model positions (red crosses) land on the observed
images, with −2 ln L = 2.22 (plain, A100 cell) and 1.13 (solved, RAL CPU cell and both solved
experiments). Source plane: χ² = 7.72 (plain) and 4.38 (solved). Nothing looked wrong. Two
presentation points, not defects in the rows:

- The source-plane figures draw a small orange marker inside each observed image that the
  legend does not name.
- The RAL CPU source-plane cell's figure (written by its breakdown) is zoomed to the image
  envelope (±2.5″); the A100 and experiment figures use the ±7.4″ frame.

## Caveats

- The image-plane stale rows predate the IP-2..IP-4c CPU levers (2026.9.26.1 and 2026.9.27.2),
  so the RAL CPU image-plane number should drop well below 24.7 ms; the A100-optimal
  expectation rests on the 2026-09 IP-4b cumulative 2.095 ms (loaded node) against 1.09 ms.
- The `simple` dataset is single-source; cluster point-source cells are 2e, not this page.
- Load averages at write were 44–59 for three RAL CPU legs (both solved experiments and the
  image-plane breakdown) and 13 for the source-plane breakdown: the six RAL legs shared one
  node. The run-time dashboard's cap of 8 reads the load at import (2.2–6.1 for every cell
  row), so no new point is refused there; the experiment rows are not on the dashboard.
- Every row stamps `autolens_version: 2026.8.17.1` (the last built version of a source
  checkout); `catalogue/release_revisions.json` publishes them at 2026.10.25.1 from their
  `library_revisions`, as in 2a and 2b. The breakdown README's version column still prints the
  stamp.

## Journal

### 2026-10-29 — batch committed (PR B)

RAL CPU array 413192 (tasks run as 413202–413207) and A100 job 413193 ran from the private
checkout at autolens_profiling 2f98e01a on tag 2026.10.25.1. 15 of 15 legs exit 0 with
`revisions OK`; 10 cells and 5 solved experiments copied into `results/`, the six stale rows
moved to `history/`. Rule met: 10/10 measurable cells current, every witness `match`. Point
source is 10/12 current; the two missing cells are the D2 planner gap. Open for the human: the
runtime cell (plain or solved), and the planner ruling shared with 2e and 2g.

### 2026-10-28 — branch ready for the batch

Manifest, submits and harness gaps on `feature/likelihood-lifecycle-p2c-point-source`; every
script smoke-run on the laptop. Rule and expected verdict recorded above before any RAL run.
