# Likelihood lifecycle phase 2c: point source

**Status:** open
**Question:** What does every planned point-source cell (`point_source_image/image_plane` and `point_source_source/source_plane`, instrument `simple`) cost on the pinned release 2026.10.25.1 on RAL CPU and the A100, and does each cell's log-likelihood match an independent eager NumPy evaluation of the same instance?
**Pre-registered rule:** every one of the 10 measurable planned cells lands `current`, with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy `AnalysisPoint(use_jax=False)` value of the same dataset, solver, model and instance (tolerance 1e-4 x |reference| in fp64, the default `parity_rtol`; see "The parity reference"). Any `diverged` witness blocks that setup until explained. The two `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells stay `missing` (decision D2). Recorded here with the campaign branch, before the batch runs (2026-10-28).
**Verdict:** the rule is met. 10 of 10 planned cells are `current` on 2026.10.25.1, each with provenance, `resources`, `fit.png` and a `match` parity witness (15/15 legs, cells and experiments, logged `match`; 0 `diverged`), so no setup is blocked. Re-checked on 2026-10-29 with the solved runtime cells (human ruling, [below](#which-likelihood-the-runtime-cell-times-ruled-solved)): the witnesses, provenance and `resources` belong to the published solved rows, and the rule still holds. The two `jax_gpu_na_fp64@hpc_ral_cpu` source-plane cells the rule left `missing` (D2) are no longer planned (#481), so the rule's 10 cells are all of point source. Against the expectation written before the runs: the image-plane likelihood is fastest on the A100 (0.847 ms per call against 1.556 ms on RAL CPU), as expected, and the RAL CPU number did drop well below the stale 24.7 ms (fused solved 2.01 ms) without passing the A100; the source-plane likelihood is fastest on RAL CPU (0.156 ms against 0.262 ms for the A100 GPU), as expected, but on the `hpc_a100` host class `jax_gpu_na_fp64` (0.262 ms) and `jax_cpu_na_fp64` (0.265 ms) are 1 % apart, inside each other's p10–p90, with `jax_gpu` ahead in both scripts by at most 4 % (breakdown 0.250 against 0.261 ms), so "`jax_cpu_na_fp64` stays optimal on both host classes" is not supported on the A100 node and this evidence does not separate the two there. Expected, as written before the runs: source plane fastest on CPU (0.147 ms RAL CPU against 0.233 ms A100, 2026.10.17.1), `jax_cpu_na_fp64` optimal on both host classes; image plane fastest on the A100 (1.09 ms against 24.7 ms, 2026.8.17.1); every witness `match`. Class designations are phase 3's decision, not this page's.
**Headline:** per call (single-jit median of 200, the solved library-default likelihood, `FitPositionsImagePairAllSolved` / `FitPositionsSourceSolved`): image plane `jax_gpu_na_fp64@hpc_a100` 0.847 ms against `jax_cpu_na_fp64@hpc_ral_cpu` 1.556 ms; source plane `jax_cpu_na_fp64@hpc_ral_cpu` 0.156 ms against 0.265 ms (`jax_cpu`) and 0.262 ms (`jax_gpu`) on the A100 node (RAL 413192, A100 413193). The free-centre plain likelihood, kept as the comparison in `experiments/plain_runtime/`, measures within 4 % of each. The two RAL CPU runtime rows are refused as dashboard trend points (load 33.4 at import, cap 8; [Caveats](#caveats)).
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the campaign branch `feature/likelihood-lifecycle-p2c-point-source` (manifest, submits, harness gaps, this page), stacked on #471; PR B #480 (rows, catalogue, coverage, this page's result), the same branch, with the planner fix #481 merged in
**Ledger:** this page and its image-plane twin [likelihood_lifecycle_p2c](../../point_source_image/image_plane/likelihood_lifecycle_p2c.md); work list [`scripts/misc/campaigns/point_source_p2c.py`](../../../../scripts/misc/campaigns/point_source_p2c.py) and its [README](../../../../scripts/misc/campaigns/README.md)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2c; issue autolens_profiling#472
**Next:** merge PR B (#480) after #471 and #481; a quiet-node RAL CPU rerun of the two solved runtime legs would give them dashboard trend points (a human call, not needed for coverage)
**Fit figures:** opened every new `fit.png` (10 files, 5 distinct images): the four observed images sit on the critical curve, the image-plane model positions land on them, and nothing looked wrong. Since the ruling every cell's `fit.png` is its solved runtime leg's. One presentation point, [below](#fit-figures).

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
| point_source_source/source_plane | `jax_cpu_na_fp64` (optimal); `jax_gpu_na_fp64` (secondary, **no task**, D2; no longer planned since #481) | `jax_cpu_na_fp64` (optimal, on the node's CPU); `jax_gpu_na_fp64` (secondary) |

Each with runtime and breakdown (`likelihood_breakdown.py`): 10 cells, 4 on RAL CPU and 6 in the
A100 job. The runtime cell is `likelihood_runtime_solved.py` since the ruling of 2026-10-29
([below](#which-likelihood-the-runtime-cell-times-ruled-solved)); the batch ran
`likelihood_runtime.py` as the cell and the solved script beside it.

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

**Ruled: fix the planner (human 2026-10-29).** #481 gives `provisional_designations.json` a
`gpu_hosts` list (`hpc_a100`) and has `catalogue_lifecycle.planned_setups` plan a GPU setup only
on a GPU host; the setup stays designated, so a measured row keeps its `secondary` class. Merged
into this branch (a merge, not a rebase) and the catalogue and coverage regenerated: planned cells
222 → 220, and the two `jax_gpu_na_fp64@hpc_ral_cpu` cells are gone from the plan, so point
source is 10/10 `current`. `MIS_DESIGNATED` is gone from the work list.

## Which likelihood the runtime cell times (ruled: solved)

The cell scripts the brief names time different likelihoods. `likelihood_runtime.py` times the
free-centre likelihoods (`PointFlux` with `FitPositionsImagePairAll` / `FitPositionsSource`,
8 free parameters); `likelihood_breakdown.py` decomposes the solved ones (`PointSolved` with
`FitPositionsImagePairAllSolved` / `FitPositionsSourceSolved`, 5 parameters), which are also the
library default (`AnalysisPoint`'s `fit_positions_cls`) and what the stale source-plane runtime
rows measured (`likelihood_runtime_solved.py`, routed to the same cell by the day-zero map). The
batch runs the brief's `likelihood_runtime.py` as the cell and, beside it on every setup,
`likelihood_runtime_solved.py` into the cell's `experiments/solved_runtime/` folder (5 extra
legs, never current), so PR B can publish either without a second batch.

**Ruled: solved (human 2026-10-29).** "Switch 2c to Solved": the library and workspace default
for a galaxy-scale point source is `FitPositionsImagePairAllSolved` (image plane; PyAutoLens
`autolens/point/model/analysis.py`, `autolens_workspace/scripts/point_source/modeling.py`), and
its source-plane counterpart is `FitPositionsSourceSolved`, so the runtime cell times those. On
each of the five runtime setups the solved row, its `runtime.png`, `parity.json` and `fit.png`
moved from `experiments/solved_runtime/` into the cell (byte-identical to the RAL pull), and the
plain row moved to `experiments/plain_runtime/`, the non-published comparison (with its
`parity.json` and `fit.png` on the three A100 setups; the RAL CPU plain witnesses were overwritten
on the node and survive only in the logs, both `match`). The work list now runs
`likelihood_runtime_solved.py` as the runtime cell and `likelihood_runtime.py` into
`experiments/plain_runtime/`, so a rerun reproduces what is published. No rerun was needed; every
breakdown already decomposes the solved likelihood (fused solved, with a plain control), so the
breakdowns are unchanged. Every runtime cell has its solved row: 5 solved rows for the 5 runtime
cells; the other 5 cells are the breakdowns.

**The numbers (batch 413192 / 413193).** Single-jit median of 200 individually timed calls after
5 warm calls, p10–p90 in brackets, ms:

| Cell | solved `likelihood_runtime_solved.py` (published) | plain `likelihood_runtime.py` (`experiments/plain_runtime/`) | breakdown fused solved / plain control | stale row superseded |
|---|---|---|---|---|
| image plane `jax_cpu_na_fp64@hpc_ral_cpu` | 1.556 (1.466–1.910) | 1.527 (1.452–1.920) | 2.014 / 1.770 | breakdown 24.69 (2026.8.17.1) |
| image plane `jax_gpu_na_fp64@hpc_a100` | 0.847 (0.833–0.906) | 0.864 (0.830–0.896) | 1.060 / 1.110 | breakdown 1.086 (2026.8.17.1) |
| source plane `jax_cpu_na_fp64@hpc_ral_cpu` | 0.156 (0.136–0.208) | 0.156 (0.137–0.212) | 0.139 / 0.149 | runtime 0.147 solved (2026.10.17.1) |
| source plane `jax_cpu_na_fp64@hpc_a100` (node CPU) | 0.265 (0.238–0.315) | 0.255 (0.226–0.282) | 0.261 / 0.264 | none (was missing) |
| source plane `jax_gpu_na_fp64@hpc_a100` | 0.262 (0.253–0.283) | 0.266 (0.229–0.285) | 0.250 / 0.250 | runtime 0.233 solved (2026.10.17.1) |

The breakdown column is the mean of a 5-call steady block, not a median. Plain and solved
differ by at most 4 % on every setup, so the choice changes no ordering between setups on a
host class except the A100-node source-plane tie, which is inside the p10–p90 spread either
way. What differs is what the cell means: the plain likelihood frees the source centre
(8 parameters), the solved one is the library default (5 parameters) and what the stale
source-plane runtime rows and every breakdown measure. Two further differences:

- **Per-call basis.** Every per-call number on this page and in the headline is the single-jit
  median, which both lanes measure the same way. The vmap blocks are not like-for-like and are
  not used: every published (solved) row ran vmap at batch 3 (image plane 1.10 ms per call RAL
  CPU, 0.41 ms A100; source plane 0.061 ms RAL CPU, 0.111 ms A100 CPU, 0.109 ms A100 GPU),
  while the image-plane plain A100 row ran batch 64 (0.025 ms per call), the image-plane plain
  RAL CPU row batch 3 (1.07 ms), and the source-plane plain rows record no vmap block. The
  dashboard's vmap column for the image-plane A100 cell therefore moves from 0.025 to 0.41 ms
  per call with the swap; that is batch 64 against batch 3, not a regression.
- **Load.** On RAL the six legs shared one node (euclid-ral-compute-10-5, 236 cores). The two
  solved runtime legs started at a 1-minute load average of 33.4 and wrote at 53.7 (image plane)
  and 43.9 (source plane); the plain runtime legs started at 2.2 and 6.1. The solved RAL CPU
  medians sit within 2 % of the plain ones, so the load did not visibly inflate them, but the
  release-sweep cap refuses both as trend points ([Caveats](#caveats)).

## Results (batch 413192 RAL CPU, 413193 A100)

All 15 legs exited 0 (`exit status: 0` in every SLURM log; the RAL array's six tasks ran as
jobs 413202–413207 and 413192_5), every row is the final write, and the submit's revision
check printed `revisions OK` for all 15 (the five library SHAs of tag 2026.10.25.1;
autolens_profiling 2f98e01a). No leg came near `--time` (1:00:00): the A100 job's nine legs summed to
442 s and the slowest RAL leg took 141 s (image-plane breakdown), below the laptop walls the
WALL-BASIS rows cite (1051 s for that script).

| Cell | job | wall (s), runtime / breakdown | peak RSS (GB), runtime / breakdown | parity witness committed | parity (all legs, from the logs) |
|---|---|---|---|---|---|
| image plane `jax_cpu_na_fp64@hpc_ral_cpu` solved runtime / breakdown | 413207 / 413202 | 50 / 141 | 1.06 / 2.34 | solved runtime, 5.3e-15 vs 7.7e-4 | plain 1.8e-15, breakdown 5.3e-15, solved 5.3e-15 |
| image plane `jax_gpu_na_fp64@hpc_a100` | 413193 | 45 / 239 | 1.18 / 2.67 | solved runtime, 1.2e-14 vs 7.7e-4 | plain 5.3e-15, breakdown 1.2e-14, solved 1.2e-14 |
| source plane `jax_cpu_na_fp64@hpc_ral_cpu` | 413192 (task 5) / 413204 | 24 / 43 | 0.57 / 1.40 | solved runtime, 1.6e-9 vs 6.0e-5 | plain 1.1e-9, breakdown 1.6e-9, solved 1.6e-9 |
| source plane `jax_cpu_na_fp64@hpc_a100` | 413193 | 7 / 46 | 0.58 / 1.41 | solved runtime, 1.6e-9 vs 6.0e-5 | plain 1.1e-9, breakdown 1.6e-9, solved 1.6e-9 |
| source plane `jax_gpu_na_fp64@hpc_a100` | 413193 | 9 / 57 | 0.69 / 1.51 | solved runtime, 1.6e-9 vs 6.0e-5 | plain 1.1e-9, breakdown 1.6e-9, solved 1.6e-9 |

|JIT − eager| against the tolerance. Every witness, committed or logged, is `match`; the
margins are the laptop smokes' (above). Walls and peak RSS for the plain runtime legs (now the
experiments) were 25 / 28 / 4 / 5 / 6 s and 0.88 / 0.98 / 0.55 / 0.54 / 0.67 GB in the order of
the table. A cell folder holds one `parity.json` and one `fit.png`, shared by its runtime and
breakdown rows. As committed in the first PR B pass, the witness was whichever leg finished last
(the solved breakdown on RAL CPU, the plain runtime in the A100 job); since the ruling every
cell's `parity.json` and `fit.png` are its solved runtime leg's, taken from the RAL pull. The
plain witnesses sit in `experiments/plain_runtime/` on the three A100 setups (3/3 `match`) and in
the logs on RAL CPU.

**Compared with the rows superseded.** Image plane on RAL CPU: 24.69 → 2.01 ms (fused solved),
the IP-2..IP-4c levers, as the caveat below expected. Image plane on the A100: 1.086 → 1.060 ms.
Source plane on RAL CPU: breakdown 0.146 → 0.139 ms; runtime (solved) 0.147 → 0.156 ms, measured
at load 33 (start) to 44 (write) on a shared node. Source plane on the A100 GPU: runtime (solved) 0.233 → 0.262 ms
(+12 %), breakdown 0.222 → 0.250 ms (+13 %); both of this batch's A100 numbers were taken on an
otherwise idle node (load ≤2.5). Not investigated here; flagged for phase 3.

## Coverage after this PR

`dashboard/coverage.md`, all dataset types. The planner fix #481, merged into this branch, plans
220 cells instead of 222 (the two D2 cells were `missing` in both counts):

| | planned | current | stale | missing |
|---|---|---|---|---|
| before 2c, without #481 (first PR B pass) | 222 | 106 | 7 | 109 |
| before 2c, with #481 | 220 | 106 | 7 | 107 |
| after (2c + #481) | 220 | 116 | 1 | 103 |

Point source: 6 stale and 4 missing before (plus the two D2 cells while they were planned);
10/10 current after. Parity witnesses: 53 → 58 `match` of 110 planned setup cells (111 before
#481), 0 `diverged`. The swap to the solved runtime cells changed no coverage count.
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
- The RAL CPU source-plane cell's figure as first committed (written by its breakdown) was
  zoomed to the image envelope (±2.5″); the A100 and experiment figures use the ±7.4″ frame.
  Since the ruling every cell's `fit.png` is its solved runtime leg's (the figures already opened
  as the solved experiments), so all five source- and image-plane cells use the runtime frame;
  the breakdown-written figure is no longer committed.

## Caveats

- The image-plane stale rows predate the IP-2..IP-4c CPU levers (2026.9.26.1 and 2026.9.27.2),
  so the RAL CPU image-plane number should drop well below 24.7 ms; the A100-optimal
  expectation rests on the 2026-09 IP-4b cumulative 2.095 ms (loaded node) against 1.09 ms.
- The `simple` dataset is single-source; cluster point-source cells are 2e, not this page.
- Load averages at write were 44–59 for three RAL CPU legs (both solved runtime legs and the
  image-plane breakdown) and 13 for the source-plane breakdown: the six RAL legs shared one
  node. The run-time dashboard's cap (`RELEASE_SWEEP_LOADAVG_CAP=8`, `hpc/release_sweep.conf`)
  reads the 1-minute load at import. The plain rows started at 2.2 and 6.1; the solved runtime
  rows, published since the ruling, started at 33.4. So `build_dashboard.py` refuses both RAL CPU
  runtime cells as trend points (listed under "refused", 58 → 60; their two single-release
  series drop, 196 → 194). The cap governs trend points only: both cells stay `current` in the
  catalogue and coverage, as the ~50 load-refused 2a points did. Their medians sit within 2 % of
  the quiet plain rows (1.556 against 1.527 ms, 0.156 against 0.156 ms), so the load did not
  visibly move them. A quiet-node rerun of the two solved RAL CPU runtime legs would put them on
  the trend; that is a human call and not needed for coverage. The A100 rows ran on
  euclid-ral-gpu-1, not the pinned euclid-ral-gpu-2, so they are flagged off-host either way.
- Every row stamps `autolens_version: 2026.8.17.1` (the last built version of a source
  checkout); `catalogue/release_revisions.json` publishes them at 2026.10.25.1 from their
  `library_revisions`, as in 2a and 2b. The breakdown README's version column still prints the
  stamp.

## Journal

### 2026-10-31 — phase 3, the JAX legs on jax 0.11.2 (autolens_profiling#495)

The A100 cells (413626) and the two RAL CPU breakdown cells the quiet rerun did not cover
(413683) were re-measured on jax/jaxlib 0.11.2 (human Q7), along with the `plain_runtime`
experiments. Witnesses are `match`. The breakdown-only RAL cells keep the quiet runtime row's
witness and fit figure. `vmap.per_call` after/before on the A100: image plane GPU 0.97, source
plane GPU 1.13, source plane node CPU 0.93. None moves a verdict. Prior rows are in
`history/2026.10.25.1_2026-10-28/`.

### 2026-10-31 — quiet-node rerun of the two solved RAL CPU runtime cells (autolens_profiling#493)

Both solved RAL CPU runtime cells ran on loaded nodes (load 33), so they were rerun on an
`--exclusive` quiet node (RAL 413319; 2/2 COMPLETED 0:0, `QUIET:` at load 1.9) on jax/jaxlib
0.11.2 with `--xla_cpu_multi_thread_eigen=false` (human ruling Q7, 2026-10-30). Witnesses
`match`. The prior rows are in `history/2026.10.25.1_2026-10-28/`.

| cell (`jax_cpu_na_fp64`, `hpc_ral_cpu`) | `vmap.per_call` before (ms) | after (ms) | after/before |
|---|---|---|---|
| image plane, simple | 1.10 | 0.95 | 0.87 |
| source plane, simple | 0.06 | 0.06 | 1.07 |

Neither moves a verdict on this page.

### 2026-10-29 — ruled: solved runtime cell; planner fix merged

Human ruling "Switch 2c to Solved": on the five runtime setups the solved rows (with their
`runtime.png`, `parity.json` and `fit.png`) moved from `experiments/solved_runtime/` into the
cells and the plain rows to `experiments/plain_runtime/`; no rerun. The work list
(`point_source_p2c.py`) and its two submits now run `likelihood_runtime_solved.py` as the
runtime cell and `likelihood_runtime.py` as the plain experiment. #481 (human ruling "fix the
planner gap") merged in: 220 planned cells, point source 10/10 `current`, coverage 116 current,
1 stale, 103 missing. Verdict re-checked with the solved numbers: rule met; image plane A100
0.847 against RAL CPU 1.556 ms, source plane RAL CPU 0.156 against A100 0.265 / 0.262 ms; the
A100-node `jax_cpu` / `jax_gpu` source-plane pair is still not separated. The two RAL CPU
runtime rows are refused as dashboard trend points (load 33.4 at import), still `current`.

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
