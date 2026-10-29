# Likelihood lifecycle phase 2c: point source, image plane

**Status:** open
**Question:** What do the four planned `point_source_image/image_plane` cells (instrument `simple`) cost on the pinned release 2026.10.25.1 on RAL CPU and the A100, and does each cell's log-likelihood match an independent eager NumPy evaluation of the same instance?
**Pre-registered rule:** the four cells (`jax_cpu_na_fp64@hpc_ral_cpu`, `jax_gpu_na_fp64@hpc_a100`, each runtime and breakdown) land `current` with provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy `AnalysisPoint(use_jax=False)` value (1e-4 x |reference| in fp64); a `diverged` witness blocks the setup until explained. Recorded before the batch runs (2026-10-28). The full rule, decisions D1-D3 and the parity reference are on the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md).
**Verdict:** the rule is met: the four cells are `current` on 2026.10.25.1 with provenance, `resources`, `fit.png` and a `match` witness (all six image-plane legs logged `match`, \|JIT − eager\| ≤ 1.2e-14 against 7.2e-4), so neither setup is blocked. Re-checked on 2026-10-29 with the solved runtime cells (human ruling): the committed witnesses and figures are the solved runtime legs' (5.3e-15 RAL CPU, 1.2e-14 A100, against 7.7e-4), and the rule still holds. As expected, the A100 is the faster host: 0.847 ms per call against 1.556 ms on RAL CPU (solved runtime), and the RAL CPU fused solved call dropped from the stale 24.7 ms to 2.01 ms without passing the A100's 1.06 ms. Expected, as written before the runs: the A100 optimal (1.09 ms against 24.7 ms in the stale 2026.8.17.1 breakdowns; the RAL CPU number should drop, but not below the A100's); every parity witness `match`.
**Headline:** `jax_gpu_na_fp64@hpc_a100` 0.847 ms per call (single-jit median of 200, the solved library default `FitPositionsImagePairAllSolved`; free-centre `FitPositionsImagePairAll` 0.864 ms, in `experiments/plain_runtime/`) against `jax_cpu_na_fp64@hpc_ral_cpu` 1.556 ms (plain 1.527 ms); A100 job 413193, RAL 413207. The runtime cell is the solved likelihood by the human ruling of 2026-10-29 (the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md#which-likelihood-the-runtime-cell-times-ruled-solved)). The RAL CPU runtime row is refused as a dashboard trend point (load 33.4 at import, cap 8) but stays `current`.
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the campaign branch `feature/likelihood-lifecycle-p2c-point-source`, then PR B #480 on the same branch (rows, catalogue, coverage, these pages), with the planner fix #481 merged in
**Ledger:** the [2c campaign page](../../point_source_source/source_plane/likelihood_lifecycle_p2c.md); work list [`scripts/misc/campaigns/point_source_p2c.py`](../../../../scripts/misc/campaigns/point_source_p2c.py)
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2c; issue autolens_profiling#472
**Next:** merge PR B (#480) after #471 and #481
**Fit figures:** opened all four image-plane `fit.png` (cells and solved experiments): the model positions land on the four observed images; nothing looked wrong. Since the ruling both cells carry the solved runtime leg's figure.

## Why this campaign

The coverage matrix lists the two image-plane breakdowns as `stale` (2026.8.17.1) and the two
runtimes as `missing`. This page holds the image-plane half of 2c; the source-plane page holds the
shared material and the two mis-designated source-plane cells.

## The cells

| Setup @ host class | Runtime | Breakdown | Runs in |
|---|---|---|---|
| `jax_cpu_na_fp64@hpc_ral_cpu` (optimal) | `likelihood_runtime_solved.py` | `likelihood_breakdown.py` | the `ral` array |
| `jax_gpu_na_fp64@hpc_a100` (optimal) | `likelihood_runtime_solved.py` | `likelihood_breakdown.py` | the A100 job (`JAX_PLATFORMS=cuda`) |

**Ruled: solved (human 2026-10-29).** "Switch 2c to Solved": the runtime cell times
`FitPositionsImagePairAllSolved` with `PointSolved` (5 parameters), the library and workspace
default for a galaxy-scale point source (PyAutoLens `autolens/point/model/analysis.py`,
`autolens_workspace/scripts/point_source/modeling.py`) and what the breakdown decomposes. The
batch ran the free-centre `likelihood_runtime.py` (`FitPositionsImagePairAll`, 8 parameters) as
the cell and `likelihood_runtime_solved.py` into `experiments/solved_runtime/`; after the ruling
the solved rows, with their `parity.json` and `fit.png`, moved into the cells and the plain rows
to `experiments/plain_runtime/` (no rerun), and the work list now runs them that way round. The
breakdowns were already solved and are unchanged.

## Fit figures

Open each new row's `fit.png` before you accept it. Note here any cell whose figure looked wrong.

Opened 2026-10-29. Both cell folders and both solved experiments show the four observed images
of `simple` on the critical curve with the model positions on them: −2 ln L = 2.22 for the
plain likelihood (A100 cell, written by its runtime leg) and 1.13 for the solved one (RAL CPU
cell, written by its breakdown leg, and both experiments). Nothing looked wrong. Since the ruling
both cells carry the solved runtime leg's figure (−2 ln L = 1.13); the plain A100 figure is in
`experiments/plain_runtime/`.

## Results (RAL 413202/413203, A100 413193)

Single-jit median of 200 calls (p10–p90), ms; breakdown is the fused 5-call steady mean:

| Setup @ host class | solved runtime (published cell) | plain runtime (`experiments/plain_runtime/`) | breakdown fused solved / plain | stale breakdown superseded |
|---|---|---|---|---|
| `jax_cpu_na_fp64@hpc_ral_cpu` | 1.556 (1.466–1.910) | 1.527 (1.452–1.920) | 2.014 / 1.770 | 24.69 (2026.8.17.1) |
| `jax_gpu_na_fp64@hpc_a100` | 0.847 (0.833–0.906) | 0.864 (0.830–0.896) | 1.060 / 1.110 | 1.086 (2026.8.17.1) |

Per-call basis: the single-jit median throughout; the vmap blocks are not like-for-like and are
not used. Both solved rows ran vmap at batch 3 (1.10 ms per call RAL CPU, 0.41 ms A100); the
plain A100 row ran batch 64 (0.025 ms) and the plain RAL CPU row batch 3 (1.07 ms). The
dashboard's A100 vmap column therefore moves from 0.025 to 0.41 ms with the swap: batch 64
against batch 3, not a regression. The RAL CPU solved runtime started at load average 33.4 and
wrote at 53.7 (the breakdown wrote at 59), on a node shared with the other RAL legs; the plain
runtime started at 2.2 and wrote at 6.1. The run-time dashboard reads the load at import
against a cap of 8, so it refuses the published RAL CPU row as a trend point; the cell stays
`current` (the cap governs trend points, not lifecycle status), and its median sits 2 % from
the quiet plain row's. Parity, per leg: runtime 1.8e-15 / 5.3e-15, breakdown 5.3e-15 /
1.2e-14, solved 5.3e-15 / 1.2e-14 (RAL CPU / A100), tolerance 7.2e-4 (plain) and 7.7e-4
(solved). The two stale breakdowns moved to `history/2026.8.17.1_2026-09-23/` (RAL CPU) and
`history/2026.8.17.1_2026-10-17/` (A100). Coverage: 4/4 image-plane cells current (2 stale and
2 missing before); unchanged by the swap.

## Journal

### 2026-10-29 — ruled: solved runtime cell

Human ruling "Switch 2c to Solved". Both setups' solved rows moved into the cells (with their
`parity.json` and `fit.png`), the plain rows to `experiments/plain_runtime/`; no rerun. Verdict
re-checked: rule met, A100 0.847 against RAL CPU 1.556 ms. The RAL CPU runtime row is refused
as a dashboard trend point (load 33.4 at import) and stays `current`. Planner fix #481 merged
into the branch (source plane only; image plane unaffected).

### 2026-10-29 — batch committed (PR B)

All six image-plane legs exit 0 with `revisions OK` on tag 2026.10.25.1; four cells and two
solved experiments copied, the two stale breakdowns moved to `history/`. Rule met. The runtime
cell's likelihood (plain or solved) waits for the human.

### 2026-10-28 — branch ready for the batch

Rule and expected verdict recorded before any RAL run.
