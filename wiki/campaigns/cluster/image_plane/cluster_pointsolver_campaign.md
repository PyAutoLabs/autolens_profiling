# Cluster PointSolver campaign ledger

**Campaign:** `cluster-pointsolver` (epic `cluster-pointsolver-speed`). Pulse task
[cluster_pointsolver_speed](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/cluster_pointsolver_speed.md).
Wiki page: [cluster_pointsolver.md](cluster_pointsolver.md).
**Phase 1:** cluster-scale robust `PointSolver` settings,
[autolens_profiling#436](https://github.com/PyAutoLabs/autolens_profiling/issues/436).
Mind prompt `active/cluster_pointsolver_robust_settings_p1.md`.
**Status:** DECIDED 2026-10-23 (human): keep 1.0″ as the production starting grid, with a validation check as mitigation (the 2026-10-22 deferral is closed). The `simple` sweep is complete and quotable.
**Follow-up #439 (2026-10-23):** the A2744 pilot leg is pulled ([A2744](#a2744-pilot-leg-job-412573-pulled-2026-10-23)) and the starting-grid question is measured ([Starting grid](#starting-grid-vs-close-member-image-439)): no fixed starting scale finds every image, so the recommendation is a validation check, not a new default. No default was changed anywhere.

## TL;DR

**Decision (human, 2026-10-23): keep 1.0″** as the production starting grid and add a validation check; this closes the 2026-10-22 deferral. See [Decision table](#decision-table-human).

**#439 (2026-10-23), both pending inputs now measured:**

- **The claim is half right.** In the autolens_visualization cluster model, image 3 of
  `point_0` (μ −0.89, 0.39″ from `main_1`'s core) is lost by the workspace grid (100×100 @
  1.0″). But it is also lost at 0.37″ and 0.3″. Of the seven scales tested (1.0–0.2″), only
  0.2″ finds it. The "~0.37″ threshold" does not reproduce on the production extent.
- **No fixed scale is safe.** Even 0.2″ misses 3 of 108 viz roots (central |μ| < 1 images
  0.17–0.33″ from an `r_core = 0` member core). Finer is not monotonically safer: 6
  draw-sources are found at a coarser scale and lost at a finer one. On A2744 the 1.0″ start
  misses fewest (2/69).
- **No separation rule holds.** "scale ≲ separation / 1.06" is contradicted. Image-to-image
  or image-to-core separation alone needs c ≈ 11–14. The smaller of the two is not
  contradicted at c = 1.66, but that rests on 80 misses in two synthetic models.
- **Recommendation:** keep 1.0″ and add a validation check (every observed image needs a
  solved counterpart, else re-solve finely around it). A fixed finer grid costs 2–8× per
  solve (laptop, diagnostic) and still misses images.


- **No row is admissible on the 101 judged draws of `dataset/cluster/simple`.** Every
  production-extent row covers all **306/306** oracle roots of the prior set (truth + 50 prior
  draws). Every one misses **2 to 6 of the 300** roots of the stress set (50 perturbed draws).
- **Every miss is a close high-|μ| image pair** 0.33–0.76″ apart, on 3 draw-sources only (stress
  draws 53, 73 and 81). The production 100×100 @ 1.0″ step-0 grid puts both images of the pair in
  one triangle, or puts neither in a kept triangle.
- **Initial scale 0.5″ and 0.7″ miss fewest (2 roots, draw 81 only).** Scale 1.0″ (the production
  control) misses 5 and scale 2.0″ misses 6. Precision (0.01 vs 0.001), capacity
  (`MAX_CONTAINING_SIZE` 20/40/60) and backend (NumPy / JIT / vmap 4 / vmap 16) change nothing. The
  uncapped containing count never exceeds 7, so the cap of 20 never binds.
- The finer start costs time. Scale 0.5″ is 6.82 ms against the control's 5.29 ms per warmed
  two-source solve (0.78× the speed). Scale 0.7″ is 7.02 ms (0.75×).
- `control_eager` is **unusable**: its projected row wall of 2675 s is over the 1800 s row cap
  (first call 21.6 s).
- Image-bounding-box extents (`bbox_*`, the PyAutoLens#764 margin) lose 49–63 stress roots. Most
  fall outside the box, because stress-draw images move beyond the truth's bounding box. They are
  diagnostic only.
- Peak RSS for the whole `simple` sweep was 4.3 GB.

## Decision table (human)

Per warmed two-source solve, CPU fp64, RAL Xeon Platinum 8490H (`euclid-ral-compute-10-4`, job
412572). Production extent is the production grid 100×100 @ 1.0″ (±49.5″), with magnification
threshold 0.1. The table shows the precision 0.001 rows. Precision 0.01 has identical coverage, so
see the full table for those rows.

| Row | Initial scale | Prior coverage | Stress misses (of 300) | Draw-sources missed | Warmed ms [90 % CI] | vs control | Compile s |
|---|---|---|---|---|---|---|---|
| `production_s0.5_p0.001` | 0.5″ | 306/306 | **2** | 81/1 | 6.82 [6.72, 6.98] | 0.78× | 6.52 |
| `production_s0.7_p0.001` | 0.7″ | 306/306 | **2** | 81/1 | 7.02 [6.94, 7.10] | 0.75× | 7.42 |
| `control` (production) | 1.0″ | 306/306 | 5 | 53/0, 73/0, 81/1 | 5.29 [5.24, 5.34] | 1.00× | 7.55 |
| `production_s2_p0.001` | 2.0″ | 306/306 | 6 | 53/0 (both), 73/0, 81/1 | 4.06 [4.01, 4.09] | 1.30× | 8.14 |

The decision is whether a 2-miss row at about 0.77× speed is better than the 5-miss production
control. Neither row is admissible, and neither passes the stress set. This ledger makes **no
recommendation** and changes no default. The open question is in [Follow-ups](#follow-ups).

**Decision (human, 2026-10-22): DEFER.** The production default stays at 1.0″. The choice is
re-opened once (i) the starting-grid draft
(`PyAutoMind/draft/research/autolens_profiling/cluster_pointsolver_starting_grid_close_member_image.md`)
has tested the same close-pair geometry on a real member galaxy and (ii) the A2744 pilot leg (job
412573) has been pulled. 0.5″ is the working answer if the choice must be made before then.

**Decision (human, 2026-10-23): KEEP 1.0″.** Both pending inputs are in hand (#439 →
[#440](https://github.com/PyAutoLabs/autolens_profiling/pull/440), merged `03d36c81`).

| Date | Choice | Evidence | Mitigation |
|---|---|---|---|
| 2026-10-23 (human, chat) | Keep the 1.0″ starting grid as the production default; the 2026-10-22 deferral is closed | `simple` table above (0.5″/0.7″ miss 2, 1.0″ misses 5 of 300 stress roots); [A2744 pilot](#a2744-pilot-leg-job-412573-pulled-2026-10-23) (1.0″ misses fewest, 2/69); [Starting grid](#starting-grid-vs-close-member-image-439) (no fixed scale finds every image, no separation rule) via [#440](https://github.com/PyAutoLabs/autolens_profiling/pull/440); Mind record [cluster-pointsolver-starting-grid.md](https://github.com/PyAutoLabs/PyAutoMind/blob/main/complete/2026/10/cluster-pointsolver-starting-grid.md); Pulse decision record [cluster-pointsolver-starting-grid](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/decisions/cluster-pointsolver-starting-grid.md) | Validation check: Mind draft `draft/feature/autolens_workspace/cluster_point_solver_validation_check.md` (unissued) |

The decision changes no library or workspace default; the production default already is 1.0″.

## Tested domain

| Item | Value |
|---|---|
| Dataset | `dataset/cluster/simple`: 2 dPIE + 10 scaling dPIE + NFW = 13 mass components; planes z = 0.5 / 1.0 / 2.0; sources `point_0` (z 1.0) and `point_1` (z 2.0), 3 images each in the truth; noise 0.005″. Regenerated on RAL by `scripts/misc/simulators/cluster.py` (simulator sha256 `34dce2637d11…`, seed 1); truth gate passes (traced difference 0.0″). |
| Draws judged | 101 = truth + 50 prior draws (seed 436, the cell's fused model priors) + 50 stress draws (seed 2436; centres ±0.5″ σ, `mass_at_200` 0.3 dex, scaling `b0_ref` 15 %, other mass parameters 10 %, no clipping). Stress betas are always back-traced from the draw. |
| Judged roots | prior 306 (102 draw-sources × 3), stress 300 (100 draw-sources; 1–7 roots each above \|μ\| 0.1) |
| Production solver | `autolens_workspace/scripts/cluster/modeling.py` shape as built for this dataset: 100×100 @ 1.0″, `pixel_scale_precision` 0.001, `magnification_threshold` 0.1, `MAX_CONTAINING_SIZE` 20 |
| Not tested | A2744 beyond the 3-draw synthetic pilot (16 diagnostic rows, job 412573; no reference tracer, no per-root record), A100, more than 2 sources on `simple` / `viz_cluster`, fp32, gradients |
| Library | private RAL checkout `PyAuto_2026.10.17.1_interf_meas`: PyAutoLens `c90eceb0`, PyAutoArray `e59d5fa9`, PyAutoGalaxy `35100def`, PyAutoFit `4b183e22`, PyAutoNerves `7f3055b7`; profiling `b39f7ab2`; jax/jaxlib 0.10.2, numpy 2.2.6 |
| Host | Xeon Platinum 8490H, `--partition=ral --exclusive`, 8 JAX threads (OMP/BLAS 1), loadavg 0.08 at start and 1.71 at end; wall 1358 s |
| Timing protocol | 20 interleaved round-robin rounds × 10 calls (3 warm), medians with a 2000-sample paired-round bootstrap 90 % CI; fresh closure + `jax.clear_caches()` per JIT route; vmap rows are batch time / B |

## Ground truth: the oracle

`scripts/cluster/image_plane/_oracle.py` samples the lens-equation residual
`|θ − α(θ) − β|` densely with the same `Tracer` multi-plane deflections (NumPy fp64). It then
Newton-polishes every basin. A root is accepted only when its residual is below 1e-10″, it moves
less than 1e-6″ under 2× finer sampling, its final Newton correction is below 5e-8″, and its
magnification is finite. It never calls the triangle solver.

- **Calibration (on RAL, in the JSON):** 6/6 cases pass. They are the 1a closed-form isothermal +
  shear quad (boundary, offset ±) and the 1c/1d near-cusp fixture down to cusp 1e-7. The maximum
  error is 2.84e-8″ against the 1e-6″ gate (laptop: 1.96e-8″).
- **Stability:** 0 unstable draw-sources out of 202. Oracle cost is 6.2 s per draw (median), 10.8 s
  at most.
- **Stored positions:** the dataset's stored positions are solver output, not truth. They lie
  0.4–0.6 mas from the oracle's truth roots.
- **Gates (all pass):** oracle calibration, `MAX_CONTAINING_SIZE` patch shapes on every route,
  patch restored, later defaults kept, reference truth gate.

**PyAutoLens `FitPositionsSource` default change** (jacobian weighting, PyAutoLens `bdb44a87c`):
this does not affect the result. None of the sweep scripts (`robust_settings_sweep.py`,
`_oracle.py`, `_judge.py`, `_datasets.py`, `_memory.py`, `oracle_figure.py`, `solver_figure.py`)
uses `FitPositions*` or a chi-squared. The sweep judges solver positions against oracle roots only.

## Supported-settings table (all 25 rows)

The source is `results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/robust_settings_sweep/breakdown.json`
(`table`, `rows`), with a PNG beside it. The tolerance is the row's last-step triangle side.
Overflow means the uncapped count is above the cap on any step. Every row passes the prior set,
except `tight` (designed to fail) and `control_eager` (not run). **No row is admissible.**

| Row | Backend | Extent | Scale | Prec | MCS | Tol ″ | Status | Prior | Stress | Overflow (uncapped max) | Max pos err ″ | Compile s | First call s | Warmed ms [90 % CI] | vs control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_s0.5_p0.01 | jit | production | 0.5 | 0.01 | 20 | 0.0156 | diagnostic | 306/306 | 298/300 | no (7) | 0.0089 | 4.67 | 0.013 | 6.06 [6.00, 6.15] | 0.87 |
| production_s0.5_p0.001 | jit | production | 0.5 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 298/300 | no (7) | 0.0011 | 6.52 | 0.010 | 6.82 [6.72, 6.98] | 0.78 |
| production_s0.7_p0.01 | jit | production | 0.7 | 0.01 | 20 | 0.0109 | diagnostic | 306/306 | 298/300 | no (7) | 0.0062 | 5.53 | 0.009 | 6.17 [6.10, 6.21] | 0.86 |
| production_s0.7_p0.001 | jit | production | 0.7 | 0.001 | 20 | 0.0014 | diagnostic | 306/306 | 298/300 | no (7) | 0.0008 | 7.42 | 0.010 | 7.02 [6.94, 7.10] | 0.75 |
| production_s1_p0.01 | jit | production | 1.0 | 0.01 | 20 | 0.0156 | diagnostic | 306/306 | 295/300 | no (7) | 0.0089 | 5.81 | 0.007 | 4.38 [4.34, 4.44] | 1.21 |
| **control** | jit | production | 1.0 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 295/300 | no (7) | 0.0011 | 7.55 | 0.008 | 5.29 [5.24, 5.34] | 1.00 |
| production_s2_p0.01 | jit | production | 2.0 | 0.01 | 20 | 0.0156 | diagnostic | 306/306 | 294/300 | no (7) | 0.0089 | 6.44 | 0.007 | 3.23 [3.20, 3.25] | 1.64 |
| production_s2_p0.001 | jit | production | 2.0 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 294/300 | no (7) | 0.0011 | 8.14 | 0.008 | 4.06 [4.01, 4.09] | 1.30 |
| bbox_s0.5_p0.01 | jit | bbox | 0.5 | 0.01 | 20 | 0.0156 | diagnostic | 306/306 | 237/300 | no (7) | 0.0089 | 5.65 | 0.007 | 3.01 [2.95, 3.05] | 1.76 |
| bbox_s0.5_p0.001 | jit | bbox | 0.5 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 237/300 | no (7) | 0.0011 | 7.40 | 0.007 | 3.92 [3.88, 3.98] | 1.35 |
| bbox_s0.7_p0.01 | jit | bbox | 0.7 | 0.01 | 20 | 0.0109 | diagnostic | 306/306 | 239/300 | no (7) | 0.0062 | 5.75 | 0.006 | 2.89 [2.86, 2.94] | 1.83 |
| bbox_s0.7_p0.001 | jit | bbox | 0.7 | 0.001 | 20 | 0.0014 | diagnostic | 306/306 | 239/300 | no (7) | 0.0008 | 7.51 | 0.007 | 3.77 [3.75, 3.81] | 1.40 |
| bbox_s1_p0.01 | jit | bbox | 1.0 | 0.01 | 20 | 0.0156 | diagnostic | 306/306 | 237/300 | no (6) | 0.0089 | 5.79 | 0.006 | 2.66 [2.63, 2.69] | 1.99 |
| bbox_s1_p0.001 | jit | bbox | 1.0 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 237/300 | no (6) | 0.0011 | 7.51 | 0.008 | 3.50 [3.47, 3.54] | 1.51 |
| bbox_s2_p0.01 | jit | bbox | 2.0 | 0.01 | 20 | 0.0156 | diagnostic | 306/306 | 251/300 | no (7) | 0.0089 | 6.39 | 0.006 | 2.77 [2.74, 2.80] | 1.91 |
| bbox_s2_p0.001 | jit | bbox | 2.0 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 251/300 | no (7) | 0.0011 | 8.17 | 0.007 | 3.64 [3.61, 3.67] | 1.45 |
| tight_s1_p0.001 | jit | tight (75 %) | 1.0 | 0.001 | 20 | 0.0020 | diagnostic (by design) | 204/306 | 196/300 | no (4) | 0.0011 | 7.68 | 0.007 | 3.37 [3.35, 3.40] | 1.57 |
| control_mcs40 | jit | production | 1.0 | 0.001 | 40 | 0.0020 | diagnostic | 306/306 | 295/300 | no (7) | 0.0011 | 7.71 | 0.009 | 6.78 [6.68, 6.86] | 0.78 |
| control_mcs60 | jit | production | 1.0 | 0.001 | 60 | 0.0020 | diagnostic | 306/306 | 295/300 | no (7) | 0.0011 | 7.72 | 0.012 | 8.92 [8.72, 8.97] | 0.59 |
| production_s2_p0.001_mcs40 | jit | production | 2.0 | 0.001 | 40 | 0.0020 | diagnostic | 306/306 | 294/300 | no (7) | 0.0011 | 8.36 | 0.009 | 5.59 [5.47, 5.78] | 0.95 |
| production_s2_p0.001_mcs60 | jit | production | 2.0 | 0.001 | 60 | 0.0020 | diagnostic | 306/306 | 294/300 | no (7) | 0.0011 | 8.40 | 0.011 | 8.14 [7.88, 8.25] | 0.65 |
| control_numpy | numpy | production | 1.0 | 0.001 | uncapped | 0.0020 | diagnostic | 306/306 | 295/300 | n/a | 0.0011 | — | 0.189 | 164.85 [162.91, 235.80] | 0.03 |
| control_eager | eager JAX | production | 1.0 | 0.001 | 20 | 0.0020 | **unusable** | — | — | — | — | — | 21.571 | — | — |
| control_vmap4 | vmap 4 | production | 1.0 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 295/300 | no (7) | 0.0011 | 8.24 | 0.020 | 3.40 [3.37, 3.43] | 1.56 |
| control_vmap16 | vmap 16 | production | 1.0 | 0.001 | 20 | 0.0020 | diagnostic | 306/306 | 295/300 | no (7) | 0.0011 | 8.20 | 0.045 | 2.35 [2.34, 2.36] | 2.25 |

- **Multiplicity:** stress agreement is 0.99 for the 0.5/0.7 rows, 0.97 for the 1.0 and 2.0 rows. No row produced a duplicated or spurious candidate.
- **Precision:** 0.01 and 0.001 have identical coverage at every scale. 0.01 is faster (1.21× at
  scale 1.0), but its position error is 8.9 mas against 1.1 mas, so it is not
  precision-equivalent (`precision_equivalent` false).
- **Capacity:** MCS 40/60 change no coverage and cost 0.59–0.95×. The uncapped max is 7 against
  the cap of 20.
- **Backend:** NumPy, JIT and vmap miss the same 5 roots, so the misses come from the geometry
  and not from the JAX path.
- **`bbox` extents:** the box is the truth's image bounding box + `2·scale + 3σ`. Stress draws
  move images outside it: 41–58 of the 49–63 misses are outside the extent. A per-model bbox is
  not a usable cluster extent without a margin that follows the model.

### #439 row sets: starting scale at production extent (laptop, diagnostic)

These rows come from `host: laptop (diagnostic)`, an Intel Core i9-10885H, CPU fp64, run
with `--release-executables` (each row timed on its own, not interleaved). The coverage
columns are valid because they are fp64 and judged against the oracle. The ms columns are
indicative only: they are not campaign timing rows, and they are not comparable to the RAL
table above. Every row uses the production extent (±49.5″), precision 0.001 and
magnification threshold 0.1. Each row is `invalid` because it misses a prior root.

| Scale ″ | Step-0 triangles | `viz_cluster` prior (truth + 8) | `viz_cluster` stress (8) | viz truth `point_0` | viz spurious | `simple` 53/73/81 (of 22) | Laptop warmed ms (viz) |
|---|---|---|---|---|---|---|---|
| 2.0 | 5 841 | — | — | — | — | 16 (6 missed) | — |
| **1.0 (workspace)** | 23 283 | 54/63 | 42/45 | 4/5 | 1 | 17 (5 missed) | 7.4 |
| 0.7 | 46 695 | 54/63 | 42/45 | 4/5 | 1 | 20 (2 missed) | 10.9 |
| 0.5 | 91 707 | 54/63 | 44/45 | 4/5 | 1 | 20 (2 missed) | 13.3 |
| 0.4 | 142 065 | 58/63 | 43/45 | 4/5 | 0 | **22** | 17.0 |
| 0.37 | 166 385 | 55/63 | 43/45 | 4/5 | 0 | 20 (2 missed) | 16.3 |
| 0.3 | 253 163 | 55/63 | 44/45 | 4/5 | 1 | **22** | 42.8 |
| 0.2 | 567 843 | **60/63** | **45/45** | **5/5** | 0 | **22** | 60.5 |

The `simple` laptop rows reproduce the RAL rows at 2.0/1.0/0.7/0.5″ exactly (6/5/2/2 missed).
See [Starting grid](#starting-grid-vs-close-member-image-439) for the analysis.

## Unresolved cases (every miss)

The same three draw-sources account for every miss in every production-extent row. Positions are
(y, x) ″, μ is the oracle's signed magnification and "sep" is the distance to the root's
nearest sibling.

| Draw / source | Missed root (μ) | Partner (μ) | Sep ″ | Missed by |
|---|---|---|---|---|
| 53 / `point_0` | (2.24, 13.05) (+56.1) | (−22.6) | 0.73 | scale 1.0 (1 root), scale 2.0 (both) |
| 73 / `point_0` | (3.85, 12.58) (+111.3) and (3.09, 12.71) (−41.3) | each other, 3rd image of a 0.76″ triplet on the tangential curve | 0.76 | scale 1.0, 2.0 |
| 81 / `point_1` | (2.32, −2.36) (−125.5) and (2.09, −2.13) (+111.4) | each other, a pair straddling the **radial** critical curve | 0.33 | every scale, 0.5–2.0 |

Draw 73 is a 7-image configuration. The 0.5″ start finds all 7 images, while the 1.0″ start finds
5. Draw 81's radial pair is missed even at 0.5″. Its separation (0.33″) is below the step-0
triangle side of every row tested.

## Figures

All figures are under [`results/breakdown/cluster/figures/`](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/).
They are local renders on the laptop, not timings.

- **Oracle verification**
  ([`oracle_figure.py`](../../../../scripts/cluster/image_plane/oracle_figure.py)). For each plane, the
  figure shows the oracle roots, the critical curves and caustics, one panel per row (control,
  `production_s0.5_p0.001`, `bbox_s2_p0.001`) with the verdict of each root, and zooms on the
  close groups:
  [53/plane 1](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/oracle_figure_figures_draw53_plane1/breakdown.png),
  [53/plane 2](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/oracle_figure_figures_draw53_plane2/breakdown.png),
  [73/plane 1](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/oracle_figure_figures_draw73_plane1/breakdown.png),
  [73/plane 2](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/oracle_figure_figures_draw73_plane2/breakdown.png),
  [81/plane 1](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/oracle_figure_figures_draw81_plane1/breakdown.png),
  [81/plane 2](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/oracle_figure_figures_draw81_plane2/breakdown.png).
- **The library's own `point_solver.png`**
  ([`solver_figure.py`](../../../../scripts/cluster/image_plane/solver_figure.py)). This is the readable
  cluster visualizer of PyAutoLens#795 (`b1754a7ec`; local canonical checkout at `b273f9c7d`). The
  figure re-solves the row's NumPy steps on the draw's tracer and draws the oracle roots as the
  hollow "observed" circles, so a missed root is a circle with no solved cross:
  - control:
    [53](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/solver_figure_figures_control_draw53_source0/breakdown.png),
    [73](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/solver_figure_figures_control_draw73_source0/breakdown.png),
    [81](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/solver_figure_figures_control_draw81_source1/breakdown.png)
  - initial scale 0.5″:
    [53](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/solver_figure_figures_production_s0.5_p0.001_draw53_source0/breakdown.png),
    [73](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/solver_figure_figures_production_s0.5_p0.001_draw73_source0/breakdown.png),
    [81](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/solver_figure_figures_production_s0.5_p0.001_draw81_source1/breakdown.png)

  On draw 73 the control keeps 5 step-0 triangles and the 0.5″ start keeps 7. On draw 81 both
  keep 1 triangle and never see the radial pair.

The local `dataset/cluster/simple` file digests differ from RAL's. Both were regenerated by the
same seeded simulator, and the bytes differ. The figures take the model structure from that
dataset and every parameter vector from the JSON, so they draw the judged draws.

## Pilot (job 412567)

Job 412567 ran the `simple` leg on 5 draws (truth + 2 prior + 2 stress) at profiling `23b72de2`:
every production/control row was admissible 30/30 and every `bbox` row missed 2. The 5-draw pilot
was too small to reach the close-pair draws. The full sweep's 50 stress draws found them. The pilot
JSON and PNG are committed beside the sweep for its wall projection.

## A2744 pilot leg (job 412573, pulled 2026-10-23)

**Diagnostic only, never quotable.** The leg ran
`hpc/batch_cpu/submit_breakdown_cluster_robust_settings_pilot_ral_cpu_fp64` (`--pilot`) at
profiling `b39f7ab2` on the private RAL checkout `PyAuto_2026.10.17.1_interf_meas`, Xeon
Platinum 8490H, `ral` partition, CPU fp64. `sacct`: **FAILED**, elapsed 03:33:07, MaxRSS
85.5 GB. The log is committed as
[`cluster_pointsolver_2026_10_18_ral_job_412573_a2744_pilot.out`](../../../../results/misc/logs/cluster_pointsolver/cluster_pointsolver_2026_10_18_ral_job_412573_a2744_pilot.out)
(1951 lines). No JSON was written, so the progress lines below are the whole record: there
are no per-root positions, no timing rounds and no backend rows.

- **Model.** `a2744`: 191 mass components, 8 planes, 7 sources at z 1.688–5.662. The
  reference is the A2744-geometry synthetic prior-median model (`prior-median-smoke`), not a
  fitted cluster: no A2744 reference tracer exists (see [Tested domain](#tested-domain)).
- **Draws.** 3 judged (draw 0 reference, 1 prior, 2 stress): 69 considered roots, 49 prior
  and 20 stress. Oracle calibration passes (2.842e-08″); every draw-source is stable; the
  oracle costs 84–172 s per draw.
- **Capacity.** The uncapped containing count reaches 7 (scales 0.5/0.7), 9 (1.0/2.0) and
  8 (`tight`), so the cap of 20 never binds.

| Row | Scale ″ | Prec | Prior | Stress | Missed (of 69) | Max pos err ″ | RSS after row GB | Peak GB |
|---|---|---|---|---|---|---|---|---|
| production_s0.5_p0.01 | 0.5 | 0.01 | 48/49 | 18/20 | 3 | 0.0077 | 8.3 | 22.1 |
| production_s0.5_p0.001 | 0.5 | 0.001 | 48/49 | 18/20 | 3 | 0.0009 | 12.4 | 42.0 |
| production_s0.7_p0.01 | 0.7 | 0.01 | 48/49 | 16/20 | 5 | 0.0059 | 13.1 | 42.0 |
| production_s0.7_p0.001 | 0.7 | 0.001 | 48/49 | 16/20 | 5 | 0.0007 | 16.6 | 53.1 |
| production_s1_p0.01 | 1.0 | 0.01 | **49/49** | 18/20 | **2** | 0.0077 | 16.8 | 53.1 |
| **control** (120×120 @ 1.0″) | 1.0 | 0.001 | **49/49** | 18/20 | **2** | 0.0009 | 20.1 | 56.7 |
| production_s2_p0.01 | 2.0 | 0.01 | 47/49 | 14/20 | 8 | 0.0077 | 20.8 | 56.7 |
| production_s2_p0.001 | 2.0 | 0.001 | 47/49 | 14/20 | 8 | 0.0009 | 24.5 | 67.6 |
| bbox_s0.5_p0.01 | 0.5 | 0.01 | 48/49 | 10/20 | 11 | 0.0077 | 24.0 | 67.6 |
| bbox_s0.5_p0.001 | 0.5 | 0.001 | 48/49 | 10/20 | 11 | 0.0009 | 26.6 | 67.6 |
| bbox_s0.7_p0.01 | 0.7 | 0.01 | 48/49 | 11/20 | 10 | 0.0059 | 27.4 | 67.6 |
| bbox_s0.7_p0.001 | 0.7 | 0.001 | 48/49 | 11/20 | 10 | 0.0007 | 30.6 | 67.6 |
| bbox_s1_p0.01 | 1.0 | 0.01 | 49/49 | 11/20 | 9 | 0.0077 | 30.8 | 67.6 |
| bbox_s1_p0.001 | 1.0 | 0.001 | 49/49 | 11/20 | 9 | 0.0009 | 33.9 | 70.7 |
| bbox_s2_p0.01 | 2.0 | 0.01 | 47/49 | 12/20 | 10 | 0.0077 | 34.7 | 70.7 |
| bbox_s2_p0.001 | 2.0 | 0.001 | 47/49 | 12/20 | 10 | 0.0009 | 38.1 | 81.5 |
| tight_s1_p0.001 | 1.0 | 0.001 | — | — | — | — | 39.4 | 81.5 |
| control_mcs40 | 1.0 | 0.001 | — | — | — | — | 39.4 | 81.5 |
| control_mcs60 | 1.0 | 0.001 | — | — | — | — | 39.4 | 81.5 |

Every judged row is `diagnostic` (no reference tracer). No row has a duplicated or spurious
candidate. The last three rows are **invalid**: `JaxRuntimeError: INTERNAL: Failed to
materialize symbols` after hundreds of `LLVM compilation error: Cannot allocate memory`.
The job then died on `LLVM ERROR: Unable to allocate section memory!` before the backend
rows and the timing rounds.

- **Ordering is dataset-dependent.** On A2744 the production 1.0″ start misses fewest (2/69,
  both stress roots) and is the only scale with every prior root found. 0.5″ misses 3 (one
  of them a prior root that 1.0″ finds), 0.7″ 5 and 2.0″ 8. This is the opposite of `simple`,
  where 0.5″/0.7″ miss fewest. Precision changes nothing here either. Three draws on a
  synthetic model cannot carry a decision.
- **RSS attribution.** RSS climbs 8.3 → 39.4 GB over 16 rows (2–4 GB per row, more at
  precision 0.001) while `jax.live_arrays()` stays at 5 arrays (~0 MB). Peak RSS climbs
  22 → 81.5 GB. Live data is not the cause. Each row keeps one compiled 191-component
  multi-plane solver executable for the interleaved timing phase, and each compile adds a
  transient on top of the retained code. The LLVM JIT then runs out of section memory.
- **Harness finding (fixed in #439).** Release each row's executable before the next row,
  or run one row per process. `robust_settings_sweep.py --release-executables` (#439) times
  each row on its own, drops the executable and runs `jax.clear_caches()` + `gc.collect()`.
  An A2744 full sweep needs that flag **and** explicit human authorization; none is given,
  and nothing was resubmitted.

## Starting grid vs close member image (#439)

[autolens_profiling#439](https://github.com/PyAutoLabs/autolens_profiling/issues/439) tests
an unverified claim from PyAutoLens#795: in the autolens_visualization cluster model, image 3
of `point_0` sits 0.39″ from a member core and is lost by any starting grid coarser than
~0.37″. If true, the workspace cluster example's 100×100 @ 1.0″ grid
(`autolens_workspace/scripts/cluster/modeling.py`) misses it.

**Model (`--dataset viz_cluster`).** `_datasets.load_viz_cluster` copies the constants of
`autolens_visualization/true_models/cluster.py`. The copy was taken at viz `7a1b825b`; the
case also records the file's sha256 `0b404bb4…` and a sha256 of the copied constants. The
model has 2 main dPIEMassSph (σ/r_core/r_cut), 10 scaling dPIEMassSph members on one shared
σ_ref with `r_core = 0`, and an NFWMCRLudlowSph host: 13 components, sources at z 1.0/2.0.
It is not `simple`, which uses dPIEMassB0Sph. The positions are derived by the oracle on the
explicit-constants truth tracer, never by the triangle solver. The truth gate passes: the
model tracer equals the explicit tracer to 0.0″, the positions back-trace to 1.7e-14″, and
the model has 13 lens galaxies. The truth has 5 `point_0` images and 2 `point_1` images.
Image 3 is (10.36, 8.149) with μ −0.89, a demagnified central image 0.39″ from `main_1`'s
core (10, 8).

**Run.** Laptop, CPU fp64, 17 judged draws (truth + 8 prior + 8 stress), wall 810 s, peak
RSS 1.8 GB. The row set is in the [table above](#439-row-sets-starting-scale-at-production-extent-laptop-diagnostic);
the JSON is
[`robust_settings_sweep_viz_cluster_laptop_cpu_fp64.json`](../../../../results/misc/likelihood_breakdown/cluster/robust_settings_sweep_viz_cluster_laptop_cpu_fp64.json).
The `simple` close-pair draws 53/73/81 were re-run at 8 scales
([`…_simple_draws53_73_81_laptop_cpu_fp64.json`](../../../../results/cluster/image_plane/simple/jax_cpu_na_fp64/local_laptop/experiments/robust_settings_sweep_draws53_73_81/breakdown.json),
`--draw-ids`, 448 s).

**Result.**

1. **Reproduced at 1.0″, refuted at ~0.37″.** The workspace grid loses truth image 3. So do
   0.7, 0.5, 0.4, 0.37 and 0.3″. Only 0.2″ finds it. The coarsest scale that finds every
   truth image is therefore 0.2″: 567 843 step-0 triangles, 60.5 ms laptop warmed median
   against 7.4 ms at 1.0″ (diagnostic). The viz producer's own 400×400 @ 0.2″ grid (369 937
   triangles) is consistent with this; the ~0.37″ figure was probably measured on that
   producer's ±40″ extent, where the lattice phase differs.
2. **The same image fails on every prior draw.** In each of the 9 prior draws, `point_0`'s
   central image near `main_1` (μ −0.4 to −1.1, 0.17–0.49″ from the core) is missed at 1.0″.
   0.2″ still misses it on 3 draws (cores at 0.17, 0.22 and 0.33″).
3. **Spurious candidates at singular cores.** At 1.0, 0.7, 0.5 and 0.3″ the solver returns a
   candidate exactly on a scaling member's centre: (15.0, 9.0008) on stress draw 9, and
   (5.5009, −6.4999) on draw 16 at 0.7″. Its lens-equation residual is 0.04″, so it is not
   an image. With `r_core = 0` the deflection is discontinuous at the core, and the
   containment test passes there.
4. **Finer is not monotonically safer.** Six draw-sources are found at a coarser scale and
   lost at a finer one. Examples: `simple` 81's radial pair (0.33″) is found at 0.4″ and lost
   at 0.37″; viz prior draw 3 is found at 0.4″ and lost at 0.37/0.3″. Which images survive
   depends on where the step-0 lattice falls relative to the image, not only on the
   triangle size. The A2744 pilot shows the same thing at row level: 1.0″ misses 2, 0.5″
   misses 3.

**Separation rule** ([`separation_rule.md`](../../../../results/misc/likelihood_breakdown/cluster/separation_rule.md),
[`.json`](../../../../results/misc/likelihood_breakdown/cluster/separation_rule.json),
[`.png`](../../../../results/misc/likelihood_breakdown/cluster/separation_rule.png);
`scripts/cluster/image_plane/separation_rule.py`). The rule is tested per root and per
draw-source over 236 draw-sources: the viz run, all 101 phase-1 `simple` draws and the
53/73/81 re-run, merged by draw id with no conflict. 80 (root, scale) misses were observed.
For a rule "safe when scale ≤ separation / c", a miss at scale s rules out every c ≤ sep / s,
so the table gives the smallest c that no miss rules out:

| Separation | c no miss rules out | Binding miss |
|---|---|---|
| nearest image (per root) | 13.9 | viz prior 1, central image 2.79″ from any other image, missed at 0.2″ |
| nearest member core (per root) | 11.1 | `simple` 81 radial pair, 4.1″ from any core, missed at 0.37″ |
| min(image, core) (per root) | 1.66 | viz prior 5, 0.33″ from `main_1`, missed at 0.2″ |

- The claimed "scale ≲ separation / 1.06" is **contradicted**.
- Neither separation alone predicts the safe scale: the central images are far from other
  images, and the radial pair is far from any core.
- The smaller of the two separations, with c ≈ 1.7, is not contradicted. It rests on only
  two synthetic models, a coarse grid of tested scales and lattice-phase luck (point 4).
  So it is **not stated as a rule**.
- A2744 cannot enter the rule: the pilot wrote no per-root record. Its rows appear only in
  the PNG's right panel, at the four phase-1 scales.

**Recommendation (for the human; no default changed).**

- **Do not adopt a fixed finer grid.** 0.2″ costs ~8× per solve (laptop) and still misses
  3/108 viz roots. 0.5″ does not even find the claim's image.
- **Do not adopt a separation-based default.** The data do not support one.
- **Add a validation check** to the workspace cluster example's best-fit validation and
  visualization. The solver is not in the likelihood: `FitPositionsSourceSolved` traces the
  observed images. So the check can be expensive. Every observed image needs a solved image
  within a few σ. If one is missing, re-solve on a small fine grid (for example ±1″ at 0.05″)
  around it, and say so. Also drop any solved candidate whose source-plane residual exceeds
  the solver tolerance, which removes the singular-core spurious points.
- Both changes are drafted as Mind prompts (they are not filed here):
  `draft/feature/autolens_workspace/cluster_point_solver_validation_check.md` and
  `draft/feature/autolens/point_solver_spurious_core_candidates.md`.

## Follow-ups

1. ~~**A2744 pull + RSS attribution.**~~ **Closed 2026-10-23 (#439):** pulled, attributed to
   retained executables plus compile transients, harness fixed (`--release-executables`); see
   [A2744](#a2744-pilot-leg-job-412573-pulled-2026-10-23). A full A2744 sweep still needs a
   reference tracer and explicit human authorization.
2. ~~**Starting-grid question.**~~ **Answered 2026-10-23 (#439):** see
   [Starting grid](#starting-grid-vs-close-member-image-439). The two Mind drafts above carry the
   follow-on work. The human's deferred decision can now be taken with both inputs in hand.
3. ~~**The human's settings decision.**~~ **Closed 2026-10-23:** keep 1.0″ + validation check
   (see [Decision table](#decision-table-human)). The human's upcoming Pulse profiling programme
   may still revisit the cluster goal.
4. Phase 2 (workspace_test CI witnesses) distils only accepted claims, so nothing from this phase
   qualifies yet.
5. **Next:** the spurious-core candidates library task
   (`draft/feature/autolens/point_solver_spurious_core_candidates.md`, start_dev in progress), then
   the workspace validation check (`draft/feature/autolens_workspace/cluster_point_solver_validation_check.md`).

## Reproduce

```bash
# on RAL (explicit human authorization per job; never the gpu partition)
sbatch hpc/batch_cpu/submit_breakdown_cluster_robust_settings_sweep_ral_cpu_fp64
# laptop, local renders only
python scripts/cluster/image_plane/oracle_figure.py results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/robust_settings_sweep/breakdown.json --draw 53 73 81
python scripts/cluster/image_plane/solver_figure.py results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/robust_settings_sweep/breakdown.json --draw 53 73 81
# #439, laptop (CPU fp64; ~14 + ~8 min)
python scripts/cluster/image_plane/robust_settings_sweep.py --dataset viz_cluster --n-draws 8 --n-stress 8 \
  --scales 1.0,0.7,0.5,0.4,0.37,0.3,0.2 \
  --configs production_s0.7_p0.001,production_s0.5_p0.001,production_s0.4_p0.001,production_s0.37_p0.001,production_s0.3_p0.001,production_s0.2_p0.001 \
  --rounds 5 --calls 5 --warm 2 --n-stream 4 --release-executables \
  --config-name laptop_cpu_fp64 --output-dir results/breakdown/cluster
python scripts/cluster/image_plane/robust_settings_sweep.py --dataset simple --draw-ids 53,73,81 \
  --scales 2.0,1.0,0.7,0.5,0.4,0.37,0.3,0.2 \
  --configs production_s2_p0.001,production_s0.7_p0.001,production_s0.5_p0.001,production_s0.4_p0.001,production_s0.37_p0.001,production_s0.3_p0.001,production_s0.2_p0.001 \
  --rounds 5 --calls 5 --warm 2 --n-stream 4 --release-executables \
  --config-name draws53_73_81_laptop_cpu_fp64 --output-dir results/breakdown/cluster
python scripts/cluster/image_plane/separation_rule.py \
  results/misc/likelihood_breakdown/cluster/robust_settings_sweep_viz_cluster_laptop_cpu_fp64.json \
  results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/robust_settings_sweep/breakdown.json \
  results/cluster/image_plane/simple/jax_cpu_na_fp64/local_laptop/experiments/robust_settings_sweep_draws53_73_81/breakdown.json \
  --a2744-log results/misc/logs/cluster_pointsolver/cluster_pointsolver_2026_10_18_ral_job_412573_a2744_pilot.out
```

## Appendix: Pulse Knowledge draft for `tasks/cluster_pointsolver_speed.md`

Paste after this PR merges, in the follow-up Pulse PR. That PR also updates `campaigns.yaml`
next/updated, the check-in section and the board.

> ## Knowledge (2026-10-22): phase 1, cluster-scale robust PointSolver settings
>
> **Sources.** [autolens_profiling#436](https://github.com/PyAutoLabs/autolens_profiling/issues/436);
> ledger `wiki/campaigns/cluster/image_plane/cluster_pointsolver_campaign.md`; sweep JSON
> `results/cluster/image_plane/simple/jax_cpu_na_fp64/hpc_ral_cpu/experiments/robust_settings_sweep/breakdown.json` (RAL job 412572,
> Xeon 8490H, CPU fp64, profiling `b39f7ab2`, PyAutoLens `c90eceb0`).
>
> **Ground truth.** The oracle is independent of the triangle solver. It samples the multi-plane
> lens-equation residual densely, Newton-polishes each basin and certifies roots below 1e-10″
> residual and 1e-6″ stability. It is calibrated to 2.84e-8″ on the 1a closed-form quad and the
> 1c/1d near-cusp fixture.
>
> **Result (`simple`, 13 components, 3 planes, 2 sources, 101 draws).** No setting is admissible.
> Production-extent rows cover 306/306 prior roots and miss 2–6 of 300 stress roots. Every miss
> is a close high-|μ| pair, 0.33–0.76″ apart, on stress draws 53, 73 and 81 (one pair straddles
> the radial critical curve). Initial scale 0.5″/0.7″ misses 2 at 0.78×/0.75× the control's
> speed. The production 1.0″ misses 5 (5.29 ms per warmed two-source solve) and 2.0″ misses 6
> (1.30×). Precision 0.01 vs 0.001, `MAX_CONTAINING_SIZE` 20/40/60 and backend
> (NumPy/JIT/vmap) do not change coverage. The uncapped containing count reaches at most 7, so
> there is no overflow at cap 20. Eager JAX is unusable (projected row wall over cap). Truth-bbox
> extents miss 49–63 stress roots because the images move with the model. Peak RSS 4.3 GB.
>
> **Verdict.** DEFERRED (human, 2026-10-22). The 2-miss, ~0.77×-speed 0.5″/0.7″ start vs the
> 5-miss 1.0″ production control is re-decided after the starting-grid draft and the A2744 pull.
> No default was changed; 0.5″ is the working answer if the choice is forced earlier.
>
> **Follow-up (#439, 2026-10-23).** A2744 pilot job 412573 pulled: FAILED after 16 diagnostic
> rows on LLVM `Cannot allocate memory` (RSS 8.3 → 39.4 GB, live arrays ~0: retained executables
> + compile transients; harness fix `--release-executables`). On that synthetic 3-draw A2744 model
> 1.0″ misses fewest (2/69), the opposite of `simple`. On the autolens_visualization cluster model
> the workspace 1.0″ grid loses `point_0`'s central image 0.39″ from `main_1`'s core, and so does
> every scale down to 0.3″; only 0.2″ finds it, and 0.2″ still misses 3/108 roots. Finer is not
> monotonically safer, and no single-separation rule holds ("scale ≲ sep/1.06" contradicted).
> Recommendation: keep 1.0″ and add a validation check (observed image without a solved
> counterpart → local fine re-solve; drop high-residual candidates). Laptop rows, diagnostic timings.
>
> **Open.** The human's deferred decision (both inputs now in hand); the two Mind drafts
> (workspace validation check; spurious candidates at `r_core = 0` member cores). The human's
> Pulse profiling programme may revisit this goal.
