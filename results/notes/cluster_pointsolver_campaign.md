# Cluster PointSolver campaign ledger

**Campaign:** `cluster-pointsolver` (epic `cluster-pointsolver-speed`). Pulse task
[cluster_pointsolver_speed](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/cluster_pointsolver_speed.md).
Wiki page: [cluster_pointsolver.md](../../wiki/campaigns/cluster_pointsolver.md).
**Phase 1:** cluster-scale robust `PointSolver` settings,
[autolens_profiling#436](https://github.com/PyAutoLabs/autolens_profiling/issues/436).
Mind prompt `active/cluster_pointsolver_robust_settings_p1.md`.
**Status:** DECIDED 2026-10-22: DEFERRED (default stays 1.0″ until the starting-grid draft and the A2744 pull; 0.5″ is the working answer if forced). The `simple` sweep is complete and quotable.
The A2744 pilot leg has not been pulled (see [A2744](#a2744-pilot-leg-not-pulled)). No default
was changed anywhere.

## TL;DR

**Decision (human, 2026-10-22): deferred** — default stays 1.0″ until the starting-grid draft and the A2744 pull; see [Decision table](#decision-table-human).


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

## Tested domain

| Item | Value |
|---|---|
| Dataset | `dataset/cluster/simple`: 2 dPIE + 10 scaling dPIE + NFW = 13 mass components; planes z = 0.5 / 1.0 / 2.0; sources `point_0` (z 1.0) and `point_1` (z 2.0), 3 images each in the truth; noise 0.005″. Regenerated on RAL by `scripts/misc/simulators/cluster.py` (simulator sha256 `34dce2637d11…`, seed 1); truth gate passes (traced difference 0.0″). |
| Draws judged | 101 = truth + 50 prior draws (seed 436, the cell's fused model priors) + 50 stress draws (seed 2436; centres ±0.5″ σ, `mass_at_200` 0.3 dex, scaling `b0_ref` 15 %, other mass parameters 10 %, no clipping). Stress betas are always back-traced from the draw. |
| Judged roots | prior 306 (102 draw-sources × 3), stress 300 (100 draw-sources; 1–7 roots each above \|μ\| 0.1) |
| Production solver | `autolens_workspace/scripts/cluster/modeling.py` shape as built for this dataset: 100×100 @ 1.0″, `pixel_scale_precision` 0.001, `magnification_threshold` 0.1, `MAX_CONTAINING_SIZE` 20 |
| Not tested | A2744 (pilot leg not pulled), A100, more than 2 sources, source redshifts above 2, fp32, gradients |
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

The source is `results/breakdown/cluster/robust_settings_sweep_simple_hpc_ral_cpu_fp64.json`
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

All figures are under [`results/breakdown/cluster/figures/`](../breakdown/cluster/figures/).
They are local renders on the laptop, not timings.

- **Oracle verification**
  ([`oracle_figure.py`](../../scripts/cluster/image_plane/oracle_figure.py)). For each plane, the
  figure shows the oracle roots, the critical curves and caustics, one panel per row (control,
  `production_s0.5_p0.001`, `bbox_s2_p0.001`) with the verdict of each root, and zooms on the
  close groups:
  [53/plane 1](../breakdown/cluster/figures/oracle_figure_simple_hpc_ral_cpu_fp64_draw53_plane1.png),
  [53/plane 2](../breakdown/cluster/figures/oracle_figure_simple_hpc_ral_cpu_fp64_draw53_plane2.png),
  [73/plane 1](../breakdown/cluster/figures/oracle_figure_simple_hpc_ral_cpu_fp64_draw73_plane1.png),
  [73/plane 2](../breakdown/cluster/figures/oracle_figure_simple_hpc_ral_cpu_fp64_draw73_plane2.png),
  [81/plane 1](../breakdown/cluster/figures/oracle_figure_simple_hpc_ral_cpu_fp64_draw81_plane1.png),
  [81/plane 2](../breakdown/cluster/figures/oracle_figure_simple_hpc_ral_cpu_fp64_draw81_plane2.png).
- **The library's own `point_solver.png`**
  ([`solver_figure.py`](../../scripts/cluster/image_plane/solver_figure.py)). This is the readable
  cluster visualizer of PyAutoLens#795 (`b1754a7ec`; local canonical checkout at `b273f9c7d`). The
  figure re-solves the row's NumPy steps on the draw's tracer and draws the oracle roots as the
  hollow "observed" circles, so a missed root is a circle with no solved cross:
  - control:
    [53](../breakdown/cluster/figures/solver_figure_simple_hpc_ral_cpu_fp64_control_draw53_source0.png),
    [73](../breakdown/cluster/figures/solver_figure_simple_hpc_ral_cpu_fp64_control_draw73_source0.png),
    [81](../breakdown/cluster/figures/solver_figure_simple_hpc_ral_cpu_fp64_control_draw81_source1.png)
  - initial scale 0.5″:
    [53](../breakdown/cluster/figures/solver_figure_simple_hpc_ral_cpu_fp64_production_s0.5_p0.001_draw53_source0.png),
    [73](../breakdown/cluster/figures/solver_figure_simple_hpc_ral_cpu_fp64_production_s0.5_p0.001_draw73_source0.png),
    [81](../breakdown/cluster/figures/solver_figure_simple_hpc_ral_cpu_fp64_production_s0.5_p0.001_draw81_source1.png)

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

## A2744 pilot leg (not pulled)

**A2744 pilot leg (job 412573): outcome and memory trace not yet pulled.** RAL login was
unavailable on 2026-10-22. The row-1 live read on 2026-10-18 showed peak 22 GB, RSS 8.3 GB and
live arrays ~0, which points to a per-row transient plus retained executables. Two more caveats:
- The A2744 reference tracer does not exist yet (no Lenstool best fit or workspace max-likelihood
  result), so that leg used the `prior-median-smoke` synthetic model and its rows are diagnostic
  only.
- On the 2026-10-18 pilot (job 412567) the A2744 production rows missed 3–5 of 69 roots at about
  410–600 s per row.

The `simple` sweep's RSS rises across rows in the same way: 1.5 GB → 4.3 GB over 24 rows, while
live arrays stay at 2–103 (tens of KB). The retained compiled executables explain this. They are kept on
purpose so the timing rounds can interleave.

## Follow-ups

1. **A2744 pull + RSS attribution.** Pull the job 412573 JSON, the `.out` and the memory trace once
   RAL login works. Then attribute the per-row memory growth (transient against retained
   executables). No resubmission is planned. Any new A2744 job needs explicit human authorization.
2. **Starting-grid question** (filed by the human): Mind draft
   `draft/research/autolens_profiling/cluster_pointsolver_starting_grid_close_member_image.md`.
   A 1.0″ starting grid may lose image 3 of `point_0`, which sits 0.39″ from a member galaxy's
   core. It is related to the close-pair misses here, and it is not done in this phase.
3. **The human's upcoming Pulse profiling programme** may revisit the cluster goal and this
   decision table. Until then, the settings decision above stays open.
4. Phase 2 (workspace_test CI witnesses) distils only accepted claims, so nothing from this phase
   qualifies yet.

## Reproduce

```bash
# on RAL (explicit human authorization per job; never the gpu partition)
sbatch hpc/batch_cpu/submit_breakdown_cluster_robust_settings_sweep_ral_cpu_fp64
# laptop, local renders only
python scripts/cluster/image_plane/oracle_figure.py results/breakdown/cluster/robust_settings_sweep_simple_hpc_ral_cpu_fp64.json --draw 53 73 81
python scripts/cluster/image_plane/solver_figure.py results/breakdown/cluster/robust_settings_sweep_simple_hpc_ral_cpu_fp64.json --draw 53 73 81
```

## Appendix: Pulse Knowledge draft for `tasks/cluster_pointsolver_speed.md`

Paste after this PR merges, in the follow-up Pulse PR. That PR also updates `campaigns.yaml`
next/updated, the check-in section and the board.

> ## Knowledge (2026-10-22): phase 1, cluster-scale robust PointSolver settings
>
> **Sources.** [autolens_profiling#436](https://github.com/PyAutoLabs/autolens_profiling/issues/436);
> ledger `results/notes/cluster_pointsolver_campaign.md`; sweep JSON
> `results/breakdown/cluster/robust_settings_sweep_simple_hpc_ral_cpu_fp64.json` (RAL job 412572,
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
> **Open.** The A2744 pilot leg (job 412573) has not been pulled. RAL login was unavailable on
> 2026-10-22. Row 1 read on 2026-10-18: peak 22 GB, RSS 8.3 GB, live arrays ~0. The pull and RSS
> attribution are owed. The starting-grid / close-member-image question is in its own Mind draft
> (`cluster_pointsolver_starting_grid_close_member_image.md`). The human's Pulse profiling
> programme may revisit this goal.
