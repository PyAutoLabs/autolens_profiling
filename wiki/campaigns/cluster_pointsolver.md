# Cluster PointSolver speed-up

**Status:** open
**Question:** What does a representative cluster-scale `PointSolver` likelihood cost on released code, where is the time, and which levers (deflection cost, grid extent, per-source batching) pay at cluster scale?
**Pre-registered rule:** phase 1: a setting is admissible only when every oracle root (independent lens-equation oracle, not the triangle solver) of every judged prior AND stress draw is covered by exactly one candidate within the row's last-step triangle side, multiplicity agrees, no spurious candidate and no uncapped containing count above the cap. Otherwise the contract inherits the point-source campaign measurement contract (exact revisions, interleaved A/B on identical hardware, fused production control, correctness gate before any speed claim, negative results recorded, one bounded phase per issue and PR).
**Verdict:** phase 1 DEFERRED (human, 2026-10-22; default stays 1.0″): no `PointSolver` setting is admissible on `simple` (101 draws). Production rows cover 306/306 prior roots and miss 2–6 of 300 stress roots, all close high-|μ| pairs. #439 (2026-10-23) supplies both pending inputs: no fixed starting scale finds every image (viz cluster model: 0.2″ still misses 3/108; finer is not monotone; A2744 pilot 1.0″ best), no single-separation rule holds; recommendation = keep 1.0″ + a validation check. No default changed.
**Headline:** initial scale 0.5″/0.7″ misses 2 stress roots at 0.78×/0.75× the production control's speed; the control (1.0″) misses 5 at 5.29 ms; RAL Xeon 8490H job 412572. Carried evidence: two-source cluster solved likelihood 47.36 → 9.085 ms (5.21x) from the static step-0 lattice, RAL Xeon 8490H job 350636.
**Library PRs:** none yet.
**Profiling PRs:** phase 1 #438 (autolens_profiling#436), merged; #439 starting-grid follow-up PR, open.
**Ledger:** [cluster_pointsolver_campaign.md](../../results/notes/cluster_pointsolver_campaign.md); carried cluster rows live in [point_source_cpu_campaign.md](../../results/notes/point_source_cpu_campaign.md) (phases 1–3).
**Mind contract:** epic `cluster-pointsolver-speed`; PyAutoPulse task `tasks/cluster_pointsolver_speed.md` (migrated from `draft/research/autolens_profiling/cluster_pointsolver_speed.md`); phase 1 `active/cluster_pointsolver_robust_settings_p1.md`.
**Next:** the human's decision on the phase-1 table with the #439 inputs; the Mind drafts `draft/feature/autolens_workspace/cluster_point_solver_validation_check.md` and `draft/feature/autolens/point_solver_spurious_core_candidates.md`; any A2744 full sweep needs a reference tracer, `--release-executables` and explicit human authorization.

## Why this campaign

Split out of the point-source CPU campaign on 2026-09-26 by human decision: "we will do cluster use
case as a separate epic … starting from working out the data and likelihood_breakdown." Phases 1–3
of the image-plane campaign had carried two-source cluster rows alongside the single-source cell;
from phase 4a on, that campaign is single-source only and everything cluster-scale lives here.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 robust settings | 2026-10-17/22 | Which `PointSolver` settings find every image at cluster scale, judged against an independent oracle? | per-root coverage + multiplicity + overflow on prior AND stress draws (above) | DEFERRED (human 2026-10-22, default stays 1.0″): no row admissible on `simple`; 0.5″/0.7″ miss 2, 1.0″ (production) 5, 2.0″ 6 of 300 stress roots; precision/capacity/backend change nothing; eager unusable. A2744 pilot (pulled #439): FAILED after 16 diagnostic rows (LLVM memory, retained executables), 1.0″ misses fewest (2/69). #439 starting grid (laptop, viz cluster model): workspace 1.0″ loses `point_0`'s central image 0.39″ from a core, so does every scale to 0.3″, only 0.2″ finds it; no fixed scale or single-separation rule is safe → validation check recommended | 412567 (pilot), 412572 (sweep), 412573 (A2744 pilot leg, FAILED, pulled 2026-10-23) | #438 (#436); #439 PR |
| 2 deflection cost | — | dPIE/NFW deflection cost (JAX deflection cell first) | — | sketch only | — | — |
| 3 grid extent | — | Cluster grid-extent guidance with completeness evidence | — | sketch only | — | — |
| 4 | — | Whatever the phase-1 split exposes | — | sketch only | — | — |

## What shipped and where it is

Nothing yet. The image-plane levers that also sped up the cluster cell (PyAutoArray#569, #570,
PyAutoLens#749, released 2026.9.26.1) are recorded on the
[image-plane page](point_source_image_plane_cpu.md).

## Carried evidence (from image-plane phases 1–3)

- IP-1 RAL baseline (Xeon 8490H): cluster fused solved 134.52 ms.
- IP-2 vertex dedup (job 350582, EPYC 7763): 155.22 → 78.32 ms (1.98x); the control is +15–20 % from the host alone, so pin the node.
- IP-3 static lattice (job 350636, 8490H pinned): 47.36 → 9.085 ms (5.21x).
- The cluster step-0 lattice now deflects 46 516 of 276 507 unique vertices; step 0 ≈ 51 % of cluster FLOPs (FLOP estimate, not measured); deflections of the 13-component lens dominate.
- `scripts/lens/deflections/` is NumPy-only with no dPIE spec — a JAX dPIE/NFW deflection cell is a prerequisite for the deflection lever.

## Open / parked / drafts

- The campaign contract is the PyAutoPulse task `tasks/cluster_pointsolver_speed.md`.
- A2744 full sweep: parked (no reference tracer; needs `--release-executables` and explicit human authorization).
- Mind drafts from #439 (not filed): workspace cluster validation check; PyAutoLens spurious candidates at `r_core = 0` member cores.

## Caveats

- The current cluster profiling case (`dataset/cluster/simple`, auto-simulated: 13 mass components, 3 planes, 2 sources) is a starting point, not a verdict.
- The existing cell runs `pixel_scale_precision=0.01`, not production `0.001`; phase 1 must measure both or justify the choice.
- The old "cluster solve is at its deflection bound" claim was extrapolated, not measured.

## Journal

### 2026-10-23 — #439: starting grid vs close member image; A2744 pilot pulled

The A2744 pilot (RAL job 412573) is pulled: FAILED after 16 diagnostic rows, LLVM `Cannot
allocate memory`, RSS 8.3 → 39.4 GB with live arrays ~0. The cause is retained compiled
executables plus compile transients, and the harness now has `--release-executables`. On that
synthetic 3-draw model 1.0″ misses fewest (2/69), the opposite of `simple`.

The autolens_visualization cluster model (copied constants, `--dataset viz_cluster`) was swept
on the laptop: CPU fp64, 17 draws, scales 1.0–0.2″, production extent. The workspace 1.0″ grid
does lose `point_0`'s image 3 (μ −0.89, 0.39″ from `main_1`'s core). So does every scale down
to 0.3″, including the claimed 0.37″ threshold; only 0.2″ finds it. 0.2″ still misses 3/108
roots, all central images within 0.17–0.33″ of a core. Six draw-sources are found at a coarser
scale and lost at a finer one. The solver also returns spurious candidates on `r_core = 0`
member centres. "Scale ≲ separation / 1.06" is contradicted, and no single separation predicts
the safe scale. The recommendation is to keep 1.0″ and add a validation check; two Mind drafts
carry it. Laptop timings are diagnostic only. See the
[ledger](../../results/notes/cluster_pointsolver_campaign.md#starting-grid-vs-close-member-image-439).

### 2026-10-22 — phase 1: no admissible setting on `simple` (interpretation pending)

The sweep ran as RAL job 412572: Xeon 8490H, CPU fp64, 25 rows, 101 draws, wall 1358 s. Every
row was judged against an independent lens-equation oracle, calibrated to 2.84e-8″. No row is
admissible. The production-extent rows cover all 306 prior roots and miss 2–6 of 300 stress
roots. Every miss is one of three close high-|μ| pairs (0.33–0.76″; draw 81's pair straddles the
radial critical curve). An initial scale of 0.5″ or 0.7″ misses 2 roots at about 0.77× the
control's speed. The production 1.0″ misses 5 and 2.0″ misses 6. Precision, capacity and backend
change nothing (uncapped max 7 < 20). Eager JAX is unusable (row cap). Image-bounding-box extents
lose 49–63 roots. The library's readable cluster `point_solver.png` (PyAutoLens#795) shows the
misses directly. The decision table is in the
[ledger](../../results/notes/cluster_pointsolver_campaign.md#decision-table-human) and is left to
the human. No default was changed. The A2744 pilot leg (job 412573) was not pulled because RAL
login was unavailable.

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.
