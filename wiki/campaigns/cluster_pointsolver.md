# Cluster PointSolver speed-up

**Status:** open
**Question:** What does a representative cluster-scale `PointSolver` likelihood cost on released code, where is the time, and which levers (deflection cost, grid extent, per-source batching) pay at cluster scale?
**Pre-registered rule:** phase 1: a setting is admissible only when every oracle root (independent lens-equation oracle, not the triangle solver) of every judged prior AND stress draw is covered by exactly one candidate within the row's last-step triangle side, multiplicity agrees, no spurious candidate and no uncapped containing count above the cap. Otherwise the contract inherits the point-source campaign measurement contract (exact revisions, interleaved A/B on identical hardware, fused production control, correctness gate before any speed claim, negative results recorded, one bounded phase per issue and PR).
**Verdict:** phase 1 DEFERRED (human, 2026-10-22; default stays 1.0″ pending the starting-grid draft and the A2744 pull): no `PointSolver` setting is admissible on `simple` (101 draws). Production rows cover 306/306 prior roots and miss 2–6 of 300 stress roots, all close high-|μ| pairs. No default changed.
**Headline:** initial scale 0.5″/0.7″ misses 2 stress roots at 0.78×/0.75× the production control's speed; the control (1.0″) misses 5 at 5.29 ms; RAL Xeon 8490H job 412572. Carried evidence: two-source cluster solved likelihood 47.36 → 9.085 ms (5.21x) from the static step-0 lattice, RAL Xeon 8490H job 350636.
**Library PRs:** none yet.
**Profiling PRs:** phase 1 PR (autolens_profiling#436), open.
**Ledger:** [cluster_pointsolver_campaign.md](../../results/notes/cluster_pointsolver_campaign.md); carried cluster rows live in [point_source_cpu_campaign.md](../../results/notes/point_source_cpu_campaign.md) (phases 1–3).
**Mind contract:** epic `cluster-pointsolver-speed`; PyAutoPulse task `tasks/cluster_pointsolver_speed.md` (migrated from `draft/research/autolens_profiling/cluster_pointsolver_speed.md`); phase 1 `active/cluster_pointsolver_robust_settings_p1.md`.
**Next:** the human's decision on the phase-1 table; pull the A2744 pilot leg (job 412573) and attribute its memory; the starting-grid draft `draft/research/autolens_profiling/cluster_pointsolver_starting_grid_close_member_image.md`.

## Why this campaign

Split out of the point-source CPU campaign on 2026-09-26 by human decision: "we will do cluster use
case as a separate epic … starting from working out the data and likelihood_breakdown." Phases 1–3
of the image-plane campaign had carried two-source cluster rows alongside the single-source cell;
from phase 4a on, that campaign is single-source only and everything cluster-scale lives here.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 robust settings | 2026-10-17/22 | Which `PointSolver` settings find every image at cluster scale, judged against an independent oracle? | per-root coverage + multiplicity + overflow on prior AND stress draws (above) | DEFERRED (human 2026-10-22, default stays 1.0″): no row admissible on `simple`; 0.5″/0.7″ miss 2, 1.0″ (production) 5, 2.0″ 6 of 300 stress roots; precision/capacity/backend change nothing; eager unusable; A2744 leg not pulled | 412567 (pilot), 412572 (sweep), 412573 (A2744 pilot leg, not pulled) | phase 1 PR (#436) |
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
- A2744 pilot leg (job 412573): outcome and memory trace not yet pulled — RAL login unavailable 2026-10-22. Row-1 live read on 2026-10-18: peak 22 GB / RSS 8.3 GB, live arrays ~0 → per-row transient + retained executables.
- `draft/research/autolens_profiling/cluster_pointsolver_starting_grid_close_member_image.md` — does a 1.0″ starting grid lose image 3 of `point_0`, 0.39″ from a member galaxy's core?

## Caveats

- The current cluster profiling case (`dataset/cluster/simple`, auto-simulated: 13 mass components, 3 planes, 2 sources) is a starting point, not a verdict.
- The existing cell runs `pixel_scale_precision=0.01`, not production `0.001`; phase 1 must measure both or justify the choice.
- The old "cluster solve is at its deflection bound" claim was extrapolated, not measured.

## Journal

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
