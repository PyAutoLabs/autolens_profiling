# Cluster PointSolver speed-up

**Status:** draft
**Question:** What does a representative cluster-scale `PointSolver` likelihood cost on released code, where is the time, and which levers (deflection cost, grid extent, per-source batching) pay at cluster scale?
**Pre-registered rule:** not yet written; the contract inherits the point-source campaign measurement contract (exact revisions, interleaved A/B on identical hardware, fused production control, correctness gate before any speed claim, negative results recorded, one bounded phase per issue and PR).
**Verdict:** not started.
**Headline:** none yet. Carried evidence: two-source cluster solved likelihood 47.36 → 9.085 ms (5.21x) from the static step-0 lattice, RAL Xeon 8490H job 350636.
**Library PRs:** none yet.
**Profiling PRs:** none yet.
**Ledger:** none yet; carried cluster rows live in [point_source_cpu_campaign.md](../../results/notes/point_source_cpu_campaign.md) (phases 1–3).
**Mind contract:** epic `cluster-pointsolver-speed`; `draft/research/autolens_profiling/cluster_pointsolver_speed.md`.
**Next:** phase 1 — work out representative cluster data and bring `scripts/cluster/likelihood_breakdown/` to a released-code baseline (pinned 8490H + A100 row) before any lever is ranked.

## Why this campaign

Split out of the point-source CPU campaign on 2026-09-26 by human decision: "we will do cluster use
case as a separate epic … starting from working out the data and likelihood_breakdown." Phases 1–3
of the image-plane campaign had carried two-source cluster rows alongside the single-source cell;
from phase 4a on, that campaign is single-source only and everything cluster-scale lives here.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 data + baseline | — | Which cluster datasets are representative, and what does the released-code breakdown say? | not yet written | not started | — | — |
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

- The whole campaign is the draft `draft/research/autolens_profiling/cluster_pointsolver_speed.md`.

## Caveats

- The current cluster profiling case (`dataset/cluster/simple`, auto-simulated: 13 mass components, 3 planes, 2 sources) is a starting point, not a verdict.
- The existing cell runs `pixel_scale_precision=0.01`, not production `0.001`; phase 1 must measure both or justify the choice.
- The old "cluster solve is at its deflection bound" claim was extrapolated, not measured.

## Journal

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.
