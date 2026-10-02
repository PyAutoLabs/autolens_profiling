# Point-source image-plane CPU speed-up

**Status:** open
**Question:** Where does the single-source image-plane `PointSolver` likelihood (JAX, CPU fp64) spend its time, and which changes cut it without changing the answer?
**Pre-registered rule:** per phase (see the table); the common contract is repeated, interleaved in-process A/B medians with bootstrap CIs on a quiet RAL CPU node, a gain above 2× the minimum detectable improvement, bit-identical correctness gates, compile ≤ +20 % and no A100 regression.
**Verdict:** IP-2 (vertex dedup), IP-3 (static lattice) and IP-4b (step-0 gather) accepted and shipped; IP-4a found the extent/scale lever (2.37x) but the human ruled it a per-workspace setting, not a library default; IP-4c raised `MAX_CONTAINING_SIZE` 15 → 20 accepting a +6.2 % scalar slowdown for correctness headroom (human decision).
**Headline:** IP-4b 1.44x per call (2.27x under vmap-16), RAL Xeon 8490H job 357321 — node loadavg ~200, ratios stand, absolute ms do not; quiet EPYC 7702 supplementary row 1.61x (job 357335).
**Library PRs:** PyAutoArray#569, PyAutoArray#570, PyAutoLens#749 (released 2026.9.26.1); PyAutoArray#580, PyAutoArray#584, PyAutoLens#753 (released 2026.9.27.2).
**Profiling PRs:** #293, #298, #301, #305, #318 (folder split), #321, #330, #335.
**Ledger:** [point_source_cpu_campaign.md](../../results/notes/point_source_cpu_campaign.md); instrument note [point_source_shared_likelihood_breakdown.md](../../results/notes/point_source_shared_likelihood_breakdown.md).
**Mind contract:** epic `point-source-cpu-speed`; campaign contract in the `## Original prompt` of `complete/2026/09/point-source-cpu-p4.md`; phase records `complete/2026/09/point-source-cpu-p{1,2,3,4}.md`, `pointsolver-step0-gather.md`, `pointsolver-mcs-headroom.md`.
**Next:** the extent/scale lever via `draft/feature/autolens/pointsolver_extent_sanity_check.md` and `draft/feature/autolens_workspace/pointsolver_grid_extent_per_package.md`; carried leftovers in `draft/research/autolens_profiling/pointsolver_cpu_speed_campaign_remainder.md`.

## Why this campaign

The campaign was filed on 2026-09-17 as the CPU half of a point-source performance plan: a shared
likelihood breakdown first (so the CPU and GPU campaigns measure on one instrument rather than two
copies), then "redundant-sort removal and measured iteration" on the JAX `PointSolver`. A
2026-09-17 research note (recovered into
[`point_source_cpu_2026_09_17_reported/`](../../results/notes/point_source_cpu_2026_09_17_reported/README.md))
had located a throwaway `jnp.unique` vertex dedup and a step-0 lattice that was rebuilt every call.
Each phase was issued one at a time, and each re-filed the remainder pointing back at its record.
On 2026-09-26 the human narrowed the campaign to **single-source only**; the two-source cluster rows
carried through phases 1–3 moved to [Cluster PointSolver](cluster_pointsolver.md).

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| Shared breakdown | 2026-09-19 | One instrument for CPU + GPU: cumulative prefixes of the eight solver iterations against the fused likelihood | instrument only; fused likelihood is the authoritative control | laptop CPU fp64 baseline committed | laptop | #293 |
| IP-1 RAL baseline | 2026-09-23 | Quotable unoptimized cost on RAL CPU; freeze the revisions the GPU campaign reproduces | not recorded (baseline) | simple fused solved 24.69 ms | 350580 | #298 |
| IP-2 vertex dedup | 2026-09-23/24 | Does removing the throwaway `jnp.unique` dedup pay? | interleaved A/B above the minimum detectable improvement; gates bit-identical; no compile regression | ACCEPT: 4.47x simple / 1.98x cluster per call, RAL EPYC 7763 | 350582, 350587 (A100) | PyAutoArray#569, #301 |
| IP-3 static lattice | 2026-09-24 | Does precomputing the static step-0 lattice pay? | stop rule (PyAutoArray#568): reject if simple gain < max(5 %, 2× MDI) with no cluster gain, any gate fails, compile +20 %, or memory/GPU regression | ACCEPT: 2.0–2.1x simple, 5.2x cluster on 8490H; A100 only 1.01–1.07x; tie case passed by human decision | 350636, 350637 (A100) | PyAutoArray#570, PyAutoLens#749, #305 |
| IP-4a re-baseline + sweep | 2026-09-26 | Where does the released 2026.9.26.1 call spend time; which solver configs are admissible and faster? | every config gated on image completeness before timing | step 0 = 66 % of the call; extent/scale ±2.5″/0.4 = 2.37x scalar [2.28, 2.59], 5.55x vmap-16 — NOT shipped (per-workspace setting, human decision); latent MCS overflow 17 > 15 found | 356365, 356367 (356366 superseded) | #321 |
| IP-4b step-0 gather | 2026-09-27 | Can step-0 containment avoid the `(N,3,2)` triangle gather? | bit-identical output on every route; compile ≤ +20 %; no A100 regression | ACCEPT: 1.44x single / 2.27x vmap-16 (8490H, loaded); 1.61x quiet EPYC; containment −60–66 % | 357321, 357322 (A100), 357335 (EPYC) | PyAutoArray#580, #330 |
| IP-4c MCS headroom | 2026-09-27 | Raise `MAX_CONTAINING_SIZE` from 15 to about 20, measured | smallest N ≥ 18 with uncapped max ≤ N − 3, compile ≤ +20 %, ≤ ~+5 % scalar median | rule_candidates empty (mcs20 +6.2 % on cost leg); human chose N = 20 for correctness | 358976, 359102 (A100) | PyAutoArray#584, PyAutoLens#753, #335 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#569 | remove throwaway vertex dedup | `681938ae` | 2026.9.26.1 |
| PyAutoArray#570 | static step-0 lattice precompute | `7fa8d271` | 2026.9.26.1 |
| PyAutoLens#749 | static lattice (PointSolver side) + tie test | `86054bbc` | 2026.9.26.1 |
| PyAutoArray#580 | step-0 structured containment (issue PyAutoArray#579) | `4383ea81` | 2026.9.27.2 |
| PyAutoArray#584 | `MAX_CONTAINING_SIZE` 15 → 20 | `9428eca2` | 2026.9.27.2 |
| PyAutoLens#753 | MCS 20 (PyAutoLens side) | `e92bde01` | 2026.9.27.2 |

Cumulative per call: IP-1 24.69 ms → IP-4a 2.095 ms (~11.8x) on the same cell but different
nodes — indicative only, not a quotable A/B.

## Open / parked / drafts

- `draft/feature/autolens/pointsolver_extent_sanity_check.md` — library construction-time warning on grid extent.
- `draft/feature/autolens_workspace/pointsolver_grid_extent_per_package.md` — per-package extent in the workspaces.
- `draft/research/autolens_profiling/pointsolver_cpu_speed_campaign_remainder.md` — carried phase 1–3 leftovers.
- `draft/research/autolens_profiling/point_source_image_plane_gpu_breakdown.md` — the GPU sibling ([page](point_source_gpu_breakdown.md)).
- No end-to-end search fit has ever been timed for the image-plane likelihood.

## Caveats

- **Laptop rows are not quotable.** IP-2's laptop 6.0x overstates the gain (loaded laptop); RAL's 4.47x / 1.98x are the quotable figures (ledger, phase 2 A/B section). IP-3's laptop witness is load-inflated and marked NOT quotable.
- **Loaded-node headline.** IP-4b's quotable 8490H row (job 357321) ran at loadavg 199.9 → 189.9 on 236 cores; interleaved route ratios stand, absolute ms and the gather spread are inflated and not comparable to IP-4a's 1.824 ms control. The quiet EPYC 7702 row is a different CPU: its ratios are valid, its absolute ms are not comparable to IP-4a.
- **Job 356366 is discarded**; 356367 supersedes it (same draws and protocol).
- **The 11.8x cumulative figure crosses nodes** and is indicative only.
- **IP-4c is a deliberate slowdown** (+6.2 % scalar on the 8490H) accepted for correctness headroom.

## Journal

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.

### 2026-10-02 — epic reconciliation and the existing inference phase

Image-plane phases through **IP-4c are merged**; no extent default was changed.
The next unissued image-plane member is the construction-time extent sanity check,
followed by per-package settings. The cluster campaign remains separate.

The epic already has one issued member: [autolens_inference#15](https://github.com/PyAutoLabs/autolens_inference/issues/15),
the approved single-source Nautilus source-plane search leaf. On resumption, RAL
array 367140 seeds 1–4 were all COMPLETED (0:0); together with probe 366937 they
recover all five truth parameters within 0.74σ across seeds 0–4. Search walls are
50.38–58.51 s for 4,700–4,850 evaluations. Warmed batch timing estimates
0.0405–0.0447% of search wall in steady likelihood evaluation (4.59–5.01 µs/eval).
This is source-plane evidence, **not an image-plane PointSolver end-to-end timing**.
The inference branch is being prepared for review; no second phase was issued.
See the [inference journal](https://github.com/PyAutoLabs/autolens_inference/blob/feature/point-source-search-nautilus-leaf/wiki/project/state.md#2026-10-02--point-source-nautilus-admission-bar-five-seeds-recovered)
for all rows and limitations. The estimate uses one prior-median vector and fixed
batch size, not an instrumented fit decomposition; it does not establish the
wall-share of a gradient sampler. blackjax forward mode still needs its own
admission measurement. The A100 throughput row already landed on 2026-09-28.

Release reconciliation: local release tags contain PyAutoArray `9428eca2` and PyAutoLens `e92bde01` in **2026.9.27.2**, including the earlier step-0 gather change.
