# Point-source A100 breakdown

**Status:** open
**Question:** Where does the image-plane point-source `PointSolver` likelihood spend its time on the A100 (fp64 and mixed precision, primal and gradient, single call and `vmap`), and which measured changes cut it without changing the answer?
**Pre-registered rule:** the 2026-09-19 campaign contract: per iteration, baseline → one hypothesis → bounded prototype → correctness gate → repeated interleaved full-likelihood A/B; keep a change only for a repeatable material gain above the stated minimum detectable improvement, with no correctness failure and no unacceptable compile or memory regression; stop when the remaining cost is explained and no worthwhile measured lever remains, or on a concrete external blocker.
**Verdict:** phase 0+1 (lean) measured; launch-bound confirmed; go/no-go pending the human (no image-plane fit timed yet)
**Headline:** scalar 0.910 ms (CUDA graphs on; 1.097 off), 173 kernels/call, device busy 57 % of wall, vmap-256 0.0120 ms/L (81x); A100 job 366916 (euclid-ral-gpu-1, exclusive); forward-mode gradient NaN
**Library PRs:** none
**Profiling PRs:** branch `feature/point-source-gpu-p01` (#350)
**Ledger:** [point_source_gpu_breakdown_2026_09.md](../../results/notes/point_source_gpu_breakdown_2026_09.md); instrument [point_source_shared_likelihood_breakdown.md](../../results/notes/point_source_shared_likelihood_breakdown.md)
**Mind contract:** `draft/research/autolens_profiling/point_source_image_plane_gpu_breakdown.md` (filed 2026-09-17, contract 2026-09-19); prerequisite `complete/2026/09/point-source-shared-breakdown.md` (#293).
**Next:** human go/no-go against the admission bar; prerequisite one autolens_inference image-plane fit measurement; route the forward-mode NaN through intake

## Why this campaign

On 2026-09-17 the human asked for the point-source image-plane chi-squared to get the same
treatment as imaging and interferometer, "mostly using the A100s on RAL". The survey that day
found a `likelihood_runtime/` tier but no point-source breakdown and no A100 row; the cluster
breakdown timed `solver.solve` as one block. On 2026-09-19 a consolidation split the work into
three tasks: a shared breakdown first, then a CPU campaign ([Point-source image-plane
CPU](point_source_image_plane_cpu.md)) and this GPU campaign, each issued one bounded phase at a time.

The shared instrument shipped as #293 (2026-09-20, `complete/2026/09/point-source-shared-breakdown.md`):
eight solver iterations opened into cumulative prefixes, with the fused production likelihood as the
authoritative control and a laptop CPU fp64 reference (solved fused 62.687 ms/call). The contract
says this campaign must first run that instrument on the A100 against the preserved unoptimized
library revisions, and never label CPU timing as A100 evidence. The September 17 hypotheses
(launch-latency bound, sort-based `unique` dominant, unrolled-step compile, implicit-diff gradient
cost) stay hypotheses until phase 1 measures them.

## Phases

From the draft's campaign contract. Phases 0 and 1 ran as one lean job (human, 2026-09-28).

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 0+1 combined (lean) | 2026-09-28 | Current-main A100 baseline (CUDA graphs on/off), vmap 1–256, trace-attributed bottleneck map, fp32 what-if, reverse/forward gradient with FD check | fiducial bit-exact; vmap lanes equal scalar; trace join complete; FD agreement at smooth points (strict) | launch-bound: 173 kernels, busy 57 % of 0.910 ms; FLOP levers below MDI; forward jacfwd NaN; FD strict fails 2/6 smooth points (finding) | 366913 (contended, caveat), 366915 (superseded), **366916** | #350 |
| 0 A100 baseline | skipped (human 2026-09-28) | Does the shared harness run correctly on the A100; what are the fp64 and separately labelled mp reference rows on the preserved unoptimized revisions? | numerical agreement with the CPU reference and the production likelihood, precision-specific tolerances recorded; no optimization before this baseline | | | |
| 1 A100 bottleneck map | not started | Fused device timeline attributed to source operations; `vmap` 1/4/16 (larger within memory headroom); fp64 vs mp; compile vs cold vs warm; simple and cluster configurations; primal vs `custom_jvp` gradient | gradient throughput usable only after finite-difference agreement at smooth points; distinct parameter inputs; synchronized timing | | | |
| 2 Bounded optimization iterations | not started | Levers in contract order: redundant-sort removal on A100 (shared with CPU), `vmap` break-even if launch-bound, the measured sort/gather or static lattice, loop form for unrolled compile, implicit-gradient Jacobian, deflections | the iteration contract in the header; each lever published as accepted / rejected / deferred, re-profile after each acceptance | | | |
| Completion | not started | Close with baseline/final artifact pairs, trace-derived map, batching/precision/compile/gradient tables and `results/notes/point_source_gpu_breakdown_2026_09.md` | stop rule in the header; if no A100 access, the campaign stays pending | | | |

## What shipped and where it is

The bottleneck-map cell `scripts/point_source_image/likelihood_breakdown/gpu_bottleneck_map.py`, its stage map `_point_solver_stage_map.py`, the submit `hpc/batch_gpu/submit_breakdown_point_source_image_gpu_bottleneck_map_a100_fp64` and the job-366916 results (branch `feature/point-source-gpu-p01`). No library change.

## Open / parked / drafts

- `draft/research/autolens_profiling/point_source_image_plane_gpu_breakdown.md` — this campaign, unstarted.
- `draft/research/autolens_profiling/point_solver_profiling_cells.md` — more point-source cells for the cluster arc; the draft says not to merge scopes.

## Caveats

- **The CPU campaign has already touched the phase 2 levers.** Vertex dedup (PyAutoArray#569), the static step-0 lattice (PyAutoArray#570, PyAutoLens#749), the step-0 gather (PyAutoArray#580) and MCS 20 (PyAutoArray#584, PyAutoLens#753) shipped from [Point-source image-plane CPU](point_source_image_plane_cpu.md) with A100 rows (jobs 350587, 350637, 357322, 359102); the static lattice gave only 1.01–1.07× on the A100. Those are CPU-campaign gates, not this campaign's baseline or bottleneck map.
- **Preserved revisions.** The unoptimized revisions phase 0 must reproduce were frozen by the CPU campaign's IP-1 (job 350580, #298); the library has moved on since, so phase 0 needs those checkouts on RAL.
- **The CPU reference is loaded-host.** The shared breakdown's CPU row shared its host with a workspace smoke job; it defines that row only, not a comparison across processes or dates.
- **Prefix rows are not additive.** Step rows are successive differences of independently compiled prefixes (the largest CPU prefix was 453.6 ms against a 62.7 ms fused call); only the fused likelihood is an end-to-end time.
- **No mixed-precision switch exists** in the `PointSolver`; an `hpc_a100_mp` row must state which dtype or policy it changes.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the 2026-09-17 request, the 2026-09-19 three-task split and campaign contract
(its iteration and stop rules as the pre-registered rule), and the four planned phases, with #293
cited as the shared instrument. Noted that the CPU campaign has since shipped several of the
contract's phase 2 levers with A100 rows. Open: the campaign is unissued and has no
A100 baseline.

### 2026-09-28 — phase 0+1 combined (lean), A100 job 366916

The human skipped the contract's phase-0 reproduction of the preserved unoptimized revisions
(the CPU campaign's A100 rows 350587 / 350637 / 357322 / 359102 already cover it) and asked for
one job on current main. Result: launch-bound confirmed — 173 kernels per call, device busy
57 % of the 0.910 ms production wall, CUDA graphs already worth 0.187 ms, vmap-256 81x per
likelihood with no OOM. Deflections and the step-0 lattice are below the 5.45 % MDI even at a
100 % saving; launch count (≤ 1.69x), the neighbourhood sort (≤ 1.24x) and the reverse
gradient's implicit Jacobian (≤ 1.92x per gradient) have room. fp32 what-if 1.27x scalar,
none batched. Findings: `jax.jacfwd` is NaN on this likelihood while `AnalysisPoint` defaults
to forward mode (PyAutoLens#752); two smooth points fail the strict FD rule (kept failing
after a briefly widened gate was reverted); scalar +8.6 % vs job 359102, not bisected. First
run 366913 shared its node and is a caveat only. No image-plane fit has been timed, so the
go/no-go against the admission bar needs one autolens_inference measurement first.
