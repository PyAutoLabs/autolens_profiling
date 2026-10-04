# Point-source source-plane χ² speed-up

**Status:** parked
**Question:** Can the single-source source-plane χ² point-source likelihood (`FitPositionsSourceSolved`, plain `FitPositionsSource` as control) and its gradient be made cheaper on CPU and A100?
**Pre-registered rule:** per phase; 2a/2b/2c used the same shape — GO if a route saves ≥ 0.05 ms AND ≥ 15 % of the call on RAL CPU (8490H), CIs excluding the bar, correctness gate green.
**Verdict:** 2a pytree-flatten lever NO-GO (0.0385 ms < 0.05 ms bar); 2b forward-mode gradient GO on a host re-based by the human (EPYC, not the pre-registered 8490H); 2c no forward/reverse crossover through n=24; 2d made the mode analysis-declared; 2e confirmed the saving arrives through the library. Campaign core complete; remaining candidates parked.
**Headline:** forward-mode batched step 0.36–0.58x of reverse on quiet RAL EPYC 7702 (job 359192); L24 solved compile 95.9 s → 11.2 s (8.6x).
**Library PRs:** PyAutoFit#1649, PyAutoLens#752 (released in 2026.9.27.2).
**Profiling PRs:** #317, #318 (folder split), #323, #327, #331, #336; runtime refresh #349 (issue).
**Ledger:** [point_source_source_plane_campaign.md](../../results/notes/point_source_source_plane_campaign.md)
**Mind contract:** epic `point-source-cpu-speed`; `draft/research/autolens_profiling/point_source_source_plane_chi_squared_speed.md`; records `complete/2026/09/point-source-source-plane-{breakdown,p2a,p2b,p2c,p2e}.md`, `point-source-gradient-mode.md`.
**Next:** parked — blackjax NUTS/SMC forward-mode `value_and_grad`, requires an admission measurement for a gradient sampler. The Nautilus leaf has five recovered seeds and is in open PR autolens_inference#17 under issue #15; its estimated steady likelihood share is 0.0405–0.0447%. A100 `vmap` throughput row done 2026-09-28.

## Why this campaign

Filed on 2026-09-26 from the user's request to "continue on going work to speed up the JAX source
plane chi squared point solver … first task will be to write a likelihood_breakdown and then speed
up from there", mirroring the image-plane campaign. The same day's scope steer made it
single-source only (cluster work belongs to [Cluster PointSolver](cluster_pointsolver.md)). Once
phase 2a showed the forward call sits at its dispatch floor, the campaign turned to the gradient:
the solved likelihood contains an inner forward-mode lensing Hessian, so reverse-mode
`value_and_grad` runs reverse-over-forward through every mass profile for only 5 parameters.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| SP-1 breakdown | 2026-09-26 | Build the source-plane breakdown instrument; rank the residue | instrument only | laptop only, load ~16 on 8 cores — a lead, no quotable number; fixed per-call cost ≈ whole call | laptop | #317 (+#318 split) |
| SP-2a pytree A/B | 2026-09-26 | Is ModelInstance pytree flattening worth a PyAutoFit fast path? | GO if saving ≥ 0.05 ms AND ≥ 15 % of fused call on RAL CPU (issue #322) | NO-GO: 0.0385 ms (26.5 %) solved; fused solved 0.1465 ms on 8490H | 356368, 356369 | #323 |
| SP-2b backward pass | 2026-09-27 | Does forward-mode `jacfwd` beat reverse `value_and_grad`? | GO if ≥ 0.05 ms AND ≥ 15 % vs `rev` on 8490H, CIs excluding the bar (issue #325) | GO, −38…46 % on RAL CPU; deciding row re-based to EPYC 7702 by the human after 8490H job 357380 could not run quiet | 357380 (cancelled), 357381, 357382 | #327 |
| SP-2c crossover | 2026-09-27 | Where does forward mode stop winning as n_params grows? | measure the crossover n* (issue #329) | no crossover through n=24 (solved) / 27 (plain) on any host; design memo: analysis-declared `gradient_mode` | 358770, 358771 | #331 |
| SP-2d library | 2026-09-27 | Put the mode into the libraries | not recorded (library change; default stays `reverse` except `AnalysisPoint`) | `gradient_mode` declared by the analysis, search override | — | PyAutoFit#1649, PyAutoLens#752 |
| SP-2e confirmation | 2026-09-27 | Does the saving arrive through `MultiStartAdam` unchanged? | not recorded (confirmation run, issue #334) | yes: step 0.36–0.58x of reverse; `fit()` solved lane 12.04 → 10.54 s, plain lane 10.86 → 11.04 s (compile-dominated); gradients agree ≤ 2.1e-9 over PRNGKey 0..15 | 359192 (EPYC), 359193 (A100, excluded) | #336 |
| Runtime refresh | 2026-09-28 | Re-measure the runtime cell at release 2026.9.27.2 on the reference node; A100 vmap throughput | not recorded (refresh + measurement, issue #349) | RAL CPU single JIT 0.258 ms (matches 2b's 0.26 ms); A100 vmap 5.6 µs/call at b64 (≈ 114×), launch-bound to b1024; A100 single JIT steady 0.27 ms (cell's 0.642 ms is a warm-up artefact) | 366911, 366912, 366914 (diag) | #349 (issue) |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoFit#1649 | `gradient_mode` in the search layer | `867af1c6` | 2026.9.27.2 |
| PyAutoLens#752 | `AnalysisPoint` declares forward mode | `b3c9b68e` | 2026.9.27.2 |

## Open / parked / drafts

- Parked: blackjax NUTS / SMC forward-mode `value_and_grad` — its admission bar (likelihood share of a fit, eval count) needs a point-source search leaf in `lens/autolens_inference/scripts/point_source/searches/`, which holds only a README.
- Done 2026-09-28: A100 `vmap` throughput row — 5.6 µs/call at batch 64 (≈ 114× the single call), 0.29 µs/call at batch 1024 (diagnostic), job 366912 / 366914.
- Carried library bugs filed through intake (see ledger verdict): `Galaxy` duplicate PyTreeDef registration; `PowerLawMultipole` m=1 at slope 2.

## Caveats

- **SP-1 has no quotable number**: laptop at load ~16 on 8 cores; treat its rows as the magnitude of a ~0.4 ms call.
- **SP-2b host swap**: pre-registered on the 8490H; decided on the quiet EPYC 7702 (job 357381) by human re-base on 2026-09-27; job 357380 cancelled. EPYC absolute ms are higher (forward 0.26 vs 0.146 ms), so savings are quoted as ratios first.
- **SP-2a threshold** was set against the load-inflated ~0.44 ms laptop call; revisiting the 0.05 ms bar is a policy call.
- **SP-2e A100 job 359193** ran beside another 8-CPU job; the A100 L24 solved cell is not used for any claim.
- **Runtime cell A100 `single_jit` is warm-up-contaminated**: 0.642 ms committed (job 366912, one warm call then mean of 10) vs a steady 0.267 ms median (diagnostic job 366914, same node); quote `vmap.per_call`, not `single_jit`, for the A100 row. The method is unchanged for dashboard comparability. Option (a) was chosen by the human on 2026-10-04 (issue #371):
  - the cell now also writes `full_pipeline_single_jit_median_ms` (with p10/p90; ≥ 5 warm calls, 200 timed calls) beside the unchanged `single_jit`;
  - the dashboard labels GPU `single_jit` headlines "first block after compile";
  - committed rows are not re-based, and the new field first appears at the next release sweep.

  Contract: PyAutoPulse task [runtime_cell_single_jit_gpu_warmup](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/runtime_cell_single_jit_gpu_warmup.md). The imaging release-sweep cells share the old method and are not yet wired.
- **End-to-end walls are compile-dominated** at L5 / 20 steps; the plain lane's `fit()` is slightly slower in forward mode (within noise).

## Journal

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.

### 2026-09-28 — library PRs released

PyAutoFit#1649 (`867af1c6`) and PyAutoLens#752 (`b3c9b68e`) are in release 2026.9.27.2: `git tag
--contains` names 2026.9.27.2 as the first tag in both repos (the PyAutoFit tag points at the
merge commit itself). Header and "What shipped" updated; `AnalysisPoint`'s forward-mode gradient
is now what a user of the released stack gets by default (autolens_profiling#349).

### 2026-09-28 — runtime refresh + A100 vmap throughput row

`likelihood_runtime/source_plane_solved.py` re-run on `euclid-ral-gpu-2` at the 2026.9.27.2 library
commits (scratch clones; `library_revisions` = tag commits), both rows carrying `device.provenance`
and qualified on the dashboard. RAL CPU (job 366911, load 0.16): single JIT 0.258 ms, vmap(b3)
0.097 ms/call; matches phase 2b's 0.26 ms forward call on this host. A100 (job 366912, load 1.16):
vmap(b64) 5.6 µs/call ≈ 114× the single call — the parked throughput row, now done. The cell's A100
single JIT of 0.642 ms is a post-compile warm-up artefact: diagnostic job 366914 on the same node
measured a steady 0.267 ms median single call (floor 0.132 ms), 1.2× phase 2a's 0.222 ms (same fused
call shape, different node: gpu-1), and batch walls flat from 64 to 1024 (0.29 µs/call at 1024). The A100 leg joins the release
sweep. Caveat: source-checkout rows are labelled `autolens_version` 2026.8.17.1 (the build-time
stamp), so the dashboard cannot yet separate releases for them. blackjax forward mode stays parked
until autolens_inference has a point-source search leaf. Ledger: "Runtime refresh on 2026.9.27.2".

## 2026-10-02 — epic reconciliation and the existing inference phase

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
Review: [inference PR #17](https://github.com/PyAutoLabs/autolens_inference/pull/17) and [ledger PR #361](https://github.com/PyAutoLabs/autolens_profiling/pull/361); both open, no second phase issued.
See the [inference journal](https://github.com/PyAutoLabs/autolens_inference/blob/feature/point-source-search-nautilus-leaf/wiki/project/state.md#2026-10-02--point-source-nautilus-admission-bar-five-seeds-recovered)
for all rows and limitations. The estimate uses one prior-median vector and fixed
batch size, not an instrumented fit decomposition; it does not establish the
wall-share of a gradient sampler. blackjax forward mode still needs its own
admission measurement. The A100 throughput row already landed on 2026-09-28.
