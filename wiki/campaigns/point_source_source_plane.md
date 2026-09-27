# Point-source source-plane χ² speed-up

**Status:** parked
**Question:** Can the single-source source-plane χ² point-source likelihood (`FitPositionsSourceSolved`, plain `FitPositionsSource` as control) and its gradient be made cheaper on CPU and A100?
**Pre-registered rule:** per phase; 2a/2b/2c used the same shape — GO if a route saves ≥ 0.05 ms AND ≥ 15 % of the call on RAL CPU (8490H), CIs excluding the bar, correctness gate green.
**Verdict:** 2a pytree-flatten lever NO-GO (0.0385 ms < 0.05 ms bar); 2b forward-mode gradient GO on a host re-based by the human (EPYC, not the pre-registered 8490H); 2c no forward/reverse crossover through n=24; 2d made the mode analysis-declared; 2e confirmed the saving arrives through the library. Campaign core complete; remaining candidates parked.
**Headline:** forward-mode batched step 0.36–0.58x of reverse on quiet RAL EPYC 7702 (job 359192); L24 solved compile 95.9 s → 11.2 s (8.6x).
**Library PRs:** PyAutoFit#1649, PyAutoLens#752 (merged, UNRELEASED).
**Profiling PRs:** #317, #318 (folder split), #323, #327, #331, #336.
**Ledger:** [point_source_source_plane_campaign.md](../../results/notes/point_source_source_plane_campaign.md)
**Mind contract:** epic `point-source-cpu-speed`; `draft/research/autolens_profiling/point_source_source_plane_chi_squared_speed.md`; records `complete/2026/09/point-source-source-plane-{breakdown,p2a,p2b,p2c,p2e}.md`, `point-source-gradient-mode.md`.
**Next:** parked candidates — blackjax NUTS/SMC forward-mode `value_and_grad`; an A100 `vmap` throughput row.

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

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoFit#1649 | `gradient_mode` in the search layer | `867af1c6` | UNRELEASED |
| PyAutoLens#752 | `AnalysisPoint` declares forward mode | `b3c9b68e` | UNRELEASED |

## Open / parked / drafts

- Parked: blackjax NUTS / SMC forward-mode `value_and_grad`.
- Parked: A100 `vmap` throughput as its own row.
- Carried library bugs filed through intake (see ledger verdict): `Galaxy` duplicate PyTreeDef registration; `PowerLawMultipole` m=1 at slope 2.

## Caveats

- **SP-1 has no quotable number**: laptop at load ~16 on 8 cores; treat its rows as the magnitude of a ~0.4 ms call.
- **SP-2b host swap**: pre-registered on the 8490H; decided on the quiet EPYC 7702 (job 357381) by human re-base on 2026-09-27; job 357380 cancelled. EPYC absolute ms are higher (forward 0.26 vs 0.146 ms), so savings are quoted as ratios first.
- **SP-2a threshold** was set against the load-inflated ~0.44 ms laptop call; revisiting the 0.05 ms bar is a policy call.
- **SP-2e A100 job 359193** ran beside another 8-CPU job; the A100 L24 solved cell is not used for any claim.
- **End-to-end walls are compile-dominated** at L5 / 20 steps; the plain lane's `fit()` is slightly slower in forward mode (within noise).

## Journal

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.
