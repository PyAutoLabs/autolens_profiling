# Phase 3 — MultiStartProdigy reliability (MGE, broad priors, positions OFF): results

Running record for Phase 3 (`../PROGRAMME.md` §4) / CP-3 (§9). Wave 1 ran
2026-08-23: RAL A100 fp64, jobs 338523–338526 (SLURM arrays, 20 runs), all
COMPLETED, every artifact's per-lane block `valid: true`. Config per the
programme doctrine: clip=prior_box, no scaler, no momentum reset,
auto-convergence ON, n_steps ceiling 3000, seed via `SEARCHES_SEED` +
per-arm `unique_tag` (the identifier fix — see the submit-script headers),
positions OFF. Per-lane records via PyAutoFit PR#1515.

## Classification rule

A lane HIT = `lane_best_log_posterior ≥ 31786.782 − 2` (the Nautilus truth
bar minus the Phase-1 tolerance). The rule is insensitive to the threshold:
every hit lane sits in a tight cluster at 31787.79–31787.83 (the known
Prodigy MAP, ~1.0 above the bar — log-posterior vs logL offset noted), and
the nearest wrong-basin best is 26076 — a >5,700-nat gap. Any threshold in
[~26100, ~31785] classifies identically. Run SUCCESS = ≥1 hit lane.

## p_hit and reliability (the CP-3 deliverable)

| tier | runs OK | pooled p̂_hit | Wilson 95% | implied reliability n=16 | n=64 | n=256 |
|---|---:|---:|---|---:|---:|---:|
| n16 ×5 seeds | **1/5** | 1/80 = 0.0125 | [0.002, 0.068] | 0.18 | 0.55 | 0.96 |
| n64 ×5 seeds | **5/5** | 15/320 = 0.0469 | [0.029, 0.076] | 0.54 | 0.95 | >0.9999 |
| n256 ×5 seeds | **5/5** | 61/1280 = 0.0477 | [0.037, 0.061] | 0.54 | 0.96 | >0.9999 |

- **[H3.2] answered:** p_hit ≈ **0.047** (n64 and n256 agree tightly), i.e.
  *below* the Adam-anchored O(0.1–0.2) guess but far from "much lower".
  The n16 tier's own p̂ (0.0125) sits low but its CI includes 0.047; the
  seed-lottery historical picture (1-in-5 runs succeed at 16 starts)
  reproduced exactly.
- **99% reliability needs n ≈ 96 starts** (ln 0.01 / ln(1−0.047)); n=128
  gives ~99.8%, n=256 gives ≥99.99% (CI lower bound).
- **Cost:** wall per run 106–284 s on the A100 (auto-convergence stopped
  every run at 138–206 steps; the 3000 ceiling was never touched — the
  whole 20-run wave cost ≈0.9 A100-hours against a 5.5 h worst-case
  budget). At n=256 that is **~172–225 s vs Nautilus's 831 s** on the same
  tier and NSS's 840–6,341 s — reliable Prodigy is ~4× faster than the
  cheapest nested-sampling row, positions-off, from broad priors.
- **Gate B (part 1) reading — provisional, pending adversarial review and
  human ratification:** the gate's failure condition ("no n_starts ≤ 256
  gives ≥99% reliability at cost below the nested-sampling budget") is
  **not met** — n≥128 clears 99% at a quarter of Nautilus's wall. On this
  evidence MultiStartProdigy(n≥128, prior-box clip, auto-converge)
  qualifies as a *global MAP searcher* for the MGE-class parametric cell,
  positions-off. Caveats before the gate is called: MAP only (no
  posterior/evidence — nested sampling still owns those), single cell
  (MGE/HST), and the §3 rule that a gate decision needs an independent
  falsify-the-interpretation pass first.

## [H3.3] trapped-lane accounting (ell_comps / prior-bound pinning)

Pinned lanes (≥1 parameter exactly on a prior bound) at final / at best:

- n16: 7–12 of 16 per run (≈44–75%) · n64: 25–32 of 64 (≈39–50%) ·
  n256: 115–130 of 256 (≈45–51%).
- Roughly **half of all lanes end pinned** — the trapping is live and
  large under the current clip=prior_box config, yet hit lanes coexist
  with it (hits are never pinned lanes' — the per-lane records allow the
  corner-vs-annulus and parameter-name breakdown; raw names + magnitudes
  are in every artifact for the Phase 4 positions-on re-measurement).
  Per the 2026-08-20 absorption directive this is measured, not closed,
  here: the positions-on repeat decides the fix question.

## Diagnostic arm — θ_E ~ U(0.2, 8) (target_class 3, mechanism probe)

Same n16 tier, seeds 0–4, only the Einstein-radius prior changed
(`Uniform(0,8) → Uniform(0.2,8)`, recorded in each artifact's
`target_override` block; artifacts quarantined under `diagnostic_theta_e/`).

| | campaign n16 | diagnostic n16 |
|---|---:|---:|
| runs OK | 1/5 | **3/5** |
| pooled p̂_hit | 0.0125 | 0.0375 |

Removing the prior wall's intersection with the θ_E→0 degenerate basin
**triples the run success rate at 16 starts** — directional support for
[H3.1] (the failure is basin discovery via the θ_E=0 attractor, not bound
handling). Not conclusive at 5 seeds (CIs overlap); the two remaining
failures land in *other* wrong basins (bests 26076 and −128,637), i.e.
the θ_E wall is not the only attractor. This stays a mechanism note —
the prior change is target-changing and is not a recommendation.

## Convergence detector

All 20 runs stopped on `converged` (never `max_steps`), at 138–206 steps.
No stopped-run has a best lane below its tier's cluster pattern, but the
confusion-matrix question (stopped-correct vs stopped-wrong-basin) is only
half-answerable positions-off: every wrong-basin run ALSO reported
"converged" — the detector cannot distinguish a wrong-basin plateau from
convergence, exactly as PROGRAMME §1.2 warned. Reliability must come from
n_starts, not from the stop rule.

## Next

- Adversarial review pass on the Gate B (part 1) reading, then the human
  gate call (DECISIONS.md).
- Phase 4: the same trio ± PositionsLH (needs positions plumbing in
  `_setup.py` — not built), including the H3.3 re-measurement per engine.
- Optional cheap sharpening: a 5-seed n128 tier to pin the 99% crossing
  directly rather than by arithmetic.
