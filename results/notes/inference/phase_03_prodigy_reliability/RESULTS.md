# Phase 3 — MultiStartProdigy reliability (MGE, broad priors, positions OFF): results

Running record for Phase 3 (`../PROGRAMME.md` §4) / CP-3 (§9). Wave 1 ran
2026-08-23: RAL A100 fp64, jobs 338523–338526 (SLURM arrays, 20 runs), all
COMPLETED, every artifact's per-lane block `valid: true`. Config per the
programme doctrine: clip=prior_box, no scaler, no momentum reset,
auto-convergence ON, n_steps ceiling 3000, seed via `SEARCHES_SEED` +
per-arm `unique_tag` (the identifier fix — see the submit-script headers),
positions OFF. Per-lane records via PyAutoFit PR#1515.

**This write-up is the post-adversarial-review version** (same day; the
original overclaimed in four places — the full attack record and what
broke is `ADVERSARIAL_REVIEW.md`, kept as provenance for the gate call).

## Classification rule

A lane HIT = `lane_best_log_posterior ≥ 31786.782 − 2` (the Nautilus truth
bar minus the Phase-1 tolerance; prior contribution measured at −0.092 in
every successful run, so the effective logL threshold is 31784.87). The
lane-level margin is NOT large: hit lanes span 31785.01–31787.83 and the
nearest non-hit lane sits **0.028 nats** below the cut, so the lane-level p̂
swings **0.0406→0.0609 (50 %)** across thresholds bar→26100. (Both figures
corrected 2026-08-27 from the raw per-lane arrays: the previously recorded
"0.251" is the hit-to-non-hit *gap*, not the distance to the cut, and the
upper swing was transcribed as 0.0588.) The threshold-stable
quantity is **run-level success — 11/15 campaign runs at every threshold
in that range**. Run SUCCESS = ≥1 hit lane.

## p_hit and reliability (the CP-3 deliverable)

Observed hits (5 seeds per tier): n16 = 1,0,0,0,0 of 16 · n64 = 4,3,2,5,1
of 64 · n256 = 13,8,17,13,10 of 256. Runs successful: **n16 1/5 · n64 5/5
· n256 5/5**.

**The tiers are NOT independent samples** (adversarial finding 7): the
broad-start draw uses `default_rng(seed)` row-by-row, so at the same seed,
lane *i* has the same initial conditions in every tier — n16 and n64 are
prefixes of n256's draw table (hit-index overlap 14.2× enriched,
p = 1.2e-08; several lane records bit-identical across tiers). The wave
therefore contains **~1,280 distinct initial conditions (the n256 tier),
not 1,680**, and n-dependence of p_hit is *unmeasured* by this design.
The n16 tier's low count is one unlucky 80-draw block, reproduced verbatim
as lanes 0–15 inside the other tiers (1/80 there too).

Corrected headline, from the n256 tier alone:

- **p̂_hit = 61/1280 = 0.048**, Clopper–Pearson 95% [0.037, 0.061] —
  a **lower bound**: hits are budget-censored by the stop rule (89.6% of
  hit lanes peak within 10 steps of the stop; nine truth-basin lanes sit
  0.03–12.2 nats short, still climbing; a longer budget gives ≈0.055).
  p̂ is a property of (Prodigy, prior_box clip, this auto-convergence
  rule), not of Prodigy alone.
- Dispersion across runs is indistinguishable from binomial (Tarone
  p > 0.6), but the stop rule couples lanes by construction; the 95%
  upper bound on within-run correlation is ρ ≤ 0.0057.
- **Reliability, joint-95% worst case over (p, ρ): n=96 → 0.92,
  n=128 → 0.95, n=256 → 0.990.** Independence-model point estimates:
  0.991 / 0.998 / 0.999996. The 99%-at-n≈96 arithmetic is a point
  estimate only; the CI-lower crossing is n ≈ 124; **the tier that
  demonstrably clears 99% under every model the data allows is n = 256 —
  the one actually measured.** (No model-free route to 99% exists at
  5 seeds: 10/10 successes bounds run reliability below only at 0.69.)

## Cost

n256 runs: 172–225 s total wall (viz disabled) vs the recorded Nautilus
A100 row's **772.7 s sampler wall** (its 831 s total includes 58.6 s viz)
→ **3.4–4.5×**. Caveats that must ride with that number: §1.2 also
records Nautilus at 523 s on the same node (→ 2.3–3.0×; unreconciled);
the Nautilus row is v2026.5.21.1 on a different XLA config (drift measured
at +30% on an NSS control — favours Prodigy, i.e. conservative); neither
side splits compile per §3 (OLS: ~144 s of each Prodigy wall is fixed
overhead — marginal compute at n256 is ~52 s); and Prodigy is MAP-only
(`log_evidence` NaN) — this is a *budget* comparison per Gate B's wording,
not a like-for-like replacement of posterior+evidence. Whole wave: 20 runs,
0.96 A100-h against a 5.5 h worst-case budget (auto-convergence stopped
every run at 138–206 steps; ceiling never touched).

Known artifact defect: `likelihood_evals` in MultiStart artifacts records
`samples.total_samples` (= n_starts+1), not evaluations — the Phase-3
budget-matched eval table is UNMET pending a `_metrics.py` fix (true
gradient-eval count at n256 ≈ 178×256 ≈ 45.6k vs Nautilus's 63.8k logL
evals).

## Parameter recovery (adversarially verified — the strongest leg)

All 80 hit lanes across seeds, tiers, and arms recover the same solution:
θ_E ∈ [1.599476, 1.599881] (0.014% spread), shear γ ∈ [0.0482, 0.0498],
centres ≤ 8.5e-4, worst parameter spread 3.8e-3. **Zero impostors; zero
hit lanes pinned on any prior bound.** Gap: §3's "recovery within stated
tolerances of simulator truth" cannot be formally evaluated — no truth
vector or tolerance doc exists yet (Phase 1, which owns `_targets.py` and
per-target tolerances, has not started although Phase 3 declares a
dependency on it).

## [H3.3] trapped-lane accounting (ell_comps / prior-bound pinning)

Pinned lanes (≥1 parameter exactly on a bound) at final: n16 7–12/16,
n64 25–32/64, n256 115–130/256 — **roughly half of all lanes end pinned**
under clip=prior_box, and no pinned lane is ever a hit. Raw per-lane
pinned-parameter names and ell_comps magnitudes are recorded in every
artifact for the Phase-4 positions-on re-measurement (measured, not
closed, per the 2026-08-20 absorption directive). Trap for aggregators:
the diagnostic arm's prior override *reorders* `parameter_names` (θ_E at
index 14 vs 8) — never assume a shared parameter order across arms.

## Non-physical ellipticity lanes (added 2026-08-27)

A scan of all 40 MultiStart artifacts carrying per-lane blocks (6,240 lanes)
finds that **1,252 lane best points (20.1 %)** and **1,964 final points
(31.5 %)** sit **outside the ellipticity unit disk** (|`ell_comps`| ≥ 1),
max |e| = 1.41421 — the (±1, ±1) corner of the per-component box prior. Only
lens-mass and source/lens-light Gaussian `ell_comps` are involved. The
mechanism is the box itself: the prior is an independent
`TruncatedGaussian(−1, 1)` per component, so **21.5 % of its volume is
non-physical**; `validate_ell_comps` returns silently on JAX tracers, so the
jitted likelihood is finite and differentiable there; and `prior_box`
clipping is faithful to that (wrong) box, so a lane that walks into the
corner is held in it.

- **The p̂ numerator and the 15/15 record are clean**: **0 of 246 hit lanes**
  are out of the disk. Nothing above is contaminated.
- **But 216/1,280 wave-1 n256 lanes** spent their budget in a non-physical
  corner. Re-based on physical lanes only, **p̂ = 61/1064 = 0.057** — a
  second, independent reason the published 0.048 is a **lower bound**
  (the first being budget censoring by the stop rule).
- **Disjoint from the other two failure populations.** These are not the
  θ_E→0 lanes (median θ_E 4.5 at their best points) and not the CP-4 λ⁴
  NaN population (0 of the out-of-disk replay draws are cholesky-NaN).
- **Cumulative lane-level p̂ across all 15 positions-off n256 runs** (wave-1
  seeds 0–4 + fresh seeds 105–114, 3,840 lanes): **193/3840 = 0.0503** under
  the coded rule — consistent with wave 1's 0.048 and subject to the same
  two lower-bound reasons.
- Nested samplers propose in the same 21.5 % of prior volume and simply
  down-weight it; there is no evidence they converge there, and none that
  anything rejects it.

Positions-on doubles the population (17 % → 29 % of best points, 26 % → 53 %
of final points — `../phase_04_positions/RESULTS.md` "Stage 3"), and the same
corner is where six Phase 8B arms crashed at results-write
(`../phase_08_regularization/RESULTS.md` "8B — run history and harvest").
The library follow-up — a joint disk constraint or reparameterisation the
clipper and model can honour, *not* making `validate_ell_comps` fire on
tracers — is filed as PyAutoMind draft
`feature/autogalaxy/ell_comps_joint_disk_constraint.md` (PROGRAMME §9b W10)
and is not implemented here.

## Diagnostic arm — θ_E ~ U(0.2, 8) (target_class 3): NO SUPPORT for H3.1

Original claim withdrawn after review. At lane level the diagnostic p̂
(3/80 = 0.0375) is *lower* than the campaign's 0.048 (Fisher p = 1.00) —
removing the θ_E=0 wall intersection did **not** raise per-start hit
probability. The apparent run-success improvement (3/5 vs 1/5) is
baseline arithmetic: unchanged p_hit already predicts 2.7/5 at n=16; the
campaign n16's 1/5 is the mild outlier (P(≤1)=0.14), not the diagnostic's
3/5 (P(≥3)=0.58). The arms are also not draw-matched (model reorder ⇒
fresh draws). H3.1 remains open; this probe was uninformative.

## Convergence detector

All 20 runs stopped on `converged` (never `max_steps`) at 138–206 steps —
including every wrong-basin run: the detector cannot distinguish a
wrong-basin plateau from convergence (as §1.2 warned), so reliability must
come from n_starts. The failures are genuine basin failures (zero
global-best improvement over their final 50 steps), and the earliest hit
anywhere lands at step 151 — n16_seed3's stop at 138 is the one run that
halted before a hit was possible (a real small-n censoring mechanism).

## Gate B part 1 — the reading put before the human

**The failure condition ("no n_starts ≤ 256 gives ≥99% reliability at cost
below the nested-sampling budget") is NOT met — demonstrated at n = 256:**
5/5 seeds, ≥99.0% reliability at the joint-95% worst case (≥99.99% under
independence), 172–225 s vs 772.7 s, every hit recovering the reference
parameters. Provisional pending the human call, with mandatory caveats:
(a) demonstrated at n=256 only — no smaller n reaches 99% at 95%
confidence; (b) tiers share draws — effective sample ~1,280, n-dependence
unmeasured; (c) p̂ is budget/detector-conditional and a lower bound;
(d) compile not split, baseline version/flags differ, two unreconciled
Nautilus walls (772.7/523 s); (e) no `target_id` — §3 comparability
unverified; Phase 1 dependency unmet; (f) MAP-only; (g) single cell,
positions-off, one tier; (h) `likelihood_evals` field wrong for
MultiStart.

## Fresh-seed strengthening tier (2026-08-24, RAL 339065[0-4], 339066[0-9])

Required by the adversarial review before the gate could be read below
n=256 and to tighten the n256 CI. Seeds 100–104 (n128) and 105–114 (n256)
draw fresh RNG streams — no lane-index draw is shared with the wave-1
tiers. All 15 runs COMPLETED, version stamp 2026.8.17.1, no resume marker.

| tier | seeds | hits | max logL range | sampler wall |
|---|---|---:|---|---|
| n128 | 100–104 | **5/5** | 31787.886 – 31787.918 | 137–198 s |
| n256 | 105–114 | **10/10** | 31787.881 – 31787.918 | 165–184 s |
| n256 cumulative | 0–4, 105–114 | **15/15** | 31787.881 – 31787.918 | 165–225 s |

Every hit sits on the +1.10 to +1.14 plateau above the Nautilus bar
(31786.782) — no impostor basins. Wilson 95 % lower bound on per-run
success: n256 15/15 → 0.80 (was 0.57 at 5/5); n128 5/5 → 0.57. Under the
lane-independent model the n128 tier's implied p_hit is consistent with
wave 1's 0.048 (P(0 hits | n=128) ≈ 0.2 %, so 5/5 is expected, not
informative about p at this depth). Caveat (a) "n=256 only" therefore
stands: the n128 tier is now measured with independent streams, but five
runs cannot demonstrate ≥99 % at 95 % confidence; calling the gate at
n=128 needs ~30 fresh seeds.

Wall reconciliation: the same-night Nautilus re-baseline on the current
stack (RAL 339070) gives 707 s sampler / 775 s total on the A100 (fp64),
matching the 772.7/831 s references; the 523 s figure does not reproduce
and is retired. The n256 speed-up is therefore **3.1–4.3×** on
sampler wall (165–225 s vs 707 s).

## Next

- Human gate call on the narrowed reading above (DECISIONS.md entry).
- ~~5-seed n128 tier with independent streams~~ done 2026-08-24 (5/5);
  a ≥30-seed n128 tier would be needed to actually call the gate at n=128.
- Fix `_metrics.py` `likelihood_evals` for MultiStart artifacts.
- ~~Reconcile the Nautilus walls~~ done 2026-08-24: re-baseline 707/775 s; 523 s retired.
- Phase 4: the same trio ± PositionsLH (needs positions plumbing in
  `_setup.py` — not built), including the H3.3 re-measurement per engine.
