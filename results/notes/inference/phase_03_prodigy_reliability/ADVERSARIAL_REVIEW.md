# Adversarial review — Phase 3 / CP-3 Gate B part 1 (2026-08-23)

Independent falsify-the-interpretation pass (PROGRAMME §3) on the wave-1
write-up, run by a fresh agent against the artifacts at branch commit
`14e1fdb`. All hit counts and pinning tables reproduced exactly before the
attacks. Verbatim findings, kept as provenance for the gate call; the
corrected claims live in `RESULTS.md`.

## 1. Binomial independence — UPHELD, with a bound that must be stated

Dispersion indistinguishable from Binomial: variance ratios 1.013 / 0.874 /
1.007 (n16/n64/n256), Tarone C(α) Z = −0.37 pooled (p > 0.6), beta-binomial
ρ̂ = 0.000000. But a coupling mechanism exists by construction (the
convergence detector watches the *global* best, stopping all lanes
together), and 5 runs/tier has little power: the 95% one-sided profile
upper bound is **ρ ≤ 0.0057**. Reliability under the joint-95% worst case
over (μ, ρ): n=96 → **0.918**, n=128 → **0.953**, n=256 → **0.990**.
There is also no model-free route to 99%: 10/10 successes gives a
Clopper–Pearson 95% lower bound of only 0.69 on run reliability
(demonstrating 99% directly needs ~299 runs) — the 99% claim is
necessarily model-based.

## 2. The n16 anomaly — conclusion right, original reasoning broken

P(≤1 hit in 80 | p=0.0477) = 0.101; Fisher/χ² all non-significant. But the
decisive evidence is stronger and different: **lane indices 0–15 produced
exactly 1 hit in 80 slots inside the n256 tier and inside the n64 tier —
identical to the n16 tier's 1/80** (see finding 7: the tiers share draws).
The n16 deficit is one unlucky block of initial conditions, reproduced
verbatim under all three budgets. One genuine small-n mechanism: the
earliest recorded hit anywhere is step 151, and n16_seed3 stopped at 138 —
the only run of 20 that halted before a hit was possible.

## 3. Cost comparison — WOUNDED (four defects, three conservative)

- Nautilus's 831 s **includes 58.6 s viz**; Prodigy ran viz-disabled.
  Like-for-like: **772.7 s vs 172–225 s → 3.4–4.5×**.
- Version/stack mismatch: Nautilus row is v2026.5.21.1 without XLA flags;
  Prodigy is v2026.8.17.1 with autotune off. The same NSS config re-run
  across that gap moved 679 s → 883 s (+30%) — drift favours Prodigy
  (conservative), but "same tier" is not "same stack".
- §3's compile-split rule violated by both: OLS over the 15 campaign runs
  gives wall ≈ **144 s fixed + 1.16 ms/lane-step** (R²=0.16); work spans
  21× while wall spans 2.5×; n16_seed0 is slower than n256_seed3.
- §1.2 also records Nautilus at **523 s** on the same node (→ 2.3–3.0×);
  the two baselines must be reconciled before the gate.
- No `target_id` on any artifact (either side) — §3 comparability
  unverifiable. Circumstantial: the +1.13-nat bar overshoot reproduces the
  recorded laptop-era row.
- `likelihood_evals` in MultiStart artifacts records `samples.total_samples`
  (= n_starts+1), not evaluations — the Phase-3 budget-matched eval table
  is unmet and the field is actively misleading.

## 4. Classification rule — conclusion upheld, stated justification broken

The write-up's "hit lanes cluster at 31787.79–31787.83; nearest wrong basin
26076; >5,700-nat gap" conflated per-RUN global bests with per-LANE bests.
Measured over all lanes: hit lanes span **31785.006–31787.828**, nearest
non-hit **31784.754** — a **0.251-nat** gap; p̂ swings 45% (0.0406→0.0588)
across thresholds bar→26100. What is genuinely threshold-stable is
**run-level success: 11/15 at every threshold** in that range. Prior
contribution quantified: max lane log-posterior sits exactly 0.092 below
the run's max logL in all 14 successful runs — benign.

## 5. Parameter recovery — UPHELD, strongly

All 80 hit lanes across seeds/tiers/arms: θ_E ∈ [1.599476, 1.599881]
(max dev from 1.5997 = 2.24e-4, 0.014%), γ₁ ∈ [0.04817, 0.04895],
γ₂ ∈ [0.04933, 0.04980], centres ≤ 8.5e-4, worst parameter spread 3.8e-3.
**Zero impostors; 0/80 hit lanes pinned at best or final.** Gap: §3 wants
recovery vs *simulator truth with stated tolerances* — no truth vector or
tolerance doc exists (Phase 1, which owns `_targets.py`/tolerances, is
"not started" although Phase 3 declares a dependency on it).

## 6. Auto-convergence — UPHELD; direction conservative

89.6% of hit lanes peak in the last 10 steps; median hit peaks 2 steps
before the stop; all 14 successful runs still improving at stop. Nine
truth-basin lanes sit below the bar by 0.03–12.2 nats, all peaking within
10 steps of the stop → a longer budget raises p̂ to ≈0.055. **p_hit is a
lower bound, conditional on this stop rule.** The failures are genuine
basin failures (zero improvement over the final 50 steps), not
budget failures.

## 7. Cross-tier lane reuse — BROKEN (biggest find)

**Tiers share initial conditions by lane index at the same seed** —
`default_rng(seed)` row *i* is identical regardless of `n_starts`
(mechanism confirmed in `_samplers.py::multi_start_seed`; only chaotic
fp64 reduction-order noise separates tiers, since the vmap batch shape
changes). Evidence: numerically identical lane records across tiers at
matching (seed, index); 8.4% of 320 matched pairs agree to <1e-9 (shifted
control: 0/320); hit-index overlap n64↔n256 is 14.2× enrichment,
p = 1.2e-08; n16's hit index is a subset of n64's in 5/5 seeds.
Consequences: **~1,280 distinct draws, not 1,680**; "n64 and n256 agree"
is not independent corroboration; the n16/n64 tiers add no information
about n-dependence. Corrected from n256 alone: p̂ = 61/1280 = 0.0477,
CP95 [0.0367, 0.0608]; n for 99% = 95 (point) / **124** (CI-lower).
The `unique_tag` scheme itself is fine — no `.completed` collisions.

## 7b. θ_E diagnostic arm — BROKEN

At lane level the diagnostic p̂ (3/80 = 0.0375) is *lower* than the
campaign's (Fisher p = 1.00) — the prior change did not raise per-start
hit probability. The "tripling" of run success is baseline arithmetic:
unchanged p_hit predicts 2.70/5 successes at n=16; the diagnostic's 3/5 is
unremarkable (P(≥3)=0.58) and the campaign n16's 1/5 is the low outlier
(P(≤1)=0.14). The arms are also not draw-matched (the prior override
reorders `parameter_names`: θ_E at index 14 vs 8 — a cross-arm
aggregation trap). **H3.1 receives no support from this arm.**

Bonus: summed walls are 3,443 s = 0.96 A100-h (not "≈0.9").

## Verdict

**Gate B part 1 ("failure condition NOT met") is SAFE to put before the
human, in the narrower form:** the demonstrated tier is **n = 256** —
5/5 seeds, 172–225 s wall vs Nautilus 772.7 s sampler wall (3.4–4.5×),
every hit recovering the reference parameters to 1.4e-4 in θ_E;
reliability **≥ 99.0% at the joint-95% worst case** over p-uncertainty and
the largest lane correlation the data cannot exclude, ≥ 99.99% under
independence. p_hit is under-estimated by the stop rule (conservative).

Mandatory caveats (a)–(h) and the required n128 sharpening (with per-tier
RNG stream offsets) are recorded in `RESULTS.md`.
