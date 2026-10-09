# Measurement tools

**Status:** open
**Question:** Do this repo's timing tests and profiling gates treat measurement noise correctly, so that an uncertain measurement never qualifies a result silently?
**Pre-registered rule:** PASS only when the interval clears the budget, FAIL only when it clears the other way, otherwise INCONCLUSIVE (never silent, never a measured pass); applied so far to the ABBA overhead (T1, P1), the dashboard drift (P7) and the A/B go / lever rules C6, C7, P2 plus C10's tie sets, on paired whole-round bootstrap intervals (fix phase 4); the runtime headline is the steady median where recorded and a timeout marker is INCONCLUSIVE until it qualifies (fix phase 5)
**Verdict:** phase 1 (audit): 31 timing assertions/gates — 11 SOUND, 10 FRAGILE, 10 UNSAFE-SILENT; after fix phases 1–2 (#404, #405): 15 / 8 / 8 (T1, P1, P6, P7 SOUND-with-caveats); after fix phase 3 (#406): 32 rows, 16 / 11 / 5 (C7, C10, P2 UNSAFE-SILENT → FRAGILE; new witness row T7); after fix phase 4 (#407): 33 rows, 20 / 8 / 5 (C6, C7, C10 SOUND-with-caveats; new witness row T8); after fix phase 5 (#PRNUM): 34 rows, 23 / 7 / 4 (P8, P9 SOUND-with-caveats; new witness row T9)
**Headline:** the #361 CI overhead guard returns INCONCLUSIVE for any true overhead from ~0.8 % to ~5.4 % at CI scatter (3 blocks, s.d. 0.014); GitHub runner, Actions run 36985476995 (no RAL job)
**Library PRs:** none
**Profiling PRs:** #361 (t-bound CI guard, merged); #402 (phase 1 audit, merged); #404 (fix phase 1, merged); #405 (fix phase 2, merged); #406 (fix phase 3, merged); #407 (fix phase 4, merged); #PRNUM (fix phase 5, open)
**Ledger:** [timing_noise_audit_2026_10.md](../../results/notes/timing_noise_audit_2026_10.md)
**Mind contract:** Pulse campaign `measurement-tools`, task `tasks/timing_noise_audit.md`; Mind `active/timing_noise_audit_phase6_median_headline_gpu_marker.md`; issue #362
**Next:** fix phase 6 (P3 warm-up flag, P5 witness band), then phase 3b (C1 / C3 / C4 / C5 intervals)

## Why this campaign

PR #361's CI failed on a timing guard at ratio 1.031073665 against 1.031, while equality and
coverage passed. The guard had already been relaxed once (1.03 → 1.031) to get a merge through.
Issue #362 asks for the whole mechanism to be audited, not the threshold to be raised again: every
timing assertion and production gate, with its estimator, samples, pairing, host qualification,
cutoff justification and noise model, and PASS / FAIL / INCONCLUSIVE semantics in which
inconclusive never qualifies a result.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 — inventory | 2026-10-08 | What timing assertions and gates exist, and which can qualify a result on noise? | audit only; no gate changes | 31 rows: 11 SOUND, 10 FRAGILE, 10 UNSAFE-SILENT; six ranked fix phases | none | #402 |
| 2 — fix phase 1 (T1 + P1) | 2026-10-08 | Can the CI test and the numba cell share one interval verdict for the ABBA overhead? | PASS / FAIL only when the one-sided t bound clears the 12 ms budget, else INCONCLUSIVE | one `abba_overhead_verdict`; 6 of 17 RAL rows re-judge INCONCLUSIVE | none | #404 |
| 3 — fix phase 2 (P6 + P7) | 2026-10-08 | Does the dashboard qualify only reference-host trend points and stop publishing in-band single samples as a null? | laptop / no-loadavg / no-host rows unqualified; in-band with a single-sample endpoint → `insufficient` | qualified records 2 → 2; `flat` 4 → 0 (→ `insufficient`); 6 `improved` keep status with a caveat | none | #405 |
| 4 — fix phase 3 (C6, C7, C10, P2) | 2026-10-08 | Do the pre-registered A/B go / lever rules read their intervals, and does argmax refuse to name a tied winner? | GO only when the whole interval clears the bar, NO_GO / NO_LEVER only when it is wholly on the bad side, else INCONCLUSIVE (≥ 5 rounds); tie set when the leaders' intervals overlap | no go / no-go / NO_LEVER call changed (C7 4, C6 24, P2 1 rows all resolve); 7 of 9 committed solver-sweep "best" are tie sets; C1 / C3 / C4 / C5 deferred to phase 3b | none | #406 |
| #407 | paired whole-round bootstrap `round_bootstrap.round_median_ratio` / `round_median_saving` (`effective_n` = rounds) for every point-source A/B cell's reported interval (C6, C7, C8, C9 / C10, gradient-mode library, static lattice, vertex dedup); `tie_set` names no best below 5 rounds; C6, C7, C10 SOUND-with-caveats | pending | n/a (profiling repo) |
| 5 — fix phase 4 (C6, C8, C9 and the reported CIs) | 2026-10-08 | Do the A/B intervals respect the round structure and the pairing of routes? | resample whole rounds, the same indices for every route; `effective_n` = rounds; no `best` below 5 rounds | witness: iid CI 0.14–0.39× (median 0.23) the round CI's width, coverage 0.33 vs 0.85 at nominal 0.90; committed RAL calls unchanged; one laptop C6 GO → INCONCLUSIVE; laptop 3-round sweep names no best | none | #407 |
| 6 — fix phase 5 (P8 + P9) | 2026-10-09 | Does the runtime headline survive a post-compile transient, and can one timeout on a noisy host still label a cell GPU-only? | median beside the block mean, headline where present, drift on like estimators only; GPU-only only for a marker that passes `qualify`, else inconclusive and re-measured | witness: block mean 2.40× vs median 1.00× on one injected transient; 0 committed rows carry a median (no badge changed); 4 / 4 committed markers re-judge inconclusive; the one rendered "GPU-only" cell (laptop OOM) → "did not finish (inconclusive)" | none | #PRNUM |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| #361 | one-sided Student-t verdict for the CI overhead test | 2026-10-02 | n/a (profiling repo) |
| #404 | one shared `abba_overhead_verdict` (ms excess, one-sided t bounds, PASS / FAIL / FAIL_GROSS / INCONCLUSIVE) for the CI test and the fixed-light numba cell; T1 and P1 now SOUND-with-caveats; the 413.301 ms row's own blocks re-judge INCONCLUSIVE | 2026-10-08 | n/a (profiling repo) |
| #405 | dashboard qualification (`is_reference_host_class`: laptop, no-loadavg and no-host rows unqualified) and drift wording (in-band + single-sample → `insufficient`; `drifted` / `improved` keep status with a single-sample caveat, human decision 2026-10-08); P6 and P7 SOUND-with-caveats | 2026-10-08 | n/a (profiling repo) |
| #406 | one shared A/B verdict `ab_verdict.ab_rule_verdict` (GO / NO_GO / INCONCLUSIVE on intervals, ≥ 5 rounds, correctness gates first) and `tie_set` for C6, C7, C10 and P2; P2 gains a paired-block interval; C7 a saved-ms bootstrap; C7, C10, P2 FRAGILE | 2026-10-08 | n/a (profiling repo) |
| #PRNUM | `timing.headline_steady_median` in the 11 block-mean runtime cells (median beside the legacy key); dashboard headline = median where recorded, estimator labelled, drift like-with-like (`estimator-mismatch` → `insufficient`); INCONCLUSIVE timeout markers with host / loads, `build_dashboard.marker_verdict` (GPU-only only when `qualify` passes), `--skip-existing` re-measures unqualified markers; P8, P9 SOUND-with-caveats | pending | n/a (profiling repo) |

## Open / parked / drafts

- Fix phases 1–6 in the [ledger](../../results/notes/timing_noise_audit_2026_10.md), each one PR
  with a deterministic synthetic witness. Phase 3b (C1 / C3 / C4 / C5 intervals) was split off
  fix phase 3; a family-wise (Holm / Bonferroni) policy for multi-route rules is a follow-up.
- Mind draft `draft/bug/autolens_profiling/call_accounting_ci_timing_threshold.md` (raise the
  threshold) is superseded by fix phase 1: the audit rejects raising the budget.

## Caveats

- The lister finds comparisons against literals and UPPER_CASE constants; two-variable comparisons,
  config-read caps, argmax selections and categorical rules were found by reading.
- Cross-repo: the Brain profiling conductor's compile-drift rule shares the dashboard drift's
  point-vs-point limitation; neither Brain nor Pulse was edited. The live Pulse registry reads the
  v2 `dashboard/catalogue.json` (since 2026-10-05), whose producer hard-codes `qualified: False`;
  the phase 1 claim that P6's gaps flow into Pulse described the v1 path only. A future v2
  qualification must reuse `is_reference_host_class`.

## Journal

### 2026-10-08 — phase 1 inventory

Read every timing assertion in `scripts/misc/test/` and every production and campaign timing gate;
wrote the ledger with per-row estimator, samples, pairing, clock, host, cutoff, noise model, FP/FN
risk, owner and verdict, the proposed PASS / FAIL / INCONCLUSIVE semantics, and six fix phases.
Added `scripts/misc/tooling/list_timing_assertions.py` (`--check` in `lint.yml`) so the inventory
cannot drift. Nothing else changed. Next: fix phase 1 (shared overhead verdict).

### 2026-10-08 — fix phase 2 (P6 + P7), phase 3 of #362

`build_dashboard.qualify` now leaves laptop rows, rows with no load average and `hpc_*` rows with
no host unqualified with a reason (`is_reference_host_class` holds the host-class rule). A
comparison inside the 2× band publishes `flat` only when both endpoints carry a repeat summary;
none does today, so the 4 laptop `flat` comparisons became `insufficient`. `drifted` / `improved`
keep their status with a "single-sample endpoint(s)" reason (human decision 2026-10-08). Qualified
records unchanged (2). Pulse's v1 validator passes. Next: fix phase 3 (A/B rule INCONCLUSIVE).

### 2026-10-08 — fix phase 3 (C6, C7, C10, P2), phase 4 of #362

One shared `ab_verdict.ab_rule_verdict` now judges the pytree phase-2b rule (C7, with a new
saved-ms bootstrap), the backward-pass phase-2c rule (C6) and the fixed-light numba promotion (P2,
on a paired-block t interval); `tie_set` replaces the solver sweep's argmax (C10), whose vmap rows
follow the labelled point leader (flagged decision). Re-judging the committed rows changed no go /
no-go / NO_LEVER call; 7 of 9 committed "best admissible" names are tie sets (IP-4a's 2.37x
leader is one of five). C1 / C3 / C4 / C5 need new intervals and are phase 3b. Next: fix phase 4
(round bootstrap).

### 2026-10-08 — fix phase 4 (C6, C8, C9 and the reported CIs), phase 5 of #362

The point-source A/B cells' iid, unpaired call bootstraps are replaced by one paired whole-round
bootstrap (`round_bootstrap.py`), recording `effective_n` = rounds. Witness: with a per-round
offset, the iid 90 % interval is 0.14–0.39× (median 0.23) the round interval's width and covers the
true ratio a third of the time (round: 0.85). Re-judged on round intervals, no RAL call changed;
one laptop C6 GO became INCONCLUSIVE and the 3-round laptop sweep names no best. Next: phase 3b.

### 2026-10-09 — fix phase 5 (P8 + P9), phase 6 of #362

Every runtime cell that headlines the 10-call block mean now writes the steady median beside it
(one helper, `timing.headline_steady_median`), and the dashboard headlines that median where a row
records it, labelled, with drift refusing to compare a median with a block mean. A sweep timeout
now records INCONCLUSIVE with host and load and renders "GPU-only" only when it qualifies; an
unqualified marker is re-measured by `--skip-existing`. No committed row carries a median, so no
badge changed; all 4 committed markers re-judge inconclusive and the one rendered "GPU-only" README
cell (a laptop OOM) now reads "did not finish (inconclusive)". No v2 field changed. Next: fix
phase 6 (P3, P5), then phase 3b.
