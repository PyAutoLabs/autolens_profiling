# Measurement tools

**Status:** open
**Question:** Do this repo's timing tests and profiling gates treat measurement noise correctly, so that an uncertain measurement never qualifies a result silently?
**Pre-registered rule:** PASS only when the interval clears the budget, FAIL only when it clears the other way, otherwise INCONCLUSIVE (never silent, never a measured pass); proposed in phase 1 and not yet applied to any gate
**Verdict:** phase 1 (audit): 31 timing assertions/gates — 11 SOUND, 10 FRAGILE, 10 UNSAFE-SILENT; after fix phases 1–2 (#404, #PRNUM): 15 / 8 / 8 (T1, P1, P6, P7 SOUND-with-caveats)
**Headline:** the #361 CI overhead guard returns INCONCLUSIVE for any true overhead from ~0.8 % to ~5.4 % at CI scatter (3 blocks, s.d. 0.014); GitHub runner, Actions run 36985476995 (no RAL job)
**Library PRs:** none
**Profiling PRs:** #361 (t-bound CI guard, merged); #402 (phase 1 audit, merged); #404 (fix phase 1, merged); #PRNUM (fix phase 2, open)
**Ledger:** [timing_noise_audit_2026_10.md](../../results/notes/timing_noise_audit_2026_10.md)
**Mind contract:** Pulse campaign `measurement-tools`, task `tasks/timing_noise_audit.md`; Mind `active/timing_noise_audit_phase3_qualify_drift.md`; issue #362
**Next:** fix phase 3 of the ledger — INCONCLUSIVE states for the A/B rules (C7, C10, then C1/C3/C4/C5, P2)

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
| 3 — fix phase 2 (P6 + P7) | 2026-10-08 | Does the dashboard qualify only reference-host trend points and stop publishing in-band single samples as a null? | laptop / no-loadavg / no-host rows unqualified; in-band with a single-sample endpoint → `insufficient` | qualified records 2 → 2; `flat` 4 → 0 (→ `insufficient`); 6 `improved` keep status with a caveat | none | #PRNUM |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| #361 | one-sided Student-t verdict for the CI overhead test | 2026-10-02 | n/a (profiling repo) |
| #404 | one shared `abba_overhead_verdict` (ms excess, one-sided t bounds, PASS / FAIL / FAIL_GROSS / INCONCLUSIVE) for the CI test and the fixed-light numba cell; T1 and P1 now SOUND-with-caveats; the 413.301 ms row's own blocks re-judge INCONCLUSIVE | 2026-10-08 | n/a (profiling repo) |
| #PRNUM | dashboard qualification (`is_reference_host_class`: laptop, no-loadavg and no-host rows unqualified) and drift wording (in-band + single-sample → `insufficient`; `drifted` / `improved` keep status with a single-sample caveat, human decision 2026-10-08); P6 and P7 SOUND-with-caveats | pending | n/a (profiling repo) |

## Open / parked / drafts

- Fix phases 1–6 in the [ledger](../../results/notes/timing_noise_audit_2026_10.md), each one PR
  with a deterministic synthetic witness.
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
