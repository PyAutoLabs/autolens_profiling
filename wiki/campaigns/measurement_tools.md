# Measurement tools

**Status:** open
**Question:** Do this repo's timing tests and profiling gates treat measurement noise correctly, so that an uncertain measurement never qualifies a result silently?
**Pre-registered rule:** PASS only when the interval clears the budget, FAIL only when it clears the other way, otherwise INCONCLUSIVE (never silent, never a measured pass); proposed in phase 1 and not yet applied to any gate
**Verdict:** phase 1 (audit): 31 timing assertions/gates inventoried — 11 SOUND, 10 FRAGILE, 10 UNSAFE-SILENT; no gate changed
**Headline:** the #361 CI overhead guard returns INCONCLUSIVE for any true overhead from ~0.8 % to ~5.4 % at CI scatter (3 blocks, s.d. 0.014); GitHub runner, Actions run 36985476995 (no RAL job)
**Library PRs:** none
**Profiling PRs:** #361 (t-bound CI guard, merged); phase 1 audit PR (pending)
**Ledger:** [timing_noise_audit_2026_10.md](../../results/notes/timing_noise_audit_2026_10.md)
**Mind contract:** Pulse campaign `measurement-tools`, task `tasks/timing_noise_audit.md`; Mind `active/timing_noise_audit_phase1_inventory.md`; issue #362
**Next:** phase 2 of the audit — one shared PASS/FAIL/INCONCLUSIVE overhead verdict for the CI test and the fixed-light numba cell, in the instrument's unit (fix phase 1 of the ledger), then the dashboard qualification gaps

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
| 1 — inventory | 2026-10-08 | What timing assertions and gates exist, and which can qualify a result on noise? | audit only; no gate changes | 31 rows: 11 SOUND, 10 FRAGILE, 10 UNSAFE-SILENT; six ranked fix phases | none | phase 1 PR |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| #361 | one-sided Student-t verdict for the CI overhead test | 2026-10-02 | n/a (profiling repo) |

## Open / parked / drafts

- Fix phases 1–6 in the [ledger](../../results/notes/timing_noise_audit_2026_10.md), each one PR
  with a deterministic synthetic witness.
- Mind draft `draft/bug/autolens_profiling/call_accounting_ci_timing_threshold.md` (raise the
  threshold) is superseded by fix phase 1: the audit rejects raising the budget.

## Caveats

- The lister finds comparisons against literals and UPPER_CASE constants; two-variable comparisons,
  config-read caps, argmax selections and categorical rules were found by reading.
- Cross-repo: the Brain profiling conductor's compile-drift rule shares the dashboard drift's
  point-vs-point limitation; Pulse requires `qualified` for accepted evidence. Neither was edited.

## Journal

### 2026-10-08 — phase 1 inventory

Read every timing assertion in `scripts/misc/test/` and every production and campaign timing gate;
wrote the ledger with per-row estimator, samples, pairing, clock, host, cutoff, noise model, FP/FN
risk, owner and verdict, the proposed PASS / FAIL / INCONCLUSIVE semantics, and six fix phases.
Added `scripts/misc/tooling/list_timing_assertions.py` (`--check` in `lint.yml`) so the inventory
cannot drift. Nothing else changed. Next: fix phase 1 (shared overhead verdict).
