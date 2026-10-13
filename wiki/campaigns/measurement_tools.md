# Measurement tools

**Status:** open
**Question:** Do this repo's timing tests and profiling gates treat measurement noise correctly, so that an uncertain measurement never qualifies a result silently?
**Pre-registered rule:** PASS only when the interval clears the budget, FAIL only when it clears the other way, otherwise INCONCLUSIVE (never silent, never a measured pass); applied so far to the ABBA overhead (T1, P1), the dashboard drift (P7) and the A/B go / lever rules C6, C7, P2 plus C10's tie sets, on paired whole-round bootstrap intervals (fix phase 4); the runtime headline is the steady median where recorded and a timeout marker is INCONCLUSIVE until it qualifies (fix phase 5); a row whose warm-up never settled is INCONCLUSIVE for every timing verdict, and the production witness judges PASS / FAIL only on a reference host class (fix phase 6); the memo-policy, scaling and log-det gates (C1, C3, C4, C5) read paired round-bootstrap intervals, and C3's six targets are one Holm family at family-wise 90 % (fix phase 3b); one family-wise policy (Holm at family-wise 90 %, family = the comparisons one verdict or claim rests on) judges C6, C10's tie sets and C11's kill gate, C12's MDI is a paired round bootstrap, and P2 is INCONCLUSIVE when its two rows drift (fix phase 9); the README, the remaining cells and the wall basis use the same median headline, and `flat` needs endpoints backed by ≥ 2 independent runs (fix phase 10)
**Verdict:** phase 1 (audit): 31 timing assertions/gates — 11 SOUND, 10 FRAGILE, 10 UNSAFE-SILENT; after fix phases 1–2 (#404, #405): 15 / 8 / 8 (T1, P1, P6, P7 SOUND-with-caveats); after fix phase 3 (#406): 32 rows, 16 / 11 / 5 (C7, C10, P2 UNSAFE-SILENT → FRAGILE; new witness row T7); after fix phase 4 (#407): 33 rows, 20 / 8 / 5 (C6, C7, C10 SOUND-with-caveats; new witness row T8); after fix phase 5 (#408): 34 rows, 23 / 7 / 4 (P8, P9 SOUND-with-caveats; new witness row T9); after fix phase 6 (#409): 35 rows, 26 / 5 / 4 (P3, P5 SOUND-with-caveats; new witness row T10); after fix phase 3b (#410): 36 rows, 31 / 5 / 0 (C1, C3, C4, C5 SOUND-with-caveats; new witness row T11) — no row is UNSAFE-SILENT; after fix phase 9 (#411, stacked on #410): 37 rows, 34 / 3 / 0 (P2, C11 SOUND-with-caveats; new witness row T12); after fix phase 10 (stacked on #411), the last PR phase: 38 rows, 35 / 3 / 0 (no verdict moved; new witness row T13) — PR phases complete pending merge
**Headline:** the #361 CI overhead guard returns INCONCLUSIVE for any true overhead from ~0.8 % to ~5.4 % at CI scatter (3 blocks, s.d. 0.014); GitHub runner, Actions run 36985476995 (no RAL job)
**Library PRs:** none
**Profiling PRs:** #361 (t-bound CI guard, merged); #402 (phase 1 audit, merged); #404 (fix phase 1, merged); #405 (fix phase 2, merged); #406 (fix phase 3, merged); #407 (fix phase 4, merged); #408 (fix phase 5, merged); #409 (fix phase 6, merged); #410 (fix phase 3b, open); #411 (fix phase 9, open, stacked on #410); fix phase 10 (open, stacked on #411)
**Ledger:** [timing_noise_audit_2026_10.md](../../results/notes/timing_noise_audit_2026_10.md)
**Mind contract:** Pulse campaign `measurement-tools`, task `tasks/timing_noise_audit.md`; Mind `active/timing_noise_audit_phase8_phase3b_intervals.md`, `active/timing_noise_audit_phase9_familywise_policy.md`, `active/timing_noise_audit_phase10_headline_completion.md`; issue #362
**Next:** the audit's PR phases are complete pending merge (#410 → #411 → phase 10). What is left is human: C1's 5-round minimum on 4 recorded repeats and C5's "both estimators" rule (flagged in #410); C11's 4-round kill gate (`--reps` ≥ 6 or a 4-round rule) and the GPU memo's "below MDI" readings at a 0.94 % paired MDI (flagged in #411); the setup-baseline / reference-host decision a v2 qualification waits on; measuring `single_jit_repeats` (compute). Outside this repo: the Brain `COMPILE_DRIFT_RATIO` draft (Mind `draft/bug/pyautobrain/profiling_compile_drift_point_vs_point.md`)

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
| 5 — fix phase 4 (C6, C8, C9 and the reported CIs) | 2026-10-08 | Do the A/B intervals respect the round structure and the pairing of routes? | resample whole rounds, the same indices for every route; `effective_n` = rounds; no `best` below 5 rounds | witness: iid CI 0.14–0.39× (median 0.23) the round CI's width, coverage 0.33 vs 0.85 at nominal 0.90; committed RAL calls unchanged; one laptop C6 GO → INCONCLUSIVE; laptop 3-round sweep names no best | none | #407 |
| 6 — fix phase 5 (P8 + P9) | 2026-10-09 | Does the runtime headline survive a post-compile transient, and can one timeout on a noisy host still label a cell GPU-only? | median beside the block mean, headline where present, drift on like estimators only; GPU-only only for a marker that passes `qualify`, else inconclusive and re-measured | witness: block mean 2.40× vs median 1.00× on one injected transient; 0 committed rows carry a median (no badge changed); 4 / 4 committed markers re-judge inconclusive; the one rendered "GPU-only" cell (laptop OOM) → "did not finish (inconclusive)" | none | #408 |
| 7 — fix phase 6 (P3 + P5) | 2026-10-09 | Does any timing verdict read a row whose warm-up never settled, and does the production witness judge a laptop row against an 8-core RAL range? | unsettled warm-up → INCONCLUSIVE for P1, P2 and dashboard qualification (FAIL_GROSS kept); witness PASS / FAIL only on `is_reference_host_class`, else INCONCLUSIVE "off reference host class (<class>)"; band unchanged | witness: flat / recorded-flat / step settle, recorded queueing ramp / geometric / 5 %-per-call ramps never settle and turn a PASS, a FAIL, a GO and a NO_LEVER INCONCLUSIVE; 0 of 28 committed warm-ups unsettled (no verdict moved); 8 / 8 committed witness verdicts (4 PASS, 4 FAIL, all laptop) → INCONCLUSIVE | none | #409 |
| 8 — fix phase 3b (C1, C3, C4, C5) | 2026-10-09 | Do the memo-policy, scaling and log-det gates read an interval, and is C3's six-target conjunction judged family-wise? | each gate's paired round-bootstrap 90 % interval on its own pairing unit (≥ 5 rounds); GO / PASS / NO_LEVER / FAIL only when resolved, else INCONCLUSIVE; C3 as one Holm family; C5 needs both estimators resolved | memo-policy NO_LEVER, scaling PASS ×5 and A100 "no lever" ×6 all resolve unchanged; 128 / 128 committed C1 classifications → INCONCLUSIVE (4 < 5 repeats; a df = 3 t would resolve all as published); 2 laptop log-det rows → INCONCLUSIVE | none | #410 |
| 9 — family-wise policy (C6, C10, C11), C12, P2 drift | 2026-10-09 | Are the multi-comparison gates judged family-wise, does C11's kill gate read an interval, is C12's MDI paired, and does P2 see drift between its rows? | Holm at family-wise 90 % over the comparisons one verdict or claim rests on (conjunction / "any" / "best" shapes); C11 rfft2 / kernel > 1.3 on paired round intervals; C12 MDI from within-round halves; P2 INCONCLUSIVE on ±2.5 % within-row drift or a 2.0 load change | EPYC phase-2c GO routes unchanged; one laptop C6 NO_GO → INCONCLUSIVE; IP-4a tie set 5 → 7, MCS {mcs18, mcs20} + mcs24; both committed kill gates "passed" → INCONCLUSIVE by construction (4 < 5 rounds); A100 MDI 5.45 % → 0.94 %; s4b NO_LEVER not drifted | none | #411 (stacked on #410) |
| 10 — headline completion (P7, P8, S1) | 2026-10-09 | Is the median headline used everywhere a block mean headlines, does the wall basis count its calls, and can an in-band comparison ever be `flat`? | README = the dashboard's estimator rule (median where recorded, labelled); every block-mean cell takes the median (≤ 2 s per call); a median cell's wall gains a 50 s bound per run; `flat` only between endpoints of ≥ 2 independent runs (one host, one estimator) | witness: 1.17× between two 3-run endpoints → `flat`, the same numbers from single runs → `insufficient`; 0 committed rows carry a median or repeat runs (README, dashboard, catalogue unchanged); `mge_mass` keys keep their meaning; one A100 submit's `--time` 20 → 27 min | none | stacked on #411 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| #361 | one-sided Student-t verdict for the CI overhead test | 2026-10-02 | n/a (profiling repo) |
| #404 | one shared `abba_overhead_verdict` (ms excess, one-sided t bounds, PASS / FAIL / FAIL_GROSS / INCONCLUSIVE) for the CI test and the fixed-light numba cell; T1 and P1 now SOUND-with-caveats; the 413.301 ms row's own blocks re-judge INCONCLUSIVE | 2026-10-08 | n/a (profiling repo) |
| #405 | dashboard qualification (`is_reference_host_class`: laptop, no-loadavg and no-host rows unqualified) and drift wording (in-band + single-sample → `insufficient`; `drifted` / `improved` keep status with a single-sample caveat, human decision 2026-10-08); P6 and P7 SOUND-with-caveats | 2026-10-08 | n/a (profiling repo) |
| #406 | one shared A/B verdict `ab_verdict.ab_rule_verdict` (GO / NO_GO / INCONCLUSIVE on intervals, ≥ 5 rounds, correctness gates first) and `tie_set` for C6, C7, C10 and P2; P2 gains a paired-block interval; C7 a saved-ms bootstrap; C7, C10, P2 FRAGILE | 2026-10-08 | n/a (profiling repo) |
| #407 | paired whole-round bootstrap `round_bootstrap.round_median_ratio` / `round_median_saving` (`effective_n` = rounds) for every point-source A/B cell's reported interval (C6, C7, C8, C9 / C10, gradient-mode library, static lattice, vertex dedup); `tie_set` names no best below 5 rounds; C6, C7, C10 SOUND-with-caveats | 2026-10-08 | n/a (profiling repo) |
| #408 | `timing.headline_steady_median` in the 11 block-mean runtime cells (median beside the legacy key); dashboard headline = median where recorded, estimator labelled, drift like-with-like (`estimator-mismatch` → `insufficient`); INCONCLUSIVE timeout markers with host / loads, `build_dashboard.marker_verdict` (GPU-only only when `qualify` passes), `--skip-existing` re-measures unqualified markers; P8, P9 SOUND-with-caveats | 2026-10-09 | n/a (profiling repo) |
| #409 | shared `warmup_gate.warmup_unsettled_reason`: an unsettled warm-up makes `abba_overhead_verdict` (P1), the promotion decision (P2) and `build_dashboard.qualify` INCONCLUSIVE / unqualified; `witness_verdict(..., host_class)` PASS / FAIL only on a reference host class (P5); P3, P5 SOUND-with-caveats | 2026-10-09 | n/a (profiling repo) |
| #410 | `interval_gates` (C1 `matched_classification` / `matched_decision_counts`, C3 `memo_policy_family`, C4 `breakdown_reconciliation`, C5 `logdet_lever_verdict`) on the shared round bootstrap and verdict; `ab_verdict.holm_levels` / `holm_family_verdict` / `bootstrap_criterion` (the family-wise policy) and `conjoin_verdicts`; C1, C3, C4, C5 SOUND-with-caveats | pending | n/a (profiling repo) |
| phase 10 (stacked on #411) | `build_dashboard.headline_reading` shared by the dashboard and the README runtime tables (median labelled); `headline_steady_median(prefix=, block_mean_is=)` in the interferometer breakdown cells, the datacube cell (`full_pipeline_cube_single_jit_median*`, new latent v2 metric `cube_single_jit_median`) and the two `mge_mass` cells (old key kept); `wall/rates.headline_median_extra_wall_s` (50 s) added per invocation by `check_submits`; `aggregate.py` repeat runs → `build_dashboard.repeat_summary` → `single_jit_repeats`, so `flat` is reachable | pending | n/a (profiling repo) |
| #411 (stacked on #410) | the family-wise policy stated in `ab_verdict` and applied through `family_gates` (`phase2c_family` C6, `sweep_tie_set` / `ab_verdict.holm_tie_set` C10, `kill_gate_family` C11), `round_split_half_mdi` (C12, plus paired round intervals for the cell's in-process ratios) and `between_row_drift` (P2); P2, C11 SOUND-with-caveats | pending | n/a (profiling repo) |

## Open / parked / drafts

- Fix phases 1–6, 3b, 9 and 10 in the [ledger](../../results/notes/timing_noise_audit_2026_10.md),
  each one PR with a deterministic synthetic witness; its **End state** section lists every row's
  final verdict and what is deliberately left. The family-wise policy exists since 3b
  (`ab_verdict.holm_family_verdict`, used by C3); phase 9 states it once and applies it to C6 /
  C10 / C11.
- Human questions from 3b (flagged in #410): keep C1's 5-round minimum on its 4 recorded repeats
  (every classification INCONCLUSIVE) or allow a 4-repeat paired t / re-run with ≥ 5 repeats; and
  C5's one-block `jit_profile` estimator, which keeps a lever from ever resolving `True` under the
  pre-registered "both estimators" rule.
- Human questions from phase 9 (flagged in its PR): the numba-interferometer kill gate ("passed")
  is INCONCLUSIVE by construction on its 4 timed rounds — re-run with `--reps` ≥ 6 or allow a
  4-round rule; the GPU go / no-go memo's "below MDI" readings (static lattice, deflections) lose
  their support at the paired MDI of 0.94 %.
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

### 2026-10-09 — fix phase 6 (P3 + P5), phase 7 of #362

One shared check, `warmup_gate.warmup_unsettled_reason`, now makes a row whose warm-up never settled
INCONCLUSIVE for the ABBA overhead verdict (FAIL_GROSS still fires), the fixed-light promotion
(never a GO or a measured NO_LEVER) and dashboard qualification. The 3-vs-3 / 10 % / 12-call rule
is kept and its limits pinned as witnesses (a ≤ 3 %/call ramp settles; a step after call 6 is
unseen; a period-2 ±15 % oscillation never settles). `witness_verdict` takes the row's host class
and judges PASS / FAIL only on `is_reference_host_class`. Re-judged: 0 of 28 committed warm-ups
unsettled; all 8 committed witness verdicts (laptop, untagged) → INCONCLUSIVE, so the
"rectangular rows pass, so the protocol is sound" reading in `production_representative_cells.md`
no longer has a timing witness behind it. Next: phase 3b.

### 2026-10-09 — fix phase 3b (C1, C3, C4, C5), phase 8 of #362

The last four UNSAFE-SILENT gates read intervals through one module, `interval_gates.py`, on the
shared paired round bootstrap and `ab_rule_verdict`: C1's memo / cold classification over its
repeats, C3's six memo-policy targets as one Holm family (new `ab_verdict.holm_family_verdict`,
for phase 9), C4's ±5 % reconciliation (status PASS needs a resolved PASS) and C5's log-det lever
(`clears_threshold` True / False only when resolved). Re-judged: the memo-policy NO_LEVER, the 5
scaling PASSes and the 6 A100 "no lever" rows all resolve unchanged; all 128 committed C1
classifications are INCONCLUSIVE (4 repeats < 5), so the memo-policy note's false accept / reject
counts and the "residual is not a reliable proxy" reading lose interval support (a dated note sits
there; no human decision rests on them); 2 laptop log-det rows become INCONCLUSIVE. No row is
UNSAFE-SILENT. Next: phase 9.

### 2026-10-09 — fix phase 9 (family-wise policy, C12, P2 drift)

Stated one family-wise policy in `ab_verdict` (family = the comparisons one verdict or claim rests
on; Holm at family-wise 90 %; conjunction / "any" / "best" shapes; unadjusted reading recorded) and
applied it through the new `family_gates` module to C6 (one host's 16 criteria), C10
(`holm_tie_set`, the leader vs every other candidate) and C11 (the kill gate on paired round
intervals, kernels × cells). C12's MDI moved to within-round halves on the paired round bootstrap;
P2 records between-row drift. Re-judged as facts: the phase-2c decision and the s4b NO_LEVER stand;
the IP-4a tie set grows to seven; the numba-interferometer kill gate and the GPU memo's "below MDI"
readings lose their interval support (dated notes in both ledgers). Next: phase 10.

### 2026-10-09 — fix phase 10 (headline completion), the last PR phase

The README runtime tables read the dashboard's estimator rule (`build_dashboard.headline_reading`,
median labelled). The interferometer breakdown cells, the datacube cell and the two `mge_mass`
cells take the headline median (the `mge_mass` cells keep their old key, already a median, and say
what it is). `wall/rates.py` bounds the median's extra calls at 50 s per run and `check_submits.py`
adds it per invocation; one A100 submit's `--time` rose 20 → 27 min. `aggregate.py` lists
`<config>.repeat<k>.json` runs and `build_dashboard.repeat_summary` turns ≥ 2 independent runs into
the repeat summary `flat` needs (witness: 1.17× is `flat` between 3-run endpoints, `insufficient`
between single runs). No committed row carries a median or repeat runs, so nothing rendered
changed. Counts 35 / 3 / 0 over 38 rows. PR phases complete pending merge; Next is the human
decisions listed above.
