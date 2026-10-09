# Timing-noise audit, phase 1: inventory of timing assertions and gates (2026-10)

Issue: [autolens_profiling#362](https://github.com/PyAutoLabs/autolens_profiling/issues/362)
(phase 1 plan in its 2026-10-08 comment). Mind: `active/timing_noise_audit_phase1_inventory.md`.
Pulse task: `tasks/timing_noise_audit.md` (campaign `measurement-tools`). Campaign page:
[Measurement tools](../../wiki/campaigns/measurement_tools.md).

**Fix phase 1 shipped (phase 2 of #362, PR #404, 2026-10-08):** rows T1 and P1 now share one verdict,
`scripts/misc/likelihood_breakdown/overhead_verdict.py::abba_overhead_verdict`, and are
SOUND-with-caveats; see "Fix phase 1 — shipped" under (b). The rest of this note is the phase 1
audit as written, with the T1 / P1 / P2 / T3 rows and the counts updated.

**Fix phase 2 shipped (phase 3 of #362, PR #405, 2026-10-08):** rows P6 and P7 are
SOUND-with-caveats. `build_dashboard.qualify` no longer qualifies laptop rows, rows whose
provenance lacks a load average, or `hpc_*` rows with no host; a comparison inside the 2× band
with a single-sample endpoint is published `insufficient`, not `flat`. See "Fix phase 2 —
shipped" under (b), which also corrects this note's original claim that P6 flows into
PyAutoPulse.

**Fix phase 3 shipped (phase 4 of #362, PR #406, 2026-10-08):** the pre-registered A/B rules
of rows C6, C7 and P2 and the argmax of C10 share one verdict,
`scripts/misc/likelihood_breakdown/ab_verdict.py` (`ab_rule_verdict`, `tie_set`): GO only when
the whole interval clears the bar, NO_GO / NO_LEVER only when it is wholly on the bad side,
INCONCLUSIVE otherwise, and a tie set instead of a single "best" when the leaders' intervals
overlap. C7, C10 and P2 move from UNSAFE-SILENT to FRAGILE (their intervals are still iid or
unpaired in time; fix phase 4). Re-judging the committed rows changed no go / no-go call; 7 of 9
committed solver-sweep "best" configurations are tie sets. C1 / C3 / C4 / C5 are deferred to
**phase 3b**. See "Fix phase 3 — shipped" under (b).

**Fix phase 4 shipped (phase 5 of #362, PR #407, 2026-10-08):** every point-source A/B cell's
reported interval (C6, C7, C8, C9 / C10 and the CIs of `gradient_mode_library_ab`,
`static_lattice_ab`, `vertex_dedup_ab`) is now a **paired whole-round bootstrap**,
`scripts/misc/likelihood_breakdown/round_bootstrap.py` (`round_median_ratio`,
`round_median_saving`), with the number of rounds recorded as `effective_n`; a tie set over
fewer than 5 rounds names no `best`. C6, C7 and C10 become SOUND-with-caveats. Re-judging the
committed rows on round intervals changed one non-deciding laptop C6 call (GO → INCONCLUSIVE) and
nothing on RAL. See "Fix phase 4 — shipped" under (b).

**Fix phase 5 shipped (phase 6 of #362, PR #408, 2026-10-09):** every runtime cell that headlines
`jit_profile`'s 10-call block mean also writes the steady median beside it through one helper,
`scripts/misc/likelihood_breakdown/timing.py::headline_steady_median`; the dashboard headlines the
median where a row records it, labels the estimator, and drift never compares a median with a
block mean (P8). A `--per-run-timeout` timeout writes an INCONCLUSIVE marker with host, loads and
timeout; a marker renders "GPU-only", and `--skip-existing` honours it, only when it qualifies
under `qualify`; otherwise it renders inconclusive and is re-measured (P9). P8 and P9 become
SOUND-with-caveats. No committed row carries a median, so no dashboard headline or drift badge
changed; the one rendered "GPU-only" cell (a hand-written laptop OOM marker) now reads "did not
finish (inconclusive)". See "Fix phase 5 — shipped" under (b).

**Fix phase 3b shipped (phase 8 of #362, PR #410, 2026-10-09):** rows C1, C3, C4 and C5 read
intervals through one module, `scripts/misc/likelihood_breakdown/interval_gates.py`, built on the
shared `round_bootstrap` and `ab_verdict` tools, and each gains an INCONCLUSIVE state. C3's six
targets are one **Holm** family (`ab_verdict.holm_family_verdict`, the policy phase 9 reuses). All
four become SOUND-with-caveats; no row is UNSAFE-SILENT any more. Re-judging the committed rows
changed no recorded verdict (memo-policy NO_LEVER, scaling PASS ×5, no log-det lever on the A100)
but every one of the 128 committed C1 matched classifications is now INCONCLUSIVE (4 repeats < the
5-round minimum), so the memo-policy note's "false accept / false reject" counts and the design
conclusion drawn from them lose their interval support. See "Fix phase 3b — shipped" under (b).

**Fix phase 9 shipped (phase 9 of #362, stacked on #410, 2026-10-09):** one documented
family-wise policy (Holm at family-wise 90 %, family = the comparisons one verdict or claim rests
on; section (a)) now judges C6 (one host's routes × lanes × criteria), C10 (the leader against
every other configuration, `ab_verdict.holm_tie_set`) and C11 (pinned numba kernels × sma / alma
cells; its ratio > 1.3 gets a paired round-bootstrap interval), through
`scripts/misc/likelihood_breakdown/family_gates.py`. C12's split-half MDI and the cell's in-process
ratios move onto the paired round bootstrap; P2 records between-row drift and is INCONCLUSIVE when
the rows drift. P2 and C11 become SOUND-with-caveats (34 / 3 / 0 over 37 rows). Re-judging the
committed rows: the phase-2c GO routes on the deciding EPYC row are unchanged; IP-4a's tie set grows
from 5 to 7; **the numba-interferometer kill gate ("passed") is INCONCLUSIVE by construction**
(4 timed rounds < 5) and **the A100 MDI the GPU go / no-go memo quotes (5.45 %) is 0.94 % on paired
rounds**, so that memo's "below MDI" readings lose their support. See "Fix phase 9 — shipped"
under (b).

**Phase 1 was an audit. Nothing was changed in it:** no gate, budget, tolerance, cutoff, estimator,
repeat count or production instrument was edited. The only code added is the read-only lister
`scripts/misc/tooling/list_timing_assertions.py`, which keeps this inventory in step with the code.
No compute job was run.

## Trigger

PR #361's CI failed `test_call_accounting_covers_a_real_likelihood_call` at an ABBA overhead ratio
of 1.031073665 against a 1.031 budget (blocks 1.0218, 1.0243, 1.0471). Equality and the 99.95 %
coverage check passed. The old guard compared the observed block *range* to the whole budget. It did
not compare the uncertainty to the *excess* over the budget, which here was 0.007 percentage points.
#361 replaced it with a one-sided Student-t rule (`_ci_overhead_verdict`, commit `036615c`). The
budget had already been relaxed once, from 1.03 to 1.031, in `d5462d4` (2026-09-19, "test: relax
profiling overhead tolerance", the PR #288 failure at 1.030266). That relaxation is the pattern the
issue forbids: raising a budget until CI goes green.

What the #361 rule does with its own trigger data (computed here, not re-run):

| quantity | value |
|---|---|
| mean of the 3 block ratios | 1.031074 |
| sample s.d. / standard error | 0.01391 / 0.00803 |
| one-sided 95 % t critical value, df = 2 | 2.920 |
| one-sided 95 % bounds | [1.00763, 1.05452] |
| verdict | INCONCLUSIVE (test passes) |
| mean needed to FAIL at this scatter | > 1.0545 (a 5.4 % overhead) |
| mean needed to PASS at this scatter | ≤ 1.0076 |
| blocks needed for a ±0.7 pp half-width at this s.d. | about 12 |

So the repaired test almost never fails falsely, but it also has almost no power. At CI scatter, a
true overhead anywhere from about 0.8 % to 5.4 % comes back INCONCLUSIVE, and the test passes. The
INCONCLUSIVE is only `print`ed, and `pytest -q` captures stdout, so it is silent in CI.

## How to read the inventory

**Kinds.** *CI test* is an assertion in `scripts/misc/test/` run by `lint.yml` on a GitHub runner.
*Production qualification* is code that labels a published result or a dashboard point (PASS,
qualified, GPU-only) or kills a row. *Campaign gate* is a pre-registered go/no-go or lever rule
inside a campaign cell. *Submit basis* is the HPC `--time` justification. *Resource guard* skips
work that would be too expensive. *Estimator* is a setting that produces a number some gate reads.

**Verdicts.**

- **SOUND**: the uncertainty is part of the decision, or the decision is deterministic or
  structural, or nothing downstream reads it as a measured result.
- **FRAGILE**: the uncertainty is considered but handled imperfectly. Examples: an iid bootstrap
  over dependent samples, no INCONCLUSIVE state, an underpowered CI rule, a winner chosen by argmax
  that a human then reviews. Or the output is advisory, and a noisy run can flip it in either
  direction.
- **UNSAFE-SILENT**: a point estimate, or a qualification that does not look at the measurement,
  produces PASS / GO / NO_LEVER / qualified / GPU-only. Downstream code or readers take that label
  as a measured result, and the label does not carry the uncertainty.

**Key.** Each lister key is written in backticks. The format is `<path>::<function>` for a
comparison (`<module>` for module-level code) and `<path>::<CONSTANT>` for a constant. The lister
keys on these, not on line numbers. Line numbers are as of `origin/main` `766f820`, except rows
T1, T3, P1 and P2, which are as of the fix phase 1 branch.

## Summary

| ID | Kind | Location (file:line) | What is gated | Cutoff | Verdict |
|---|---|---|---|---|---|
| T1 | CI test | `scripts/misc/test/test_fixed_light_numba.py::test_call_accounting_covers_a_real_likelihood_call` (667) via `scripts/misc/likelihood_breakdown/overhead_verdict.py::abba_overhead_verdict` (102) | ABBA instrumentation overhead in ms of excess on a real tiny fit | the cell's `MAX_INSTRUMENTATION_OVERHEAD_MS` 12.0 through the fixture's own clean mean; FAIL_GROSS > 1.5; witnesses at `scripts/misc/test/test_fixed_light_numba.py::BUDGET_MS` 12.0 (asserted equal to the cell's) | SOUND-with-caveats |
| T2 | CI test | `scripts/misc/test/test_call_accounting.py` (94–330) | exclusive/inclusive additivity, > 0 timings, re-entrancy | 1e-9 identities, `> 0` | SOUND |
| T3 | CI test | `scripts/misc/test/test_fixed_light_numba.py` 489–665 (shared-verdict witnesses) and 1059–1391 (overhead-gate, promotion and warm-up unit tests); since fix phase 6 also the P3 warm-up witnesses (`scripts/misc/test/test_fixed_light_numba.py::test_warmup_witness_sequences`, `scripts/misc/test/test_fixed_light_numba.py::test_an_unsettled_warmup_makes_the_overhead_verdict_inconclusive`, `scripts/misc/test/test_fixed_light_numba.py::test_an_unsettled_arm_makes_the_promotion_inconclusive`, `scripts/misc/test/test_fixed_light_numba.py::test_every_committed_warmup_record_settled`) | the shared verdict and the cell's gate statements on synthetic inputs; flat / ramp / step warm-up sequences through the cell's own `_warm_to_steady_state` into P1 and P2 | pins 12.0 ms, 3 blocks, 1.03, warm-up 3/0.10/12 | SOUND |
| T4 | CI test | `scripts/misc/test/test_timing_steady_median.py` | `steady_median_profile` on an injected clock | exact | SOUND |
| T5 | CI test | `scripts/misc/test/test_wall_check_submits.py::test__the_41x_spread_that_caused_the_loss` (298) | wall estimate arithmetic from the rates table | `> 8` | SOUND |
| T6 | CI test | `scripts/misc/test/test_build_dashboard.py` (132, 260–285) | qualification and drift on synthetic rows | exact | SOUND (pins P6/P7 semantics) |
| T7 | CI test | `scripts/misc/test/test_ab_verdict.py` (shared A/B verdict witnesses; `scripts/misc/test/test_ab_verdict.py::test_a_15_percent_point_estimate_straddling_the_bar_is_inconclusive` and the bars `scripts/misc/test/test_ab_verdict.py::GO_MIN_SAVED_MS`, `scripts/misc/test/test_ab_verdict.py::GO_MIN_FRACTION`) | `ab_rule_verdict` / `tie_set` on seeded synthetic samples, the cells' own lifted rules, and the committed C6 / C7 / C10 rows | exact, seeded | SOUND (pins the fix phase 3 semantics) |
| T9 | CI test | `scripts/misc/test/test_headline_estimator_and_marker.py` (P8 / P9 witnesses: `scripts/misc/test/test_headline_estimator_and_marker.py::test_injected_transient_moves_the_block_mean_2p4x_and_not_the_median`, `scripts/misc/test/test_headline_estimator_and_marker.py::test_drift_never_compares_a_median_with_a_block_mean`, `scripts/misc/test/test_headline_estimator_and_marker.py::test_an_unqualified_marker_is_never_rendered_gpu_only`, `scripts/misc/test/test_headline_estimator_and_marker.py::test_skip_existing_re_measures_an_unqualified_marker`) | the headline median on an injected clock, the dashboard estimator / drift rule, `marker_verdict` on synthetic and the 4 committed markers, `sweep.py --skip-existing` with a stubbed subprocess | exact | SOUND (pins the fix phase 5 semantics) |
| T8 | CI test | `scripts/misc/test/test_round_bootstrap.py` (round-bootstrap witnesses: `scripts/misc/test/test_round_bootstrap.py::test_the_round_interval_covers_the_true_ratio_under_a_fixed_seed`, `scripts/misc/test/test_round_bootstrap.py::test_round_coverage_is_near_nominal_and_iid_coverage_is_not`, `scripts/misc/test/test_round_bootstrap.py::test_rounds_are_paired_across_arms`, `scripts/misc/test/test_round_bootstrap.py::test_the_committed_pytree_ral_cpu_saving_stays_below_the_bar`) | `round_median_ratio` / `round_median_saving` on seeded correlated rounds, each cell's lifted helper, and the committed C6 / C7 / C10 rows | exact, seeded | SOUND (pins the fix phase 4 semantics) |
| T10 | CI test | `scripts/misc/test/test_warmup_and_witness_band.py` (P5 / P3-dashboard witnesses: `scripts/misc/test/test_warmup_and_witness_band.py::test_a_reference_host_class_row_keeps_its_pass_or_fail`, `scripts/misc/test/test_warmup_and_witness_band.py::test_a_laptop_row_is_inconclusive_whatever_the_band_says`, `scripts/misc/test/test_warmup_and_witness_band.py::test_the_committed_witness_rows_rejudge_as_inconclusive`, `scripts/misc/test/test_warmup_and_witness_band.py::test_a_point_whose_warmup_never_settled_is_unqualified`) | `witness_verdict` on reference-class and laptop host classes, the 8 committed witness rows, `qualify` on points carrying a warm-up record | exact | SOUND (pins the fix phase 6 semantics) |
| T11 | CI test | `scripts/misc/test/test_interval_gates.py` (C1 / C3 / C4 / C5 and Holm witnesses: `scripts/misc/test/test_interval_gates.py::test_c5_a_point_saving_above_threshold_with_a_straddling_interval_is_inconclusive`, `scripts/misc/test/test_interval_gates.py::test_c3_holm_witness_unadjusted_go_but_family_wise_inconclusive`, `scripts/misc/test/test_interval_gates.py::test_c1_a_point_below_097_with_a_straddling_interval_is_inconclusive`, `scripts/misc/test/test_interval_gates.py::test_c4_status_pass_requires_a_resolved_pass`) | `interval_gates` and `holm_family_verdict` on fixed synthetic repeats with fixed seeds, the cells' wiring, and the committed C1 / C3 / C4 / C5 rows | exact, seeded | SOUND (pins the fix phase 3b semantics) |
| T12 | CI test | `scripts/misc/test/test_family_gates.py` (family-wise and remainder witnesses: `scripts/misc/test/test_family_gates.py::test_holm_witness_unadjusted_names_a_best_and_the_family_does_not`, `scripts/misc/test/test_family_gates.py::test_c6_holm_witness_unadjusted_go_but_family_wise_not`, `scripts/misc/test/test_family_gates.py::test_c11_holm_witness_unadjusted_passed_but_family_wise_inconclusive`, `scripts/misc/test/test_family_gates.py::test_c12_slow_drift_inflates_the_iid_floor_and_not_the_paired_one`, `scripts/misc/test/test_family_gates.py::test_p2_a_load_change_across_the_rows_drifts`) | `holm_tie_set`, `phase2c_family`, `kill_gate_family`, `round_split_half_mdi`, `between_row_drift` on fixed synthetic inputs and seeds, the cells' wiring, and the committed C6 / C10 / C11 / C12 / P2 rows | exact, seeded | SOUND (pins the fix phase 9 semantics) |
| P1 | production qualification | `scripts/imaging/pixelized/fixed_light_numba.py::<module>` (2122–2130; raise at 2267) via `scripts/misc/likelihood_breakdown/overhead_verdict.py::abba_overhead_verdict` | ABBA overhead in ms: one-sided t bounds vs budget; raises only on FAIL / FAIL_GROSS, INCONCLUSIVE keeps the row | `scripts/imaging/pixelized/fixed_light_numba.py::MAX_INSTRUMENTATION_OVERHEAD_MS` 12.0; `scripts/imaging/pixelized/fixed_light_numba.py::MIN_BLOCKS_FOR_OVERHEAD_ASSERT` 3; `scripts/imaging/pixelized/fixed_light_numba.py::REFERENCE_OVERHEAD_RATIO` 1.03 (recorded) | SOUND-with-caveats |
| P2 | campaign gate | `scripts/imaging/pixelized/fixed_light_numba.py::<module>` (2402–2560) via `scripts/misc/likelihood_breakdown/ab_verdict.py::ab_rule_verdict` and `scripts/misc/likelihood_breakdown/family_gates.py::between_row_drift` (559) | promotion `timing_candidate` / `INCONCLUSIVE` / `NO_LEVER` | whole-call speedup ≥ 0.05 on a 90 % paired-block interval (≥ 5 blocks) and both P1 PASS; a P1 INCONCLUSIVE gives INCONCLUSIVE; since fix phase 9 INCONCLUSIVE when the rows drift (a row's clean median moving > `scripts/misc/likelihood_breakdown/family_gates.py::ROW_DRIFT_TOLERANCE` 0.025 between its halves, or the 1-minute load moving > `LOAD_CHANGE_TOLERANCE` 2.0 across the rows) | SOUND-with-caveats |
| P3 | production protocol | `scripts/imaging/pixelized/fixed_light_numba.py::_warm_to_steady_state` (1745); read since fix phase 6 by `scripts/misc/likelihood_breakdown/warmup_gate.py` `warmup_unsettled_reason`, called by `abba_overhead_verdict` (P1), the promotion block (P2) and `build_dashboard.qualify` (P6) | warm-up "steady" flag; an unsettled row is INCONCLUSIVE for every timing verdict that reads it | `scripts/imaging/pixelized/fixed_light_numba.py::WARMUP_WINDOW` 3, `scripts/imaging/pixelized/fixed_light_numba.py::WARMUP_TOLERANCE` 0.10, `scripts/imaging/pixelized/fixed_light_numba.py::WARMUP_MAX_CALLS` 12 | SOUND-with-caveats |
| P4 | production qualification | `scripts/imaging/pixelized/fixed_light_numba.py` 2247 | unattributed fraction of the instrumented call | `MAX_UNATTRIBUTED_FRACTION` 0.05 | SOUND |
| P5 | production qualification | `_production_config.py::witness_verdict` (869; 886 since fix phase 6); `_production_config.py::timing_summary` (792) | production-representative cold-eval witness; PASS / FAIL only on a reference host class (`is_reference_host_class`), else INCONCLUSIVE (fix phase 6) | `WITNESS_FACTOR` 1.5 × the production cold-eval range | SOUND-with-caveats |
| P6 | production qualification | `scripts/misc/tooling/build_dashboard.py::qualify` (448) via `scripts/misc/tooling/build_dashboard.py::is_reference_host_class` (437) | `qualified` flag on every dashboard / v1 `profiling-summary` point (the project dashboard and badge; the live Pulse registry reads the v2 catalogue) | `RELEASE_SWEEP_LOADAVG_CAP` 8.0, `RELEASE_SWEEP_NODE`, `REFERENCE_HOST_CLASS_PREFIXES` (`hpc_`) | SOUND-with-caveats |
| P7 | production qualification | `scripts/misc/tooling/build_dashboard.py::drift` (512) and `scripts/misc/tooling/build_dashboard.py::_summary_comparison` (742) | release-to-release drift badge (drifted / improved / flat / insufficient) | `scripts/misc/tooling/build_dashboard.py::DRIFT_RATIO` 2.0 and `scripts/misc/tooling/build_dashboard.py::DRIFT_FLOOR_S` 1 ms; `flat` only with repeat-summary endpoints; endpoints on different headline estimators → `insufficient` (fix phase 5) | SOUND-with-caveats |
| P8 | estimator | `scripts/misc/likelihood_breakdown/timing.py` `jit_profile` (92) and `headline_steady_median` (269), called by the 11 runtime cells' local `jit_profile` blocks (e.g. `scripts/imaging/delaunay/likelihood_runtime.py`); read by `scripts/misc/tooling/build_dashboard.py::_point` (296) | `full_pipeline_single_jit` (legacy block mean, kept) and `full_pipeline_single_jit_median*` beside it; the dashboard headline P7 compares | block mean of 10 after the first call; median of `n_timed` individually timed calls after `scripts/misc/likelihood_breakdown/timing.py::MIN_STEADY_WARM` 5 warm calls, `n_timed` = `scripts/misc/likelihood_breakdown/timing.py::HEADLINE_MEDIAN_BUDGET_S` 30 s / block mean in [20, 200]; no median above `scripts/misc/likelihood_breakdown/timing.py::HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S` 2 s per call | SOUND-with-caveats |
| P9 | production qualification | `scripts/misc/likelihood_runtime/sweep.py` `timeout_marker` (271), `--skip-existing` (462); `scripts/misc/tooling/build_dashboard.py::marker_verdict` (476); `scripts/misc/tooling/build_readme.py` | `--per-run-timeout` writes an INCONCLUSIVE `.unusable.json` marker; "GPU-only" only for a qualified marker | the timeout; `qualify` on the marker's host and max(load at start, load at timeout) | SOUND-with-caveats |
| P10 | production qualification | `scripts/misc/tooling/baseline_readiness.py` `check_observation` (351) | structural screening of fresh baseline observations | load cap per allocated CPU, repetitions ≥ 2, warm-up, median aggregation | SOUND |
| C1 | campaign gate | `scripts/imaging/pixelized/fixed_light_numba_memo_policy.py::matched_counterfactual` (101) via `scripts/misc/likelihood_breakdown/interval_gates.py::matched_classification` (90) and `scripts/misc/likelihood_breakdown/interval_gates.py::matched_decision_counts` (149) | `classification` beneficial / harmful / neutral / INCONCLUSIVE (`warm_beneficial_3pct` / `warm_harmful_3pct` only when resolved), counted as resolved classifications plus an INCONCLUSIVE count, never as false-accept / false-reject rates | 0.97 / 1.03 on a 90 % paired round-bootstrap interval of the memo / cold median ratio, ≥ 5 repeats; the cell records `scripts/imaging/pixelized/fixed_light_numba_memo_policy.py::MATCHED_REPEATS` 4, so every classification is INCONCLUSIVE by construction | SOUND-with-caveats |
| C2 | campaign gate | `scripts/imaging/pixelized/fixed_light_numba_memo_policy.py::evaluate` (241, 248) | descriptive cross-lane 3 % classifications | 1.03 / 0.97 on medians | FRAGILE |
| C3 | campaign gate | `scripts/imaging/pixelized/fixed_light_numba_memo_policy.py::main` (486–493) via `scripts/misc/likelihood_breakdown/interval_gates.py::memo_policy_family` (200) and `scripts/misc/likelihood_breakdown/ab_verdict.py::holm_family_verdict` (509) | memo-policy verdict GO / NO_LEVER / INCONCLUSIVE | six targets (≤ 0.95 vs memo, ≤ 1.03 vs cold / memo), each a 90 % paired round-bootstrap interval over the repeats, one Holm family at family-wise 90 %: GO only if all six resolve in favour, NO_LEVER only if one resolves against | SOUND-with-caveats |
| C4 | campaign gate | `scripts/imaging/pixelized/fixed_light_numba_scaling.py::evaluate_group` (160) via `scripts/misc/likelihood_breakdown/interval_gates.py::breakdown_reconciliation` (257) and `scripts/misc/likelihood_breakdown/interval_gates.py::conjoin_reconciliations` (300); status by `scripts/imaging/pixelized/fixed_light_numba_scaling.py::run_status` (192) | `breakdown_reconciles` PASS / FAIL / INCONCLUSIVE; status PASS only on a resolved PASS in every lane, INCONCLUSIVE when only the reconciliation is unresolved | observed / clean median − 1 within ±0.05 on a 90 % paired round-bootstrap interval over the repeats (≥ 5) | SOUND-with-caveats |
| C5 | campaign gate | `scripts/imaging/pixelized/fixed_light_trace.py::<module>` (3708, 3758–3761) via `scripts/misc/likelihood_breakdown/interval_gates.py::logdet_lever_verdict` (316) | logdet lever `clears_threshold` True / False only when resolved, else `"INCONCLUSIVE"` | `scripts/misc/likelihood_breakdown/logdet_reuse_injection.py::LOGDET_LEVER_MS` 0.5 ms on both estimators: the interleaved saving on a 90 % paired round-bootstrap interval (7 rounds); the one-block `jit_profile` saving has no interval, so a lever resolves only as False or INCONCLUSIVE | SOUND-with-caveats |
| C6 | campaign gate | `scripts/point_source_source/source_plane/backward_pass_ab.py::_phase2c_rule` (1003) | phase-2c GO | `scripts/point_source_source/source_plane/backward_pass_ab.py::GO_MIN_SAVED_MS` 0.05 ms and `scripts/point_source_source/source_plane/backward_pass_ab.py::GO_MIN_FRACTION` 0.15, point estimate and 90 % CI; GO / NO_GO / INCONCLUSIVE via `ab_rule_verdict` since fix phase 3, on paired round-bootstrap intervals since fix phase 4; since fix phase 9 judged family-wise via `scripts/misc/likelihood_breakdown/family_gates.py` `phase2c_family` (113): the host's timed routes × lanes × both criteria are one Holm family at family-wise 90 %, a route GO only when GO in every lane | SOUND-with-caveats |
| C7 | campaign gate | `scripts/point_source_source/source_plane/pytree_input_ab.py::_phase2b_rule` via `scripts/misc/likelihood_breakdown/ab_verdict.py::ab_rule_verdict` | phase-2b GO / NO_GO / INCONCLUSIVE | `scripts/point_source_source/source_plane/pytree_input_ab.py::GO_MIN_SAVED_MS` 0.05 ms and `scripts/point_source_source/source_plane/pytree_input_ab.py::GO_MIN_FRACTION` 0.15, on 90 % paired round-bootstrap intervals (≥ 5 rounds) | SOUND-with-caveats |
| C8 | campaign gate | `scripts/point_source_source/source_plane/gradient_mode_crossover.py` `_crossover` (763) | crossover summary ("rev wins from…", "n* = …") | ratio = 1 crossing of point medians; bootstrap fractions and `n*` CI from paired round bootstraps since fix phase 4 | FRAGILE |
| C9 | campaign gate | `scripts/point_source_image/image_plane/solver_config_sweep.py::<module>` (1934–1943) | MCS `rule_candidates` | compile ≤ 1.20, median ≤ 1.05 vs control (point; its round-bootstrap interval `row_over_control_median_ci90` recorded beside it since fix phase 4) | FRAGILE |
| C10 | campaign gate | `scripts/point_source_image/image_plane/solver_config_sweep.py` `_fastest` via `scripts/misc/likelihood_breakdown/ab_verdict.py::tie_set` | `best_admissible` (a resolved leader, else `None`) and `best_admissible_tie_set` | tie set at the family-wise level since fix phase 9 (`scripts/misc/likelihood_breakdown/family_gates.py` `sweep_tie_set` (273) → `scripts/misc/likelihood_breakdown/ab_verdict.py` `holm_tie_set` (702)): a `best` only when the leader's paired round-bootstrap speed-up interval separates from every other candidate's at the Holm-adjusted level; the unadjusted 90 % tie set recorded beside it; no `best` below 5 rounds | SOUND-with-caveats |
| C11 | campaign gate | `scripts/misc/numba_interferometer/bakeoff.py::main` (858) via `scripts/misc/likelihood_breakdown/family_gates.py` `kill_gate_family` (354); the old point rule kept as `scripts/misc/likelihood_breakdown/family_gates.py::kill_gate_point` (330) | numba-vs-rfft2 kill gate `passed` / `tripped` / INCONCLUSIVE | rfft2 / kernel > 1.3 on a paired round-bootstrap interval per pinned kernel × sma / alma cell, one Holm family at family-wise 90 %: `passed` if any member resolves above, `tripped` only if every member resolves below; ≥ 5 timed rounds (the default `--reps 5` times 4, so INCONCLUSIVE by construction) | SOUND-with-caveats |
| C12 | campaign gate | `scripts/point_source_image/image_plane/gpu_bottleneck_map.py` `_mdi` (396) via `scripts/misc/likelihood_breakdown/family_gates.py` `round_split_half_mdi` (483) | minimum detectable improvement (split-half noise floor) for a human go/no-go | 90 % CI half-width; since fix phase 9 of a paired round bootstrap of each round's first half of calls over its last half (NaN below 5 rounds) | SOUND |
| S1 | submit basis | `scripts/misc/wall/check_submits.py::RATE_TOLERANCE` (91), `HEADROOM_FLOOR` (95), `budget < needed` (437) | `--time` ≥ estimated wall × headroom | 5 % rate match; headroom 1.25 / 1.5 / 3.0 | SOUND |
| R1 | resource guard | `scripts/imaging/pixelized/fixed_light.py::measure_system` (773), `scripts/imaging/pixelized/fixed_light.py::COND_BUDGET_S` 5 s | skip the second `cond()` | 5 s | SOUND |
| R2 | resource guard | `scripts/misc/numba_interferometer/bakeoff.py::run_cell` (430), `scripts/misc/numba_interferometer/bakeoff.py::ALMA_HIGH_PAIR_LOOP_BUDGET_S` 180 s | skip an extrapolated alma_high kernel | 180 s | SOUND |

**Counts.**

| Kind | Rows | SOUND | FRAGILE | UNSAFE-SILENT |
|---|---|---|---|---|
| CI test | 12 | 12 | 0 | 0 |
| Production qualification / protocol / estimator | 9 | 9 | 0 | 0 |
| Campaign gate (incl. P2 promotion) | 13 | 10 | 3 | 0 |
| Submit basis | 1 | 1 | 0 | 0 |
| Resource guard | 2 | 2 | 0 | 0 |
| **Total** | **37** | **34** | **3** | **0** |

Counts after fix phase 9, which added the CI-test row T12 and moved P2 and C11 from FRAGILE to
SOUND-with-caveats (counted as SOUND, like T1, P1, P3, P5–P9, C1, C3–C7 and C10). After fix phase
3b the totals were 31 / 5 / 0 over 36 rows (T11 added; C1, C3, C4, C5 UNSAFE-SILENT →
SOUND-with-caveats). After fix phase 6 the totals were 26 / 5 / 4 over 35 rows (T10 added; P3 and P5 FRAGILE →
SOUND-with-caveats). After fix phase 5 the
totals were 23 / 7 / 4 over 34 rows (T9 added; P8 FRAGILE and P9 UNSAFE-SILENT →
SOUND-with-caveats). After fix phase 4 the totals were 20 / 8 / 5 over 33 rows (T8 added; C6, C7, C10 FRAGILE →
SOUND-with-caveats). After fix phase 3 the totals were
16 / 11 / 5 over 32 rows (T7 added; C7, C10 and P2 UNSAFE-SILENT → FRAGILE). After fix
phase 2 the totals were 15 / 8 / 8 over 31 rows; after fix phase 1, 13 / 9 / 9 (P6 UNSAFE-SILENT,
P7 FRAGILE); at phase 1, 11 / 10 / 10 (T1 FRAGILE, P1 UNSAFE-SILENT). The production row group is
P1 and P3–P10, nine rows. P2 is a promotion rule, so it is counted with the campaign gates. No
row can still label a result from a point estimate or without looking at the measurement (at
phase 1 there were ten; C1, C3, C4 and C5 left with fix phase 3b). The three FRAGILE rows (C2, C8,
C9) are descriptive labels or human-picked candidates whose intervals are recorded beside a point
headline; none qualifies a result by itself, and none is in the phase 9 / 10 plan.

## Per-row detail

Each entry gives: what is measured; estimator; samples, repeats and warm-up; pairing and order;
clock; host qualification; cutoff and the justification the code states; the noise model the
cutoff implies; false-positive risk (a flaky FAIL or a spurious GO); false-negative risk (a real
regression passes, or a real lever is reported as NO_LEVER); owner; and the verdict.

### T1 — `test_call_accounting_covers_a_real_likelihood_call` (CI test)

- **Role after phase 2 (human decision 2026-10-08):** coverage, cached-site-count and gross-breakage guard. On this ~15–17 ms fixture the shared 12 ms budget can only fail through the 1.5 gross guard; the ms budget is resolved on the cell's 225–415 ms production rows by the same function. Documented in the test's docstring.
- **Measured:** ABBA ratio of instrumented to clean `FitImaging.figure_of_merit` wall time on the
  30×30, 67-parameter tiny fixture, sparse-numba S3 system.
- **Estimator:** mean of 3 block ratios, each `mean(B1, B2) / mean(A1, A2)`. One-sided 95 %
  Student-t bounds, df = 2.
- **Samples / warm-up:** 1 untimed warm-up call (numba compile), then 3 blocks × 4 calls.
- **Pairing:** ABBA within each block, so linear drift cancels. The NNLS memo is cleared before
  every call.
- **Clock:** `time.perf_counter`, CPU only, with no device synchronisation needed.
- **Host:** none. It runs on any GitHub `ubuntu-latest` runner or any laptop. Load is not recorded.
- **Cutoff:** `CI_OVERHEAD_RATIO = 1.031`; FAIL_GROSS when the mean exceeds 1.5. The code states
  two justifications: "contended host" (the 1.03 → 1.031 relaxation, `d5462d4`), and #361's "fail
  only a resolved exceedance".
- **Noise model:** block ratios iid normal. The CI scatter observed in #361 was s.d. 0.014.
- **FP risk:** low. FAIL requires the lower bound to exceed the budget. With heavy-tailed blocks
  the s.d. inflates and the bound widens, which makes FAIL rarer still.
- **FN risk:** high. FAIL needs a mean above about 1.054 at CI scatter, so overheads between 3.1 %
  and about 5.4 % pass as INCONCLUSIVE. INCONCLUSIVE is a `print` that pytest captures, so it is
  silent.
- **Owner:** test helper. It does **not** share code or units with the production gate P1, which
  uses a 12 ms point estimate. The same instrument has two budgets in two units, and nothing
  reconciles them.
- **Also asserted:** coverage ≥ 0.95 and cached-site counts. These are SOUND: coverage is a ratio
  taken within the same instrumented calls, and the counts are deterministic.
- **Observed during this audit (laptop, WSL2, i9-10885H; load average 3.7–5.5; not CI):** three
  isolated runs of this test gave:
  - blocks [1.061, 1.210, 1.108], INCONCLUSIVE, bounds [0.998, 1.255];
  - blocks [0.852, 1.023, 0.878], INCONCLUSIVE;
  - blocks [0.711, 0.923, 0.738], **PASS**, upper bound 0.985.

  The third is a PASS on a mean ratio of 0.79, which says the instrumented call ran 21 % *faster*
  than the clean call. No instrumentation can do that, so the rule passed on a measurement that
  fails a basic validity check. One run of the whole `scripts/misc/test/` suite failed this test.
  The run used `-x` and the message was not captured, so the failing verdict is unverified. A
  rerun of the whole suite passed (1140 passed, 6 skipped). On this host the test is flaky in both
  directions.
- **Verdict (phase 1):** FRAGILE. It qualifies no result, but:
  - it has almost no power against the budget it names;
  - its INCONCLUSIVE is invisible in CI;
  - it can PASS on physically impossible ratios below 1;
  - its budget unit diverges from the instrument's.
- **After fix phase 1: SOUND-with-caveats.** The test calls the cell's own
  `abba_overhead_verdict` (identity asserted) with the cell's 12 ms budget, converted through the
  fixture's own clean mean. FAIL / FAIL_GROSS fail the test; INCONCLUSIVE (including a mean ratio
  resolved below 1) raises a visible `OverheadInconclusiveWarning` in the pytest summary.
  `CI_OVERHEAD_RATIO` and `_ci_overhead_verdict` are gone. Caveats:
  - the fixture's clean call is ~15–17 ms (laptop, 2026-10-08), so the 12 ms budget is ~75 % of
    the call. A ms FAIL needs a lower bound above 12 ms, which means a mean ratio above ~1.8; the
    1.5 gross guard fires first. On this fixture the test therefore fails only on FAIL_GROSS. That
    is the cost of judging the instrument in its own unit: the instrument costs ~1.4–3.4 ms here,
    a large ratio of a short call, and a 3.1 % ratio budget would fail it on every quiet host
    (one laptop run resolved x1.22, bounds [+2.4, +4.4] ms);
  - three blocks, so the interval assumes iid normal block ratios it cannot check.

### T2–T6 (CI tests, SOUND)

- **T2** `test_call_accounting.py`: `abs(Σ excl − incl_root) ≤ 1e-9`, `excl_s > 0` after a 4 ms
  busy-burn, `incl ≤ wall + 1e-9` around the same calls. These are arithmetic identities over the
  same clock reads, so host noise cannot break them. These keys are lister `validity` hits.
- **T3** `test_fixed_light_numba.py` 911–1182 executes the cell's own gate statements, lifted from
  its AST, on synthetic ratios, call lengths and warm-up sequences. It is deterministic. It pins
  the P1/P3 constants, so any change to them has to be made deliberately.
- **T4** `test_timing_steady_median.py` uses an injected clock. It is deterministic and includes
  the "median robust to a transient, block mean not" witness for P8.
- **T5** `test_wall_check_submits.py` is deterministic arithmetic on `wall/rates.py`.
- **T6** `test_build_dashboard.py` uses synthetic rows. It is deterministic. It pinned the phase 1
  P6/P7 semantics; fix phase 2 updated it on purpose and added the fix phase 2 witnesses.

### P1 — fixed-light numba ABBA overhead gate (production qualification)

- **Measured:** `overhead_ms = (mean ABBA ratio − 1) × clean mean call ms`, for every decomposed
  row of `fixed_light_numba.py`.
- **Estimator:** point estimate (mean of block ratios × mean of clean calls), with no interval.
- **Samples:** `n_blocks = ceil(N_REPEATS / 2)` blocks of ABBA (4 calls each), after the P3
  warm-up.
- **Pairing:** ABBA. Memos are cleared within blocks.
- **Clock:** `perf_counter`. Single-thread pinning (`pin_thread_env`).
- **Host:** none gating. `/proc/loadavg` is recorded at the start and end of the row, but not
  checked.
- **Cutoff:** 12.0 ms. The stated justification is "1.03 × the ~400 ms call it was set on;
  measured fixed cost 6–8 ms, so the budget sits ~1.5× above". `MIN_BLOCKS_FOR_OVERHEAD_ASSERT = 3`;
  below that the status is RECORDED. That rule is stated as "three blocks put the SE of the mean
  block ratio near 1 %".
- **Noise model:** an SE of about 1 % of a 225–415 ms call, which is 2–4 ms against a 4–6 ms
  margin. The cell documents ±15 % call-to-call scatter.
- **FP risk:** moderate. A FAIL raises `AssertionError`, and the row and its JSON are lost on
  noise. This already happened once with the ratio gate (RAL job 343355).
- **FN risk:** moderate. A real 13–15 ms overhead can PASS on a low draw, and nothing records that
  the PASS was unresolved.
- **Owner:** production instrument. It diverges from T1.
- **Verdict (phase 1):** UNSAFE-SILENT. A PASS from a point estimate is written as
  `instrumentation_overhead_status: PASS`, and P2 reads it as a qualification.
- **After fix phase 1: SOUND-with-caveats.** The status is the shared verdict's PASS / FAIL /
  FAIL_GROSS / INCONCLUSIVE, with `instrumentation_overhead_lower_ms`, `_upper_ms`,
  `_confidence` and `_verdict_reason` beside it. The cell raises only on FAIL or FAIL_GROSS;
  INCONCLUSIVE (fewer than 3 blocks, a straddling interval, or a mean ratio resolved below 1)
  keeps the row and its JSON. `RECORDED` is retired for this field (no reader consumed it).
  Caveats: the interval assumes iid normal block ratios; at the 8-block legs of levers 1–3 the
  interval is ~±13 ms wide, so those rows would now read INCONCLUSIVE (see the re-judged rows
  under fix phase 1); the gross guard (mean ratio > 1.5) now applies to the cell too and fires
  even below 3 blocks.

### P2 — fixed-light numba promotion decision (campaign gate)

- **Measured:** `(b.call_ms − d_perm.call_ms) / b.call_ms`. Each `call_ms` is a row's clean mean.
- **Estimator:** difference of two point means, with no interval.
- **Pairing:** none. The rows are measured one after the other, in separate passes.
- **Cutoff:** speedup ≥ 0.05 and both P1 statuses PASS. Justification: "the ≥ 5 % whole-call
  timing rule". Since fix phase 1, a speedup ≥ 0.05 with either P1 INCONCLUSIVE is written as
  status `INCONCLUSIVE` (with `abba_unresolved_rows`), never `timing_candidate` and never a
  measured NO_LEVER. The speedup itself is still a point difference (fix phase 3).
- **FP risk:** reduced by `requires_separate_witness_pass`; `promotion_ready` stays false.
- **FN risk:** high. Drift between rows can hide a 5–10 % lever, and the result is then written as
  "a valid **measured** NO_LEVER result, not a failed job".
- **Verdict (phase 1):** UNSAFE-SILENT, because an unresolved difference is published as a measured
  NO_LEVER.
- **After fix phase 3: FRAGILE.** The ≥ 5 % rule reads a 90 % two-sided Student-t interval on the
  per-block speedup, `1 − mean_j(d_perm block j / b block j)`, through the shared
  `ab_rule_verdict` (≥ `MIN_AB_ROUNDS` = 5 blocks). The rows walk the same validated instance
  stream, so block j of each row evaluates the same instances and the blocks are paired by
  instance (a decomposed row has two clean calls per block). NO_LEVER now requires the whole
  interval below 5 %; a straddling interval or fewer than 5 blocks is INCONCLUSIVE with the
  resolvable effect recorded (`whole_call_speedup_paired_blocks`, `speedup_verdict`,
  `resolvable_effect`); the P1 conjunction is unchanged. Still FRAGILE: the two rows are separate
  passes in time, so drift between them is not cancelled and the interval does not cover it.
- **After fix phase 9: SOUND-with-caveats.** The block records `between_row_drift`
  (`family_gates.between_row_drift`): each row's clean-call median of its last half over its first
  half (drift if beyond ±2.5 %, half the 5 % bar) and the range of every recorded 1-minute load
  (the run's start / end and, from now on, each row's `load_average_at_row_start` / `_end`; drift
  if > 2.0). A drifting host makes the promotion INCONCLUSIVE before the timing-candidate /
  NO_LEVER branches. Caveats: a step change between the rows that leaves both rows internally flat
  and the load unchanged is still invisible (only an interleaved design cancels it), and a missing
  load is recorded as unassessed, not as drift.

### P3 — warm-up to steady state (protocol)

- **Rule:** the median of the last 3 calls must be within 10 % of the median of the previous 3,
  with at most 12 calls. The whole sequence is recorded, and "NEVER SETTLED" is logged.
- **Noise model:** with the cell's documented ±15 % scatter, a 3-vs-3 median comparison at 10 %
  can stop early on a ramp or run to the cap on a flat stream. The `steady` flag is noisy in both
  directions.
- **Verdict:** FRAGILE. It is recorded and does not gate, but the rows after it do not condition
  on it.
- **After fix phase 6: SOUND-with-caveats.** One shared check,
  `likelihood_breakdown.warmup_gate.warmup_unsettled_reason`, returns a reason beginning "warm-up
  never settled" for a record whose `steady` is false or missing, and `None` for a settled record
  or no record. Every timing verdict that reads a post-warm-up row applies it: the ABBA overhead
  verdict (P1, `warmup=` keyword; after the gross guard, so FAIL_GROSS still fires), the promotion
  decision (P2; INCONCLUSIVE whatever the speedup interval, with `warmup_unsettled_rows`) and
  dashboard qualification (P6; unqualified with the reason, for any payload carrying a warm-up
  record). P4 (unattributed fraction), the cached-site count and the dispatch asserts read the
  same rows but are coverage / correctness checks whose numerator and denominator come from the
  same calls; they are not timing verdicts and are not weakened. The `steady` flag and the
  sequence are recorded as before. **The rule is kept** (3 vs 3, 10 %, 12 calls). Its limits,
  pinned as witnesses: a ramp of ≤ 3 % per call moves ≤ 9 % per window and is called steady at
  call 6 (seeded, with 5 % noise on top: settles 99 % of the time at 3 %/call, 77 % at 4 %/call);
  a step after the warm-up stops is never seen by it (the ABBA design cancels only linear drift
  within a block, and P2's between-row drift is P2's own caveat); a stationary period-2 ±15 %
  oscillation never settles. iid scatter rarely fails it (seeded: 0 % at 5 % s.d., 0.4 % at 10 %,
  2.3 % at 15 %), and a false "unsettled" now costs only a re-run, never a false verdict. A
  stricter rule would need measured ramps to justify it; none of the 28 committed warm-ups is
  unsettled.

### P4 — unattributed fraction ≤ 5 % (SOUND)

Both numerator and denominator come from the same instrumented calls, so host drift cancels to
first order. The cutoff is a design budget for the site spec, not a noise budget.

### P5 — production-representative witness (`_production_config.witness_verdict`)

- **Measured:** the median of `n_cold` cold evaluations, after one discarded compile call
  (`timing_summary`). It is compared to the logged production cold-eval range: Euclid job 342301,
  0.69–1.13 s; HST job 342311, 0.52–0.81 s.
- **Cutoff:** a band from range low / 1.5 to range high × 1.5, set by human decision 2 on #235
  (2026-09-08).
- **Host:** none. A laptop row is judged against an 8-core RAL production range.
- **Noise:** the band is wide, about 2.3–3.3× end to end, so call noise rarely flips it. Host
  class can.
- **Callers:** `scripts/imaging/{rectangular,delaunay}/likelihood_{runtime,breakdown}_numba.py`.
- **Verdict:** FRAGILE. The PASS / FAIL is binary with no INCONCLUSIVE, and host class is not part
  of the rule.
- **After fix phase 6: SOUND-with-caveats.** `witness_verdict(cold_eval_median_s, instrument,
  host_class)` judges PASS / FAIL only when the host class (the row's `--config-name`) passes
  `build_dashboard.is_reference_host_class`, the rule P6 and P9 use. Otherwise the verdict is
  INCONCLUSIVE with the reason "off reference host class (<class>)" (`untagged` when the cell ran
  without `--config-name`). The record gains `host_class`, `in_band` (where the median fell,
  whatever the host) and `reason`. The band is unchanged (human decision 2 on #235). Caveats: the
  class is the config label, not a measured host; the verdict is one median of `n_cold` calls
  with no interval (the 2.3–3.3× band dwarfs call noise); and a reference-class miss can still be
  dataset realism rather than configuration (`production_representative_cells.md`).

### P6 — dashboard / `profiling-summary` qualification (`build_dashboard.qualify`)

- **Rule:** a row is refused if its load average exceeds 8.0. It is unqualified if it has no
  provenance block. An `hpc_*` row is unqualified if its host differs from `RELEASE_SWEEP_NODE`.
- **Gaps (verified by calling `qualify` on synthetic points):**
  1. a `local_cpu_*` laptop row with provenance is **qualified**, although
     `hpc/release_sweep.conf` itself says "the laptop drifts 2.5× between runs";
  2. a row whose provenance carries no load average is qualified;
  3. an `hpc_*` row with `host: None` passes the host pin.
- **Measurement content:** none. Qualification never looks at sample count, spread or estimator.
  It is a provenance check.
- **Downstream (corrected after fix phase 2):** the v1 `dashboard/summary.json` feeds the project
  dashboard and badge. The live PyAutoPulse registry has read `dashboard/catalogue.json`
  (`profiling-summary@2`) since the registry switch of 2026-10-05 (PyAutoPulse `3c7bed2`), and its
  producer `scripts/misc/tooling/build_catalogue.py` hard-codes `qualified: False`. So P6's gap did
  not flow into Pulse live, and fixing it is not a Pulse contract change. The phase 1 text
  ("Pulse requires `qualified: true` … so P6's gaps flow into the organ") described the v1 path.
- **Today:** 0 of 97 local records and 2 of 62 hpc records in `dashboard/summary.json` are
  qualified, so the gap is latent. It applies to the first laptop row written with a provenance
  block.
- **Verdict (phase 1):** UNSAFE-SILENT.
- **After fix phase 2: SOUND-with-caveats.** `qualify` checks, in order: above the cap → refused;
  no provenance; not a reference host class (`is_reference_host_class`, `hpc_*` only: "laptop
  rows never qualify as trend points"); no load average; no host; off the pinned node. Each is
  unqualified with that reason. The caveat: qualification is still a provenance property and
  never looks at sample count or spread. A future v2 (`catalogue.json`) qualification must reuse
  the same host-class rule rather than a second copy.

### P7 — release drift badge (`build_dashboard.drift`)

- **Measured:** the ratio of the last two releases' `single_jit_s`, one number per release (P8).
- **Cutoff:** ≥ 2.0× and ≥ 1 ms is `drifted`; ≤ 0.5× and ≥ 1 ms is `improved`; anything else is
  `steady`, published as `flat`. The ratio is borrowed from the Brain profiling conductor's
  `COMPILE_DRIFT_RATIO` ("generous on purpose: host load alone has produced 7× errors").
- **Noise model:** none, beyond the generous band. A qualified 1.9× regression is published as
  `flat` (verified synthetically).
- **FP risk:** P8's documented post-compile transient (A100 source plane: block mean 0.642 ms
  against a steady median of 0.267 ms, 2.4×) on its own exceeds the 2× band.
- **Comparison qualification:** the `qualified` on a comparison is the conjunction of its two
  endpoints' P6 flags, and nothing more.
- **Verdict (phase 1):** FRAGILE. The band is declared and generous, but `flat` names a policy
  band, not a measured null, and the estimator underneath can produce 2× by itself.
- **After fix phase 2: SOUND-with-caveats.** Inside the band is `flat` ("within the 2x policy
  band; not a measured null") only when both endpoints carry a repeat summary of the compared
  metric; otherwise `insufficient` ("single-sample endpoint(s): within the 2x policy band is not a
  measured null"). No producer writes a repeat summary yet, so every in-band comparison is
  `insufficient` today. `drifted` / `improved` keep their status as gross-band flags with the
  reason "single-sample endpoint(s)" (human decision 2026-10-08). The caveat: the estimator (P8)
  can still produce a 2× flag by itself until fix phase 5. Since fix phase 5 a series headlined by
  the steady median compares medians only; a median against a block-mean endpoint is
  `insufficient` ("endpoints use different headline estimators"). Legacy-only series keep the
  block-mean caveat until their cells are re-run.

### P8 — single-jit headline estimator

`jit_profile`'s `steady_per_call_s` is the mean of one block of 10 calls, taken right after the
first call. That makes it a single sample of a block mean, with no repeats. The bias is documented
in `timing.py`'s own docstring and is "kept unchanged for continuity". The opt-in
`steady_median_profile` (≥ 5 warm and N timed calls, median with p10/p90) is the noise-aware
alternative. Several cells keep local copies of `jit_profile`, for example
`scripts/imaging/delaunay/likelihood_runtime.py`. **FRAGILE** (phase 1).

**After fix phase 5: SOUND-with-caveats.** The 11 runtime cells whose headline is the
`jit_profile` block call `headline_steady_median` on the compiled pipeline right after reading
the block, and write `full_pipeline_single_jit_median` (s), `_median_ms`, `_p10_ms`, `_p90_ms` and
`_median_protocol` beside the unchanged `full_pipeline_single_jit`. The dashboard headline is the
median where present, labelled "steady median", with the block mean beside it; drift never
compares a median with a block mean. Caveats: one median is still one run's summary, so a
comparison is still single-sample (P7); no committed row carries a median yet; a call slower than
2 s keeps the legacy headline (the median's 25 extra calls would cost minutes); the breakdown cells
(`interferometer/mge/likelihood_breakdown.py`, `misc/likelihood_breakdown/interferometer_pixelized.py`),
the datacube cell (`scripts/datacube/delaunay/likelihood_runtime.py`, a 3-call cube block,
`full_pipeline_cube_single_jit`) and the two `mge_mass` cells (which already write a median *under*
the legacy key) are not opted in. See (b) item 5.

### P9 — sweep `--per-run-timeout` → "GPU-only"

A single wall-clock run that exceeds the timeout writes `.unusable.json`. The dashboard renders it
as "GPU-only", and `--skip-existing` honours it, so the cell is never re-measured. It is one
sample on an unqualified host (the laptop is documented at 7× load error). **UNSAFE-SILENT**
(phase 1). A categorical scientific label comes from one noisy observation and persists.

**After fix phase 5: SOUND-with-caveats.** A timeout writes `verdict: INCONCLUSIVE` with the host,
the load average at the start of the run and at the timeout, the timeout and the elapsed time.
`build_dashboard.marker_verdict` renders "GPU-only" only when the marker passes `qualify` (so
`is_reference_host_class`, a recorded host and load, the higher load ≤ the cap, the pinned node);
anything else renders "timed out (inconclusive)" or "did not finish (inconclusive)", and
`sweep.py --skip-existing` re-measures it. Caveat: a qualified marker is still one observation on
a qualified host (qualification is provenance, not statistics, as for P6), and `sweep.py` runs
only `local_*` configs, so no marker it writes can qualify today. See (b) item 5.

### P10 — baseline readiness screening (SOUND)

`check_observation` requires a load per allocated CPU under a frozen cap, no GPU peers, a fresh
process, witnessed synchronisation, a warm-up count, raw samples and a median aggregation. Its
output is `candidate_for_human_review`, and its report states "structural screening only". One
limit: repetitions ≥ 2 is a weak floor for a median, but nothing is accepted automatically.

### C1–C3 — CPU memo policy (`fixed_light_numba_memo_policy.py`)

- **Design:** six lane orders (a Latin-square-like schedule) cycled across `--n-repeats` (default
  6). Each traversal uses fresh caches. Single-thread pinning. `perf_counter_ns`. The estimator is
  the median of repeats. The load average is recorded after the run.
- **C1:** `matched_counterfactual` is solver-only, with 4 paired repeats. It classifies memo/cold
  < 0.97 as beneficial and > 1.03 as harmful. Those flags are summed into `false_accepts` and
  `false_rejects`, which are written as if they were measured decision errors. There is no
  interval: a 3 % band on 4 repeats is inside call scatter. **UNSAFE-SILENT.**
- **C2:** cross-lane 3 % classifications. They are explicitly labelled "descriptive …, not causal
  false-decision rates". **FRAGILE.**
- **C3:** six point-ratio targets must all hold for GO; otherwise the verdict is NO_LEVER. The
  conjunction lowers the false-GO rate but raises the false-NO_LEVER rate, and no multiple-
  comparison logic is stated. NO_LEVER is published as a verdict. **UNSAFE-SILENT.**
- **Since fix phase 3b:** C1 classifies a transition only when the 90 % paired round-bootstrap
  interval of memo / cold (repeat = round) resolves against the 3 % band, and the counts are
  resolved classifications plus an INCONCLUSIVE count; with the cell's 4 repeats (< 5) every
  classification is INCONCLUSIVE by construction. C3's six targets are intervals over the
  repeats judged as one Holm family; GO / NO_LEVER only when resolved. Both
  **SOUND-with-caveats** (caveats: C1 cannot resolve until the repeats reach 5; C3's bootstrap
  has 6 rounds, so Holm's first level, 98.3 %, sits near the extremes of 462 distinct resamples).

### C4 — numba scaling breakdown reconciliation

`|median(instrumented totals) / median(clean totals) − 1| ≤ 0.05` feeds `breakdown_passed`, which
feeds `status: PASS`. The two medians come from interleaved but separate traversals. There is no
interval, and the 5 % band is a point comparison. **UNSAFE-SILENT** on the PASS side. The FAIL
side is loud: `INCOMPLETE_OR_FAIL`.

**Since fix phase 3b:** `breakdown_reconciles` is PASS / FAIL / INCONCLUSIVE on a 90 % paired
round-bootstrap interval over the repeats (the clean and instrumented traversals of a repeat
pair); status PASS needs every lane resolved PASS, and an unresolved reconciliation with every
other gate green is status INCONCLUSIVE. **SOUND-with-caveats** (the four lanes of a cell are a
conjunction at 90 % each, conservative for PASS; no family-wise adjustment).

### C5 — logdet lever `clears_threshold` (`fixed_light_trace.py`)

The rule is pre-registered (#303): a lever clears when **both** the `jit_profile` saving and the
interleaved-median saving are ≥ 0.5 ms against the same-process library control. Two point
estimates of the same quantity both clearing is weaker evidence than it looks: their errors share
the control. The code labels it "Never a gate", but it qualifies a lever "worth a PyAutoArray
prompt". **UNSAFE-SILENT.**

**Since fix phase 3b:** the rule reads intervals. The interleaved saving gets a 90 % paired
round-bootstrap interval over its 7 rounds (a round = one 10-call block per arm); the
`jit_profile` saving is one block per arm and has no interval, so that estimator is INCONCLUSIVE
by construction. Their conjunction (`ab_verdict.conjoin_verdicts`) makes `clears_threshold`
`False` when the interleaved interval resolves below 0.5 ms and `"INCONCLUSIVE"` otherwise; with
this protocol it can never be `True`. **SOUND-with-caveats** (that limit is the caveat).

### C6 — backward-pass phase-2c rule (FRAGILE; the best-designed gate found)

- **Rule:** GO needs the point estimate to clear the bar, the 90 % bootstrap CIs to exclude it
  (saved-ms CI low ≥ 0.05 ms, ratio CI high ≤ 0.85), and the correctness gate green. Only
  `hpc_ral_cpu_fp64` decides. The order is round-robin with the start rotated each round.
- **Weaknesses:**
  1. the bootstrap resamples individual calls iid, ignoring the round structure, so with
     within-round autocorrelation the CI is too narrow;
  2. the resampling is unpaired although rounds pair the routes;
  3. there is no INCONCLUSIVE state, so an overlapping CI reads `go: false`;
  4. several routes and lanes are tested without correction.
- **After fix phase 3:** weakness 3 is fixed. The rule is the shared `ab_rule_verdict` on the same
  two criteria and bars (saved ≥ 0.05 ms, ratio ≤ 0.85) with the correctness gate as a gate: a red
  gate is NO_GO; an overlapping CI is INCONCLUSIVE with its resolvable effect, not `go: false`
  as a measured negative. Weaknesses 1, 2 (fix phase 4) and 4 remain, so it stays FRAGILE.
- **After fix phase 9:** weakness 4 is fixed. The phase-2c decision picks the GO routes, so one
  host's run is one family: every timed (route, lane) × {saved ms, ratio} criterion (16 on a full
  run) under Holm at family-wise 90 %, read from paired round draws with the cell's seeds (equal to the
  recorded 90 % intervals for rows written since fix phase 4; the older committed rows store iid
  intervals, so their re-judgement recomputes the round draws). A (route, lane) is GO when both its criteria resolve in favour at
  their Holm levels; a route is GO for the host only when GO in every lane (`per_route`,
  `go_routes`); `verdict_unadjusted` keeps the per-criterion 90 % reading. Caveat: only one host
  decides; other hosts are separate families, recorded, not pooled.

### C7 — pytree phase-2b rule (UNSAFE-SILENT)

GO is decided on the point estimate only (`saved ≥ 0.05 ms and ≥ 15 %`). The 90 % CIs are
computed and printed beside it but are not part of the rule.

**After fix phase 3: FRAGILE.** The cell now bootstraps the saved milliseconds as well
(`saved_ms_pytree_minus_flat_vector_ci90`, the C6 estimator) and maps the ratio's interval to the
fraction saved (`1 − 1/ratio`). `_phase2b_rule` is the shared `ab_rule_verdict` on both criteria
(bars unchanged; ≥ 5 rounds), written per lane as `verdict` (GO / NO_GO / INCONCLUSIVE),
`verdict_reason` and `resolvable_effect`; `go` is true only for GO. Still FRAGILE: the
bootstrap resamples calls iid (fix phase 4).

**After fix phase 4: SOUND-with-caveats.** Both intervals are the paired whole-round bootstrap
(`round_median_ratio`, `round_median_saving`; `effective_n` = 20 rounds). Caveats: the two lanes
and two criteria are judged at 90 % each with no family-wise adjustment, and only the RAL CPU row
decides.

### C8 — gradient-mode crossover (FRAGILE)

The summary string comes from the point-median ratio curve. The bootstrap fraction of each outcome
and an `n*` 90 % CI are recorded beside it, so the uncertainty is published, but the headline
string does not carry it.

**After fix phase 4:** the per-rung bootstrap curves are paired whole-round bootstraps
(`round_median_ratio(..., return_boots=True)`), so the bootstrap fractions and the `n*` interval
no longer assume iid calls. Still FRAGILE: the headline string reads the point curve.

### C9, C10 — point-source solver sweep (`solver_config_sweep.py`)

- **C9:** `rule_candidates` uses point thresholds (compile ≤ 1.20, median ≤ 1.05 against control).
  The decision rule says "human picks N at the step-2 checkpoint". **FRAGILE.** Since fix phase 4
  the row records `row_over_control_median_ci90` (the round-bootstrap interval) beside the point
  the rule reads; the rule itself is unchanged (the human picks; IP-4c chose MCS 20 at +6.2 %,
  outside the 5 % point bar, for correctness headroom).
- **C10:** `best_admissible = argmax(point speedup)` over many admissible configurations. Their
  bootstrap CIs are computed but ignored, so statistically tied configurations still produce one
  named "best", which is a winner's-curse effect. Downstream vmap rows are measured for it.
  **UNSAFE-SILENT** (phase 1).
- **C10 after fix phase 3: FRAGILE.** `_fastest` is the shared `tie_set` over the candidates'
  90 % speed-up intervals. `best_admissible` names a configuration only when its interval is clear
  of every other candidate's and is `None` otherwise; `best_admissible_tie_set` (and the
  any-precision twin) records the point leader and its tie set. The vmap rows and the uncapped
  counts are measured for the point leader, labelled `point_leader` beside the `tie_set`. Still
  FRAGILE: the intervals are iid bootstraps (fix phase 4), and the tie set applies no
  multiple-comparison adjustment.
- **C10 after fix phase 4: SOUND-with-caveats.** The speed-up intervals are paired whole-round
  bootstraps and `_fastest` passes `n = N_ROUNDS`: below 5 rounds every candidate is in the tie
  set and no `best` is named. Caveat: no family-wise adjustment over the configurations, which
  makes the tie set narrower than an adjusted family would.
- **C10 after fix phase 9:** the caveat is removed. The claim "the leader is best" rests on the
  k − 1 leader-vs-candidate comparisons, so they are one family: `holm_tie_set` reads every
  interval at `1 − α/m` (m = candidates not yet separated), excludes those that no longer overlap
  the leader's and steps down; a `best` only when all separate. The family-wise tie set always
  contains the 90 % one (nested draws). Caveat: the leader is chosen by argmax first (a
  data-chosen reference), and separation is judged by overlap of two vs-control intervals rather
  than a paired leader-vs-candidate ratio, which is conservative.

### C11 — interferometer numba bake-off kill gate (FRAGILE)

The kill gate asks whether the best numba kernel beats `rfft2_numpy` by more than 1.3× (medians of
rounds) at sma or alma. The 30 % margin is large against typical scatter, but there is no
interval, the "best" kernel is chosen by argmin among several, and the result is a binary
tripped / passed.

**After fix phase 9: SOUND-with-caveats.** `kill_gate_family`: every pinned numba kernel × sma /
alma cell is a member `median(rfft2) / median(kernel)` over the cell's timed rounds (each kernel
runs once per round, round-robin, so rounds pair the arms; round 0 discarded as before) with a
paired round-bootstrap interval against > 1.3; the members are one Holm family. "passed" (numba
survives) is an "any" claim — some member resolves above the margin at its Holm level; "tripped"
(kill) needs every member resolved below; anything else, or fewer than 5 timed rounds, is
INCONCLUSIVE. The point rule is kept as `kill_gate_point`. Caveat: the cell's default `--reps 5`
times 4 rounds, so every bake-off at the default is INCONCLUSIVE by construction until it is run
with `--reps` ≥ 6 (a human decision; not changed here).

### C12 — GPU bottleneck map minimum detectable improvement (SOUND)

`_mdi` bootstraps a split-half null A/B of one route against itself. It publishes the effect size
the run can resolve, which is the right quantity for a human go/no-go. One caveat: the two halves
are adjacent in time, so slow drift inflates the reported floor, which is the conservative
direction.

**After fix phase 9 (still SOUND):** the null A/B is each round's first half of calls against its
last half, bootstrapped over whole rounds with the same indices for both halves
(`round_split_half_mdi`): the two adjacent slots a real interleaved A/B occupies. Drift across the
run cancels, so the floor is the one a paired A/B of this size can resolve; it is NaN below 5
rounds or 2 calls per round (the `--quick` laptop witnesses). The in-process interleaved ratios
(OFF / ON, gradient / primal, scalar and vmap) use the paired round bootstrap; the fp32 what-if
ratio compares two separate processes, which share no rounds, so it stays an unpaired iid
resampling of calls, labelled `resampling: "iid calls, unpaired …"` (reported, never gated).

### S1, R1, R2 (SOUND)

- **S1 (`check_submits`):** the `--time` budget must be at least the estimated wall times a
  headroom floor (1.25 measured-wall, 1.5 rates, 3.0 unmeasured). The rates come from the slowest
  arm on the same cell. `RATE_TOLERANCE` (5 %) is a citation-match tolerance, not a noise cutoff.
  One limit: most rate rows are single-job observations. The headroom absorbs that, and the cost
  of a miss is lost compute, not a mis-qualified result.
- **R1 and R2:** resource guards. A skipped measurement is recorded as skipped, never as a result.

## Estimator settings found by the lister (no automated verdict)

These constants set repeat counts or bootstrap sizes for cells whose timings are reported, not
gated. They matter only through a gate that reads them (listed above), or through a human reading
a ledger.

| Key | Value | Read by |
|---|---|---|
| `scripts/cluster/image_plane/likelihood_breakdown.py::N_REPEATS` | 3 | reported breakdown |
| `scripts/datacube/delaunay/likelihood_runtime_shared_preloads.py::N_REPEATS` | 5 | reported runtime |
| `scripts/imaging/mge_mass/likelihood_runtime_jax.py::N_REPEATS` | 3 | reported runtime |
| `scripts/imaging/mge_mass/likelihood_runtime_jax.py::N_STEADY` | 10 | reported runtime |
| `scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py::N_REPEATS` | 5 | reported runtime |
| `scripts/multi_dataset/delaunay/likelihood_runtime_shared_preloads.py::N_REPEATS` | 5 | reported runtime |
| `scripts/point_source_image/image_plane/likelihood_breakdown.py::N_REPEATS` | 5 | reported breakdown |
| `scripts/point_source_source/source_plane/likelihood_breakdown.py::N_REPEATS` | 500 | reported breakdown |
| `scripts/imaging/pixelized/fixed_light_numba_levers_l2_witness.py::W5_N_REPEATS` | 20 | W5 timing (gate is correctness) |
| `scripts/imaging/pixelized/fixed_light_numba_levers_l3_witness.py::W5_N_REPEATS` | 20 | W5 timing (gate is correctness) |
| `scripts/imaging/pixelized/fixed_light_numba_s4_witness.py::W4_N_REPEATS` | 20 | W4 kernel timing (reported) |
| `scripts/lens/deflections/basis.py::WITNESS_REPEATS` | 20 | memo on/off medians (printed) |
| `scripts/misc/hazards/mge_faddeeva.py::COST_REPEATS` | 5 | hazard cost table |
| `scripts/misc/likelihood_breakdown/timing.py::MIN_STEADY_WARM` | 5 | P8 median (the headline where present since fix phase 5) |
| `scripts/misc/likelihood_breakdown/timing.py::HEADLINE_MEDIAN_BUDGET_S` | 30.0 | P8: timed-call budget of the headline median (`n_timed` in [20, 200]) |
| `scripts/point_source_image/image_plane/gpu_bottleneck_map.py::BOOTSTRAP_SAMPLES` | 2000 | C12 |
| `scripts/point_source_image/image_plane/solver_config_sweep.py::BOOTSTRAP_SAMPLES` | 2000 | C9/C10 (recorded CI) |
| `scripts/point_source_image/image_plane/static_lattice_ab.py::BOOTSTRAP_SAMPLES` | 2000 | reported CI (gates are correctness) |
| `scripts/point_source_image/image_plane/vertex_dedup_ab.py::BOOTSTRAP_SAMPLES` | 2000 | reported CI (gates are correctness) |
| `scripts/point_source_source/source_plane/backward_pass_ab.py::BOOTSTRAP_SAMPLES` | 2000 | C6 |
| `scripts/point_source_source/source_plane/gradient_mode_crossover.py::BOOTSTRAP_SAMPLES` | 2000 | C8 |
| `scripts/point_source_source/source_plane/gradient_mode_library_ab.py::BOOTSTRAP_SAMPLES` | 2000 | reported CI |
| `scripts/point_source_source/source_plane/pytree_input_ab.py::BOOTSTRAP_SAMPLES` | 2000 | C7 (its intervals decide since fix phase 3) |

Every bootstrap above uses a fixed seed (12345). That makes the CI reproducible. It does not
make it valid for dependent samples. Since fix phase 4 the point-source A/B cells above resample
paired whole rounds through `scripts/misc/likelihood_breakdown/round_bootstrap.py::ROUND_BOOTSTRAP_SAMPLES`
(2000) rather than iid calls; since fix phase 9 `gpu_bottleneck_map` does too for every
in-process interleaved ratio and its MDI, keeping an iid `_median_ratio` only for the fp32
what-if ratio across two separate processes (no shared rounds to pair; labelled).

## Lister hits that are not timing gates

The lister over-reports on purpose. These keys matched its token heuristic but compare
iteration counts, budgets in iterations, geometry or formatting, not a measured time:

- `scripts/imaging/pixelized/fixed_light.py::<module>`: `PASS_BUDGET` / `SAFE_BUDGET` are
  active-set iteration budgets.
- `scripts/imaging/pixelized/fixed_light_cpu_kernels.py::<module>`,
  `scripts/imaging/pixelized/fixed_light_draws.py::<module>`,
  `scripts/imaging/pixelized/fixed_light_library.py::<module>`: the same iteration budgets.
- `scripts/imaging/pixelized/fixed_light_trace.py::<module>` also holds `certified_budget < 1`,
  an iteration budget. Its timing comparison is C5.
- `scripts/imaging/pixelized/nautilus_batch_capture.py::<module>`: a sample count (`len(...) > 1`).
- `scripts/misc/hazards/_likelihood.py::ell_comps_radius_from_axis_ratio`: a geometric axis ratio.
- `scripts/misc/likelihood_breakdown/fixed_light_numpy_solvers.py::nnls_factor_reuse`: NNLS
  iteration caps.
- `scripts/misc/likelihood_breakdown/interferometer_pixelized_numpy.py::LEVER_WALK_STEP`: a
  parameter random-walk step.
- `scripts/misc/likelihood_breakdown/psf_cube_injection.py::patched`: an over-sampling size.
- `scripts/misc/nnls_warm_start/nnls_iterations_matrix.py::findings`: an iteration ratio, a
  deterministic count.
- `scripts/misc/test/test_matrix_free_steps.py::test_jacobi_diag_estimate_is_positive_and_exact_on_the_func_block`
  and `scripts/misc/test/test_matrix_free_steps.py::test_pdip_matrix_free_matches_the_cell_nnls`:
  a diagonal ratio and an iteration-count difference.
- `scripts/misc/tooling/build_readme.py::_format_time` and
  `scripts/misc/tooling/build_readme.py::_render_runtime_table`: unit formatting.

**What the lister cannot see.** It finds comparisons against a literal or an UPPER_CASE constant.
It misses comparisons of two variables, lowercase caps read from config (P6's
`point["loadavg"] > cap`, S1's `budget < needed`), argmax selections (C10, C11's best kernel) and
categorical rules (P9's timeout). Those rows were found by reading and are inventoried above. A
new gate of that shape needs a hand-added row.

## Cross-repo mechanisms found (read-only; nothing edited)

- **PyAutoBrain profiling conductor** (`agents/conductors/profiling/_profiling.py`):
  `COMPILE_DRIFT_RATIO = 2.0` and `COMPILE_DRIFT_FLOOR_S = 1.0` judge this repo's
  `scripts/misc/jax_compile/pins.json` warm-compile rows. A pin is one record (the most recent
  warm row), and an observation is one record. The comparability key includes the hostname. P7
  borrows the 2× ratio. This is a shared timing gate living outside this repo, and the same
  point-vs-point limitation applies.
- **PyAutoPulse** (`pulse/catalogue.py`) requires `qualified: true` for accepted evidence.
  Corrected after fix phase 2: the live registry reads this repo's v2 `dashboard/catalogue.json`
  (since 2026-10-05, PyAutoPulse `3c7bed2`), whose producer hard-codes `qualified: False`, so P6's
  gaps did not reach the organ live; the v1 `summary.json` feeds the project dashboard and badge.
- **PyAutoHeart** unit-test timing (`heart/checks/script_timing.py`, Heart #276/#277) is not
  imported or read by this repo. There is no shared mechanism, so it is out of scope.
- The README roadmap item "Regression-watch indicator … regressed (>5%)" is an unbuilt gate. If it
  is built as written, it would be a point-estimate 5 % cutoff on P8's estimator, which this audit
  rates UNSAFE-SILENT in advance.

## (a) Proposed PASS / FAIL / INCONCLUSIVE semantics

One rule shape for every gate class. A measured effect `Δ` (overhead, saving, ratio − 1, drift)
is compared to a pre-registered budget `B`, using an interval `[L, U]` at a stated confidence:

- **PASS** when the whole interval is on the good side of `B` (`U ≤ B` for "must not exceed").
- **FAIL** when the whole interval is on the bad side of `B` (`L > B`).
- **FAIL_GROSS** when the point estimate exceeds a pre-registered gross bound, whatever the
  interval width. Noise never masks a catastrophic regression, and this preserves today's gross
  guards.
- **INCONCLUSIVE** otherwise, and whenever the sample count is below the method's minimum or the
  validity checks fail (non-finite, non-positive or missing samples, host not qualified, warm-up
  never settled).

INCONCLUSIVE is never silent and never counts as a PASS. It is written into the result JSON as
the status. In CI it is reported through a visible channel, not a captured `print`. A
qualification, promotion, GO or `qualified: true` requires PASS. NO_LEVER or no-go requires FAIL;
an INCONCLUSIVE comparison is reported as INCONCLUSIVE with the resolvable effect size (the C12
MDI idea), never as a measured negative.

Per gate class:

| Class | Interval | Minimum n | INCONCLUSIVE consequence |
|---|---|---|---|
| CI overhead test (T1) | one-sided t on block ratios, budget in the **same unit** as the instrument | sized to the budget's excess: ~12 blocks at CI scatter for ±0.7 pp | test passes, emits a visible warning with the bounds; never claims the budget held |
| production overhead (P1) | the same shared function as T1 | `MIN_BLOCKS_FOR_OVERHEAD_ASSERT`, sized from the recorded spread | row kept, status INCONCLUSIVE; no promotion; no raise |
| A/B lever / go rules (P2, C1, C3–C7, C9–C11) | paired, block-level (round-level) bootstrap or t on paired differences | ≥ 5 paired rounds | `INCONCLUSIVE` replaces NO_LEVER / no-go; MDI recorded |
| release drift (P7) | needs ≥ 2 repeat summaries per release (P8 median + p10/p90) | ≥ 2 per endpoint | `insufficient` (already in the contract) instead of `flat` |
| qualification (P6, P9) | not statistical: provenance completeness + host class | n/a | unqualified, with a reason; a missing load average or host is unqualified, not qualified |

Limits that must be stated beside any interval:

- **Small-sample Student-t.** With n = 3 the one-sided 95 % critical value is 2.92, not 1.645. The
  interval is valid only if block ratios are roughly normal and independent. At n ≤ 5 it cannot
  detect non-normality, so a guard is only as good as its blocks are iid.
- **Non-normal and outlier timings.** Wall times are right-skewed, with GC pauses, frequency
  steps and scheduler preemption. Prefer medians or trimmed means per block, and intervals on
  block summaries rather than raw calls. A percentile bootstrap of a median from 4–20 samples has
  poor coverage. Report its n.
- **Dependence and drift.** Calls within a round are autocorrelated, and load drifts across a
  run. iid resampling of calls understates the interval. Resample whole rounds or blocks, and pair
  arms within a round (ABBA or a rotated round-robin). A design that cancels only linear drift
  (ABBA) does not cancel step changes. Record load at both ends, and treat a load change as
  INCONCLUSIVE.
- **Multiple comparisons.** C3 (6 targets), C6 (routes × lanes), C9/C10 (many configurations) and
  C11 (several kernels × cells) make many comparisons per verdict. A conjunction of PASSes inflates
  the INCONCLUSIVE / NO_LEVER rate. A "best of k" argmax inflates the winner's apparent effect.
  Either pre-register a single primary comparison or adjust (for example, Bonferroni or Holm on
  the per-comparison confidence), and never name a "best" when the leaders' intervals overlap.
  (Since fix phase 3b, C3 is judged by Holm, `ab_verdict.holm_family_verdict`.)

  **The family-wise policy (fix phase 9; stated once, in `ab_verdict`'s module docstring).** The
  *family* is the set of comparisons one verdict or one claim rests on — every comparison that, if
  it came out the other way, would change that verdict or claim; comparisons feeding separate,
  independently acted-on verdicts are separate families, and one host's run is one family (other
  hosts are recorded, not pooled). The *procedure* is Holm's step-down at family-wise 90 % on
  two-sided intervals read from the same bootstrap draws at every level (nested). The *verdict
  shape* is the claim's: a conjunction reads `holm_family_verdict`'s verdict; an "any" claim is GO
  if any member resolves GO at its Holm level and NO_GO only if every member resolves NO_GO; a
  "best" is named only when the leader separates from every other candidate at the Holm-adjusted
  level (`holm_tie_set`). The unadjusted 90 % reading is always recorded beside, never used.
  A conjunction of a few criteria inside one target (C4's lanes, C5's two estimators) is an
  intersection-union test and is not adjusted (conservative for PASS / GO).

  | Rule | Family | Members | Claim shape |
  |---|---|---|---|
  | C3 memo policy | one evaluation run | the six targets | conjunction (GO iff all six) |
  | C6 phase 2c | one host's run | timed routes × lanes × {saved ms, ratio} (16) | per route: GO iff GO in every lane; the decision = the GO routes |
  | C10 best admissible | one sweep, one candidate list | the leader vs each other candidate (k − 1) | best iff every comparison separates |
  | C11 kill gate | one bake-off | pinned numba kernels × sma / alma cells (≤ 24) | "passed" iff any member > 1.3×; "tripped" iff all ≤ 1.3× |
- **Host qualification is not statistics.** No interval turns a laptop row into a reference-host
  row. P6 must stay a separate gate from the measurement verdict.

## (b) Proposed fix phases (priority order), each with a deterministic synthetic witness

Each phase is one bounded PR. None of them raises a budget, retries to green or weakens a
correctness or gross-regression guard.

1. **Shared overhead verdict (T1 + P1); the #361 guard done properly.**
   - **Change:** move `_ci_overhead_verdict` into `scripts/misc/likelihood_breakdown/` as one
     function used by both the test and `fixed_light_numba.py`. Express the budget in the
     instrument's unit (ms of excess over the clean call). Compare the interval to the
     **excess** over the budget. The cell writes PASS / FAIL / FAIL_GROSS / INCONCLUSIVE and raises
     only on FAIL or FAIL_GROSS. P2 promotion requires PASS. INCONCLUSIVE in CI becomes a visible
     pytest warning.
   - **Witness (synthetic, no timing):** block sets with known answers:
     - clear pass `[1.009, 1.010, 1.011]`;
     - clear fail `[1.049, 1.050, 1.051]`;
     - the #361 blocks → INCONCLUSIVE;
     - boundary `[1.031] × 3` (zero variance);
     - n < 3 → INCONCLUSIVE;
     - gross `[1.6]` → FAIL_GROSS;
     - the implausible local blocks `[0.711, 0.923, 0.738]` → INCONCLUSIVE (a mean ratio resolved
       below 1 is a host-noise signature, not a measured pass);
     - non-finite or non-positive → ValueError;
     - the 224 ms / 400 ms / 413 ms RAL rows from T3, converted to ms, keep today's PASS / FAIL /
       PASS.

     The witness also asserts that the test and the cell import the same function object.
   - **Shipped (phase 2 of #362, PR #404, 2026-10-08).** As implemented:
     - verdict rule, in order: invalid input → `ValueError`; mean ratio > 1.5 → FAIL_GROSS;
       n < 3 → INCONCLUSIVE; upper bound of the excess < 0 ms (mean ratio *resolved* below 1) →
       INCONCLUSIVE, "host-noise signature"; upper ≤ budget → PASS; lower > budget → FAIL;
       else INCONCLUSIVE. Bounds are one-sided 95 % Student-t on the block ratios, df = n − 1,
       converted to ms by the row's clean mean. A mean below 1 whose upper bound is still ≥ 0
       is judged on its bounds (it is consistent with a near-zero cost).
     - witness results (budget 12 ms; ratio-only sets read at the 400 ms calibration call):
       clear pass PASS; clear fail FAIL; #361 blocks INCONCLUSIVE (bounds [3.05, 21.8] ms);
       `[1.031] × 3` PASS at a budget equal to its point and FAIL one ulp below; `[1.0, 1.02]`
       and `[1.01]` INCONCLUSIVE; `[1.6]` and `[0.8, 1.6, 2.4]` FAIL_GROSS; `[0.711, 0.923,
       0.738]` INCONCLUSIVE (host-noise signature); invalid input ValueError.
     - the three pinned RAL rows, as point estimates (32 zero-spread blocks): 1.0366 at
       224.330 ms = 8.21 ms PASS; 1.037 at 400 ms = 14.8 ms FAIL; 1.01467 at 413.301 ms =
       6.06 ms PASS — unchanged.
     - **verdict change, recorded as a fact:** the 413.301 ms row judged on its own eight
       recorded blocks (0.944–1.087) has bounds [−7.30, +19.42] ms and is **INCONCLUSIVE**, not
       the PASS it was published with. Re-judging every committed decomposed RAL row the same
       way (`results/breakdown/imaging/fixed_light_numba_*hpc_ral*.json`, 17 decomposed rows):
       INCONCLUSIVE are the 8-block rows at 299.7, 413.3, 267.4, 302.7 and 407.9 ms (levers 1–2,
       their controls and `sparse_b_warm`) and route a of `sparse_t1` at 926 ms; PASS are the
       32-block rows (lever 3 and its control, s4 ×3, s4b ×2), both `sparse_dnp` rows and routes
       b and c of the 8-block `sparse_t1` (upper bounds 3.3 and 1.0 ms). 11 PASS, 6
       INCONCLUSIVE, no FAIL; all 17 were published PASS. The three 12-block laptop rows
       (`local_cpu ... sparse_t1`, mean ratios 0.957–0.986) are INCONCLUSIVE; the smoke rows
       (1 block) were RECORDED and are now INCONCLUSIVE. No committed JSON was rewritten; their timing numbers are unaffected,
       only the claim that the instrument cost was resolved inside 12 ms.
     - the 224.330 ms row's blocks were lost with its JSON (job 343355), so its interval
       verdict cannot be computed.

2. **Dashboard qualification and drift wording (P6 + P7); the organ-facing fix.**
   - **Change:** a row is qualified only if its host class is declared as a reference host (laptop
     rows never qualify as trend points), and a missing load average or host makes it unqualified
     with a reason. The `flat` label gains an explicit meaning ("within the 2× policy band") or
     becomes `insufficient` when either endpoint lacks a repeat summary. This is a
     `profiling-summary` contract change, coordinated with Pulse. (Corrected when shipped: it is
     not; see below.)
   - **Witness:** synthetic points for laptop + provenance → unqualified; `loadavg: None` →
     unqualified; `hpc` + `host: None` → unqualified; 1.9× qualified → not `flat`-as-null;
     single-sample endpoints → `insufficient`.
   - **Shipped (phase 3 of #362, PR #405, 2026-10-08).** As implemented:
     - `qualify` order: above the cap → refused; no provenance; not a reference host class
       (`is_reference_host_class(config)`, `REFERENCE_HOST_CLASS_PREFIXES = ("hpc_",)`) →
       "not a reference host class; laptop rows never qualify as trend points"; no load average →
       "provenance carries no load average"; no host → "provenance carries no host"; off the
       pinned node. All unqualified, none refused.
     - repeat summary: the point field `REPEAT_SUMMARY_FIELD` (`single_jit_repeats`, ≥ 2). No
       producer writes it, so every current endpoint is single-sample (P8). The opt-in
       `single_jit_median_s` does not count: it summarises a different estimator from the one
       compared.
     - `_summary_comparison`: `steady` → `insufficient` with a single-sample endpoint, else `flat`
       with "within the 2x policy band; not a measured null". **Human decision (2026-10-08, "Ill
       go with your recomendation"):** `drifted` / `improved` keep their status (gross-band
       signals, like FAIL_GROSS) and gain "single-sample endpoint(s)". Not taken: mapping them to
       `insufficient` too. Status vocabulary and the v1 grammar are unchanged; PyAutoPulse's v1
       validator (`pulse/summary.py`) returns no errors on the regenerated file.
     - the badge headline reads "none drifted >= 2x (single-sample endpoints: not a measured
       null)" instead of "no drift"; the page's table says "within 2x band" instead of "steady".
     - measured effect on the committed tree (`dashboard/summary.json` before → after): 159
       records, qualified 2 → 2 (both `point_source_source/source_plane_solved` @2026.8.17.1,
       `hpc_ral_cpu_fp64` and `hpc_a100_fp64`, host `euclid-ral-gpu-2`, load 0.16 / 1.16); no
       record's qualification or reason changed; comparisons `flat` 4 → 0, `insufficient`
       135 → 139, `improved` 6 → 6 (each now with the caveat), `drifted` 0 → 0.
     - correction: the live PyAutoPulse registry reads `dashboard/catalogue.json`
       (`profiling-summary@2`, since PyAutoPulse `3c7bed2`, 2026-10-05), whose producer
       `build_catalogue.py` hard-codes `qualified: False`. P6's gap therefore never flowed into
       Pulse live, and this phase is not a Pulse contract change. A future v2 qualification must
       reuse `is_reference_host_class`.
3. **A/B rule semantics (C7, C10, then C1/C3/C4/C5, P2).**
   - **Change:** every pre-registered go / lever / NO_LEVER rule gains an INCONCLUSIVE state using
     its already-computed CIs (C7, C10) or block-level intervals (C1, C3, C4, C5, P2). Argmax
     selection reports a tie set when the leaders' intervals overlap.
   - **Witness:** deterministic sample arrays with seeded bootstraps:
     - a clear 30 % saving → GO;
     - a clear 0 % → FAIL / no-go;
     - a 15 % point estimate with a CI straddling the bar → INCONCLUSIVE;
     - two configurations with overlapping CIs → tie set, no single "best".
   - **Shipped (phase 4 of #362, PR #406, 2026-10-08), C6 / C7 / C10 / P2; C1 / C3 / C4 / C5
     deferred to phase 3b.** As implemented:
     - **the rule** (`scripts/misc/likelihood_breakdown/ab_verdict.py::ab_rule_verdict`), in
       order: a red correctness gate → NO_GO; invalid input (non-finite point or bound, lower >
       upper, a bad sample count) → INCONCLUSIVE; n < `MIN_AB_ROUNDS` (5, the minimum in (a)) →
       INCONCLUSIVE; per criterion GO when the whole interval and the point clear the bar, NO_GO
       when the whole interval is on the bad side, else INCONCLUSIVE; the conjunction is NO_GO if
       any criterion is NO_GO, GO if all are GO, INCONCLUSIVE otherwise. INCONCLUSIVE carries
       each criterion's resolvable effect (`mdi`, the interval half-width, the C12 idea). Bars are
       the cells' pre-registered values; none was raised.
     - **tie sets** (`scripts/misc/likelihood_breakdown/ab_verdict.py::tie_set`): the point
       leader plus every candidate whose interval overlaps the leader's (a non-finite interval
       cannot be excluded); a single `best` only when the tie set is the leader alone.
     - **wired:** C7 (`_phase2b_rule`, with a new saved-ms bootstrap), C6 (`_phase2c_rule`, the
       correctness gate as a gate), P2 (paired-block t interval,
       `scripts/misc/likelihood_breakdown/ab_verdict.py::paired_block_ratio_interval`) and C10
       (`_fastest` → `tie_set`). Each cell's `from likelihood_breakdown.ab_verdict import ...` is
       asserted to bind the same function objects (`test_ab_verdict.py`, T7).
     - **witness results** (seeded lognormal calls, seeded 2000-sample bootstraps): a clear 30 %
       saving → GO; a clear 0 % → NO_GO; a 15 % point with ratio interval straddling 0.85 (6 rounds,
       12 % scatter) → INCONCLUSIVE, "not a measured negative", MDI recorded; two configurations at
       0.50 / 0.51 ms → a tie set with no `best`, a clear leader → named; n = 4 → INCONCLUSIVE;
       NaN / inverted bounds / bad n → INCONCLUSIVE; a red gate → NO_GO. P2 on the cell's own lifted
       block: a 6 % point with ±4 % paired scatter → INCONCLUSIVE; 4 blocks → INCONCLUSIVE.
     - **re-judged committed rows (recorded as facts; no JSON rewritten, no decision reversed):**

       | Rule | Committed rows | Published | Re-judged | Changed |
       |---|---|---|---|---|
       | C7 phase 2b | `pytree_input_ab_hpc_ral_cpu_fp64` solved / plain (decides) | no-go / no-go | NO_GO / NO_GO: saved [0.0367, 0.0408] / [0.0439, 0.0469] ms, wholly < 0.05 ms | no |
       | C7 phase 2b | `pytree_input_ab_hpc_a100_fp64` solved / plain | go / go | GO / GO: saved [0.0572, 0.0613] / [0.0546, 0.0596] ms, fraction ≥ 0.195 | no |
       | C6 phase 2c | `backward_pass_ab_{hpc_ral_gpunode_cpu,hpc_a100,local_cpu}_fp64`, 24 timed lane × route rows | 15 go, 9 no-go | 15 GO, 9 NO_GO; every interval resolves | no |
       | P2 promotion | `fixed_light_numba_..._s4b_warm_t1` (the one committed b / d_perm pair) | NO_LEVER, −0.80 % | NO_LEVER: 32 paired blocks, speedup [−1.31 %, −0.32 %], wholly < 5 % | no |
       | C10 best | `solver_config_sweep_hpc_ral_cpu_fp64` | `e2.5_s0.4` | tie set {e2.5_s0.4, e2.5_s0.2, e3_s0.4, e3_s0.3, e4_s0.4}; any-precision adds e3_s0.5, e4_s0.5 | **yes** |
       | C10 best | `solver_config_sweep_laptop_cpu_fp64` | `e3_s0.4` | tie set {e3_s0.4, e2.5_s0.4, e3_s0.3}; any-precision adds e4_s0.5, e3_s0.5 | **yes** |
       | C10 best | `solver_config_sweep_mcs_{hpc_ral_cpu,laptop_cpu}_fp64` | `mcs18` | tie set {mcs18, mcs20} | **yes** |
       | C10 best | `solver_config_sweep_step0_hpc_ral_a100_fp64` | `step0_gather` | tie set {step0_gather, step0_structured, step0_components} | **yes** |
       | C10 best | `solver_config_sweep_step0_hpc_ral_cpu_epyc7702_fp64` | `step0_structured` | tie set {step0_structured, step0_components} | **yes** |
       | C10 best | `solver_config_sweep_step0_hpc_ral_cpu_fp64` | `step0_components` | tie set {step0_components, step0_structured} | **yes** |
       | C10 best | `solver_config_sweep_mcs_hpc_ral_a100_fp64`, `solver_config_sweep_step0_laptop_cpu_fp64` | `mcs20`, `step0_components` | resolved: same | no |

       No go / no-go / NO_LEVER call changed; every one resolves on its own interval. Seven of
       nine committed "best admissible" names are tie sets. The recorded human decisions stand
       and none rested on the C10 name: IP-4a's "extent/scale ±2.5″/0.4 = 2.37x" (not shipped,
       per-workspace setting, human decision) is the point leader of a five-member tie set, so
       its "best" is now INCONCLUSIVE among {e2.5_s0.4, e2.5_s0.2, e3_s0.4, e3_s0.3, e4_s0.4};
       IP-4c chose MCS 20 for correctness headroom, not speed; IP-4b shipped `structured` as the
       step-0 default on its ratios against the gather route (1.44× / 1.61×, which resolve), but
       its RAL CPU step-0 sweeps cannot separate `structured` from `components` (tie sets on both
       nodes), so the choice between those two is not a measured speed difference.
     - **decision taken (flagged in the PR):** C10's vmap rows and uncapped counts are measured
       for the tie set's **point leader** (labelled `point_leader`, never "best"), not every
       member. This keeps the job's protocol and the `check_submits` wall basis unchanged;
       measuring every tie member is the reversible alternative.
     - **multiple comparisons:** only tie sets (the phase-3 text). No Holm / Bonferroni
       adjustment of the per-criterion confidence is applied; C6's routes × lanes, C10's
       configurations and C3's six targets are judged at 90 % each. A family-wise policy is a
       follow-up.
     - **phase 3b (not shipped here):** C1 (`matched_counterfactual`'s 3 % flags on 4 repeats), C3
       (the six-target memo-policy GO / NO_LEVER), C4 (`breakdown_reconciles` ±5 %) and C5 (the
       logdet lever `clears_threshold`) have no interval at all; each needs its own block-level
       interval over a different data layout. They keep their phase 1 verdicts (UNSAFE-SILENT) and
       are the next A/B fix, through the same `ab_rule_verdict`.
   - **Shipped (phase 8 of #362, PR #410, 2026-10-09), C1 / C3 / C4 / C5 — "phase 3b".** As
     implemented:
     - **one module** (`scripts/misc/likelihood_breakdown/interval_gates.py`), one function per
       gate, each on its own pairing unit and through the shared tools (`round_median_ratio` /
       `round_median_saving` for the 90 % paired round bootstrap, `ab_rule_verdict` for the
       verdict; the minimum stays `MIN_AB_ROUNDS` 5). Bars, bands, repeat counts and protocols
       are unchanged. The three cells call it; T11 asserts they do and that no point-only
       comparison is left.
       - **C1** `matched_classification(samples_ms)`: memo / cold over the transition's repeats
         (a repeat is the round, cold and memo alternating inside it); `beneficial` when the
         interval and point are ≤ 0.97, `harmful` when ≥ 1.03, `neutral` when the interval is
         wholly inside the band, else INCONCLUSIVE. `matched_decision_counts` replaces
         `false_accepts` / `false_rejects` with `resolved_harmful_accepted`,
         `resolved_beneficial_rejected` (and their two complements), `resolved_neutral` and
         `inconclusive`, under a qualification that they are not decision-error rates.
         `warm_beneficial_3pct` / `warm_harmful_3pct` are kept, true only when resolved.
       - **C3** `memo_policy_family(evaluations)`: each of the six targets is
         `median(guarded) / median(reference)` over the evaluation's repeats (a repeat runs the
         three lanes once in a rotated order); the six are one Holm family. The cell writes
         `performance_family` (levels, steps, adjusted and unadjusted verdicts),
         `performance_targets` (per-target GO / NO_GO / INCONCLUSIVE at its Holm level) and
         `performance_targets_point` (the old point booleans, kept for continuity); the verdict
         is GO / NO_LEVER / INCONCLUSIVE when the gates hold, `INCOMPLETE_OR_FAIL` otherwise.
       - **C4** `breakdown_reconciliation(observed, clean)`: `ratio − 1` against ±0.05 as two
         criteria; `breakdown_reconciles` is PASS / FAIL / INCONCLUSIVE (a string, was a bool;
         no reader in the tree keyed on it) with the interval beside it in
         `breakdown_reconciliation`; `breakdown_passed` needs every lane PASS (lanes joined by `conjoin_reconciliations`, i.e. `conjoin_verdicts`);
         `run_status` gives PASS / INCONCLUSIVE / INCOMPLETE_OR_FAIL.
       - **C5** `logdet_lever_verdict(interleaved, saving_jit_profile_ms)`: the interleaved
         saving's interval over its 7 rounds; the `jit_profile` estimator is INCONCLUSIVE by
         construction (one block per arm); the two are joined by the new
         `ab_verdict.conjoin_verdicts`, so a resolved interleaved NO_GO is not hidden by the
         other estimator's missing interval. `clears_threshold` is `True` / `False` only when
         resolved, `"INCONCLUSIVE"` otherwise (`None` still marks the control row);
         `interval_verdict` records both estimators.
     - **the family-wise policy** (`scripts/misc/likelihood_breakdown/ab_verdict.py`, for phase 9
       to reuse on C6 / C10 / C11): `holm_levels(k, confidence=0.90)` → the step-down
       confidences `1 − α/k … 1 − α`; `holm_family_verdict(members, *, n, min_n=5,
       confidence=0.90, gates=None)` → `FamilyVerdict`, where `members` maps each target's name
       to a callable `confidence → Criterion` (for a bootstrap, `bootstrap_criterion(name, point,
       boots, bar, direction)` reads two-sided percentiles of the same draws, so intervals are
       nested). Procedure: judge the m unresolved targets at `1 − α/m`; every target resolved
       (GO or NO_GO) is final and m drops; stop when none resolves (the rest INCONCLUSIVE). The
       family verdict is the `ab_rule_verdict` conjunction of the final per-target verdicts; the
       unadjusted verdict (every target at 90 %) is recorded beside it. Holm is applied to both
       directions; for GO alone an intersection-union conjunction is already level-α
       unadjusted, so a family GO is conservative — stated, kept for one policy.
     - **witnesses** (`test_interval_gates.py`, T11; fixed arrays, fixed seeds): Holm levels for
       k = 6 are 98.33 / 98 / 97.5 / 96.67 / 95 / 90 %; Holm resolves a two-target family that
       Bonferroni leaves open; six targets each resolving at 90 % but none at 98.33 % are
       unadjusted GO and family-wise INCONCLUSIVE (exact normal intervals, and again through
       `memo_policy_family` on synthetic sequence totals: point = bar / 1.0175, 1 % spread); one
       target resolved against makes the family NO_GO whatever the rest. C1: a 0.96 point (the old
       flag fired) with interval straddling 0.97 → INCONCLUSIVE; 0.50 / 1.70 / 1.00 with 0.2 %
       spread → beneficial / harmful / neutral; 4 repeats → INCONCLUSIVE at any effect. C3: all
       clear → GO; one target at 1.06 vs 1.03 → NO_LEVER; a 0.94 point (old GO) with ±3.75 %
       scatter → INCONCLUSIVE. C4: 1.002 → PASS; 1.10 and 0.90 → FAIL; a 1.04 point (old PASS)
       and a 1.06 point (old FAIL) with ±3 % scatter → INCONCLUSIVE; 4 repeats → INCONCLUSIVE;
       status PASS only with every lane PASS. C5: a 0.6 ms point with interval straddling 0.5 →
       INCONCLUSIVE; 0.1 ms resolved → `False`; a 2.0 ms saving resolved on the interleaved
       interval is still INCONCLUSIVE (no `jit_profile` interval).
     - **re-judged committed rows (facts; no JSON rewritten, no compute run):**

       | Rule | Committed rows | Published | Re-judged | Changed |
       |---|---|---|---|---|
       | C3 memo policy | `fixed_light_numba_memo_policy_..._hpc_ral_cpu_fp64_..._s5b` (job 343413), 6 repeats | NO_LEVER (4 targets met, broad holdout 1.0528 vs cold, nearby holdout 1.1166 vs memo missed) | NO_LEVER: all six resolve at Holm's first level (98.33 %), the same way as their points; graded / permuted vs memo [0.619, 0.621] / [0.645, 0.646]; vs cold [0.985, 0.987] / [1.005, 1.008]; broad holdout [1.052, 1.056]; nearby holdout [1.114, 1.117] | no |
       | C1 matched | the same row, 128 eligible transitions (graded 38, permuted 40, nearby 28, broad 22), 4 repeats each | 128 classified (38 beneficial, 90 harmful); "false accepts" 1, "false rejects" 14 | 128 INCONCLUSIVE (4 < 5 repeats); resolved counts all 0 | **yes** |
       | C4 scaling | `fixed_light_numba_scaling_delaunay_..._s6_n{500,1000,1500,2500,4000}`, 4 lanes each, 6 repeats | 20 reconcile; 5 × status PASS | 20 PASS (every interval within ±0.0062); 5 × status PASS | no |
       | C5 log-det, A100 | 6 lever rows (Delaunay / rectangular × k32 / k64 / k256) | `clears_threshold` False ×6 | False ×6: interleaved savings resolve below 0.5 ms (widest upper bound 0.152 ms, rectangular k32) | no |
       | C5 log-det, laptop RTX 2060 | Delaunay k32 / k64 / k256 | False ×3 | k32 False ([−53.7, −6.3] ms); k64 and k256 INCONCLUSIVE ([−88.4, +34.1], [−153.3, +124.4] ms) | **yes** (2 laptop rows) |
       | C5 controls | Delaunay / rectangular A100, Delaunay laptop | `None` (control) | `None`; for the record the A100 control savings resolve below 0.5 ms and the laptop one does not | no |

       Decisions: the memo-policy NO_LEVER (and the phase 7 shelving, which rests on "no
       promotable residual-precheck policy") stands, resolved on intervals. The phase-6 scaling
       PASS stands. The log-det "no lever" on the A100 stands; the two changed C5 rows are
       laptop screen rows, and the phase-4 verdict was taken on the A100 rows. **One recorded
       reading loses its interval support:** `fixed_lens_light_numba_memo_policy_2026_09.md`
       ("Decisions and design implications") draws "a residual … is not a reliable proxy … it
       can reject useful sets and accept costly ones" from the C1 counts (1 false accept, 14
       false rejects, `seed1_broad_03` at 2.514×). All 128 classifications are now INCONCLUSIVE
       by the 5-round minimum. For context, not as a verdict: a two-sided 90 % paired Student-t
       on the 4 per-repeat ratios (df = 3) resolves all 128 exactly as published (nearest bound
       0.35 from the band; `seed1_broad_03` [2.511, 2.525]), so the effects are large; whether a
       4-repeat t interval may resolve C1, or the repeats are raised to ≥ 5, is a human decision
       (flagged in the PR). No recorded human decision rests on the C1 counts. That ledger carries
       a dated re-judgement note.
     - **decisions taken (flagged in the PR):** (1) C1 keeps the audit's 5-round minimum, so every
       4-repeat classification is INCONCLUSIVE by construction (the reversible alternative: a
       C1-specific paired t at n = 4, or `MATCHED_REPEATS` ≥ 5 on a new run); (2) Holm is applied
       to C3 in both directions, so a family GO is conservative; (3) C5's one-block `jit_profile`
       estimator has no interval, so under the pre-registered "both estimators" rule a lever can
       resolve `False` but never `True` until that basis records repeats (or the rule is
       re-registered on the interleaved estimator alone — a human decision); (4) C4's
       `breakdown_reconciles` changes type (bool → PASS / FAIL / INCONCLUSIVE) and the scaling
       status gains INCONCLUSIVE.
     - **contract:** no `profiling-summary@2` field, metric or meaning change; `build_catalogue.py`
       is not edited (its catalogue is re-rendered for the source hash only). None of the three
       cells produces a dashboard point.
     - **reader notes (review, low):** `clears_threshold` may now be the string
       `"INCONCLUSIVE"`, which is truthy — compare with `is True` (no reader in this repo or
       PyAutoPulse keys on it). C1's band edges are now inclusive (`≤ 0.97` / `≥ 1.03`, the
       `ab_rule_verdict` convention) where the old point flags were strict; this matters only at
       the exact edge. `FamilyVerdict.steps` / `member_confidence` record the step-down even when
       `n < min_n` (the verdict is then INCONCLUSIVE regardless), and `adjusted.confidence`
       records the family-wise 90 %, not each target's Holm level (that is `member_confidence`).
     - **limits:** a percentile bootstrap over 6–7 rounds is coarse (462 / 1716 distinct
       resamples), and Holm's 98.33 % level reads near their extremes; the four C4 lanes of a
       cell and C5's two estimators are conjunctions without an adjustment (conservative for
       PASS / GO).
     - **not changed (remainders):** phase 9 — the family-wise policy applied to C6 (routes ×
       lanes), C10 (configurations) and C11 (kernels × cells, its ratio > 1.3 given an interval)
       through `holm_family_verdict`, C12 moved to the paired round bootstrap where its layout
       allows, and P2's between-row drift recorded (INCONCLUSIVE when the rows drift); phase 10 —
       headline completion (the runtime README on the median headline; the breakdown, datacube
       and `mge_mass` cells opted in; `wall/rates.py` counting the median's calls;
       `single_jit_repeats` support). Not in this audit's PRs: a v2 qualification (waits on the
       reference-host decision) and measuring `single_jit_repeats` (needs compute).
   - **Shipped (phase 9 of #362, stacked on PR #410, 2026-10-09), the family-wise policy for C6 /
     C10 / C11, C12 on paired rounds, P2's between-row drift — "phase 9".** As implemented:
     - **the policy** (section (a); stated once in `ab_verdict`'s module docstring): family = the
       comparisons one verdict or claim rests on; Holm step-down at family-wise 90 % on nested
       intervals from the same bootstrap draws; the claim's own shape (conjunction, "any", "best");
       the unadjusted 90 % reading recorded beside, never used; below 5 units INCONCLUSIVE.
     - **one module** (`scripts/misc/likelihood_breakdown/family_gates.py`) on the shared tools
       (`holm_family_verdict`, `bootstrap_criterion`, the new `ab_verdict.holm_tie_set` /
       `bootstrap_interval` / `FamilyTieSet`, `round_median_ratio`, and `round_median_saving`, which
       gains `return_boots`). Bars, margins, repeat counts, seeds and protocols are unchanged.
       - **C6** `phase2c_family`: one host's timed (route, lane) × {saved ms, ratio} criteria (16)
         as one Holm family on the cell's own draws (`seed + i`, `seed + 100 + i`; the 90 % intervals equal
         those recorded by rows written since fix phase 4); `_phase2c_rule` writes `verdict` (family-wise),
         `verdict_unadjusted`, `family_member_confidence`, `family_criteria`, and at the top
         `per_route` (GO only when GO in every lane), `go_routes`, `go_routes_unadjusted`, `family`.
       - **C10** `sweep_tie_set` → `holm_tie_set`: the leader vs each other candidate, one family;
         read every interval at `1 − α/m`, exclude the candidates that no longer overlap, step
         down; `best_admissible` only when all separate. `best_admissible_tie_set` carries the
         family-wise members, `excluded_at`, `steps` and the `unadjusted` 90 % tie set.
       - **C11** `kill_gate_family`: pinned numba kernel × sma / alma cell members, paired round
         bootstrap of rfft2 / kernel over the timed rounds, one Holm family; "passed" if any
         resolves > 1.3, "tripped" if all resolve ≤ 1.3, else INCONCLUSIVE; `kill_gate_point`
         records the old rule. `kill_gate` becomes `passed` / `tripped` / `INCONCLUSIVE`.
       - **C12** `round_split_half_mdi`: each round's first `n_calls // 2` calls vs its last
         `n_calls // 2`, paired round bootstrap (`mdi_same_program` is its `mdi`;
         `mdi_same_program_detail` the record). `_median_ratio(..., n_rounds=)` is the paired round
         bootstrap for OFF / ON and the scalar / vmap gradient-over-primal ratios; the fp32
         what-if ratio (two processes) stays unpaired iid, labelled.
       - **P2** `between_row_drift`: within-row drift (last-half / first-half clean median − 1,
         tolerance ±0.025) and the 1-minute load range (run start / end, plus the new per-row
         `load_average_at_row_start` / `_end`; tolerance 2.0); `drifted` → promotion INCONCLUSIVE
         before the candidate / NO_LEVER branches; the record is `between_row_drift`.
     - **witnesses** (`test_family_gates.py`, T12; fixed inputs and seeds): a leader at 1.10 and five
       rivals at 1.02 (s.e. 0.02, exact normal intervals) — unadjusted names the leader, Holm at 98 %
       keeps all six; a clear leader separates; a step-down (far rival out at 95 %, near one at 90 %);
       non-finite candidates never separate; family ⊇ unadjusted; n = 4 keeps every candidate. C6 on
       synthetic rounds (two routes at a true 0.83 ratio, ±3 % round noise): unadjusted names `fwd`
       GO, family-wise none (8 members); 0.60 → both GO; 1.00 → NO_GO; 4 rounds → INCONCLUSIVE. C11 on
       24 kernels at a true 1.30× over 7 timed rounds: point and unadjusted "passed", family-wise
       INCONCLUSIVE; 2.0× → passed; 0.8× → tripped; 4 timed rounds → INCONCLUSIVE by construction.
       C12 with a 10 % linear drift across 20 × 20 calls: old iid split-half MDI > 4 %, paired
       < 1 %; NaN below 5 rounds or 2 calls. P2: stationary rows do not drift; an 8 % within-row
       trend, a 1.0 → 4.5 load change drift (and through the cell's own block a timing candidate
       and a NO_LEVER both become INCONCLUSIVE, T3); missing records are unassessed. Every cell
       imports the shared function objects; the bake-off holds no point comparison.
     - **re-judged committed rows (facts; no JSON rewritten, no compute run, no decision
       reversed):**

       | Rule | Committed rows | Before (phase 4 / point) | Family-wise (phase 9) | Changed |
       |---|---|---|---|---|
       | C6 EPYC (`hpc_ral_gpunode_cpu_fp64`, decides) | 8 lane × route rows, 16 criteria | GO routes `fwd`, `fwd_analytic`, `rev_analytic` | the same: 15 criteria resolve at 99.375 %; plain `rev_analytic`'s ratio straddles 0.85 there (upper 0.855) and resolves at the last step (90 %, [0.821, 0.847]) | no |
       | C6 A100 | 8 rows | GO `fwd`, `fwd_analytic` | the same (all 16 at the first level) | no |
       | C6 laptop | 8 rows | solved `rev_analytic` INCONCLUSIVE (since phase 4), plain `rev_analytic` NO_GO | both INCONCLUSIVE (plain ratio [0.850, 0.913] at 95 %) | **yes** (plain `rev_analytic`, non-deciding) |
       | C10 RAL CPU (IP-4a) | precision-equivalent / any precision | tie sets of 5 / 7 | 7 / 9: adds `e3_s0.2`, `e4_s0.3` | **yes** (no best named before or after) |
       | C10 MCS RAL CPU, MCS laptop | | {mcs18, mcs20} | {mcs18, mcs20, mcs24} | **yes** (no best before or after) |
       | C10 other 5 sweeps | | as phase 4 (laptop 3 rounds: every candidate; MCS A100 `mcs20` and step-0 laptop `step0_components` resolved; step-0 RAL / EPYC / A100 tie sets) | identical | no |
       | C11 kill gate | `bakeoff_v2026.8.17.1.json` (4 cells), `bakeoff_machine_drift_sma_v2026.8.17.1.json` (2 cells), 4 timed rounds each | **passed** (point: `direct_conv` 5.93× / 2.83× / 1.54× / 0.99×) | **INCONCLUSIVE** by construction (4 < 5 rounds); context, not a verdict: at a 4-round minimum the family-wise gate reads passed (sma Delaunay `direct_conv` [5.18, 6.11] at its Holm level, 99.58 %) | **yes** |
       | C12 MDI | A100 fp64 (job 366916) / fp32 what-if | 5.45 % / 6.52 % | 0.94 % ([0.9906, 1.0048]) / 1.00 % | **yes** |
       | C12 MDI | laptop `--quick` fp64 / fp32 (3 rounds × 3 calls) | 10.6 % / 5.3 % | NaN (3 < 5 rounds) | **yes** (not quotable rows) |
       | C12 OFF / ON | A100 fp64 / fp32; laptop fp64 / fp32 | [1.197, 1.211] / [1.355, 1.376]; [0.955, 1.067] / [1.002, 1.100] | [1.194, 1.211] / [1.352, 1.378]; [0.995, 1.055] / [1.020, 1.073] | widths only |
       | C12 gradient / primal | A100 fp64, laptop | iid intervals | not re-judgeable: the grad legs store stats, not per-call samples | — |
       | P2 promotion | `..._s4b_warm_t1` (the one b / d_perm pair) | NO_LEVER | NO_LEVER; not drifted (within-row −1.08 % / +1.00 %, load 1.0 → 1.0) | no |

       **Decisions whose support changes (flagged; nothing reversed; dated notes in the ledgers):**
       (1) **the numba-interferometer kill gate** (`numba_interferometer_verdict.md` section 5,
       "Passed"; the epic continued to the in-situ step and reinstated a numba path on
       `direct_conv`) is INCONCLUSIVE by the 5-round minimum; the effect is large and the in-situ
       rows corroborate it, so whether a 4-round family may decide, or `--reps` ≥ 6 is re-run, is a
       human decision. (2) **The GPU go / no-go memo's "below MDI"** (`point_source_gpu_breakdown_2026_09.md`:
       static lattice ≤ 2 %, deflections ≤ 2.9 % / 4.4 % "below MDI even at a 100 % saving") read an
       MDI inflated by drift across the run; on paired rounds the MDI is 0.94 %, so those ceilings
       are small but resolvable by an interleaved A/B of that size; the memo's ranking of levers
       (launch / fusion, sort, implicit Jacobian first) is unchanged. (3) **IP-4a's "2.37×"** was
       already the point leader of a tie set (phase 3); the tie set grows to seven. The phase-2c
       decision (`fwd` GO on the deciding row) and the s4b NO_LEVER stand.
     - **decisions taken (flagged in the PR):** (1) C6's family is one host's 16 criteria (routes
       × lanes × both criteria) and a route is GO only in every lane, the shape the 2026-09-27
       decision was taken in; (2) Holm on both directions everywhere (conservative for a
       conjunction GO, as in 3b); (3) C10 separation stays "intervals vs control do not overlap",
       now at Holm levels, rather than a new paired leader-vs-candidate ratio (conservative; the
       tie set can only grow); (4) C11 keeps the 5-round minimum, so the default `--reps 5` is
       INCONCLUSIVE by construction (the reversible alternative: `--reps 6` or a 4-round rule);
       (5) P2's tolerances (±2.5 % within-row, 2.0 load) are new, stated constants, and a missing
       load is unassessed rather than drift; (6) C12's paired MDI uses within-round halves (the
       slots a real A/B occupies), and the cross-process fp32 ratio stays iid.
     - **contract:** no `profiling-summary@2` field, metric or meaning change; `build_catalogue.py`
       is not edited (its catalogue is re-rendered for the source hash only). None of the five
       cells produces a dashboard point.
     - **limits:** the leader in C10 is chosen by argmax before the family is formed; Holm over 16 –
       24 members reads 99.4–99.6 % percentiles of 2000 draws (~6–8 draws in a tail); P2's
       within-row check misses a step change exactly between the rows when both rows are flat and
       the load is unchanged.
     - **not changed (remainders):** phase 10 — headline completion (the runtime README on the
       median headline; the breakdown, datacube and `mge_mass` cells opted in; `wall/rates.py`
       counting the median's calls; `single_jit_repeats` support). Not in this audit's PRs: a v2
       qualification (waits on the setup-baseline / reference-host decision) and the Brain
       `COMPILE_DRIFT_RATIO` rule (draft `draft/bug/pyautobrain/profiling_compile_drift_point_vs_point.md`
       in Mind; Brain is not edited here). The FRAGILE rows C2, C8 and C9 are descriptive or
       human-picked and are not planned.
4. **Bootstrap structure (C6, C8, C9 and the reported CIs).**
   - **Change:** resample whole rounds, paired across routes, instead of iid calls. Record the
     number of rounds as the effective n.
   - **Witness:** synthetic rounds with injected within-round correlation, where the iid
     bootstrap's 90 % CI is visibly narrower than the round bootstrap's. Assert the round
     bootstrap covers the known true ratio, under a fixed seed.
   - **Shipped (phase 5 of #362, PR #407, 2026-10-08).** As implemented:
     - **the bootstrap** (`scripts/misc/likelihood_breakdown/round_bootstrap.py`): calls are
       reshaped to `(n_rounds, n_calls)` in round order; each of 2000 draws resamples round
       indices with replacement, **the same indices for both arms**, and takes the median of the
       pooled calls of the drawn rounds; 90 % percentile interval; fixed seeds (the cells' own).
       Every result carries `resampling: "paired whole rounds"` and `effective_n = n_rounds`.
       Keys and seeds are unchanged, so fix phase 3's verdicts read the new intervals unchanged.
     - **wired:** `_median_ratio` / `_median_saving_ms` / `_ratio_boots` are thin wrappers over the
       shared functions in `backward_pass_ab` (C6), `pytree_input_ab` (C7),
       `gradient_mode_crossover` (C8), `gradient_mode_library_ab`, `solver_config_sweep` (C9 /
       C10, including the vmap ratios), `static_lattice_ab` and `vertex_dedup_ab`. `tie_set`
       gains `n` / `min_n`: below 5 rounds nothing is excluded and no `best` is named. C9 records
       `row_over_control_median_ci90`. Not changed: `gpu_bottleneck_map` (C12's split-half MDI and
       mixed layouts), recorded as a remainder.
     - **witness** (`test_round_bootstrap.py`, T8): 20 rounds × 20 calls with a per-(round, arm)
       lognormal offset (s.d. 0.08) over call noise (0.02), true ratio 0.8. The iid interval is
       ~0.14–0.39 × (median 0.23) the round interval's width over the 40 witness seeds; on the pinned seed the round interval covers 0.8
       and the iid one misses it; over 40 seeded datasets round coverage is 0.85 and iid 0.33 at a
       nominal 0.90. A shared round effect cancels under pairing (width < 0.05). Each cell's
       lifted helper returns exactly the shared result.
     - **re-judged committed rows on round intervals (facts; no JSON rewritten, no decision
       reversed):**

       | Rule | Rows | Fix phase 3 (iid) | Fix phase 4 (round) | Changed |
       |---|---|---|---|---|
       | C7 RAL CPU (decides) | solved / plain saved ms | [0.0367, 0.0408] / [0.0439, 0.0469], NO_GO | [0.0360, 0.0414] / [0.0438, 0.0470], NO_GO | no |
       | C7 A100 | solved / plain | GO / GO | GO / GO | no |
       | C6 `hpc_ral_gpunode_cpu_fp64` (decides) | 8 lane × route rows | 6 GO, 2 NO_GO | 6 GO, 2 NO_GO (intervals ~1.3× wider) | no |
       | C6 `hpc_a100_fp64` | 8 rows | 4 GO, 4 NO_GO | 4 GO, 4 NO_GO | no |
       | C6 `local_cpu_fp64` | solved `rev_analytic` | GO, ratio [0.810, 0.846] | **INCONCLUSIVE**, ratio [0.805, 0.854] straddles 0.85 | **yes** |
       | C6 `local_cpu_fp64` | the other 7 rows | 4 GO, 3 NO_GO | unchanged | no |
       | C8 crossover | 3 configs × 2 lanes × 2 shapes | "fwd wins through n=24 / 27", fraction `none` 1.0 | identical | no |
       | C10 RAL CPU | tie set | {e2.5_s0.4, e2.5_s0.2, e3_s0.4, e3_s0.3, e4_s0.4} | identical | no |
       | C10 laptop (3 rounds) | tie set | {e3_s0.4, e2.5_s0.4, e3_s0.3} | every candidate (3 < 5 rounds) | **yes** |
       | C10 other 7 sweeps | tie set / best | as fix phase 3 | identical | no |

       Widths of the committed C10 speed-up intervals, round / iid (median per file): RAL CPU
       1.04, MCS RAL CPU 1.12, MCS laptop 1.36, step-0 EPYC 1.47, step-0 RAL CPU 1.84, step-0
       laptop 2.29, A100 ≈ 1. The one changed C6 call is a laptop row; phase 2c is decided by
       the RAL CPU row, which is unchanged, so no recorded decision rests on it.
     - **limits:** a percentile bootstrap over 20 clusters is approximate (coverage 0.85 in the
       witness at a nominal 0.90); rounds are treated as exchangeable, so a slow drift across
       rounds is absorbed as scatter, not modelled.
5. **Headline estimator and GPU-only marker (P8, P9).**
   - **Change:** opt every runtime cell into `steady_median_profile` beside the legacy block mean,
     with the dashboard reading the median where present. A timeout records INCONCLUSIVE /
     "timed out on host X at load Y" and is re-measured before being rendered as GPU-only.
   - **Witness:** an injected-clock transient (the existing T4 pattern) where the block mean moves
     2.4× and the median does not. A marker with a load above the cap is not rendered as GPU-only.
   - **Shipped (phase 6 of #362, PR #408, 2026-10-09).** As implemented:
     - **P8, the producer:** one helper, `scripts/misc/likelihood_breakdown/timing.py::headline_steady_median`,
       runs `steady_median_profile` on the cell's compiled `full_pipeline` after the legacy block
       (no `Timer` section, so `full_pipeline_single_jit` and compile / first-call sections are
       byte-for-byte what they were) and returns `full_pipeline_single_jit_median` (s, the key
       `catalogue_adapters.DIRECT` already maps to `single_jit_median`), `_median_ms`, `_p10_ms`,
       `_p90_ms` and `_median_protocol` (n_warm, n_timed, statistic, what the block mean is).
       `n_timed` is 30 s over the block mean, clamped to [20, 200]: an A100 call keeps the 200 of
       job 366914, a 1.5 s CPU call takes 20 (~38 s with its warm calls). Above a 2 s block mean
       (`HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S`) no median is taken and the legacy headline stands, so a
       48.8 s laptop call is not extended by ~20 min past `--per-run-timeout` or an HPC wall
       (review finding). Wired into the 11 runtime
       cells that headline the block: imaging delaunay / delaunay_nn / mge / rectangular,
       interferometer delaunay / mge / rectangular, point-source image-plane plain / solved,
       source-plane plain / solved (the last moved off its own #371 copy onto the helper).
     - **P8, the reader:** `build_dashboard._point` headlines the median where a row records one
       beside a single-jit headline (`headline_estimator: "steady median"`,
       `single_jit_block_mean_s` beside it; both only on such points, so every other point is
       unchanged); the table and tooltip label the estimator. `drift` returns
       `estimator-mismatch` when the two endpoints use different estimators, published
       `insufficient` with "endpoints use different headline estimators … not compared"; the
       status vocabulary is unchanged. A median is not a repeat summary: one run's median is one
       summary, so in-band comparisons stay `insufficient` (P7). The v1 record carries
       `single_jit_block_mean_s` in `measurement` and a limitation line only where the median is
       the headline.
     - **P9:** `sweep.py` writes `{cpu_unusable, verdict: "INCONCLUSIVE", outcome: "timeout",
       reason: "timed out after Ns on host X at load a/b …", timeout_seconds, elapsed_s, host,
       loadavg_at_start, loadavg_at_timeout, marker_schema: 2}` (`cpu_unusable` kept for the
       readers that key on it). `build_dashboard.marker_verdict` builds a point from the marker
       and calls `qualify`; only a qualified marker is "GPU-only". `--skip-existing` skips a
       result JSON or a qualified marker and re-measures anything else. `aggregate.py` carries the
       marker's verdict / host / loads / timeout into `comparison.json`; `build_readme.py` renders
       a qualified marker **GPU-only** and any other in italics as "timed out (inconclusive)" or
       "did not finish (inconclusive)" (no timeout recorded, e.g. an OOM).
     - **witness** (`test_headline_estimator_and_marker.py`, T9): a transient-laden first block
       (one call +37.5 ms over a 2.67 ms steady state) moves the block mean 2.40× while the median
       of 40 timed calls (one 10× spike) stays 1.00×; legacy rows flag that as `drifted`, median rows
       do not; drift between a median and a block-mean endpoint is `estimator-mismatch` →
       `insufficient`; markers above the cap (at start or at timeout), with no load, no host, on a
       laptop, off the pinned node, or written before this phase are never GPU-only; a timeout
       marker records INCONCLUSIVE with host and load; `--skip-existing` runs a cell whose only
       evidence is an unqualified marker.
     - **contract:** no `profiling-summary@2` field or metric changes. `build_catalogue.py` is not
       edited; it indexes the markers as `indexed_only` (no measured fields, before and after).
       The median reaches v2 only on rows measured after this phase, as the existing
       `runtime` / `single_jit_median` metric (statistic `median`) the adapter already declared.
       v1 (`dashboard/summary.json`, read by the project badge, not by Pulse live) gains the
       conditional `single_jit_block_mean_s` measurement and limitation described above; neither
       is present today.
     - **re-judged committed rows (facts; no JSON rewritten, no compute run):**

       | What | Committed | Before | After | Changed |
       |---|---|---|---|---|
       | rows carrying `full_pipeline_single_jit_median*` | 0 | — | — | no headline, drift badge or comparison changed; `dashboard/` re-renders identical (`--check` current, 145 series) |
       | `.unusable.json` markers | 4, all `local_cpu_*`, written 2026-07-11 (#62), none with host or load | "GPU-only" | inconclusive (not a reference host class; no host / load) | **yes** |
       | `results/runtime/interferometer/delaunay/alma/delaunay_local_cpu_fp64` (laptop OOM, 2858 s) | the only marker in a committed `comparison.json` (also the `PreOptimizationTimes` baseline copy) | README **GPU-only** | _did not finish (inconclusive)_ in `README.md` and `scripts/misc/likelihood_runtime/README.md` | **yes** |
       | `datacube/delaunay/alma` fp64 (3600 s timeout), `interferometer/delaunay/alma_high` fp64 (~1 h kill) / mp (killed, sibling > 1 h) | not aggregated into any committed `comparison.json` | not rendered | timed out / timed out / did not finish (inconclusive) if aggregated | no rendered change |

       No recorded decision rests on the alma OOM label: its `local_cpu_mp` sibling has a measured
       6.51 s row, and the GPU-first campaign doctrine did not depend on it.
     - **not changed (remainders):** the README runtime table still reads the legacy headline
       (`full_pipeline_per_call`, the aggregator's alias of the block mean); the breakdown cells,
       the datacube cell (`scripts/datacube/delaunay/likelihood_runtime.py`, whose
       `full_pipeline_cube_single_jit` is a 3-call block after compile) and the two `mge_mass`
       cells are not opted in; calls slower than 2 s get no median; `single_jit_repeats` (≥ 2
       independent runs per release) is still written by no producer; the wall basis in
       `wall/rates.py` does not include the median's extra calls (at most 25 calls of ≤ 2 s,
       ~50 s, per cell).
6. **Warm-up flag and witness band (P3, P5).**
   - **Change:** rows after an unsettled warm-up are INCONCLUSIVE for any timing verdict. P5 gains
     host class and an INCONCLUSIVE state off the reference host class.
   - **Witness:** synthetic flat / ramp / step sequences (the existing T3 helper).
   - **Shipped (phase 7 of #362, PR #409, 2026-10-09; stacked on #408).** As implemented:
     - **P3, the check:** `scripts/misc/likelihood_breakdown/warmup_gate.py`
       `warmup_unsettled_reason(warmup)`: `None` for no record (the CI fixture, the runtime
       cells) or `steady: true`; otherwise "warm-up never settled (N call(s), last-3 median … vs
       previous-3 … (x % > 10 %)): …", and "… carries no steady flag" for a record without one.
       Pure, stdlib only.
     - **P3, the consumers** (enumerated from this inventory; every row that reads a post-warm-up
       row): `abba_overhead_verdict(..., warmup=)` returns INCONCLUSIVE after the gross guard and
       before the block-count rule, so a PASS and a FAIL become INCONCLUSIVE (the cell keeps the
       row and does not raise) and FAIL_GROSS still fires; the cell passes `warmup=_warmup` and
       prints the reason beside "NEVER SETTLED". The promotion block (P2) writes
       `warmup_unsettled_rows` and INCONCLUSIVE whatever the speedup verdict, so neither a
       `timing_candidate` nor a measured NO_LEVER comes from an unsettled arm.
       `build_dashboard.qualify` leaves a point unqualified with the reason when its payload
       carries an unsettled warm-up record, top-level or under `rows[*]` (no scanned payload carries
       one today: the fixed-light rows produce no dashboard point). Not consumers: P4, the cached-site counts
       and the dispatch asserts (coverage / correctness, not timing verdicts), C1–C5 and P10
       (their own warm-up protocols, no `steady` flag).
     - **P3, the rule:** kept at 3 vs 3 / 10 % / 12 calls; its limits are stated under P3 and
       pinned as witnesses rather than changed.
     - **P5:** `witness_verdict(cold_eval_median_s, instrument, host_class)`; PASS / FAIL only on
       `is_reference_host_class(host_class)` (imported lazily from `build_dashboard`, one rule
       for P5, P6 and P9), else INCONCLUSIVE "off reference host class (<class>)"; records
       `host_class`, `in_band`, `reason`; band and `WITNESS_FACTOR` unchanged. The four callers
       (`scripts/imaging/{rectangular,delaunay}/likelihood_{runtime,breakdown}_numba.py`) pass
       `_cli.config_name` and print the verdict with its reason.
     - **witness** (T3 additions and T10, deterministic): through the cell's own
       `_warm_to_steady_state` — flat (6 calls), the recorded 2026-09-15 flat run (6), a step
       0.4 → 0.3 at call 4 (8) settle; the recorded queueing decay, a geometric ramp and a
       5 %/call ramp never settle (12); limits: a 3 %/call ramp settles (6), a step at call 9 is
       unseen (6), a period-2 ±15 % never settles (12). An unsettled record turns a clear PASS
       and a clear FAIL (and the cell's own pinned 224 ms PASS / 400 ms FAIL statements)
       INCONCLUSIVE, keeps `[1.6] × 3` FAIL_GROSS, and turns a promotion GO and a NO_LEVER
       INCONCLUSIVE on either arm. P5: reference-class rows keep PASS / FAIL (HST 0.80 s PASS,
       1.463 s FAIL; Euclid 0.70 s PASS, 0.2346 s FAIL); `local_*` and untagged rows are
       INCONCLUSIVE whatever the band says. Dashboard: an unsettled or flag-less warm-up record
       is unqualified; no record or a scalar warm-up time is unaffected.
     - **contract:** no `profiling-summary@2` field, metric or meaning change; `build_catalogue.py`
       is not edited (its catalogue is re-rendered for the source hash only). v1 points gain
       `warmup_unsettled` only where a payload carries an unsettled record (none today).
     - **re-judged committed rows (facts; no JSON rewritten, no compute run):**

       | What | Committed | Before | After | Changed |
       |---|---|---|---|---|
       | fixed-light numba rows with a warm-up record | 28 (17 RAL decomposed, 3 laptop `sparse_t1`, 8 smoke) | 28 steady (22 settle at 6 calls, all 17 RAL rows among them; 3 laptop and 3 smoke rows at 8–12) | 28 settled: no P1, P2 or dashboard verdict moves | no |
       | closest calls | laptop `c_sparse_numba` settled at 10 calls with a 10.0 % window change; smoke `a_sparse_numba` at the 12-call cap with 9.5 % | steady | steady (the rule is unchanged) | no |
       | P5 witness verdicts (2026-09-08, laptop i9-10885H, untagged, no provenance) | 8: rectangular hst / euclid × runtime / breakdown PASS; Delaunay hst / euclid × runtime / breakdown FAIL | 4 PASS, 4 FAIL | 8 INCONCLUSIVE, "off reference host class (untagged)"; `in_band` matches the published reading | **yes** |

       No go / no-go / promotion call rests on the P5 verdicts. One recorded reading loses its
       timing support: `production_representative_cells.md` "the two rectangular rows pass on
       both instruments, so the protocol itself is sound" leaned on laptop PASSes; the
       configuration match it records is unaffected, and the Delaunay misses were already
       attributed to dataset and host there (and to Phase 4 of `imaging_over_sampling`). That ledger
       carries a dated re-judgement note above its witness reading.
     - **not changed (remainders):** phase 3b (C1 / C3 / C4 / C5 intervals); the family-wise
       policy for multi-route rules; the runtime README and the breakdown / datacube / `mge_mass`
       cells still on the legacy headline; `single_jit_repeats` written by no producer; P2's
       between-row drift; a v2 qualification, which must reuse `is_reference_host_class`.

The Brain `COMPILE_DRIFT_RATIO` rule shares P7's point-vs-point limitation. If phase 2 changes the
local drift semantics, a separate Brain task should decide whether compile drift follows; this
audit does not edit Brain.

## (c) Statement of no change

Phase 1 changed no gate, budget, tolerance, cutoff, estimator, repeat count, warm-up rule,
qualification rule, production instrument, notebook, library or Heart file. It ran no compute job.
It added this note, the read-only lister `scripts/misc/tooling/list_timing_assertions.py` (with
its `--check` step in `lint.yml`), the [Measurement tools](../../wiki/campaigns/measurement_tools.md)
campaign page and index row, and pointer lines in `AGENTS.md` and `scripts/misc/tooling/README.md`.

## Reproduce

```bash
python scripts/misc/tooling/list_timing_assertions.py          # the mechanical inventory
python scripts/misc/tooling/list_timing_assertions.py --all    # + validity comparisons
python scripts/misc/tooling/list_timing_assertions.py --check  # fails if a key is missing here
```
