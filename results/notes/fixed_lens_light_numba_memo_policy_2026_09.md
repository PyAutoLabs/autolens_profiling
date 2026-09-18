# CPU memo eligibility before solving — phase 5b (#280)

**NO_LEVER under the declared promotion criteria.** The locked precheck removes
35–38% of unchanged-memo runtime on the original stress orders, but is 11.7%
slower than unchanged memo on the independent nearby walk and 5.3% slower than
cold on the independent broad sample. Numerical correctness passes throughout.
The finding changes the design direction: this residual score is insufficient
as a general memo eligibility rule. Production library behavior is unchanged.

## Scope

This profiling-only prototype follows [phase 5](fixed_lens_light_numba_memo_2026_09.md),
which found broad-proposal memo slowdowns despite numerical agreement. It leaves
PyAutoArray and its defaults unchanged. The branch depends on profiling PR #279.

## Declared experiment

The precheck evaluates the current KKT residual at the previous reconstruction,
using absolute residuals on the previous passive set and negative residuals on
inactive coordinates. The maximum violation is divided by the larger infinity
norm of the current matrix-vector product and data vector. Zero scale is
handled explicitly. No current solution, final active set, evidence or runtime
is available to this decision. A rejected entry is removed before the unchanged
cold path; an accepted entry uses the production warm solve and post-solve guard.
The prototype vector cache follows the production memo lifetime and is isolated
between traversals.

Calibration uses eight nearby and eight broad draws, seed 278. Nearby increments
are 0.05 times the original prior sigmas; broad draws use the original 5-sigma
recipe. Neither clips its offsets. The seven thresholds are 0, 0.0001, 0.001,
0.01, 0.1, 0.5 and 2. Each candidate receives three complete traversals per lane in a balanced Latin
cycle. Candidates are visited in the declared ascending grid order.
Selection minimizes the larger of guarded/cold broad time and guarded/memo nearby
time among numerically passing candidates, breaking ties toward smaller thresholds.
The declaration is written before calibration; the selected threshold and its
hash are locked before any holdout preparation.

Evaluation comprises the frozen 41-model sequence in graded and seed-278 shuffled
order, a 32-model nearby walk and 24 broad draws at independent seed 1. Six full
traversals use all six lane permutations, with independent empty memo caches.
Timings cover the entire production Analysis likelihood, including the precheck
and reconstruction-cache maintenance. Per-model S0 lens-light solves and S3
subtraction preparation are excluded, as in phase 5. Diagnostics are separate.

Correctness requires exact active-set membership and <=1e-9 relative evidence
agreement with cold solves; clean timed evidence must also agree with diagnostic
calls. No factor determinant criterion is used. Runtime targets are >=5% faster
than unchanged memo on both original orders, <=3% overhead versus cold on broad
evaluations and <=3% overhead versus memo on the nearby holdout. These are
empirical acceptance targets, not confidence bounds.

False accepts and rejects are measured separately, after each actual guarded
fit, using four alternating cold/memo solver pairs from the identical
pre-decision system and memo snapshot. These are solver-only diagnostics with
a 3% neutral band, excluding precheck overhead; they never feed the policy or
threshold selection. All actual guarded state is restored between replays.

Additional per-model accepted/rejected timing classifications are descriptive cross-lane
comparisons, not causal false-decision rates: different policies create different
memo histories. Repeat scatter and absolute/relative numerical differences are
retained in the artifact. No production recommendation follows from an incomplete
or failed campaign.

## Calibration lock

Threshold **0.5** was locked at `2026-09-18T10:33:28Z` before any
holdout was prepared. Every calibration candidate passed numerical checks.

| Threshold | Broad guarded/cold | Nearby guarded/memo | Selection score (maximum) |
|---:|---:|---:|---:|
| 0 | 1.0000 | 1.2055 | 1.2055 |
| 0.0001 | 1.0013 | 1.2038 | 1.2038 |
| 0.001 | 1.0022 | 1.2061 | 1.2061 |
| 0.01 | 1.0056 | 1.2052 | 1.2052 |
| 0.1 | 1.0045 | 1.2050 | 1.2050 |
| 0.5 | 1.0000 | 1.0388 | 1.0388 |
| 2 | 1.6004 | 1.0048 | 1.6004 |

This is a small exploratory calibration, not uncertainty-controlled optimization:
8 + 8 models, three repeats and a predeclared ascending candidate order. The
selected threshold did not itself meet the 3% nearby target in calibration;
the independent holdout remains the declared decision test.

The raw [threshold lock](../breakdown/imaging/fixed_light_numba_memo_policy_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s5b.locked.json)
has SHA256 `cf19fdd9d31a100b3a9c4a64b6fc7697febca9512f6d4de664e6b13e66a7d22c`.

## Results

Times are the median complete-traversal total divided by the number of models,
in ms/model. Negative percentages mean the guarded lane is faster.

| Sequence | Models | Cold | Existing memo | Guarded | vs cold | vs memo |
|---|---:|---:|---:|---:|---:|---:|
| graded | 41 | 385.053 | 613.080 | 379.684 | -1.39% | -38.07% |
| permuted | 41 | 384.774 | 600.198 | 387.327 | +0.66% | -35.47% |
| nearby_holdout | 32 | 270.875 | 212.352 | 237.108 | -12.47% | +11.66% |
| broad_holdout | 24 | 453.099 | 759.296 | 477.017 | +5.28% | -37.18% |

Both original-order speed targets pass. The independent broad <=3% cold-overhead
and nearby <=3% memo-overhead targets fail. The policy is still faster than
cold on nearby models, but it gives back a substantial part of the existing
memo benefit; that is why cold alone is not the nearby acceptance baseline.

Full repeat spread, `(max-min)/median` of the six sequence totals:

| Sequence | Cold spread | Memo spread | Guarded spread |
|---|---:|---:|---:|
| graded | 0.423% | 0.584% | 0.135% |
| permuted | 0.423% | 0.330% | 0.367% |
| nearby_holdout | 0.529% | 0.718% | 0.567% |
| broad_holdout | 0.346% | 0.157% | 0.369% |

The two failed performance targets are much larger than the observed repeat
scatter. These ranges are descriptive, not confidence intervals.

## Decisions and design implications

All 128 eligible matched transitions agree on final active sets. Missing memo
entries are excluded from the false-decision denominator. The final column is
diagnostic precheck time per model, including empty lookups; headline timing
also includes cache maintenance and all other policy overhead.

| Sequence | Eligible | Accept | Reject | False accept | False reject | Postguard drops | Precheck ms/model |
|---|---:|---:|---:|---:|---:|---:|---:|
| graded | 38 | 5 | 33 | 0 | 2 | 2 | 2.005 |
| permuted | 40 | 0 | 40 | 0 | 3 | 0 | 2.120 |
| nearby_holdout | 28 | 19 | 9 | 0 | 9 | 3 | 1.842 |
| broad_holdout | 22 | 1 | 21 | 1 | 0 | 1 | 1.944 |

There were zero exception retries in every lane. On the nearby holdout,
unchanged memo had 31 hits and zero postguard invalidations; guarded memo had
19 hits, nine precheck rejections and three postguard invalidations. Rejections
refresh the production cold baseline and thereby affect later guard decisions,
so the cost includes this changed cache history, not just nine isolated calls.

The one broad false acceptance is `seed1_broad_03`: score 0.474503 passes the
locked 0.5 threshold, but the matched warm solve takes 2.514 times the cold
solve. Its whole-likelihood median is 1064.737 ms guarded versus 529.089 ms
cold. A single expensive acceptance is enough to miss the broad overhead
target despite rejecting 21 other available seeds. All nine rejected nearby
seeds are beneficial in matched replays.

This supports two design conclusions. A residual measured at the previous
reconstruction is not a reliable proxy for the cost of reusing its passive set:
it can reject useful sets and accept costly ones. Any future eligibility rule
also needs to be tested with the production post-solve guard, because rejection
changes its baseline and subsequent memo lifetime. A replacement signal or
policy requires a separate study and a fresh holdout; this task did not retune
on seed 1. These are research results, not a production runtime improvement.

## Numerical validation and provenance

- All **276/276** evaluation memo/guarded-to-cold comparisons pass exact active-set
  membership and <=1e-9 relative evidence agreement. All clean timing evidences
  agree with their diagnostic traversals.
- Maximum evidence difference: **5.5661076e-10 nats** absolute,
  **5.7173334e-14** relative. Maximum reconstruction relative difference:
  **9.0090379e-13**. All **224/224** calibration comparisons also pass.
- All completeness, unchanged-lock, slope-promotion and thread-configuration
  gates pass. Performance targets are separate from these correctness gates.
- RAL job **343413**, `COMPLETED 0:0`, elapsed **01:01:18**, peak RSS
  **8155204 KiB**. AMD EPYC 7702, node `euclid-ral-gpu-1`, four allocated CPUs,
  no GPU allocation. Numba reported one runtime thread; BLAS thread environment
  variables were pinned to one. `threadpoolctl` returned no pools, so runtime
  BLAS thread count was **unavailable**, not observed as one.
- The four imported library revisions match phase 5. The 47-file private source
  snapshot verified before and after the run; all library checkouts retained
  their original revisions and were tracked-clean. The run has no git directory,
  hence `repo_revision: null`; the source archive/base/hash sidecar supplies
  provenance. Local PyAutoLens had advanced, so local smoke timing is not used.
- Final artifacts preserve all calibration candidates, timings, active sets,
  counters and matched replay results. The JSON was reformatted for review with
  parsed-data equality asserted; both raw and formatted hashes are recorded.
- 24 focused phase-5/policy tests, full-size two-model numerical and matched-state
  smoke, Ruff check/format, import, API, shell and README checks pass.
  Independent pre-run and complete empirical review are **CLEAN**, including
  recomputation of all summaries, gates, decisions and provenance checks.

[Full JSON](../breakdown/imaging/fixed_light_numba_memo_policy_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s5b.json) ·
[Timing plot](../breakdown/imaging/fixed_light_numba_memo_policy_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s5b.png) ·
[Predeclared protocol](../breakdown/imaging/fixed_light_numba_memo_policy_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s5b.declaration.json) ·
[Source/job sidecar](fixed_light_numba_s5b_source_job343413.json)
