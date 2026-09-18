# CPU NNLS memo robustness across graded draws — phase 5

> **Status correction — 2026-09-18:** The CPU epic is complete. Phase 5b subsequently found NO_LEVER and phase 7 was explicitly shelved. Assume zero dependable memo benefit for planning and leave production defaults unchanged. Historical Next suggestions below are superseded by the [current campaign summary](profiling_campaign_status_2026_09.md).

Issue: https://github.com/PyAutoLabs/autolens_profiling/issues/278. Campaign: `fixed-lens-light-numba-cpu`, phase 5. Measured 2026-09-18.

**Study status: PASS.** The production memo is compared with the same production likelihood with memo disabled. No solver or library change was made.

The memo remains numerically correct on all 82 warm/cold model comparisons, but
it is **56–60% slower over these broad-proposal stress sequences**. Graded order
costs 383 → 611 ms/call; shuffled order costs 382 → 595 ms/call. This is stable
across all five repeats in each order. It is not evidence that every posterior
or nearby-proposal workload would be slower.

The diagnostic counts identify extra solver work: warm starts increase outer
iterations from 994 to 3665 (graded) or 3093 (shuffled), and inner iterations
from 108 to 647 or 474. The guard invalidates 18/22 and 19/21 memo-seeded solves,
respectively, but only after the expensive solve has completed. There are **no
exception-triggered retries**. Nearby graded models can still benefit; the
per-model table and figure retain those improvements rather than hiding them
behind the sequence average.

## Whole-likelihood measurements

These are single-threaded HST Delaunay N=1500, fp64, CPU sparse-API calls to `AnalysisImaging.log_likelihood_function(use_jax=False)`. A sequence starts with an empty memo and evaluates each model once. Each row is the median of five complete sequence costs divided by 41, including the first cold solve.

| Order | Cold ms/call | Warm ms/call | Cold / warm | Five paired sequence ratios |
|---|---:|---:|---:|---|
| graded | 382.888 | 611.286 | 0.626x | 0.629, 0.624, 0.627, 0.627, 0.626 |
| permuted | 381.977 | 594.973 | 0.642x | 0.642, 0.643, 0.642, 0.641, 0.640 |

Five traversals per lane quantify repeat scatter on one host; this is not a sampler trajectory or a distribution over posterior models. The graded order is the manifest order; the second order is its recorded seed-278 permutation. Warm/cold lane order alternates between repeats. Setup, compilation and instrumentation are outside these timings. These are different model streams from phase 4b; do not multiply this ratio into its cumulative speedup or compare absolute milliseconds as a like-for-like change.

## Solver behaviour

| Order | Memo hits / 41 | Cold entries / 41 | Guard invalidations | Exception retries | Cold outer / inner iterations | Warm outer / inner iterations |
|---|---:|---:|---:|---:|---:|---:|
| graded | 22 | 19 | 18 | 0 | 994 / 108 | 3665 / 647 |
| permuted | 21 | 20 | 19 | 0 | 994 / 108 | 3093 / 474 |

An invalidation means the memo seed exceeded the production error guard and its entry was dropped **after the current solve**; the next evaluation with that key is cold. An exception retry means a failed memo kernel attempt was followed by a cold solve in the same evaluation. They are separate events. The unchanged library default guard is 1.5 times the error fraction of the latest dense-sign seed; no GPU PDIP fallback is involved.

## Numerical gates

| Order | Exact active-set membership | Max evidence difference (nats) | Max relative evidence difference | Max relative reconstruction difference |
|---|---:|---:|---:|---:|
| graded | 41/41 | 3.347e-10 | 2.859e-14 | 5.670e-13 |
| permuted | 41/41 | 3.056e-10 | 5.717e-14 | 9.009e-13 |

The acceptance gate requires finite evidence/reconstruction, exact passive-set membership (factor ordering can differ), and relative evidence difference <= 1e-9. Reconstruction differences are reported separately. Every clean timed evidence is also checked against the corresponding observed traversal. No factor determinant comparison is made in this phase; the approved roundoff-aware phase-4b factor check has not been substituted for these gates.

Gate record: `{"complete_campaign_cell": true, "numerical_equivalence": true, "single_thread": true, "slope_promotion": true}`.

## Construction, provenance and limitations

- The input is the frozen GPU-study manifest: 1 fiducial, 16 one-parameter walks and 24 seed-0 random models. The harness validates the exact SHA256, names, families and finite offsets. It does not use the GPU solver or its timings.
- Every model gets an S0 lens-light solve and a separately rebaked source-only dataset before timing. This matches the graded-draw study and differs from holding one lens-light subtraction fixed across an entire inference run. Preparation seconds and subtraction fluxes are recorded per draw.
- Current CPU likelihood drops are recomputed against each mass family's own fiducial. Requested targets and historical values remain under `manifest_draw`; `achieved_d_log_l_cpu` is the current value. Old GPU fields are provenance only.
- PowerLaw(slope=2) promotion relative difference is 3.279e-05, against the inherited 1e-4 check. This family check is distinct from the warm/cold 1e-9 evidence gate.
- RAL job **343398**, `euclid-ral-gpu-1`, AMD EPYC 7702 64-Core Processor. Four allocated CPUs, one likelihood thread, no GPU requested. BLAS thread environment variables were pinned to one before import, and Numba reported one runtime thread. `threadpoolctl` returned an empty pool list on this stack, so a runtime BLAS thread count was not available; the single-thread gate establishes the pinned BLAS configuration and observed Numba count. Host load before/after timing: [1.0, 0.8, 0.42] / [1.0, 0.97, 0.68].
- The job ran from a private SHA-verified source snapshot. The source sidecar below records every input in the 45-file execution snapshot, including instrument/config inputs and the CPU model queried inside this allocation. The result records imported library revisions, individual source hashes, dataset hashes and the empty threadpool inspection result.
- The library imports JAX indirectly during CPU fits; the harness imports no JAX itself and dispatch asserts `InversionImagingSparseNumba` on every diagnostic fit. These are CPU likelihood timings.

## Per-model comparison

Ratios below use each model's median clean-call time. `dense` marks a cold seed in the memo-enabled traversal. `drop` is post-solve memo invalidation.

| Model | CPU ΔS0 (nats) | Graded cold/warm | Graded seed / drop | Permuted cold/warm | Permuted seed / drop |
|---|---:|---:|---|---:|---|
| fiducial | 0.00 | 1.003 | dense / False | 1.001 | dense / False |
| walk_einstein_radius_m10 | -11.62 | 1.264 | memo / False | 0.997 | dense / False |
| walk_einstein_radius_m100 | -80.46 | 1.283 | memo / True | 1.000 | dense / False |
| walk_einstein_radius_m1000 | -963.81 | 0.998 | dense / False | 0.998 | dense / False |
| walk_einstein_radius_m1e4 | -8740.44 | 0.731 | memo / True | 0.766 | memo / True |
| walk_ell_comps_0_m10 | -10.10 | 0.996 | dense / False | 0.999 | dense / False |
| walk_ell_comps_0_m100 | -88.25 | 1.277 | memo / True | 0.998 | dense / False |
| walk_ell_comps_0_m1000 | -868.90 | 0.995 | dense / False | 0.996 | dense / False |
| walk_ell_comps_0_m1e4 | -11241.36 | 0.750 | memo / True | 0.745 | memo / True |
| walk_centre_x_m10 | -10.48 | 1.001 | dense / False | 0.423 | memo / True |
| walk_centre_x_m100 | -113.34 | 1.287 | memo / False | 1.287 | memo / False |
| walk_centre_x_m1000 | -1086.09 | 1.242 | memo / False | 0.365 | memo / True |
| walk_centre_x_m1e4 | -11188.35 | 0.765 | memo / True | 0.998 | dense / False |
| walk_slope_m10 | -18.93 | 0.997 | dense / False | 0.998 | dense / False |
| walk_slope_m100 | -85.80 | 1.288 | memo / False | 1.283 | memo / False |
| walk_slope_m1000 | -1120.02 | 1.190 | memo / True | 0.999 | dense / False |
| walk_slope_m1e4 | -11383.51 | 0.996 | dense / False | 0.381 | memo / True |
| random_00 | -7528.30 | 0.410 | memo / True | 0.803 | memo / True |
| random_01 | -41832.14 | 1.000 | dense / False | 0.659 | memo / True |
| random_02 | -45216.18 | 0.329 | memo / True | 0.999 | dense / False |
| random_03 | -56118.43 | 0.998 | dense / False | 0.605 | memo / True |
| random_04 | -13780.19 | 0.273 | memo / True | 0.785 | memo / True |
| random_05 | -11787.94 | 0.998 | dense / False | 0.839 | memo / True |
| random_06 | -60534.24 | 0.540 | memo / True | 0.998 | dense / False |
| random_07 | -33771.46 | 0.996 | dense / False | 0.999 | dense / False |
| random_08 | -40845.22 | 0.286 | memo / True | 0.998 | dense / False |
| random_09 | -39756.99 | 1.000 | dense / False | 1.001 | dense / False |
| random_10 | -31685.52 | 0.415 | memo / True | 0.408 | memo / True |
| random_11 | -48291.24 | 1.001 | dense / False | 0.629 | memo / True |
| random_12 | -38392.64 | 0.339 | memo / True | 0.436 | memo / True |
| random_13 | -32389.29 | 1.001 | dense / False | 0.998 | dense / False |
| random_14 | -9588.23 | 0.295 | memo / True | 0.298 | memo / True |
| random_15 | -44268.38 | 0.999 | dense / False | 0.398 | memo / True |
| random_16 | -15368.28 | 0.425 | memo / True | 0.233 | memo / True |
| random_17 | -97594.87 | 0.999 | dense / False | 0.561 | memo / True |
| random_18 | -20674.77 | 0.233 | memo / True | 0.720 | memo / True |
| random_19 | -97821.06 | 1.000 | dense / False | 1.000 | dense / False |
| random_20 | -48544.83 | 0.432 | memo / True | 0.999 | dense / False |
| random_21 | -46221.26 | 0.998 | dense / False | 0.180 | memo / True |
| random_22 | -53623.06 | 0.423 | memo / True | 0.999 | dense / False |
| random_23 | -56923.16 | 1.002 | dense / False | 1.000 | dense / False |

## Validation and artifacts

- Eight focused tests cover frozen-manifest integrity, memo state restoration, sequence ordering, exception retry versus invalidation, and numerical failure reporting. Ruff, formatting, shell syntax, live API audit and changed-cell import smoke pass. A two-model N1500 local smoke passed the numerical, slope-family and single-thread gates; it is explicitly incomplete as a campaign.
- Independent final review: CLEAN on the staged code and artifacts. The reviewer independently recomputed the timing ratios, numerical maxima, iteration/event totals and sample counts, and verified all 45 snapshot input hashes. The note records the runtime BLAS-inspection limitation explicitly.
- [Full result](../breakdown/imaging/fixed_light_numba_draws_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s5.json)
- [Figure](../breakdown/imaging/fixed_light_numba_draws_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s5.png)
- [Source and job record](fixed_light_numba_s5_source_job343398.json)

## Next phase

Phase 6 is source-pixel scaling and should carry both memo-on and memo-off controls.
A useful intervening optimization task is to investigate avoiding poor memo seeds
before paying for their warm solve, while retaining nearby-model benefits and
all correctness gates. This measurement supplies new evidence for a memo policy
study; the artificial stress distribution alone does not justify changing the
production default. No library change or phase-4b prototype promotion is part of
this task.
