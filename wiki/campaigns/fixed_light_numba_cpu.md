# Fixed lens light on the numba CPU path

**Status:** complete
**Question:** On the production numba CPU path, where does the fixed-lens-light (source-only) likelihood spend its time, and how fast can it be made without changing the answer?
**Pre-registered rule:** per phase (see the table); common gates are the library's own log evidence to ≤ 1e-9 relative, single-threaded (numba 1, BLAS 1) on RAL `gpu`-partition CPUs only, and no comparison crossing a thread setting. Phases 4a/4b: lever iff ≥ 5 % faster on the whole call with ABBA PASS (4b also every witness draw PASS); phase 5b had separate performance targets.
**Verdict:** fixing the lens light is 2.03x (a modelling choice); the NNLS round closed with no solver change (the memo already covers it); three non-solver PyAutoArray levers took the memo-on call to 1.80x; 4a and 4b found no lever; memo robustness (5) and memo precheck (5b) found NO_LEVER, production defaults left alone; scaling (6) showed NNLS takes over at large N. Epic closed at the user's request 2026-09-18, phase 7 shelved.
**Headline:** 413.3 → 230.0 ms (1.80x) from three library levers, route b memo ON, HST Delaunay N=1500, 1 thread, RAL `euclid-ral-gpu-1` EPYC 7702, jobs 343345 → 343356. The chain depends on bridge control job 343355, which is not in the tree. Separately, 932.4 → 459.2 ms (2.03x, job 343311, memo OFF) is the modelling choice and must not be multiplied with it.
**Library PRs:** PyAutoArray#553, PyAutoArray#554, PyAutoArray#555 (released 2026.9.19.1); PyAutoArray#557, docstring only (released 2026.9.19.1).
**Profiling PRs:** #264, #266, #269, #271, #272, #275, #277, #279, #281, #283, #285.
**Ledger:** [profiling_campaign_status_2026_09.md](../../results/notes/profiling_campaign_status_2026_09.md), [fixed_lens_light_numba_2026_09.md](../../results/notes/fixed_lens_light_numba_2026_09.md), [fixed_lens_light_levers_2026_09.md](../../results/notes/fixed_lens_light_levers_2026_09.md), [fixed_lens_light_s4_2026_09.md](../../results/notes/fixed_lens_light_s4_2026_09.md), [fixed_lens_light_numba_memo_2026_09.md](../../results/notes/fixed_lens_light_numba_memo_2026_09.md), [fixed_lens_light_numba_memo_policy_2026_09.md](../../results/notes/fixed_lens_light_numba_memo_policy_2026_09.md), [fixed_lens_light_numba_scaling_2026_09.md](../../results/notes/fixed_lens_light_numba_scaling_2026_09.md)
**Mind contract:** epic `fixed-lens-light-numba-cpu`; campaign map and close-out in `complete/2026/09/fixed-lens-light-numba-cpu.md`; phase records `complete/2026/09/fixed-light-numba-{phase1,solver,levers,s4,s4b,s5,s5b,s6}.md`; contracts follow-up `complete/2026/09/profiling-contracts.md`; shelved phase 7 `complete/archive/shelved/fixed_light_numba_s7_cpu_verdict.md`.
**Next:** none (phase 7 reopens only on an explicit user request with a fresh scope).

## Why this campaign

The JAX [fixed-lens-light](fixed_lens_light.md) epic finished on 2026-09-14. It settled the GPU,
but every CPU row in it was JAX-CPU or numpy, and the production CPU path in PyAutoArray (numba,
`InversionImagingSparseNumba`) had never been measured. Its one CPU row went the other way: the
certified active set did not beat the library's `fnnls`. So the campaign map, filed 2026-09-14,
asked a new question rather than porting the GPU result: on numba, where does the call go, and is
fixing the lens light plus a better solver worth anything? The phases were issued one at a time,
each grid set by the previous answer.

The human set the scope along the way. On 2026-09-15 they chose single-threaded only (production
runs one numba likelihood per process) and retired the thread-scaling phase. On 2026-09-16 they
paired the non-solver round with the A100 ([HST GPU residue](hst_gpu_residue.md)) and inserted it
as phase 3. On 2026-09-18 they shelved phase 7 and closed the epic, saying to assume no usable
memo benefit when sampler order changes. The issues are `autolens_profiling#263`, #265, #267,
#274, #276, #278, #280, #282 and #284.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 whole-call harness | 2026-09-14 | Decompose the whole numba call in one process (access counting), routes a/b/c × {dense, sparse_numba} | exclusive rows + `unattributed` within 5 % of the clean call; overhead ≤ 1.03; re-baked sparse operator bitwise equal | harness shipped, smoke gates PASS; timing legs **not run** (loaded laptop, human decision) | laptop (smoke, N=484) | #264 |
| 2 source-only solver | 2026-09-15/16 | Fixed-light speedup on numba; make the S3 `fnnls` solve as fast as possible | S3 solver rows ≤ 1e-9 rel vs library `fnnls`; decomposition within 5 %; no speed threshold recorded | a → b **2.03x** (932.387 → 459.220 ms, memo off); memo ON 404.597 ms (1.134x) vs factor-reuse NNLS 1.112x, so **no solver change**; residue: `regularization_matrix` 112 ms, two log-dets 40.8 + 37.8 ms | 343311 (+ laptop) | #266 |
| 3 non-solver levers | 2026-09-16 | Jit split-reg assembly, sparse log det H, shared Cholesky of F + λH; each paired with an A100 row | each lever ≤ 1e-9 rel evidence; H bit-identical to the Python `_reference`; decomposition within 1 %; no speed threshold recorded | L1 1.379x, L2 1.132x, L3 1.188x; **413.301 → 230.031 ms = 1.797x**; A100 identity on all three, so L2 and L3 help the CPU only | 343345/343346 (L1 CPU/A100), 343353/343354 (L2), 343355 (L3 bridge, FAILED), 343356/343357 (L3) | PyAutoArray#553, #554, #555; #269, #271, #272 |
| 4a curvature kernel A/B | 2026-09-17 | Two-stage vs direct vs touched-index curvature kernel on Delaunay | lever iff ≥ 5 % faster whole call with ABBA PASS (overhead gate restated as a 12 ms budget first) | **no lever**: b 230.149, direct 244.924 (+6.42 %), touched 236.913 ms (+2.94 %); two-stage stays; docstring fixed | 343394 | PyAutoArray#557, #275 |
| 4b permute-active-last | 2026-09-18 | Does A′ (one potrf on a passive-first permutation) pay? | witness W1–W6 first (Jaccard 1.0 every draw, ≤ 1e-9 rel evidence; W6 factor gate relaxed by the human to max(2e-12 nats, 32 ulp) + 1e-12 residual); lever iff ≥ 5 % whole call | **NO_LEVER**: b 226.772 vs d_perm 228.576 ms; all 8 draws PASS | 343397 | #277 |
| 5 memo robustness | 2026-09-18 | Does the `fnnls` memo warm start hold over the 41 graded draws? | exact active sets, ≤ 1e-9 rel evidence; explicitly no speedup threshold | numerically clean (82/82) but memo **56–60 % slower** on broad stress orders (graded 382.9 → 611.3 ms/call); NO_LEVER, defaults unchanged | 343398 | #279 |
| 5b memo precheck | 2026-09-18 | Reject unsuitable memo seeds before the solve (KKT-residual score) | numerical gates unchanged; targets ≥ 5 % faster than memo on both stress orders, ≤ 3 % over cold (broad) and over memo (nearby holdout); threshold locked before holdout | **NO_LEVER**: threshold 0.5 met the stress-order target (−35 to −38 %) but nearby holdout +11.66 % vs memo, broad holdout +5.28 % vs cold; 276/276 comparisons pass | 343413 | #281 |
| 6 source-pixel scaling | 2026-09-18 | Runtime and bottleneck vs N = 500–4000, memo off/on, nearby vs broad | exact active sets, ≤ 1e-9 rel evidence; timers reconcile within 5 %; no extrapolation | all 5 cells PASS; NNLS overtakes curvature as N grows; N4000 cold nearby/broad 2.138/4.312 s (~81 %/~88 % NNLS); memo −39.7 % nearby, 2.50x cold broad | 343430_0, 343445_1..4 | #283 |
| contracts | 2026-09-18 | Harden cell CLIs and timed streams; add current-summary notices | not recorded (maintenance) | unknown flags rejected; streams reset and matched; status page and bridge limitation written; no new timings | none | #285 |
| 7 HST + Euclid verdict | 2026-09-18 | Production CPU configuration on HST and Euclid | not recorded | **shelved** by the user, never issued | none | none |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#553 | lever 1: numba kernels for the split-regularization assembly (`_reference` retained; three input-mutation/insert defects fixed) | `2f6657a6` | 2026.9.19.1 |
| PyAutoArray#554 | lever 2: sparse (SuperLU) log det of H, gated at ≥ 256 pixels; stale docstring fixed | `7c230c4c` | 2026.9.19.1 |
| PyAutoArray#555 | lever 3: log det(F + λH) from the NNLS Cholesky factor; `curvature_reg_matrix` cached | `91240e43` | 2026.9.19.1 |
| PyAutoArray#557 | docstring only: two-stage curvature dispatch, Delaunay claim replaced by the 4a measurement | `192d4b70` | 2026.9.19.1 |

Production recommendation from the close-out: use the fp64 numba `_sparse` assembly with
positive-only NNLS and one BLAS/numba thread per process. Budget with zero memo benefit and cold
timings.

## Open / parked / drafts

- `complete/archive/shelved/fixed_light_numba_s7_cpu_verdict.md`: phase 7, shelved and not a pickable task.
- `draft/bug/autoarray/sparse_inversion_ignores_profile_subtracted_image.md`: out of scope throughout (blocks the sparse operator).
- `draft/research/autolens_profiling/post_certified_solver_likelihood_breakdown.md`: reuses this campaign's HST Delaunay CPU columns ([page](post_certified_breakdown.md)).
- Unfiled, recorded in the phase 2 and 3 notes: `abstract_ndarray.__getitem__` imports `jax.numpy`, so a numba-only process is not JAX-free (`jax_after_rows`).
- Deliberately not filed: `nnls_seed_factor_reuse` (phase 2 verdict 3). Reopening it needs a new measurement.

## Caveats

- **1.80x provenance limit.** The chain joins lever 3 to levers 1–2 through job **343355**'s 16-repeat control (268.681 vs lever 2's 267.448 ms). That job FAILED: its feature arm aborted on the 1.03 overhead gate, so it is a control-only record from an incomplete A/B job. Its JSON/PNG sat in untracked `output/ral_job343355_16repeats/`, are not in the committed tree, and could not be found in the #285 review. The per-lever A/Bs are committed; the exact bridge cannot be reproduced from the tree. The ledger calls 1.797x a lower bound. The product of the row ratios (1.854x) is not the headline.
- **Do not multiply 2.03x by 1.80x.** 932.4 → 459.2 ms is the modelling choice (what is held fixed; light preparation is costed separately), memo OFF. 413.3 → 230.0 ms is a memo-ON fixture. Route a was never re-measured with the memo on, so no a → production-b ratio exists.
- **Memo precheck NO_LEVER** means the numerical gates passed and the performance targets failed. It is not a correctness failure. The 5b hindsight "best lane" savings (9.64 / 2.96 ms per model) are descriptive only.
- **N4000 scaling rows are conditional synthetic sequences** (8 frozen seed-601 nearby/broad draws, lens light re-solved per model). They are not sampler throughput. Harness RSS 9.085 GiB covers eight prepared Analyses and is not per-worker memory. The log-log exponents are descriptive.
- **Absolute ms do not chain across phases 5/5b/6.** Each uses different model streams and per-model S0 → S3 preparation. Phase 4b's 226.772 ms control is a fresh control, not a source change against 230.031.
- **Laptop rows are corroboration only.** Clean spread was 70.9–187.6 % there against RAL's 0.77–16.25 %. Phase 1 has no measured timings at all.
- **Runtime BLAS thread count unavailable** in phases 5/5b/6 (`threadpoolctl` returned no pools). The env pins are recorded; the observed count is not.
- Provenance sidecars in `results/notes/`: `fixed_light_numba_s4b_source_job343397.json`, `fixed_light_numba_s5_source_job343398.json`, `fixed_light_numba_s5b_source_job343413.json`, `fixed_light_numba_s6_source_jobs.json`.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records every phase: 1–6 including 4a/4b/5b, the #285 contracts task, and phase 7 as
shelved, each with rule, result, RAL jobs and PRs. It also records all four PyAutoArray PRs from
the release sheet (all 2026.9.19.1). Resolved: PyAutoArray#557 added to Library PRs, and the
headline's job reference corrected (343355 is the bridge; the chain endpoints are 343345/343356).
Still open: the job 343355 bridge JSON is not in the tree.
