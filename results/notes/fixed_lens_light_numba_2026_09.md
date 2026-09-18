# Fixed lens light on the numba CPU path — the whole-call measurement (2026-09-15)

> **Status correction — 2026-09-18:** The CPU epic is complete and phase 7 is shelved. For planning assume zero dependable memo benefit; production defaults were not changed. Subsequent numerical and timing results supersede the follow-up suggestions below: [current campaign summary](profiling_campaign_status_2026_09.md).

autolens_profiling issue [#265](https://github.com/PyAutoLabs/autolens_profiling/issues/265),
epic `fixed-lens-light-numba-cpu` phase 2, branch `feature/fixed-light-numba-solver`.
Four legs in one SLURM job (343311) on one idle RAL host, plus two corroborating laptop legs.
**Fixing the lens light speeds the production numba CPU likelihood 2.03×** — SLaM light[1]'s
MGE converted to regular light profiles at their solved intensities and subtracted, leaving a
source-only linear system ("S3"), takes the whole `AnalysisImaging.log_likelihood_function`
from **932.4 ms to 459.2 ms** on HST Delaunay N=1500, one thread, sparse numba operator, with
**no solver change at all**; the laptop measures 2.036× independently. Both of those routes run
with the cross-evaluation NNLS memo **off**. Where the remaining 459 ms goes is the second
result: the positivity solve is 28 % of it, `F + λH` 25 %, `D`/`F` assembly 21 %, the two
log-dets 18 %, and the single largest site in the call is **`inversion.regularization_matrix`
at 112.3 ms (24 %) — larger than `fnnls_cholesky`'s 62.1 ms**. The third result kills the
solver work: the library's own memo, which is ON by default and therefore already in
production, takes the whole call to **404.6 ms (1.13×)**, and the factor-reuse NNLS built for
this phase reaches only 412.6 ms (1.11×) — within 2 % of each other. **No PyAutoArray solver
change is proposed.**

## Scope — read this before quoting a number

- **Single-threaded only.** Production for this path (`cpu_fast_modeling.py`) runs one
  single-threaded numba likelihood per process under a multiprocessing pool, so no multi-core
  per-call speedup was ever a target here; the campaign map's thread-scaling phase was retired
  by that decision (human, 2026-09-15). Every leg ran at `--threads 1` with
  `NUMBA_NUM_THREADS=1` and the whole BLAS family (`OMP`/`OPENBLAS`/`MKL`/`VECLIB`/`NUMEXPR`)
  pinned to 1, recorded per leg in `configuration.thread_env` and `numba_thread_env`.
  `cpu_over_wall` is 0.997–1.000 on every row — no pool this cell did not pin.
- **What was measured.** One pixelized-imaging likelihood call, HST, mask 3.5″, 15 361 masked
  image pixels, Hilbert/Delaunay mesh at 1500 source pixels, `adapt_split` regularization
  (0.1 / 10.0 / 0.1), fp64, `InversionImagingSparseNumba` (the numba sparse operator),
  `use_jax=False`. Routes: **a** = the joint S0 system the library solves today (1560 params:
  1500 source pixels + 60 unregularised linear-MGE columns), **b** = the source-only S3 system
  (1500 params), **c** = S3 with positivity off, **d_np** = route b with the factor-reuse NNLS
  injected at the library's own positive-only entry point. No library edits anywhere.
- **The host is a `gpu`-partition node used CPUs-only.** `euclid-ral-gpu-1`, AMD EPYC 7702
  64-Core (`os.cpu_count()` 124), `--partition=gpu --cpus-per-task=4 --mem=32gb` and **no
  `--gres`**, so the A100s stayed free. The `ral` CPU partition was saturated for days (21
  nodes at 252/252 allocated; a 2-CPU request scheduled three days out) and the idle `imp`
  partition does not mount `/mnt/ral`, where the repo, the venv and `activate.sh` live.
- **Contention.** The cell's `measured_under_contention` compares the **node-wide** 1-minute
  loadavg to the **node's** core count, not this job's cgroup — on a shared node that stamp is
  a caveat, not a gate. It is moot here: the host was idle, peak 1-minute loadavg 0.94–1.00
  against 124 cores, and all four legs stamp `timing_status: measured`,
  `contention_warning: false`. The judges actually used are the ABBA overhead gate (≤ 1.03)
  and the per-row clean spread.
- **The laptop legs are corroboration, not headline.** `DESKTOP-H143S82`, Intel i9-10885H,
  8 cores, WSL2: clean spreads 70.9–187.6 % against RAL's 0.77–16.25 %, and it could not hold the 1.03
  ABBA gate at n-repeats 8 (three failures) — hence n-repeats **24** there and **16** on RAL.
  Every millisecond below is quoted with the n-repeats it came from.
- **`log_likelihood` is not comparable across legs.** `--instances iid` rotates a call index
  and the legs have different route counts, so the last-timed instance differs (route b reads
  19651.457885 in Leg A and 18665.014334 in Legs C and D). P2/P3/P4 are per-instance
  comparisons and all pass; a reader diffing `log_likelihood` between legs will be misled.
- **No pin was ever calibrated for a numba fixed-light configuration** anywhere in this repo
  (`pinned_expected: null`). Every numerical quantity here is RECORDED; the hard verdicts are
  the structural gates below.
- **The process is not JAX-free.** `jax_after_rows.jax_in_sys_modules` is **true**: the first
  `FitImaging` pulls `jax.numpy` in through `abstract_ndarray.__getitem__`, even though this
  cell imports no JAX and `device_info_dict()` is deliberately not called. Recorded, not fixed
  — see "Next".
- **`--help` was broken repo-wide** by a literal `%` in `_profile_cli.py`'s `--memo` help
  (`argparse` raised `TypeError: %c requires int or char` for every cell on the shared
  parser). Fixed in this branch (`338b80c`). `parse_profile_cli` uses `parse_known_args`, so a
  mistyped flag is silently ignored — check `configuration.routes_selected`, `n_repeats`,
  `nnls_warm_start_env` and `formalisms_selected` in a JSON before quoting it.
- Route **c** (positivity off) is a diagnostic and is never a headline; never quote its
  milliseconds without the evidence delta below.

## Provenance

| | |
|---|---|
| Job | **343311**, `COMPLETED 0:0`, elapsed **00:05:55**, MaxRSS **13 260 788 K** of a 32 GB request |
| Node / partition | **euclid-ral-gpu-1**, `gpu` partition CPUs-only, 4 allocated CPUs, no `--gres` |
| CPU | AMD EPYC 7702 64-Core Processor, `os.cpu_count()` 124, kernel 5.14.0-687.39.1.el9_8 |
| loadavg at entry | `0.02 0.01 0.00` |
| Python / autolens | 3.12.4 / **2026.8.17.1** (laptop: 3.12.10 / 2026.8.17.1) |
| `AP_ROOT` | `/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-solver` @ `3c2ca70`, branch `feature/fixed-light-numba-solver` |
| Laptop host | `DESKTOP-H143S82`, Intel i9-10885H, 8 cores, worktree @ `50b6caf` |

Library revisions on the shared RAL install (`HPCPullPyAuto` ran first, queue empty):

| Repo | rev |
|---|---|
| PyAutoNerves | `fac8b17b962adc272e0aaae81e951d58b13bcce3` |
| PyAutoFit | `27d41e7c8d235a8aedbc1afa88cadf0863d68fe8` |
| **PyAutoArray** | **`5e2bc0f42940adfe04ff30fa42a7ce13a86ecbe0`** (== local `main`; the solver library) |
| PyAutoGalaxy | `840ffde063fae1c7bfb41986f7c62b6ca96df3b7` |
| PyAutoLens | `ccf9295f32ec3ad897d4c809c76f33987ce2b0ed` |

Config names, one per leg (the `hpc_ral_cpu_fp64_*` / `local_cpu_fp64_*` labels are
comparator-study labels, README-table-invisible by design):

| Leg | What | Config name |
|---|---|---|
| A | routes a,b,c — the headline, memo off | `hpc_ral_cpu_fp64_fixed_light_numba_sparse_t1` |
| B | solver kernels K0/K1/K1i/K3/K4 | `hpc_ral_cpu_fp64_fixed_light_numba_solvers_t1` |
| C | routes b,d_np — injected factor-reuse NNLS | `hpc_ral_cpu_fp64_fixed_light_numba_sparse_dnp_t1` |
| D | route b, memo ON | `hpc_ral_cpu_fp64_fixed_light_numba_sparse_b_warm_t1` |
| A (laptop) | routes a,b,c, n-repeats 24 | `local_cpu_fp64_fixed_light_numba_sparse_t1` |
| B (laptop) | solver kernels | `local_cpu_fp64_fixed_light_numba_solvers_t1` |

Branch history behind this note: `50b6caf` (solver-injection seam, factor-reuse NNLS, `d_np`
route, solver cell), `338b80c` (`--help` fix), `5e03560` / `3c2ca70` / `2483948` (the batch
script), `71b617c` (this harvest — 6 JSON + 6 PNG).

## Gate roll-up

| Leg | timing_status | contention (peak 1-min / cores) | P1 | S3=S0 | P2 | P3 | P4 | ABBA overhead | unattributed |
|---|---|---|---|---|---|---|---|---|---|
| A | measured | false, 0.94 / 124 | PASS | PASS | PASS | PASS | — | 1.0125 / 1.0008 / 1.0013 PASS | ≤ 0.287 % |
| B | measured | false, 0.98 / 124 | — | — | — | — | — | n/a (kernel cell) | — |
| C | measured | false, 1.00 / 124 | PASS | PASS | PASS | PASS | **PASS** | 1.0048 / 1.0000 PASS | ≤ 0.325 % |
| D | measured | false, 1.00 / 124 | PASS | PASS | PASS | PASS | — | 1.0111 PASS | 0.297 % |
| A (laptop) | measured | false, 1.70 / 8 | PASS | PASS | PASS | PASS | — | 0.9709 / 0.9862 / 0.9566 PASS | ≤ 0.220 % |
| B (laptop) | measured | false, 1.46 / 8 | — | — | — | — | — | n/a (kernel cell) | — |

- **P1** re-baked sparse operator identical to the original, 3/3 arrays — subtracting the lens
  light changes only the data, so the operator built from noise map, PSF and mask must not move.
- **S3 = S0 on the mapper block**: `log_det_curvature_reg_matrix_term` 8326.70017133483 and
  `log_det_regularization_matrix_term` 7670.876894312947 in both systems, **rel diff exactly
  0.0**, edge-zeroed 0 vs 0. This is what makes S3 a re-arrangement of the same fit.
- **P2** dense vs sparse-numba system: `D` max rel diff 1.297e-15, `F` (mapper block) 3.728e-15
  (rtol 1e-9).
- **P3** dense vs sparse-numba evidence: Δ −9.094947e-11 nats (rel 4.615e-15), **0 passive-set
  differences**, 1485 passive both sides.
- **P4** (Leg C) injected `d_np` == route b: Δ +2.182787e-11 nats (rel 1.108e-15), max |Δx|
  2.281e-12, and the injection is witnessed — `n_calls_numpy` **38**, `n_calls_jax` 0,
  `installed_outside_call_accounting: true`, `seed_source: factor_reuse_dense_sign`.
- **Leg D is genuinely warm**, not silently cold: `nnls_warm_start_mode: on`,
  `nnls_warm_start_env: "1"`, `memo_cleared_within_block: false`, and the row's
  `solver_stats["solver.fnnls_cholesky"].seed_source` is **`"memo"`** where every memo-off leg
  reads `"dense"`.
- **Leg B** equivalence: K0 0.0, K1 +5.093e-11, K1i +5.093e-11, K3 +1.455e-11 nats — every
  positive row inside the 1e-9 pin. `K3_uses_one_factorisation` PASS (1 factorisation, 15
  downdates, 10 seed negatives). K4 RECORDED.

## Headline — the fixed-lens-light speedup, whole call

Sparse numba operator, HST Delaunay N=1500, 1 thread, `--instances iid`, **memo off on every
row of this table**. `ms` are the clean (uninstrumented) calls; the decomposition below is
measured in counterbalanced ABBA blocks against them.

| host | n-rep | route | mean ms | median ms | min | max | clean spread | ABBA |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| RAL | 16 | a — S0 joint (what the library runs today) | 926.239 | **932.387** | 857.095 | 1007.575 | 16.25 % | 1.0125 |
| RAL | 16 | **b — S3 source-only (the fixed-light route)** | 459.971 | **459.220** | 452.557 | 472.643 | 4.37 % | 1.0008 |
| RAL | 16 | c — S3 positivity off (diagnostic) | 393.094 | 393.107 | 391.511 | 394.558 | 0.77 % | 1.0013 |
| laptop | 24 | a — S0 joint | 1058.308 | 1016.215 | 767.323 | 1517.625 | 70.90 % | 0.9709 |
| laptop | 24 | b — S3 source-only | 530.829 | 499.167 | 390.301 | 895.385 | 95.15 % | 0.9862 |
| laptop | 24 | c — S3 positivity off | 447.936 | 400.720 | 351.841 | 1192.212 | 187.61 % | 0.9566 |

| speedup a → b | RAL | laptop |
|---|---:|---:|
| on medians | **2.030×** | 2.036× |
| on means | 2.014× | 1.994× |

**That is the result of this phase: 2.03×, single-threaded, on the production numba path, with
no solver change.** It reproduces on a second host and on a different CPU vendor. Two
independent processes on RAL measured route b at 459.220 ms (Leg A) and 458.683 ms (Leg C) —
**0.12 % apart**, which is the strongest evidence these numbers are host-clean.

The 16.25 % spread on route a is not host jitter: route a's per-instance NNLS iteration count
varies with the instance (23 outer / 10 inner here, 29 / 12 on the laptop, `n_passive` 1521 /
1504), while route b is 0 outer / 1 inner on both hosts. Routes b and c sit at 0.77–4.37 %.

## Where the S3 call goes — decomposition, Leg A route b

Exclusive (self) time per instrumented site, grouped; rescaled by `decomposition_rescale_factor`
so the groups sum to the clean call, with `unattributed` an explicit measured remainder.
Percentages are of the 459.971 ms mean clean call.

| group | ms | % of call |
|---|---:|---:|
| Positivity solve | 129.275 | 28.11 |
| F + λH | 116.594 | 25.35 |
| D and F assembly | 97.196 | 21.13 |
| Log determinants + evidence | 81.542 | 17.73 |
| Mesh (ray trace, placement, triangulation) | 19.866 | 4.32 |
| Mapper + mapping matrix | 7.812 | 1.70 |
| Mapper weights (Delaunay) | 3.777 | 0.82 |
| Image + blurring | 2.750 | 0.60 |
| **unattributed** | **1.158** | **0.25** |

Top sites, route b:

| site | ms |
|---|---:|
| `inversion.regularization_matrix` | **112.252** |
| `sparse_numba.curvature_matrix` | 88.650 |
| `solver.fnnls_cholesky` | 62.117 |
| `solver.reconstruction_positive_only_from` | 59.811 |
| `inversion.log_det_curvature_reg_matrix_term` | 40.803 |
| `inversion.log_det_regularization_matrix_term` | 37.753 |
| `delaunay.triangulation` | 18.639 |
| `sparse_numba.psf_weighted_data` | 8.384 |
| `inversion.reconstruction` | 7.342 |

**The single largest site in the S3 call is `inversion.regularization_matrix` (112.252 ms,
24.4 %) — nearly twice `fnnls_cholesky` (62.117 ms).** No cost model in this campaign flagged
it. The laptop leg says the same thing (100.613 vs 82.051 ms), so it is not a host artefact.

Route a for contrast (same leg, 926.239 ms mean call): solve 380.955 ms (41.13 %), D+F assembly
286.486 (30.93 %), F+λH 131.914 (14.24 %), log-dets+evidence 76.701 (8.28 %). Its solve is
`fnnls_cholesky` 305.976 + the positive-only wrapper 67.001 ms over **23 outer / 10 inner**
iterations against route b's **0 / 1**, and its assembly carries an extra
`sparse_numba.linear_func_operated_mapping_matrix_dict` at 130.870 ms that S3 does not pay at
all. Fixing the lens light removes an iteration count *and* a matrix.

## Solver kernels — Leg B

The same S3 system the library hands its positive-only solver (the edge-zeroed subset;
`n = 1500`, `n_edge_zeroed = 0` on Delaunay), timed standalone. n-repeats 8, n-stream 16,
primary stream iid. K0/K3/K4 are the median of n_repeats after two warm-ups; the two memo rows
are timed **one call per stream member**, because the memo seeds from the previous evaluation
and repeating one solve would warm-start it from its own answer.

| row | RAL ms | min | max | nFact | nDown | outer | inner | n_passive | seed_source | Δlog-ev (nats) | max &#124;Δx&#124; | laptop ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| K0 library `fnnls_cholesky`, cold | 116.213 | 115.985 | 116.569 | 2 | — | 0 | 1 | 1485 | dense | 0.000e+00 | 0.000e+00 | 119.335 |
| **K1 memo warm (random-walk stream)** | **55.045** | 50.629 | 65.001 | — | — | — | — | 1485 | memo | +5.093e-11 | 2.951e-12 | 55.834 |
| K1i memo warm (iid stream — worst case) | 61.149 | 52.622 | 67.785 | — | — | — | — | 1483 | memo | +5.093e-11 | 2.621e-12 | 55.921 |
| K3 factor-reuse NNLS | 71.979 | 71.839 | 72.135 | **1** | **15** | 0 | 2 | 1485 | factor_reuse_dense_sign | +1.455e-11 | 2.281e-12 | 68.702 |
| K4 scipy unconstrained (floor, RECORDED) | 32.638 | 32.550 | 32.755 | 1 | 0 | 0 | 0 | 1490 | unconstrained | **+7.827e+01** | 5.961e-02 | 31.885 |

Speedups over K0 (RAL): K1 **2.111×**, K1i 1.900×, K3 1.615×, K4 3.561× (not equivalent).
**The fastest kernel passing the 1e-9 equivalence pin is K1, the library's own memo, at
55.045 ms.** K3 is second at 71.979 ms.

Three things to read off this table:

1. **K3 does exactly what it was designed to do and still loses.** One factorisation against
   K0's two, 15 downdates, equivalence 1.455e-11 nats — the mechanism works. It is simply not
   the fastest *equivalent* kernel at N=1500; the library's memo is, by 1.308×.
2. **The iid stream costs the memo 11 %** on a clean host (61.149 vs 55.045 ms), a penalty the
   laptop's jitter hid entirely (55.921 vs 55.834 — indistinguishable). Production under a
   Nautilus pool hands a worker successive *unrelated* instances, so **61.1 ms, not 55.0, is
   the production-relevant memo number.**
3. **K4 is a floor, never a candidate.** +78.269 nats, 10 negative entries, max |Δx| 5.96e-02.
   It exists to say how much of the solve is positivity rather than linear algebra.

## The three-way whole-call comparison, and the memo's mechanism

All RAL, all route b on the same S3 system, medians of 16 clean calls.

| row | leg | whole call, median ms | solve group ms | the solve, site by site |
|---|---|---:|---:|---|
| b, memo OFF | A | 459.220 | 129.275 | `fnnls_cholesky` 62.117 + positive-only wrapper 59.811 + `inversion.reconstruction` 7.342 |
| b, memo OFF | C | 458.683 | 128.833 | `fnnls_cholesky` 62.366 + wrapper 59.247 + reconstruction 7.215 |
| d_np, factor-reuse NNLS injected | C | 412.556 | 83.592 | `nnls_factor_reuse` 76.322 + reconstruction 7.238 + wrapper 0.028 (`fnnls_cholesky` reached 0 times) |
| **b, memo ON (what production pays)** | **D** | **404.597** | **77.769** | `fnnls_cholesky` 62.789 + **wrapper 7.545** + reconstruction 7.431 |

Against memo-off b measured *in the same process* (Leg C, 458.683 ms): **d_np is 1.112×** and
**memo-on b is 1.134×**. They are **1.97 % apart**.

**The memo's saving is in the seed, not in the active-set loop** — and only the whole-call
decomposition could show that. `solver.fnnls_cholesky` is *unchanged* between memo-off and
memo-on (62.117 → 62.789 ms) while `solver.reconstruction_positive_only_from` collapses
**59.811 → 7.545 ms**. The wrapper's own dense-sign seed (a factorise-and-solve to decide the
starting passive set) is what the memo removes; that is also why K0 reports two factorisations
and the memo rows report none. The injected `nnls_factor_reuse` recovers most of the same cost
by replacing the whole wrapper with one 76.322 ms site, which is why two very different
mechanisms land 8 ms apart.

Scaled to the call, the kernel-level 2.111× memo win is worth **1.134×** (the solve is only
28 % of the call). Any quotation of 2.1× must say it is a kernel number.

## Route c — never a bare millisecond

Route c (393.107 ms median, the fastest row in this note) turns positivity off, and
`Inversion.solve_ids_to_keep` returns `None` the moment it does, so it also silently turns edge
zeroing off and solves the **full** system while routes a, b and d_np solve the edge-zeroed one
(here Delaunay zeroes nothing, so the systems coincide). Its solution is a **different,
infeasible minimiser**: `log_evidence` 19783.987056, **+78.269 nats above route b's**
19705.717586, with 10 negative reconstruction entries carrying 9.879e-05 of the flux. A higher
evidence here is an unusable evidence, not an improvement. Route c stays a diagnostic; the
prohibition the GPU epic established carries to numba unchanged.

## Verdict

1. **Fixing the lens light is worth 2.03× on the production numba CPU path, and that is the
   result.** 932.387 → 459.220 ms on medians (2.014× on means), single-threaded, HST Delaunay
   N=1500, sparse numba operator, no library change — corroborated at 2.036× on a second host.
   Both rows are memo-off; the memo-on production configuration is measured separately below
   and no "a → production b" ratio is quoted, because route a was not re-measured with the memo
   on.
2. **The solve is the largest group in the S3 call but does not dominate it.** 28 % solve,
   25 % `F + λH`, 21 % assembly, 18 % log-dets. Even a *free* solver caps the further S3 win at
   ~28 %, so the honest headline stays a → b.
3. **The factor-reuse solver is not worth carrying into PyAutoArray.** It is a correct kernel
   (1 factorisation, 15 downdates, equivalent to 1.455e-11 nats) and it is 1.112× at whole-call
   level — but the library's cross-evaluation memo, which is **on by default and already in
   production**, is 1.134× by a different mechanism, and the two are 1.97 % apart. The
   conditional `nnls_seed_factor_reuse` feature prompt this phase was authorised to file **is
   deliberately not filed**. Leg B alone (K1 55.0 vs K3 72.0 ms as kernels) suggested the same
   conclusion; the whole-call legs settle it.
4. **What is left inside the solver is ~9 % of the call, and it is a ≤1.1× lever.** With the
   memo on, everything from the library's positive-only entry point to a solution costs
   62.789 + 7.545 = **70.334 ms** against the K4 unconstrained floor of **32.638 ms** — about
   **37.7 ms, 9.3 % of the 404.597 ms call** — inside `fnnls_cholesky` itself (the k×k
   fancy-index copy and the O(n²) gradient rebuilds). Worth knowing, not worth a campaign. (A
   kernel row and an in-call site are being compared here; both cover entry-to-solution, which
   is what makes it fair.)
5. **The untouched levers are bigger than the solver.** `inversion.regularization_matrix` is
   24 % of the S3 call on its own and the two log-dets are another 18 % — and
   `log_det_curvature_reg_matrix_term` re-factorises the very matrix (`F + λH`) the solver has
   just factorised. That, not the NNLS, is where the next lever is.
6. **The numba path does not reproduce the GPU epic's shape.** On an A100 the certified
   active-set solve fell to 4–11 ms and *stopped being the call*; on numba the library's
   `fnnls` is already competitive and the assembly/regularization terms are what is left. The
   campaign's premise — that the balance "could land anywhere" on numba — was right, and it
   landed away from the solver.

## Next

1. **`regularization_matrix` and the log-det factor reuse** — filed as
   `PyAutoMind/draft/research/autolens_profiling/fixed_light_numba_s3_regularization_logdet_levers.md`
   (epic `fixed-lens-light-numba-cpu`, phase 3): cut the 24 % `inversion.regularization_matrix`
   site and reuse the solver's Cholesky factor of `F + λH` for
   `log_det_curvature_reg_matrix_term`. Witness is this leg's route-b decomposition JSON.
2. **The memo under bad models** (campaign map phase, was "phase 3"): the memo is now known to
   be worth 1.134× at whole-call level with an 11 % iid penalty — the open question is whether
   `seed_source` stays `"memo"` and `warm_start_fallback` stays false over the seeded graded
   draw set, or whether a bad model pays the cold 116 ms path.
3. **Not filed, deliberately**: the PyAutoArray `nnls_seed_factor_reuse` feature prompt (see
   Verdict 3).
4. **Not filed yet**: the PyAutoArray bug that `abstract_ndarray.__getitem__` imports
   `jax.numpy`, so a numba-only process cannot stay JAX-free after the first `FitImaging`.
   Recorded in every JSON as `jax_after_rows`; worth a bug prompt of its own.
5. Out of scope throughout, as in the GPU epic: rectangular meshes, Euclid, the source-pixel
   sweep, and the sparse-operator/profile-subtracted-image bug
   (`PyAutoMind/draft/bug/autoarray/sparse_inversion_ignores_profile_subtracted_image.md`).

## Artifacts

```
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_sparse_t1.{json,png}         # Leg A
results/breakdown/imaging/fixed_light_numba_solvers_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_solvers_t1.{json,png} # Leg B
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_sparse_dnp_t1.{json,png}      # Leg C
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_sparse_b_warm_t1.{json,png}   # Leg D
results/breakdown/imaging/fixed_light_numba_delaunay_local_cpu_fp64_fixed_light_numba_sparse_t1.{json,png}            # Leg A, laptop
results/breakdown/imaging/fixed_light_numba_solvers_delaunay_local_cpu_fp64_fixed_light_numba_solvers_t1.{json,png}   # Leg B, laptop
```

Code and submit script:

```
scripts/imaging/likelihood_breakdown/fixed_light_numba.py          # routes a/b/c/d_np + --nnls-warm-start
scripts/imaging/likelihood_breakdown/fixed_light_numba_solvers.py  # the kernel cell (K0/K1/K1i/K3/K4)
scripts/misc/likelihood_breakdown/fixed_light_numpy_solvers.py     # nnls_factor_reuse, the injected kernel
scripts/misc/likelihood_breakdown/call_accounting.py               # the decomposition instrument
hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_delaunay_ral_hst_fp64   # all four legs, one job
```

Job logs (SLURM, not versioned in this repo): `output/output.343311.out`,
`error/error.343311.err` on `/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-solver`;
`.err` is 22 lines of benign mask-padding `UserWarning` and no traceback.

JSON keys specific to these cells: `systems` (`s0`/`s3`), `rows[*].steps` (per-site `excl_s` /
`incl_s` / `n_calls` / `group`), `rows[*].abba`, `decomposition_rescale_factor`,
`attributed_ms` / `unattributed_ms`, `solver_stats`, `injected_solver` (Leg C),
`nnls_warm_start_mode` / `memo_cleared_within_block` (Leg D), `route_c`, `jax_after_rows`,
`contention`, and — in the solver cell — `rows[K*]` with `n_factorisations`, `n_downdates`,
`seed_source`, `equivalence_pin`.
