# Fixed lens light, source-only inversion — the A100 kernel measurement (2026-09-13)

autolens_profiling issue [#248](https://github.com/PyAutoLabs/autolens_profiling/issues/248),
branch `feature/fixed-lens-light-source-only`, harvest commit `876dba4`. Five A100 legs
(342802–342806) on `euclid-ral-gpu-2`, fp64, dense: three fiducial (rectangular 1521,
Delaunay 1500, DelaunayNN 1500) and two off-fiducial (rectangular 3025, Delaunay 3000).
**The lever works, and the win is the positivity solve, not the linear algebra.** With the
MGE converted to regular light profiles at their solved intensities and subtracted
beforehand ("S3"), `cond(F+λH)` drops 4.1e10 → 1.2e7 (rect) / 2.1e6 (Delaunay), library
PDIP falls 36.8 → 25.8 ms (rect) and 37.4 → 28.3 ms (Delaunay), and a **certified
active-set positive solve** reaches the same positive solution — Δlog-evidence ≤ 1.2e-10
nats — in **4.21 ms on Delaunay (pass 2)** and **11.05 ms on rectangular (pass 7)**, against
the 25.8–28.3 ms S3 PDIP row and the 36.8–37.4 ms S0 row the library runs today. Dropping
positivity entirely (A2, one unconstrained Cholesky) costs 1.5 ms but +6.4 nats on Delaunay
and +334.9 nats on rectangular, and must not enter production before the matched-injection
witness. The expectation held on Delaunay and half-missed on rectangular: the CPU forecast
was 4–7 ms for the certified solve on both meshes; rectangular needs seven passes, not two,
because its **dual** violation set drains one or two indices at a time.

## Scope — read this before quoting a number

- **What was measured.** One pixelized-imaging likelihood call, HST, mask 3.5″, MGE-60 lens
  light, dense inversion path, on `euclid-ral-gpu-2` (NVIDIA A100 80GB PCIe), fp64. Systems:
  **S0** = the current system (60 unregularised linear-MGE columns + source pixels) and
  **S3** = the MGE converted to regular light profiles at the solved intensities and
  subtracted from the data, leaving source pixels only. Solvers: library **PDIP** (A1), the
  **certified active-set** masked-Cholesky scheme at pass budgets 1–8 (A1′), and the
  **unconstrained Cholesky** (A2, positivity dropped). Fiducial tier is the one the
  [A100 baseline note](./a100_pixelized_baseline_2026_09.md) defines.
- **What was NOT measured.** There is **no library-path row that runs the certified scheme
  or A2 inside `FitImaging.figure_of_merit`.** The library row measures S0 and S3 with the
  library's own PDIP only (51.7 → 38.4 ms rect). Every "≈ 24 ms / ≈ 14 ms per call" figure
  below is *arithmetic on measured rows* — the S3 library call minus the PDIP-row-to-
  certified-row difference — and stays a projection until phase 1 of the
  `fixed-lens-light-profiling` epic measures it. **Phase 1 has now measured it**
  ([#251](https://github.com/PyAutoLabs/autolens_profiling/issues/251), note
  [`fixed_lens_light_library_path_2026_09.md`](./fixed_lens_light_library_path_2026_09.md),
  2026-09-13): the certified route inside `FitImaging.figure_of_merit` is 25.10 / 25.39 /
  36.26 ms against the 23.7 / 25.0 / 33.1 ms projected below — the projection held. No sparse-operator leg (the sparse context
  rejects the multi-func-list S3 build and its weight map ignores the subtraction — bug
  filed, see "Next"). No CPU or consumer-GPU leg. No low-likelihood draws: every leg is the
  same good model. No PyAutoArray change — nothing in the library moved.
- **Regularization** is `Constant(1.0)` on the rectangular cell and `adapt_split`
  (0.1 / 10.0 / 0.1) on the Delaunay family, each cell's shipped default. Compare log-det
  and evidence terms within a mesh, never across.
- **Rectangular edge zeroing.** The library holds 152 (n=1521) / 216 (n=3025) rectangular
  source pixels at exactly zero via `solve_ids_to_keep`. The certified scheme starts with
  exactly that set fixed, so its reference is the library's own edge-zeroed reconstruction,
  not the full-system PDIP solution — the two differ by the **edge-zeroing penalty**, 215.18
  nats at n=1521, recorded beside the library reference and never to be read as solver error.

## Provenance and gate

| Job | Mesh | N_src | Exit | Elapsed | Node | cache_fresh | Runtime pins |
|---|---|---:|---|---:|---|---|---|
| 342802 | rectangular | 1521 | 0:0 | 00:01:05 | euclid-ral-gpu-2 | true | PASSED / PASSED |
| 342803 | delaunay | 1500 | 0:0 | 00:01:01 | euclid-ral-gpu-2 | true | PASSED / PASSED |
| 342804 | delaunay_nn | 1500 | 0:0 | 00:01:26 | euclid-ral-gpu-2 | true | PASSED / PASSED |
| 342805 | rectangular | 3025 | 0:0 | 00:01:42 | euclid-ral-gpu-2 | true | SKIPPED (non-fiducial N) |
| 342806 | delaunay | 3000 | 0:0 | 00:01:36 | euclid-ral-gpu-2 | true | SKIPPED (non-fiducial N) |

**Counts: legs = 5; result JSONs written = 5; off-node = none; `cache_fresh: false` = none;
non-zero exit = none; runtime pin FAILED = none.** Every leg reports
`autotune_cache_entries_at_start: 0`, `autolens 2026.8.17.1` and the same AP_ROOT
(`55fa192`); XLA flags are the standard cell set (`constant_folding` disabled,
`autotune_level=0`, Triton GEMM off).

Two harvest gates beyond the pins, checked on all five JSONs and PASSED on all five:

| Gate | rect | delaunay | delaunay_nn | rect n3025 | del n3000 |
|---|---|---|---|---|---|
| `hostname` = euclid-ral-gpu-2, A100 80GB | PASS | PASS | PASS | PASS | PASS |
| `device.cache_fresh` | PASS | PASS | PASS | PASS | PASS |
| runtime log-det pins (fiducial only) | PASS | PASS | PASS | n/a | n/a |
| S3 mapper-block log-dets == S0 (rel diff) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S3 `figure_of_merit` == S0 (rel diff) | 2.5e-16 | 6.7e-15 | 2.7e-15 | 2.6e-16 | 1.3e-14 |

The mapper-block gate is the one that makes S3 legitimate: converting the MGE to regular
profiles changes neither `log det(F+λH)` nor `log det(H)` on the mapper block (rel diff
exactly 0.0 on every leg), and the library likelihood is unchanged to 1e-14 relative. S3 is
a re-arrangement of the same fit, not a different model.

Branch history behind this note: `5aec350` (active-set kernels + the S3 system builder),
`3956a7e` (committed CPU probe), `55fa192` (the `fixed_light` cell + submits), `e6d64ea`
(merge of `main` after #249), `876dba4` (this harvest — 5 JSON + 5 PNG).

## Headline — per-call milliseconds, fiducial legs

Single-call steady-state (jit, per call) and `@vmap 16` per-call. The `certified` column is
the pass budget at which the scheme certifies on that mesh — its *own* budget, not a shared
one. **These rows are kernels, measured in the cell's own process; the library-call row is
the whole `figure_of_merit`. Never sum a kernel row into a library call.**

| Row | rectangular (1521) | delaunay (1500) | delaunay_nn (1500) |
|---|---:|---:|---:|
| S0 NNLS PDIP (library today) | 36.781 (21 it) | 37.394 (22 it) | 37.218 (22 it) |
| S3 NNLS PDIP (source-only) | 25.796 (15 it) | 28.347 (17 it) | 26.614 (16 it) |
| **S3 certified active-set** | **11.049** (pass 7) | **4.213** (pass 2) | **4.161** (pass 2) |
| S3 unconstrained Cholesky (A2) | 1.570 | 1.485 | 1.465 |
| — | | | |
| @vmap16 S0 PDIP | 21.721 | 21.940 | 21.862 |
| @vmap16 S3 PDIP | 14.746 | 16.077 | 15.151 |
| @vmap16 S3 certified | 6.051 | 2.171 | 2.176 |
| — | | | |
| Library call, S0 (`figure_of_merit`) | 51.669 | 70.406 | 79.144 |
| Library call, S3 (`figure_of_merit`) | 38.442 | 49.169 | 55.596 |
| S3 / S0 library ratio | 0.744 | 0.698 | 0.703 |
| — | | | |
| cond(F+λH), S0 → S3 | 4.117e10 → 1.208e7 | 4.107e10 → 2.051e6 | 4.113e10 → 2.474e6 |

Supporting rows (S0 → S3, ms): curvature+reg build (dense) 4.960 → 5.038 / 4.900 → 4.918 /
4.879 → 4.840; `log det(F+λH)` 1.169 → 1.241 / 1.159 → 1.171 / 1.211 → 1.199; `log det(H)`
1.166 → 1.201 / 1.171 → 1.204 / 1.152 → 1.171; one PDIP iteration 3.265 → 3.138 / 3.182 →
3.079 / 3.184 → 3.030. Everything except the NNLS row is flat: the source-only system is the
same size (the 60 MGE columns are ~4 % of the parameters) and only the *conditioning*, hence
the iteration count, changes. The eager `log_evidence_terms` row (904 / 959 / 901 ms at S0
against 53 / 63 / 54 ms at S3) is an **eager diagnostic, not a jit row** — do not quote it as
a cost.

**The projection, stated as a projection.** Substituting the certified row for the PDIP row
inside the library call gives 38.4 − (25.8 − 11.0) ≈ **23.7 ms** on rectangular, 49.2 −
(28.3 − 4.2) ≈ **25.0 ms** on Delaunay, 55.6 − (26.6 − 4.2) ≈ **33.1 ms** on DelaunayNN; with
A2 instead, ≈ 14.2 / 22.3 / 30.5 ms. That is arithmetic on measured rows — the library-path
row with either solver was **not measured**, and phase 1 of the epic exists to measure it.

## The certified active-set scheme — pass budget × certification

One masked full-size Cholesky per pass, static shapes, seeded with `Z0 = {x_unc < 0}` plus
(rectangular) the library's edge-zero set. `tau_rel = 1e-9`. Each budget is its own
`lax.scan` length and therefore its own compile; the ms column is the steady-state per call
at that budget.

### rectangular (n=1521, `n_fixed0` = 152 edge-zero, certifies at pass 7)

| Budget | ms | fixed / pass | primal viol. / pass | dual viol. / pass | Certified | Δlog-ev vs library (nats) |
|---:|---:|---|---|---|---|---:|
| 1 | 2.983 | 231 | 26 | 20 | no | +16.08 |
| 2 | 4.357 | 231, 237 | 26, 11 | 20, 7 | no | +5.581 |
| 3 | 5.689 | …, 241 | …, 3 | …, 4 | no | +0.3449 |
| 4 | 6.995 | …, 240 | …, 0 | …, 2 | no | −4.4e-09 |
| 5 | 8.444 | …, 238 | …, 0 | …, 2 | no | −3.3e-10 |
| 6 | 9.824 | …, 236 | …, 0 | …, 2 | no | −2.9e-11 |
| **7** | **11.049** | …, 234 | …, 0 | …, **0** | **yes** | **−3.6e-12** |
| 8 | 12.479 | …, 234 | …, 0 | …, 0 | yes | −3.6e-12 |

Read the two violation traces: **primal** violations (negative entries in the free solve)
are gone by pass 4 — 26 → 11 → 3 → 0 — and the *evidence* is already converged there
(−4.4e-09 nats). What runs to pass 7 is the **dual** trace, 20 → 7 → 4 → 2 → 2 → 2 → 0: the
last few wrongly-fixed indices leave one or two at a time. **The dual set is the tail.** A
four-pass budget (6.99 ms) buys an evidence indistinguishable from the certified one at
nano-nat level but carries no certificate; the production proposal remains "N-pass budget +
PDIP fallback when uncertified".

### delaunay (n=1500) and delaunay_nn (n=1500) — certify at pass 2

| Budget | delaunay ms | delaunay traces (fixed / primal / dual) | delaunay_nn ms | delaunay_nn traces | Δlog-ev vs library |
|---:|---:|---|---:|---|---:|
| 1 | 2.957 | 4 / 1 / 0 | 2.858 | 4 / 1 / 0 | +0.0147 / +0.0240 |
| **2** | **4.213** | 4, 5 / 1, 0 / 0, 0 | **4.161** | 4, 5 / 1, 0 / 0, 0 | **+1.2e-10 / +1.1e-11** |
| 3–8 | 5.53–12.07 | unchanged after pass 2 | 5.47–12.03 | unchanged after pass 2 | unchanged |

There is no edge zeroing on the Delaunay family (`n_fixed0 = 0`), no dual violation at any
pass, and the free set is settled after one correction: four negative entries in the
unconstrained solve, five in the certified set. This is the clean case the CPU probe
predicted.

### Off-fiducial legs (pins skipped by design)

| Leg | S0 PDIP | S3 PDIP | S3 certified | A2 | Library S0 → S3 | @vmap16 S0/S3/cert |
|---|---:|---:|---|---:|---|---|
| rectangular 3025 (216 edge-zero) | 82.860 (24 it) | 51.758 (15 it) | 20.464 @ pass 6 | 2.989 | 115.151 → 92.619 | 90.4 / 55.4 / 22.3 |
| delaunay 3000 | 76.341 (23 it) | 66.239 (20 it) | 8.508 @ pass 2 | 2.875 | 136.335 → 117.124 | 83.6 / 69.7 / 9.0 |

The ratio holds off-fiducial and the shape of the answer does not change: rectangular's dual
trace (33, 9, 9, 4, 3, 0) still sets the certifying budget at 6 while its primal trace (43,
24, 6, 2, 0, 0) is done at pass 5; Delaunay still certifies at pass 2, at 8.5 ms against a
66.2 ms S3 PDIP row — a 7.8× reduction of the reconstruction row at twice the fiducial mesh.
Δlog-evidence at certification is 0.0 nats exactly (rect 3025) and −3.6e-12 (del 3000).

## Evidence agreement per scheme

Against each mesh's reference — the library's own reconstruction of S3, which is the PDIP
solution on the edge-zeroed problem.

| Scheme | rectangular | delaunay | delaunay_nn | rect 3025 | del 3000 |
|---|---:|---:|---:|---:|---:|
| Certified active-set, at its certifying budget | −3.6e-12 | +1.2e-10 | +1.1e-11 | 0.0 | −3.6e-12 |
| A2 unconstrained Cholesky (nats) | +334.93 | +6.404 | +8.219 | +352.50 | +12.06 |
| A2, negative entries | 136 | 4 | 4 | 211 | 8 |
| A2, negative flux fraction | 0.135 % | 0.0017 % | 0.0020 % | 0.180 % | 0.0014 % |
| A2, max relative reconstruction difference | 5.74 % | 0.25 % | 0.38 % | 6.91 % | 0.35 % |
| Edge-zeroing penalty (library reference vs full-system PDIP) | 215.18 | −1.9e-10 | +9.5e-11 | 211.76 | −5.3e-10 |

The certified scheme is exact: it returns the constrained optimum, to nano-nats, in 2–7
factorisations. **A2 is not exact and is not free.** Its evidence sits *above* the
constrained optimum because it is a different (infeasible) minimiser — a positive Δ here is
an unusable evidence, not an improvement. Clipping the negative entries to zero flips the
sign (−271.7 nats rect, −0.49 Delaunay), which is the honest measure of what positivity is
buying. On rectangular the cost is 136 negative pixels carrying 0.135 % of the flux and a
5.7 % peak reconstruction change; on the Delaunay family it is 4 pixels and 0.002 %.

**A2 must not enter production before the matched-injection witness** — a simulated source
with known structure, fitted with and without positivity, showing that the negative pixels
do not change the recovered source or the inferred mass model. Until then A2 is a *cost
bound* (1.5 ms, the floor a positive solve is measured against), not a candidate.

## CPU cross-check (phase 1 probe, committed)

`results/misc/fixed_light_probe/{rectangular_1521,delaunay_1500}.json`, numpy fp64, same
datasets, `free_all` policy (the scheme the A100 cell runs):

| Quantity | rect CPU | rect A100 | delaunay CPU | delaunay A100 |
|---|---:|---:|---:|---:|
| PDIP iterations, S0 → S3 | 21 → 15 | 21 → 15 | 22 → 17 | 22 → 17 |
| cond(F+λH), S0 → S3 | 4.117e10 → 1.208e7 | identical | 4.107e10 → 2.051e6 | identical |
| Certified at pass | 7 | 7 | 2 | 2 |
| S0 certified at all? | no | not run (S3 only) | no (`free_all`) | not run |

The iteration counts, conditioning and certifying pass agree exactly between the numpy
reference and the JAX masked twin — the A100 legs measure the *cost* of a scheme the CPU
probe already validated for correctness. The probe additionally ran ±1 % parameter draws
(Einstein radius ±1 %, `ell_comps[0]` +0.01): rectangular certifies at pass 7, 7, 8 and
Delaunay at 2, 1, 2 — **no draw exceeded the 8-pass budget, so the measured fallback rate is
0/4 per mesh**, on the CPU probe only, and only around a good model.

## Verdict

1. **Fixing the lens light is worth it on its own.** The library call falls 51.7 → 38.4 ms
   (rect), 70.4 → 49.2 ms (Delaunay), 79.1 → 55.6 ms (DelaunayNN) with **no solver change at
   all** — 26–30 %, purely from removing the 60 unregularised MGE columns from the linear
   system. This is the same conditioning story #247 ended on, now paid off.
2. **The certified active-set solve is the lever.** On the Delaunay family it replaces a
   28.3 ms PDIP row with a 4.2 ms exact one (6.7×) at pass 2, and 2.2 ms per call under
   `@vmap 16`. On rectangular it is 11.0 ms at pass 7 (2.3×) — real, but less than half the
   forecast, because the dual set drains slowly. **The expectation held on Delaunay and
   half-missed on rectangular.**
3. **The tail is the dual violation set, and it is mesh-structural.** Rectangular's
   edge-zeroing puts 152 pixels in the fixed set from pass 0 and the KKT test then has to
   release the wrongly-fixed ones one or two at a time. Anything that makes the rectangular
   free set smarter at pass 0 attacks exactly this.
4. **Positivity is cheap enough to keep.** A2 saves a further ~2.7 ms (Delaunay) over the
   certified solve and costs 6.4 nats and a 0.25 % reconstruction change; on rectangular it
   saves 9.5 ms and costs 335 nats and 5.7 %. There is no case for dropping positivity on
   the Delaunay family, and no case at all before the injection witness.

## Next

The follow-up programme is the PyAutoMind epic **`fixed-lens-light-profiling`**
(`draft/research/autolens_profiling/fixed_lens_light_profiling_epic.md`, filed 2026-09-13),
worked strictly one phase at a time — each phase's grid is chosen from the previous phase's
answer:

1. **Library-path timing** of the S3 unconstrained (positive-negative) solve with the MGE
   subtracted as fixed regular profiles, and of the certified scheme — the row this note
   could not measure (`fixed_light_unconstrained_library_path.md`). **DONE 2026-09-13**
   ([#251](https://github.com/PyAutoLabs/autolens_profiling/issues/251)) — note
   [`fixed_lens_light_library_path_2026_09.md`](./fixed_lens_light_library_path_2026_09.md):
   the library call falls 50.97 → 25.10 ms (rect, 2.03×) / 65.10 → 25.39 ms (Delaunay, 2.56×) /
   72.95 → 36.26 ms (DelaunayNN, 2.01×) with the certified active set injected at the library's
   own solver entry point, and the positive-negative route stays prohibited at +334.93 nats.
2. **The same cell on CPU and on the laptop RTX 2060** (`PyAutoGPU` venv), fp64 and mixed
   precision — consumer-GPU optimisation now that the method is converging
   (`fixed_light_cpu_and_consumer_gpu.md`).
3. **Certified-scheme cost on low-likelihood model draws** — is 2 passes an artefact of a
   good model? (`fixed_light_certified_low_likelihood_draws.md`).
4. **Source-pixel scaling per hardware** (`fixed_light_source_pixel_scaling.md`).
5. **Whole-likelihood assessment, HST + Euclid × 500 / 1250 / 2500 source pixels × A100 /
   CPU / RTX 2060, dense only** — the verdict
   (`fixed_light_likelihood_assessment_hst_euclid.md`).

Out of scope throughout, and blocking the sparse path here: the w-tilde weight map ignores a
profile-subtracted image, filed as
`PyAutoMind/draft/bug/autoarray/sparse_inversion_ignores_profile_subtracted_image.md`; and
`sparse_steps.sparse_context_from` rejects the S3 build outright (it requires exactly one
func list). No PyAutoArray solver prompt is filed yet — phase 1's library-path rows decide
whether the certified scheme becomes a `Settings` option with PDIP fallback.

## Artifacts

```
results/breakdown/imaging/fixed_light_{rectangular,delaunay,delaunay_nn}_hpc_a100_fp64_fixed_light.{json,png}
results/breakdown/imaging/fixed_light_{rectangular_n3025,delaunay_n3000}_hpc_a100_fp64_fixed_light.{json,png}
results/misc/fixed_light_probe/{rectangular_1521,delaunay_1500,rectangular_400}.{json,md}   # CPU probe
```

Code:

```
scripts/misc/likelihood_breakdown/active_set_steps.py    # S3 system builder, numpy certified reference, masked-JAX twin
scripts/misc/likelihood_breakdown/fixed_light_probe_cpu.py
scripts/imaging/likelihood_breakdown/fixed_light.py      # the A100 cell
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_{pixelization,delaunay,delaunay_nn}_a100_hst_fp64
hpc/batch_gpu/submit_fixed_light.sh
scripts/misc/test/test_active_set_steps.py
```

JSON keys specific to this cell: `s0` / `s3` (`rows`, `nnls`, `log_evidence_terms`, `cond`,
`log_evidence_figure_of_merit`), `mapper_block_log_dets`, `active_set` (`budgets` — one entry
per pass budget with `n_primal_violations_per_pass`, `n_dual_violations_per_pass`,
`certified_at_pass`, `d_log_evidence_vs_library`; `edge_zeroing_penalty_nats`;
`smallest_certifying_budget`), `a2`, `vmap16`, `library_row`, `reconstruction_alternatives_ms`,
`peak_bytes`, `jit_phases`, `device.cache_fresh`, `pinned_expected` / `pinned_drift`. The
`hpc_a100_fp64_fixed_light` config label is README-table-invisible by design
(`CONFIG_TAGGED_RE`); this note and the README prose row are the index into these legs.
