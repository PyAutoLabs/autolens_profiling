# Certified positive solver, phase C1: uncertified-lane rate on real Nautilus batches

Issue [#304](https://github.com/PyAutoLabs/autolens_profiling/issues/304). Follows phase B
([certified_solver_policy_phase_b_2026_09.md](certified_solver_policy_phase_b_2026_09.md), array
350588). Phase B passed its gate on seeded draws near the fiducial with the regularization fixed.
This phase replays lanes that a real `af.Nautilus` run proposed, at the production settings of
both pixelized SLaM stages, with the regularization **free** (production priors and classes).

## Verdict

**The pre-registered gate FAILED: all 24 timed tasks failed (2 to 40 lanes each); the 6 B=100
tasks ran out of memory; the 4 ungated rate tasks completed.** The gate is the phase-B gate,
unchanged: each arm is compared per lane against the library-PDIP reference of its own composition,
at 1e-9 relative. Under the pre-registration nothing in the timed tables supports a production
recommendation, and no `certified_fallback=none` row is policy-eligible.

What the failures are, from the per-lane data (details under Diagnosis):

1. **The failures are lane-driven, not solver-driven.** Every failure in a `library PDIP` or
   `certified+PDIP` row at B ≤ 20 is on a lane whose captured log likelihood is catastrophic (≤ 0,
   i.e. between −3e4 and −1.9e5 against a peak of about +2.9e4). These lanes mostly have a large
   inner or outer coefficient (≳ 30). On them, **library PDIP disagrees with itself** at 1e-9 to
   1e-5 relative (up to 1.8 nats) between any two separately compiled programs. That covers
   arm against reference with the same solver, vmap reference against scalar reference, and this
   cell against the capture's own Fitness evaluation. Some of these programs return NaN on some
   of these lanes and a finite value on others. No lane with a positive log likelihood failed in
   these rows.
2. **The uncertified lanes are a separate population.** With `certified_fallback=none`, the
   remaining failures are lanes where the certificate failed at the budget of 16 passes. These
   lanes are under-regularized (both coefficients small, median log10 min about −4) and fit
   reasonably well (log L about 1.9e4). The returned iterate is wrong by 1e-6 to 8e-3 relative
   (up to 57 nats in a timed window, 141 nats in the rate replay). The error is the same to
   1e-11 in both compositions. This is the failure the fallback exists for, and the fallback
   catches it: at B ≤ 20 no `certified+PDIP` row failed on a lane that `none` left uncertified.
3. **At B=50 the `jit(vmap)` program returns wrong values on nearly every lane, on both meshes,
   for every solver.** The median error is about 5200 nats (Delaunay) and 6000 nats
   (rectangular); the relative error on individual lanes reaches 9x. The scalar arm and the
   capture agree with each other. The gate is **blind** to this: the arm and its own-composition
   reference share the fault. Only the ungated cross-composition column shows it, at 9.12 and
   8.59 relative. It occurs exactly in the six tasks whose `.err` logs the XLA allocator warning
   "ran out of memory trying to allocate 18.3 GiB ... not a failure". It is not a solver
   question, and it is unresolved here.
4. **B=100 is the memory limit.** The vmap library-PDIP reference asks for one 73.3 GiB
   (Delaunay) or 74.3 GiB (rectangular) allocation. XLA reports that rematerialization cannot get
   the program below about 90 GiB.

Rate (ungated, B=20 chunks, certified+none): **1.9–4.9% of lanes are uncertified overall. All of
them fall in the early half of the run; the late half has 0 and needs at most 3 to 8 passes.**
Timing is never gated. The B ≤ 20 timings are valid measurements of the programs as run. The B=50
`jit(vmap)` timings are timings of a program that returns wrong values, so they should not be
used.

Policy is the human's call. The open questions are at the end.

## Experiment

- Captures: `nautilus_batch_capture.py`, `af.Nautilus(n_live=150|75, n_batch=20,
  use_jax_vmap=True)`, library PDIP, stage pix1 (mass + shear + regularization free) and pix2
  (regularization only), capped at 40000 likelihoods and 3600 s.
  - Delaunay (AdaptSplit): reused from array `350659` tasks 0 and 2 (COMPLETED; 18640 and 1680
    lanes), captured on PyAutoArray `7fa8d271` / PyAutoFit `a7368401`.
  - Rectangular (Adapt, re-captured after rectangular pix1 moved to Adapt): array `350766` tasks
    1 and 3 (COMPLETED 16m21s and 6m09s; 19440 and 2840 lanes), on PyAutoArray `3de624b5` /
    PyAutoFit `dd9fbe0a`.
  - The npz captures stay on RAL (not committed). Every replay checks the fingerprint;
    `fingerprint_check.passed` is true in every JSON.
- Replays: array `350768` tasks 4–37, started 2026-09-25 17:39:41 BST, on `euclid-ral-gpu-1` and
  `-2`. Tasks 4–7 (rate) COMPLETED 0:0. Tasks 8–37 FAILED 1:0: the gate raise in 24 of them, OOM
  in 6. Every task's JSON and PNG were written before the raise.
- Profiling source: `be01a52` (feature/certified-solver-phase-c1-lane-rate), the AP_ROOT revision
  printed in every `.out`. Every replay JSON's `cell_source_sha256` (`06a162fc…`) is the cell at
  that commit. The gate is pre-registered in the submit header and the cell docstring at that
  commit (committed 17:04 BST, before the 17:23 BST start of 350766).
- Library mains at replay (JSON `library_revisions`): PyAutoNerves `1fa613aa`, PyAutoFit
  `dd9fbe0a`, PyAutoArray `3de624b5` (#567 and #572), PyAutoGalaxy `70a61e26`, PyAutoLens
  `86054bbc`. Unreleased.
- Device: NVIDIA A100 80GB PCIe; fp64; HST imaging; fixed lens light at S3; dense inversion;
  border relocation on; Delaunay N=1500, rectangular 39x39=1521. Fresh compilation cache per
  task; `AUTOTUNE_ENTRIES count=0` in all 36 replay and capture footers.
- Time pass: the phase-B matched pair (`jax.jit(jax.vmap(fn))` and B x `jax.jit(fn)`, one
  process) on the `timed_window`. The window is consecutive captured calls from the first
  post-prior batch (lane rows 300… for pix1, 160… for pix2), so B=16 and B=20 share lanes and
  B=20 is exactly one real Nautilus batch. The solver is selected through `al.Settings`; the
  budget is the packaged 16.

Provenance, per-task sacct rows, OOM and allocator excerpts, and artifact checksums are in
[certified_solver_phase_c1_job350768.json](certified_solver_phase_c1_job350768.json). The 28 JSON
files and 24 PNG files are under `results/breakdown/imaging/`, suffixes
`_jitvmap<B>_captured_<stage>_lib<solver>_fb<on|off>_b16_…` (time) and
`_captured_rate_libcertified_fboff_b16_<stage>_…` (rate).

## Results: the time pass (gated)

Failed lanes are split into three columns: *catastrophic* (captured log L ≤ 0), *uncertified*
(log L > 0, `certified=False`) and *other*. "nonfinite" counts the lanes where any of the four
values (two arms, two references) is NaN; a NaN lane fails the gate. The cross-composition columns
are recorded, not gated.

| task | mesh | stage | B | row | failed | catastrophic | uncertified | other | nonfinite | worst gated vmap rel | worst gated scalar rel | cross-comp rel | cross-comp nats | uncertified lanes | vmap ms/lane | scalar ms/lane |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | delaunay | pix1 | 16 | library PDIP | 4/16 | 4 | 0 | 0 | 0 | 4.30e-06 | 7.95e-08 | 5.17e-06 | 0.916 | n/a | 47.0 | 50.6 |
| 9 | delaunay | pix1 | 16 | certified+PDIP | 5/16 | 5 | 0 | 0 | 0 | 1.64e-06 | 2.67e-07 | 1.02e-05 | 1.8 | 4 | 60.2 | 38.9 |
| 10 | delaunay | pix1 | 16 | certified+none | 8/16 | 4 | 4 | 0 | 0 | 2.26e-04 | 2.26e-04 | 7.04e-06 | 1.25 | 4 | 32.3 | 32.1 |
| 11 | delaunay | pix1 | 20 | library PDIP | 4/20 | 4 | 0 | 0 | 1 | 1.40e-06 | 6.06e-07 | 4.81e-06 | 0.913 | n/a | 43.0 | 50.0 |
| 12 | delaunay | pix1 | 20 | certified+PDIP | 4/20 | 4 | 0 | 0 | 0 | 2.99e-06 | 2.78e-07 | 7.86e-06 | 1.39 | 7 | 54.4 | 44.3 |
| 13 | delaunay | pix1 | 20 | certified+none | 11/20 | 4 | 7 | 0 | 1 | 2.26e-04 | 2.26e-04 | 3.65e-06 | 0.692 | 7 | 30.5 | 34.3 |
| 14 | delaunay | pix1 | 50 | library PDIP | 14/50 | 12 | 0 | 2 | 0 | 1.60e-04 | 1.72e-06 | **9.12** | **2.07e5** | n/a | *32.6* | 50.1 |
| 15 | delaunay | pix1 | 50 | certified+PDIP | 12/50 | 12 | 0 | 0 | 1 | 1.44e-04 | 4.86e-07 | **9.12** | **2.07e5** | 24* | *40.2* | 44.2 |
| 16 | delaunay | pix1 | 50 | certified+none | 36/50 | 13 | 22 | 1 | 1 | 1.68e-03 | 3.31e-04 | **9.12** | **2.07e5** | 24* | *26.7* | 34.3 |
| 20 | rectangular | pix1 | 16 | library PDIP | 2/16 | 2 | 0 | 0 | 2 | 9.24e-07 | 1.31e-11 | 3.99e-06 | 0.709 | n/a | 33.2 | 41.4 |
| 21 | rectangular | pix1 | 16 | certified+PDIP | 2/16 | 2 | 0 | 0 | 2 | 1.11e-07 | 8.06e-11 | 5.18e-10 | 1.15e-05 | 7 | 44.4 | 42.3 |
| 22 | rectangular | pix1 | 16 | certified+none | 9/16 | 3 | 6 | 0 | 2 | 3.26e-04 | 3.26e-04 | 2.81e-10 | 6.24e-06 | 7 | 27.9 | 30.8 |
| 23 | rectangular | pix1 | 20 | library PDIP | 3/20 | 3 | 0 | 0 | 2 | 4.40e-06 | 6.81e-07 | 2.64e-06 | 0.469 | n/a | 30.6 | 41.1 |
| 24 | rectangular | pix1 | 20 | certified+PDIP | 3/20 | 3 | 0 | 0 | 3 | 1.03e-06 | 1.10e-10 | 2.42e-10 | 5.37e-06 | 10 | 40.4 | 44.7 |
| 25 | rectangular | pix1 | 20 | certified+none | 13/20 | 4 | 9 | 0 | 3 | 7.94e-03 | 7.94e-03 | 3.47e-10 | 7.68e-06 | 10 | 26.5 | 31.3 |
| 26 | rectangular | pix1 | 50 | library PDIP | 12/50 | 11 | 0 | 1 | 6 | 4.06e-05 | 3.09e-07 | **8.59** | **2.13e5** | n/a | *27.6* | 40.4 |
| 27 | rectangular | pix1 | 50 | certified+PDIP | 12/50 | 11 | 1 | 0 | 5 | 1.08e-04 | 7.36e-06 | **8.59** | **2.13e5** | 28* | *34.1* | 41.7 |
| 28 | rectangular | pix1 | 50 | certified+none | 40/50 | 13 | 27 | 0 | 5 | 2.95e-02 | 7.94e-03 | **8.59** | **2.13e5** | 28* | *22.1* | 30.4 |
| 32 | delaunay | pix2 | 20 | library PDIP | 5/20 | 5 | 0 | 0 | 0 | 5.25e-07 | 2.40e-08 | 4.23e-06 | 0.586 | n/a | 22.8 | 49.5 |
| 33 | delaunay | pix2 | 20 | certified+PDIP | 5/20 | 5 | 0 | 0 | 0 | 6.93e-07 | 9.30e-08 | 3.47e-06 | 0.482 | 8 | 34.1 | 43.6 |
| 34 | delaunay | pix2 | 20 | certified+none | 12/20 | 5 | 7 | 0 | 0 | 1.42e-04 | 1.42e-04 | 3.49e-06 | 0.484 | 8 | 13.2 | 33.1 |
| 35 | rectangular | pix2 | 20 | library PDIP | 5/20 | 5 | 0 | 0 | 0 | 1.63e-07 | 2.21e-08 | 3.50e-07 | 0.068 | n/a | 17.0 | 39.8 |
| 36 | rectangular | pix2 | 20 | certified+PDIP | 6/20 | 6 | 0 | 0 | 0 | 8.61e-08 | 1.54e-08 | 3.53e-07 | 0.0685 | 6 | 26.8 | 35.7 |
| 37 | rectangular | pix2 | 20 | certified+none | 11/20 | 5 | 6 | 0 | 0 | 1.51e-04 | 1.51e-04 | 3.87e-07 | 0.0753 | 6 | 11.5 | 28.5 |

`*` The B=50 uncertified counts and the *italic* B=50 vmap timings come from the faulty B=50
`jit(vmap)` program (see Diagnosis 3). They are recorded but not usable. The B=50 scalar columns
are unaffected.

Worst gated disagreement in nats on the catastrophic lanes at B ≤ 20: at most 0.86 nats (rectangular
pix1 B=20 library PDIP). The uncertified certified+none lanes are off by up to 57 nats (rectangular
pix1 B=20).

### Matched timing at B ≤ 20 (ms per likelihood, per lane; valid)

| mesh | stage | B | library PDIP `jit(vmap)` | certified+PDIP `jit(vmap)` | certified+none `jit(vmap)` | scalar PDIP | scalar certified+PDIP | scalar certified+none |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| delaunay | pix1 | 16 | 47.0 | 60.2 | 32.3 | 50.6 | 38.9 | 32.1 |
| delaunay | pix1 | 20 | 43.0 | 54.4 | 30.5 | 50.0 | 44.3 | 34.3 |
| delaunay | pix2 | 20 | 22.8 | 34.1 | 13.2 | 49.5 | 43.6 | 33.1 |
| rectangular | pix1 | 16 | 33.2 | 44.4 | 27.9 | 41.4 | 42.3 | 30.8 |
| rectangular | pix1 | 20 | 30.6 | 40.4 | 26.5 | 41.1 | 44.7 | 31.3 |
| rectangular | pix2 | 20 | 17.0 | 26.8 | 11.5 | 39.8 | 35.7 | 28.5 |

At the production batch (B=20, one real Nautilus batch):

- certified+none `jit(vmap)` is 1.41x faster than library PDIP `jit(vmap)` on Delaunay pix1,
  1.16x on rectangular pix1, 1.73x on Delaunay pix2 and 1.47x on rectangular pix2.
- certified+PDIP `jit(vmap)` is slower than library PDIP `jit(vmap)` in every row, as in phase B.

Against phase B's seeded lanes at Delaunay B16, every captured-lane row is slower: PDIP vmap 47.0
vs 38.9, certified+PDIP 60.2 vs 45.6, certified+none 32.3 vs 25.9. Real batches are harder, and
under vmap a certified batch runs to its slowest lane's pass count.

The scalar certified arms vary between tasks more than in phase B. For example, Delaunay pix1
scalar certified+PDIP is 38.9 ms at B16 and 44.3 ms at B20 on the same lanes plus four. These are
recorded as measured.

## Results: the rate pass (ungated)

Certified with `certified_fallback=none`, `jit(vmap)` chunks of 20 (each chunk is exactly one
captured batch), every captured lane, budget 16.

| mesh | stage | lanes | uncertified | rate | at budget | mean passes | prior phase rate | early half rate | late half rate | late max passes | uncertified vs scalar PDIP: median / max nats |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delaunay | pix1 | 18640 | 368 | 1.97% | 387 | 2.45 | 11.7% (35/300) | 3.95% | 0 | 3 | 0.28 / 31.4 (256 rechecked) |
| delaunay | pix2 | 1680 | 82 | 4.88% | 82 | 2.98 | 20.6% (33/160) | 9.76% | 0 | 3 | 0.16 / 2.81 |
| rectangular | pix1 | 19440 | 365 | 1.88% | 376 | 5.95 | 14.3% (43/300) | 3.76% | 0 | 8 | 2.69 / 141 (256 rechecked) |
| rectangular | pix2 | 2840 | 70 | 2.46% | 71 | 6.22 | 20.0% (32/160) | 4.93% | 0 | 7 | 2.02 / 4.33 |

- Every rate task reached pass 16, so the packaged budget is reached, not merely approached. The
  late half of every run certifies within at most 3 passes (Delaunay) or 8 (rectangular).
- By Nautilus phase: the exploration phase is 1.4–3.4% uncertified. The only sampling-phase rows
  (Delaunay pix2, 80 lanes) had 0.
- The uncertified lanes are under-regularized: median log10 min(inner, outer) is −3.7 to −4.6,
  and 71–83% have min coefficient < 1e-3. The certified lanes sit at −0.2 to −3.5 (17–60%).
  Uncertified lanes also fit reasonably well (median captured log L about 1.9e4 against about 2.9e4
  for the certified lanes), so they are not far down the likelihood tail where the sampler would
  discard them.
- Against the capture's own PDIP value, the median disagreement over all lanes is 3e-13
  (Delaunay) to 2e-11 (rectangular). The number of lanes above 1e-9 is 648, 147, 518 and 130
  respectively. Of those, 333, 75, 354 and 69 are uncertified; the rest are certified lanes, which
  concentrate on the catastrophic lanes of Diagnosis 1.

## Diagnosis

Per-lane tables were built from the 24 time JSONs, joining each lane to its captured parameter
vector and figure of merit through `vmap.lane_construction.lane_rows`.

**1. PDIP is not reproducible at 1e-9 on catastrophic lanes, and this is lane-driven.**

- *Same lanes fail across rows and batch sizes.* At B ≤ 20, in Delaunay pix1 the lanes 302, 303
  and 309 fail in 6/6 rows and 310 in 5/6. The rectangular pix1 lanes 304 and 314 fail in 6/6. The
  pix2 catastrophic lanes fail in 3/3.
- *All of these have captured log L ≤ 0.* Examples: −1.9e5 (Delaunay 302), −1.8e5 (309),
  −1.9e5 (rectangular 304). Most have a large coefficient: log10 inner = +2.8, +1.75, +1.5 or
  log10 outer = +2.2.
- *Every PDIP program disagrees on them, at the same order of magnitude.* In `library PDIP` rows
  the arm and the reference both run library PDIP, and the two differ by up to 4.4e-6. The vmap
  and scalar references (both PDIP) disagree with each other by up to 1.05e-5 (1.67 nats on
  Delaunay 309 at B16). This cell's vmap reference and the capture's Fitness evaluation disagree by up to
  6.8e-6, or 1.3 nats (`timed_window_vs_capture_pdip`, B ≤ 20). The untimed reporting pass and the timed arm (same
  Settings, compiled separately) disagree by up to 4.5e-6.
- *NaNs appear in some programs and not others.* Delaunay lane 309 at B=20: the PDIP vmap arm
  returns NaN while the PDIP vmap reference returns a finite value. Rectangular lanes 304 and 314:
  the scalar composition returns NaN while vmap is finite. Rectangular 317 is NaN everywhere; the
  capture recorded it as −1e99, a resampled non-finite value.
- *The composition is not the source at B ≤ 20.* Rectangular certified rows agree across
  compositions to 5e-10, but each is off from its PDIP reference by up to 1e-6. So the certified
  solve is reproducible there and PDIP is the element that varies.
- *Phase B passed* because its lanes were near-fiducial with the regularization fixed. Its floor was
  2e-10 (Delaunay) and 0 (rectangular).

**What differs between the arm and the reference in a `library PDIP` row.** Both come from
`_lane_likelihood_fn(system_s3.dataset, settings)`. The arm uses `_settings_library`
(`positive_only_solver="pdip"`, `certified_fallback="pdip"`, `certified_pass_budget=16`); the
reference uses `_settings_pdip_reference` (`positive_only_solver="pdip"` only). They are traced and
compiled separately, and the arm is evaluated inside `host_callback_probe.qhull_probe()`, which is a
no-op on rectangular. On a mapper inversion the library ignores the certified settings when the
solver is PDIP (`positive_only_solver_used` returns "pdip"; `positive_only_preconditioning_used`
returns "jacobi"). So the two are the same algorithm, compiled twice. The data do not show whether
their HLO is byte-identical, or whether one executable run twice returns the same bits.

The mechanism is a hypothesis, not measured here. These lanes may be ill-conditioned enough that
PDIP (tolerance `min(n·eps, 1e-2)` ≈ 3.3e-13, `max_iter=50`) stops on an iterate that depends on
rounding. The cell records no PDIP iteration or convergence flag (`certification_report.pdip_iter`:
"NOT AVAILABLE"), although PyAutoArray #572, now on main, exposes `converged` and `iterations` in
the PDIP `stats=` dict.

**2. Uncertified certified+none lanes are a second, separate population.** They fail only in
certified+none rows: in Delaunay pix1, lanes 301, 311, 314 and 315 fail in 2/6 rows, both of them
certified+none. Each is `certified=False, passes=16`. The vmap and scalar arms agree with each other
to 1e-11 to 1e-15 and are off from PDIP by 2.6e-6 to 7.9e-3. The iterate is deterministic and wrong,
as designed for `none`. In the certified+PDIP rows the same lanes fall back and pass.

This is the pattern behind the observation "vmap ≈ scalar to 1e-13 but both 1e-4..1e-5 from the
reference". Task 37 lanes 3, 4, 5 and 11 (rows 163, 164, 165, 171) are all `certified=False,
passes=16`. That pattern is not PDIP ill-conditioning.

**3. At B=50, `jit(vmap)` computes a different likelihood.**

- In all six B=50 tasks, the vmap arm and the vmap reference agree with each other (so most lanes
  pass the gate). But they differ from the scalar arm by more than 1e-3 on 44–50 of 50 lanes, and
  from the capture on 47–50.
- Example, Delaunay lane 300 (PDIP): vmap 18700.59, scalar 24758.51, capture 24758.51. At B=16 and
  B=20 the same lane gives 24758.51 in both compositions.
- The values are not a permutation of other lanes' values (checked on the Delaunay PDIP task).
- The B=50 tasks are the only ones whose `.err` logs `bfc_allocator.cc:317 ... ran out of memory
  trying to allocate 18.31 GiB` (18.56 GiB rectangular) "... The caller indicates that this is not
  a failure". That size is B x the 0.37 GiB PSF-convolved mapping cube; it appears 10 times per
  task, and peak use is 58.8 GB.
- The correlation is exact in this array (6/6 tasks with the warning have the fault; none of the 18 B ≤ 20 time tasks or the 4 rate tasks logs the warning or shows the fault).
  It is not a demonstrated cause.
- The per-lane gate as pre-registered cannot see a fault that the arm shares with its reference.

**4. B=100 memory limit (pre-registered: "an OOM task is itself the memory-limit row").** Tasks
17–19 (Delaunay) and 29–31 (rectangular) failed after 41–45 s at `fixed_light_trace.py:2340`, the
`jit(vmap)` library-PDIP reference, before either arm ran:

```
hlo_rematerialization.cc:3231] Can't reduce memory use below 56.48GiB (60647426270 bytes) by rematerialization; only reduced to 89.66GiB (96270652980 bytes), down from 89.67GiB
bfc_allocator.cc:514] Allocator (GPU_0_bfc) ran out of memory trying to allocate 73.30GiB (rounded to 78702055168)requested by op
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 73.30GiB. [executable_name='jit__likelihood']
```

(rectangular: 90.86 GiB after rematerialization, 74.27 GiB allocation). This fp64 cell at about
1500 source pixels cannot run B=100 as one `jit(vmap)` call on an A100 80GB.

## Open questions for the human

1. **The gate failed as pre-registered.** Should phase C1's timed pass be re-registered and re-run?
   If so, which gate? The data separate three things the 1e-9 own-composition pin mixes together:
   PDIP's own non-reproducibility on catastrophic lanes (≤ 0.86 nats, lanes about 1e5 nats below
   the peak), uncertified `none` iterates (up to 57 nats timed, 141 nats in the rate replay), and a
   composition-level fault that the pin cannot see (B=50). Options include a nats-based pin, a
   pin restricted to lanes within some Δlog L of the batch maximum, or making the cross-composition
   and capture checks gated. Those are decisions, not findings, and none is taken here.
2. **Is the C2 guard (re-run uncertified lanes through scalar certified+PDIP) warranted?** The
   uncertified rate is 1.9–4.9% overall, all of it in the early half of the run (up to 21% in the
   prior phase), and 0 in the late half. The uncertified iterates are off by a median of 0.16–2.7
   nats and at most 141 nats, on lanes that fit reasonably well, so `none` without a guard returns
   materially wrong likelihoods early in a search. Arithmetic only, from the valid B=20 timings:
   Delaunay pix1 certified+none vmap is 30.5 ms/lane. Adding 2% of lanes re-run at the measured
   scalar certified+PDIP 44.3 ms gives about 31.4 ms/lane, against library PDIP vmap at 43.0.
3. **Should B=50 `jit(vmap)` be investigated as a separate bug** (library or XLA under allocator
   pressure)? Production `n_batch=20` did not show it here: the rate pass ran at 20 with no
   allocator warning, and B ≤ 20 compositions agree away from catastrophic lanes. But any
   configuration that batches 50 or more HST-scale lanes in one `jit(vmap)` call is suspect until it
   is localised.
4. **Should the next witness record PDIP convergence per lane?** That means `stats=` `converged` and
   `iterations`, now available on main after #572. It would test whether the catastrophic lanes hit
   `max_iter` without convergence. A second check would be whether the same compiled executable
   returns identical bits when re-run on those lanes.
5. **NaN on catastrophic lanes.** Some PDIP programs return NaN on lanes that another composition
   evaluates finitely (Delaunay 309, rectangular 304 and 314). Does production care, given that
   Fitness resamples non-finite values (rectangular 317 was captured as −1e99)?
