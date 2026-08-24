# Phase 8A / CP-4 — slogdet vs cholesky A/B on the AdaptSplit NaN wall: results

Driver: `scripts/misc/searches/slogdet_ab.py` (pre-registration, cell
selection and verdict rules in its module docstring). Submits:
`hpc/batch_gpu/submit_slogdet_ab_adaptsplit_a100` (RAL 338808, A100 fp64,
29 min) and `hpc/batch_cpu/submit_slogdet_ab_adaptsplit_ral_cpu` (RAL
338807, 5 h 37 m of a 6 h limit). Both COMPLETED 2026-08-23/24; artifacts
under `slogdet_ab/` (verdict JSON + `per_draw.npz` per cell × tier, draw
sets under `draws/`). Version stamp 2026.8.17.1, jax 0.10.2.

Replay set per cell: 128 prior draws + 128 λ-transect draws + 128
truth-bar draws (+32 descent-trajectory draws on the A100 tier); anchor =
prior medians (no truth-basin vector recorded in-repo — see driver).

## Verdict table (pre-registered criteria: all four must hold)

| cell | tier | draws | chol NaN | slogdet NaN | rescued | regressed | NaN both | grad non-finite chol → slogdet | runtime ratio | criteria 1/2/3/4 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| knn | A100 | 416 | 0 | 0 | 0 | 0 | 0 | 128 → 128 | 1.03 | ✅ ✗ ✗ ✅ | **VOID** (control never walled) |
| knn | RAL CPU | 384 | 0 | 0 | 0 | 0 | 0 | 129 → 129 | 2.52 | ✅ ✗ ✗ ✗ | **VOID** (control never walled) |
| delaunay_adapt_split | A100 | 416 | 90 | 32 | **58** | **0** | 32 | 192 → 160 | 1.03 | ✗ ✗ ✗ ✅ | **FAIL** |
| delaunay_adapt_split | RAL CPU | 384 | 73 | 20 | **53** | **0** | 20 | 180 → 151 | **3.74** | ✗ ✗ ✗ ✗ | **FAIL** |

## Reading

1. **knn is not a NaN wall — confirmed, on both tiers.** As the driver's
   "WHICH CELL ACTUALLY WALLS" note predicted from #117, the pre-registered
   stressor (KNN + free AdaptSplit) produced zero Cholesky NaNs over 416
   draws spanning the prior, a λ transect to 8e5 and the truth bar. Its
   criteria pass vacuously; the verdict is VOID as pre-registered, not a
   pass. The 128/129 non-finite gradients are the λ-transect draws under
   BOTH arms and are unrelated to `log_det_method` (attribution block:
   `slogdet_introduces_no_new_grad_nans: true`).

2. **On the real wall (Delaunay + free AdaptSplit) slogdet is a partial
   mitigation, not a fix.** It rescues 64 % / 73 % of the Cholesky NaNs
   (58/90 A100, 53/73 CPU) with **zero regressions** — every draw finite
   under Cholesky stays finite under slogdet, and every draw whose gradient
   is finite under Cholesky stays finite (attribution:
   `grad_nonfinite_only_under_slogdet = 0`; 32 / 29 draws gain a finite
   gradient). But 32 / 20 draws are NaN under both arms — those are
   genuinely singular systems (the λ⁴ fragility of #104), where no log-det
   formula returns a number — and the λ-transect gradients are non-finite
   under both arms on all 128 draws. Criteria 1 and 3, scored on the
   treatment alone as registered, fail.

3. **Value equality holds on clean-PD points and breaks in the marginal
   band, as the laptop probe warned.** `clean_pd` (251 / 243 draws):
   max |Δ| 1.7e-4 / 1.0e-4 at |logL| ~ 1.6e5 (relative ~1e-9 — the target
   has not moved where the Cholesky was healthy). `marginal_band` (75 / 68
   draws, coefficients up to 8e5–9e5): max |Δ| **9,619 nats on the A100**
   vs 2.27 on CPU. The two arms disagree by thousands of nats exactly
   where the Cholesky is numerically marginal — and the disagreement is
   tier-dependent, so neither arm can be called "the value" there. The 58
   rescued draws all evaluate to logL ≤ −1.6e5 (median −2.2e5): slogdet
   turns a NaN into a finite but catastrophically bad number, which is
   what a gradient search needs (a direction) but not a category-2
   equivalence claim.

4. **Runtime: free on GPU, 3.7× on CPU.** A100 warm wall 67 → 69 s
   (1.03×); RAL CPU 1,746 → 6,522 s (3.74×, fails the 2× ceiling; compile
   also 45 → 100 s). LU-based slogdet has no cheap CPU path at this matrix
   size; the GPU hides it.

## Decision input for Gate E/F (superseded by the human call below)

The pre-registration expected a clean pass; it did not get one. Recommended
recording (DECISIONS.md 2026-08-24): **slogdet is NOT recommended as the
gradient-work default profile.** It may be enabled per-fit on GPU tiers as a
NaN-wall softener at zero cost and zero regression risk, but it does not
remove the wall, it changes the value in the marginal band by a
tier-dependent amount, and it is 3.7× slower on CPU. Phase 8B
(log-coordinate stepping — a category-1 reparameterization that keeps the
search away from the singular band altogether) is now the live candidate;
8+ (analytic log-det on fixed-topology rectangular meshes) is the only
lever that kills the Cholesky rather than dressing it.

## Human call (2026-08-24)

Adopt now, chase the rest: slogdet becomes the default for gradient-work
cells in this repo on GPU tiers (W8, library default untouched); the
NaN-under-both draws and transect gradients get their own investigation
(W7); Phase 8B proceeds in parallel; "make slogdet standard in PyAutoArray"
is a reminder owed once W7 reports. See DECISIONS.md 2026-08-24.

## Ops notes

- The CPU run's 5 h 37 m sits 23 min under its 6 h limit; any wider
  transect on CPU needs a longer limit or a smaller replay set.
- Descent-trajectory draws exist only on the A100 tier (32); on CPU the
  driver skipped them (no per-source `descent` block).
- **CORRECTED 2026-08-24 (W7, autolens_profiling#164):** `coefficient_min =
  −5.2` in the A100 marginal band is **not** a log-coefficient axis — that
  read was wrong. It is a harvest bug: six `descent`-source draws (indices
  384, 391, 393, 394, 397, 405) carry a NEGATIVE physical
  `inner_coefficient`/`outer_coefficient` (e.g. draw 405 outer=−5.22),
  produced by the unbounded Prodigy lane-checkpoint spy in `harvest_descent`
  (`slogdet_ab.py`). Ten further descent draws (406-415) are dead lanes
  (entirely-NaN checkpointed vectors). Both are now dropped at harvest —
  see "W7 addendum" below.

## W7 addendum — attribution per draw class (autolens_profiling#164)

Driver: `scripts/misc/searches/slogdet_nan_attribution.py`. Replays
individual stored draws non-jitted, drills into `inversion.*` matrices and
small `jax.grad` closures (fixed one-time JIT compile per parameter subset,
reused across every draw), and assigns each sampled draw one mechanism
label. Artifacts under `slogdet_ab/attribution/`.

Two driver artefacts identified and fixed at harvest (`slogdet_ab.py`):

1. **Descent-harvest bug.** Six `descent`-source draws (384, 391, 393, 394,
   397, 405) carried a NEGATIVE physical regularization coefficient; ten
   more (406-415) were dead lanes (entirely-NaN checkpointed vectors). Both
   are now dropped in `harvest_descent` before the row is appended to the
   replay set.
2. **Anchor-singularity artefact.** All 128 `lambda_transect` draws sit at
   the anchor (prior medians), where `ell_comps` and shear components are
   EXACTLY `(0, 0)`. `autogalaxy/convert.py:86`
   (`axis_ratio_and_angle_from`) and `:220`
   (`shear_magnitude_and_angle_from`) both take
   `sqrt(e1**2 + e2**2)`, whose gradient is undefined at the origin — a
   property of the driver's anchor choice, not evidence about the
   regularization wall. `anchor_vector` now jitters any exact-zero
   ell_comps/shear component by 1e-3 in unit-cube space.

### Classification table (sampled draws, both tiers)

Two replay modes. `--classes nan_both,transect --max-per-class 4` (8 draws,
full gradient probes) established the mechanism vocabulary and is the
source for `anchor_singularity`. `--classes all --max-per-class 15
--matrix-only` (87 A100 / 83 RAL CPU draws — every class capped at 15,
`clean_pd`/`marginal_band`/`transect`/`tier_flip`/`truth_bar` sampled
evenly, `nan_both`/`rescued` sampled evenly across the stored replay) is the
population-level classification: it skips the `jax.grad` closures (cost
note in the driver's docstring — an un-jitted grad through the full
15,361-pixel forward+backward pass is minutes/call; matrix-only is
seconds/call) and classifies purely from `cond(curvature_reg_matrix_reduced)`,
so it cannot see `anchor_singularity` (a gradient-only mechanism) — every
`lambda_transect` draw in the matrix-only table is instead classified by
its conditioning alone, which is a genuinely different (and complementary)
reading: **within the same draw class, the failure mechanism changes with
λ** — see "Reading" point 1.

| class (matrix-only, ≤15 sampled/tier) | clean | marginal_tier_flippable | genuinely_singular | dead_lane | build_error |
|---|---:|---:|---:|---:|---:|
| nan_both — A100 (n=15) | 0 | 2 | 8 | 5 | 0 |
| nan_both — RAL CPU (n=15) | 0 | 3 | 12 | 0 | 0 |
| rescued — A100 (n=15) | 7 | 2 | 6 | 0 | 0 |
| rescued — RAL CPU (n=15) | 4 | 3 | 8 | 0 | 0 |
| marginal_band — A100 (n=15) | 8 | 5 | 1 | 0 | 1 |
| marginal_band — RAL CPU (n=15) | 7 | 4 | 4 | 0 | 0 |
| transect — A100 (n=15) | 7 | 6 | 2 | 0 | 0 |
| transect — RAL CPU (n=15) | 7 | 6 | 2 | 0 | 0 |
| tier_flip — A100 (n=15) | 5 | 4 | 6 | 0 | 0 |
| tier_flip — RAL CPU (n=15) | 5 | 4 | 6 | 0 | 0 |
| truth_bar — A100 (n=15) | 15 | 0 | 0 | 0 | 0 |
| truth_bar — RAL CPU (n=15) | 15 | 0 | 0 | 0 | 0 |
| **totals — A100 (n=87)** | **41** | **18** | **22** | **5** | **1** |
| **totals — RAL CPU (n=83)** | **35** | **18** | **30** | **0** | **0** |

`cond` ranges per label (A100 / RAL CPU, pooled over the matrix-only
sample): `clean` 118 – 4.2e11 / 118 – 2.2e11; `marginal_tier_flippable`
1.1e12 – 5.4e15 / 1.1e12 – 5.4e15 (the `[1e12, 1e16)` band by construction);
`genuinely_singular` 8.4e14 – ∞ (a `cond < 1e16` genuinely_singular draw
reached that label via the LAPACK-cholesky-raised branch, not the cond
threshold — the two criteria are alternatives, not both required, per the
driver's `_classify`).

The one `marginal_band` `build_error` (A100, draw 405) is a `descent`
checkpoint with an out-of-bounds `ell_comps` magnitude (5.6, must be < 1) —
an unphysical state the harvest fix above already drops via its
negative-coefficient check on the SAME draw (outer_coefficient=−5.22), so
this is corroborating evidence for the harvest bug rather than a new one.

The 4-draw non-`--matrix-only` probe (`nan_both`={5, 61, 127, 415},
`transect`={256, 298, 341, 383}) is what grounds `anchor_singularity`:
3 of 4 transect draws (low-to-mid λ) show a finite `figure_of_merit`,
non-finite ell_comps-only gradient and finite coefficient-only gradient —
the textbook signature. The 4th (383, λ near the prior's 1e6 ceiling)
instead lands `marginal_tier_flippable` by conditioning — the mechanism
genuinely changes along the transect (see "Reading" below). Of the 4
`nan_both` draws: 3 `genuinely_singular` (5, 61, 127 — prior draws,
coefficients in the established ~4e5-9e5 band) and 1 `dead_lane` (415, a
descent checkpoint).

### Tier-attack: does conditioning explain the tier-dependent draws?

Yes, with a caveat. Draw 96 (prior, coefficients [3.5e5, 536]) is the
concrete case: on A100 it is finite under both arms with a 9,619-nat
slogdet/cholesky delta (the original Phase 8A `marginal_band` outlier); on
RAL CPU the identical input vector is NaN under both arms. Rebuilt fresh on
this machine (CPU, numpy LAPACK), its `curvature_reg_matrix_reduced` has
`cond = 4.5e18` — three orders of magnitude past float64's ~1e16 precision
floor — with the numpy LAPACK Cholesky succeeding (matrix is nominally
PD, sign=+1) while the RECONSTRUCTION linear solve (`inversion.reconstruction`)
is itself NaN despite every input (`mapping_matrix`, `data_vector`,
`curvature_matrix`, `curvature_reg_matrix_reduced`) being finite. That is
the mechanism: at `cond ~ 1e15-1e19` the log-det terms can still be computed
(both the Cholesky-diagonal-product and the LU-based `np.linalg.slogdet`
return numbers, and per this draw they agree with the ORIGINAL A100
`log_det_curvature_reg_matrix_term`, 59814.148, to 5 significant figures),
but the reconstruction solve — a full linear system with the SAME matrix —
is not, and different BLAS/LAPACK implementations (cuSOLVER on the A100 vs
OpenBLAS on RAL CPU) round differently near that edge. So a finite log-det
does not imply a finite figure-of-merit at extreme conditioning, and the
same input vector can legitimately produce different NaN/finite verdicts
across hardware — not a bug in either implementation, a property of
inverting a ~1e18-conditioned system in fp64.

Cross-tier value agreement, restricted to draws finite under the SAME arm
on both tiers over the shared 384-draw prefix (`prior`+`truth_bar`+
`lambda_transect`, harvested with the same `--seed` on both tiers and
verified `np.allclose` before comparison): slogdet max|Δ| = **6.72 nats**
(n=360, argmax draw 68), cholesky max|Δ| = **1.30 nats** (n=293, argmax
draw 57) — both far above the fp64 round-off floor the original Phase 8A
`clean_pd` slice measured (~1e-4), confirming the marginal band (not the
clean-PD population) is where cross-tier disagreement concentrates, and
that slogdet's disagreement is systematically larger than cholesky's
there.

### Reading

1. **The failure mechanism changes along the λ transect, not just across
   draw classes.** At low-to-mid λ the transect's failure is
   `anchor_singularity` — the driver's own artefact (ell_comps/shear pinned
   at exactly zero), unrelated to regularization. At the high end (λ near
   the prior's 1e6 ceiling) the SAME transect's failure becomes matrix
   conditioning (`marginal_tier_flippable`/`genuinely_singular`, 8/15 of the
   matrix-only transect sample on both tiers). The fixed anchor jitter
   removes the first regime from future runs; the second regime is the
   real regularization-conditioning signal the transect exists to probe.

2. **`nan_both` is now attributable, not just counted.** Of the 32 A100 /
   20 RAL CPU original nan-under-both draws, established analytically
   (verified against the stored artifacts, not sampled): 10 A100 dead
   lanes and 6 A100 negative-coefficient descent rows are harvest bugs,
   now dropped. The matrix-only sample of the remaining `nan_both` draws
   (15/tier) is 0% `clean`, 80% `genuinely_singular` on RAL CPU and 53%
   on A100 (the balance being the 5 dead-lane descent draws unique to that
   tier's replay set) — i.e. once the harvest bugs are removed, essentially
   every `nan_both` draw is a genuine conditioning failure at extreme λ,
   confirming established finding 3 (the λ⁴ population, `prior`-source,
   coefficients ~4e5-9e5) rather than a driver or numerical-path artefact.

3. **`marginal_tier_flippable` (`cond` in `[1e12, 1e16)`) is a real,
   sizeable population, not an edge case.** 18/87 A100 and 18/83 RAL CPU
   matrix-only draws land there, spread across every class except
   `truth_bar`. This is the band where a hardware/BLAS change can flip a
   draw's NaN verdict (per the tier-attack above) even though the matrix is
   nominally still finite and invertible — the population CP-4's marginal
   band and this investigation's tier-flip cases both draw from.

4. **`truth_bar` is clean everywhere sampled (30/30 across both tiers,
   matrix-only), and its 15/15 A100 non-`--matrix-only` gradients are
   likewise unremarked in the original Phase 8A run.** The Gaussian
   perturbation around the anchor moves every draw off the exact-zero
   ell_comps/shear point with probability 1, so it never triggers
   `anchor_singularity`, and it stays inside a coefficient range where
   conditioning is unremarkable. It is the one draw class that behaves as
   a naive reader of the original Phase 8A verdict table would expect a
   "normal" draw to behave — useful as the negative control this
   attribution needed but did not have inside Phase 8A itself.

## CP-4 re-scored on the clean subset (W7, autolens_profiling#164)

Recomputed the four pre-registered Phase 8A criteria on `delaunay_adapt_split`,
excluding the three classes established above as driver/harvest artefacts
rather than evidence about the regularization wall: `dead_lane`,
`invalid_coefficient` (coefficient below its own prior's 1e-6 lower limit),
and `anchor_singularity` (every `lambda_transect` draw, since the anchor sat
at an exact-zero ell_comps/shear component program-wide — see above).
Computed directly from the stored `per_draw.npz` arrays (ad hoc script, not
committed; numbers pasted below).

| tier | n total | excluded (dead/invalid/anchor) | n clean | crit 1 (zero slogdet NaN) | crit 2 (value equality) | crit 3 (finite grad) | crit 4 (runtime) | verdict |
|---|---:|---|---:|---|---|---|---|---|
| A100 | 416 | 10 / 6 / 128 = 144 | 272 | FAIL (22 NaN) | FAIL (48/218 exceed tol, max\|Δ\|=9,619) | FAIL (22 non-finite) | PASS (1.03×) | **FAIL** |
| RAL CPU | 384 | 0 / 0 / 128 = 128 | 256 | FAIL (20 NaN) | FAIL (40/207 exceed tol, max\|Δ\|=1.62) | FAIL (23 non-finite) | FAIL (3.74×) | **FAIL** |

**The re-score does not overturn the original Phase 8A verdict.** Excluding
every identified driver artefact still leaves 20-22 NaN-under-both draws and
a marginal band with the same order-of-magnitude value disagreement on both
tiers. The residual failures are the genuinely-singular λ⁴ population
(established below: `prior`-source draws with a regularization coefficient
in ~4e5-9e5) — exactly the systems CP-4 set out to measure, not artefacts of
how the replay set was built.
