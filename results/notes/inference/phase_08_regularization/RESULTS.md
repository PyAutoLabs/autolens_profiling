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

- `coefficient_min = −5.2` in the A100 marginal band is a **harvest bug**,
  not a log-coefficient axis — six `descent`-source draws carried a
  genuinely negative physical `inner`/`outer_coefficient` from the unbounded
  lane-checkpoint spy in `harvest_descent`, and are now dropped at harvest.
  (This bullet previously repeated the superseded log-axis reading; it now
  agrees with the corrected "Ops notes" entry above — PR#171.)

## 8B — pre-registration (2026-08-24)

W5 (issue #162). PyAutoFit half merged to main (PR#1525 — `bijector.py`,
`MultiStartGradient(bijector=...)`, opt-in `record_lane_nan_history` /
`trace_param_indices`). Driver: `scripts/misc/searches/bijector_ab.py`
(full pre-registration, arm table and readouts in its module docstring).
Submit: `hpc/batch_gpu/submit_phase8b_bijector_a100` (A100, 39-task array,
`--time=0:30:00` each). **Submit ids: 340576** (first dispatch),
**341845**, **341860**, **341874**, **341875** (reruns) — see "8B — run
history and harvest" below.

**Question.** Does stepping in `log(lambda)` for the free AdaptSplit
regularization coefficients (`log_reg`: `af.BijectorPerPath` restricted to
`"regularization."` paths backed by a `LogUniformPrior`) move the NaN-wall
position, speed up free-regularization convergence, or reduce time spent at
high coefficients — without moving the physical objective (a category-1
reparameterization; see `autofit.non_linear.bijector`'s equivalence
argument)?

**Arms (39 tasks)**:

| cell | log_det_method | bijector | seeds | n |
|---|---|---|---|---:|
| `delaunay_adapt_split` (the NaN wall — Phase 8A/CP-4) | cholesky, slogdet | none, log_reg | 0-4 | 20 |
| `knn` (finite over-regularized floor; the cell 8B's text names) | auto (W8-resolved) | none, log_reg | 0-4 | 10 |
| `knn` (secondary arm) | auto | logit | 0-4 | 5 |
| `mge` (F4 control — no regularization coefficients at all) | auto | none, log_reg | 0-1 | 4 |

Every arm: `multi_start_prodigy` (fixed-step), `n_starts=16`, `n_steps=3000`
(#117-validated pixelized budget), `batch_size=4`, `clipper=prior_box`,
`scaler=none`, `record_lane_nan_history=True`; `knn`/`delaunay_adapt_split`
also trace the two regularization coefficients
(`SEARCHES_TRACE_PARAMS`).

**Pre-registered falsification** (any two of F1-F4 -> 8B falsified; F5 halts
and is scored first — a trip means the bijector changed the physical
objective, a bug, not a science finding):

- **F1** — median first value-NaN step under `log_reg` not earlier than
  `none` on `delaunay_adapt_split`, OR value-NaN lane-steps do not fall
  below 50% of `none`'s.
- **F2** — steps-to-within-10-nats of a reference log-posterior not reduced
  >= 2x at matched seeds on `knn`. Framed against the historical
  free-vs-fixed-regularization convergence figures cited in
  `PROGRAMME.md:579` (**2,200 steps free vs 98 steps fixed**) — but see the
  driver's module docstring: this campaign has no dedicated fixed-reg
  control arm, so the actual reference used is the max `none`-arm
  log-posterior per (cell, log_det_method) group, and the artifact records
  the resolved value rather than replaying the bare 2,200/98 figures
  literally. **This is a documented deviation from a literal reading of the
  pre-registration**, flagged here for a human to confirm or override before
  the verdict is treated as final.
- **F3** — `log_reg` lanes spend >= the same fraction of steps at
  lambda > 1e4 as `none`, on either wall cell.
- **F4** — the `mge` control differs by any bit between `none`/`log_reg`
  (it must not — MGE carries zero `LogUniformPrior` "regularization." paths,
  so `log_reg` resolves to an empty `BijectorPerPath` map), OR the `knn`
  `logit` arm reproduces a pinned-lane-to-infinity pathology.
- **F5** — figure of merit at the shared initial broad-start draw (step-0
  global-best fom) differs by more than 1e-9 relative between arms at a
  matched seed.

**Verdict state: NONE.** The delaunay arms are still running and F2's
reference deviation is unruled; `--stage score` must not be run on the
current data (see the scorer note below). The run history, the arms that did
land, and the per-criterion state are recorded in the next section; a
verdict table replaces this line only when the pending arms finish and the
F2 ruling exists.

**Known gap, recorded rather than hidden**: the driver cannot recover
Prodigy's internal step-scale estimate ("final `d`",
`optax.contrib.ProdigyState.estim_lr`) from the standard results JSON
pipeline — `search_internal["opt_state"]` is not serialized by
`_per_lane.per_lane_block` (an unbounded, non-JSON-safe JAX pytree), and each
arm runs in its own subprocess (matching the real SLURM array), so there is
no in-process handle either. Every scored row carries `final_d: null` with
this note rather than a silently-dropped field.

## 8B — run history and harvest (2026-08-27): one bug, six lost arms, all recovered

**No verdict is emitted here.** The delaunay arms are still running, F2's
reference deviation is unruled, and the scorer was returning spurious
defaults on missing data (below).

### Run history

| submit | tasks | outcome |
|---|---:|---|
| **340576** | 39 | first dispatch — **35 of 39 arms lost**, run at ~12 % of the wall budget a 3000-step pixelized arm needs |
| **341845** | 15 | rerun; the knn arms that landed come from here |
| **341860** | 14 | 13 of 14 tasks ended `PREFLIGHT: giving up after 12 requeues` on a MIG-mode A100 — PR#181's guard fires correctly, its requeue cap is too low |
| **341874** | 13 | knn rerun, in flight at harvest |
| **341875** | 20 | delaunay rerun, in flight at harvest |

Across the campaign **45 of 62 tasks never produced a step**: 31 starved on
the MIG-mode A100 and 14 earlier ones died with `CUDA_ERROR_NO_DEVICE`.
Raising the requeue cap to ~60 is PR#181's open follow-up.

### Crash root cause — out-of-unit-disk `ell_comps` at results-write

Six of the seven arms that reached their write step died with the same
chained exception: `ModelParameterException: ell_comps must satisfy
e0² + e1² < 1`, magnitudes **1.03–1.414** (1.41421 = the (±1, ±1) box
corner). The chain:

- the `ell_comps` prior is an independent per-component box, so **21.5 % of
  the prior volume is non-physical**;
- `validate_ell_comps` returns silently on JAX tracers, so the jitted
  likelihood is finite and differentiable in the corner and lanes settle
  there (`−0.999998` is `PriorBoxClipper`'s 1e-6 inset — the clipper is
  faithful to a wrong box);
- on completion `Result.instance` materialises through `SamplesSummary`,
  which holds one sample and inherits the raising policy; the recovery
  added by PyAutoFit#1486 applies to `Samples`, not to the path that runs
  (**PyAutoFit#1535**; the same early return is the suspect in the older,
  still-open PyAutoFit#1487, and the #1535 PR fixes both);
- `updater._save_samples` then catches the exception and silently skips
  `samples.csv`, autolens `save_results` catches only `AttributeError`, and
  the process dies before `.completed`.

**Bijector and log-det method are innocent**: `none` and `logit`,
`cholesky` and `slogdet` all appear among the crashes.

### Recovery — offline, zero GPU time

`search_internal.dill` survives because the crash pre-empts its deletion,
and `MultiStartGradient.samples_via_internal_from` rebuilds full `Samples`
from it. The six arms were rebuilt through the driver's own
`collect_metrics` / `per_lane_block` / `_build_summary`, marked
`recovered_offline: true`, and verified two ways: every −½·`best_fom`
matches that arm's final `prodigy step 3000/3000` log line to 4 d.p., and
the knn arm's `target_id` is byte-identical to its successful sibling's.
A bare rerun would short-circuit to 0 steps and crash identically, so the
pending arms are left to run — they will crash the same way and leave a
recoverable dill.

### Per-arm results

| cell · log-det · bijector · seed | best log_post | max logL | alive | constrained | value-NaN steps | \|e\| at best | wall | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| knn · none · s0 | 28873.80 | — | 16/16 | 4/16 | 0 | — | 1.8 h | json |
| knn · none · s2 | 28863.84 | — | 16/16 | 4/16 | 0 | — | 1.8 h | json |
| knn · log_reg · s3 | 30559.28 | 30557.03 | 16/16 | 9/16 | 0 | — | 1.8 h | json |
| knn · logit · s1 | 28672.21 | 28683.33 | 16/16 | 10/16 | 0 | 1.414 | 1.8 h | recovered |
| delaunay · slogdet · none · s0 | −146872.80 | −146802.60 | 16/16 | 5/16 | 0 | 1.078 | 1.8 h | recovered |
| delaunay · cholesky · none · s1 | 20395.84 | 22945.06 | 13/16 | 3/16 | 9,226 | 1.414 | 3.6 h | recovered |
| delaunay · slogdet · log_reg · s0 | 28004.15 | 28127.40 | 13/16 | 7/16 | 6,381 | 1.228 | 3.7 h | recovered |
| delaunay · slogdet · log_reg · s1 | 2.1e53 (diverged; 2nd lane 29871) | — | 12/16 | 9/16 | 7,105 | 1.414 | 3.1 h | recovered |
| delaunay · cholesky · log_reg · s2 | 29894.80 | 29977.66 | 7/16 | 6/16 | 26,690 | 1.032 | 4.0 h | recovered |

Still running at harvest: delaunay `cholesky·log_reg·s3` (30117 @ step
1600), `slogdet·none·s1` (28881 @ 2660), `slogdet·none·s2` (−132469, frozen
since step 10, @ 920), `slogdet·log_reg·s3` (30286 @ 2450). The four MGE F4
controls landed earlier (no logs in this harvest).

### Signal

- **Every `log_reg` arm on both wall cells clears +28,000 by step
  100–500**; no `none` arm on delaunay reaches a positive log-posterior
  before step ~2000, and `slogdet·none·s0` never does.
- On knn: `log_reg` (30559) >> `none` (28874 / 28864) > `logit` (28672).
- Substantive F2 material on knn: steps-to-(reference − 10 nats) =
  **2631 / 2633 (`none`) vs 263 (`log_reg`)** — 10× against a >=2× bar, but
  **with no matched seed**, so it is not yet a scored pass.
- A preliminary memory note ("none at −154k/−137k, log_reg/logit at
  +20.8k/+28.0k") is **mis-attributed** and should not be cited: −154k is a
  step-100 waypoint, −137k matches nothing in the harvest, and +20.4k is a
  `none` arm.

### Falsification criteria — partial, no verdict

| criterion | state |
|---|---|
| **F5** (objective inert under the bijector) | **CLEAN.** Step-0 global-best fom bit-identical across bijectors on the MGE control: 423546.4213174847 / 380535.00536054926. The bijector provably leaves the physical objective alone. |
| **F4** (mge control differs by any bit) | **AMENDED, informational.** It trips as written (16/16 lanes differ in final params), but `best_fom` is bit-identical on the winning lane and maxL agrees to 1e-14. Byte-identity is unachievable for a reparameterised 3000-step optimizer, and F5 already carries the inertness proof; F4 is restated as "`best_fom` and max log-likelihood equivalent within fp64 on the winning lane" and is **not** counted as a trip. |
| **F1** (NaN-wall position on delaunay) | **Scorable only now** — the driver never wrote the delaunay JSONs; the recovered set supplies them. Not scored here. |
| **F2** (steps-to-reference on knn) | **Partially scorable.** 10× improvement, but unmatched seeds, and the reference itself is a **documented deviation** (max `none`-arm log-posterior per group, not a fixed-regularization control) that **needs a human ruling** before any verdict — recorded as owed in `../DECISIONS.md` 2026-08-27. |
| **F3** (fraction of steps at λ > 1e4) | **Not falsified** (knn only, n=1): 12.5 % / 6.25 % under `none` (whole lanes parked for 3000 steps) vs 0 % under `log_reg`. |

### Scorer defect (fixed in this PR)

`bijector_ab.py` returned **spurious verdicts on missing data**: `score_f1`
gave a PASS (`bool(None) or bool(None)`) and `score_f2` a FAIL
(`falsified = median_ratio is None or …`). Combined with F4's byte-identity
trip, `--stage verdict` on today's data would have emitted
"falsified_criteria_count=2 → close, no rescoping to logit" — an
artefact-driven false close. Both now return **UNSCORABLE** on missing
inputs, and F4 carries the fp64-equivalence wording. **Do not run
`--stage verdict` until the pending arms land and the F2 reference is
ruled on.**

### Scorer diagnostic readout after the fix (2026-08-27, 13 rows, no verdict)

`score_rows` **halts at F5** on `delaunay_adapt_split[slogdet]` seed 0: step-0 global-best fom
357347.020 (`none`) vs 357343.242 (`log_reg`), rel 1.06e-5. F5 is therefore clean on the MGE
control only; on the slogdet delaunay cell the two arms do not start from the same objective and
the halt is the correct response. Scored individually with F5 bypassed (diagnostic only):
F1[cholesky] falsified · F1[slogdet] UNSCORABLE · F2 UNSCORABLE (no matched knn seed) · F3
falsified · F4 falsified (seed 0 `best_fom`/maxL agree to 9.8e-15; seed 1 disagrees at 1.7e-2).
This machine reading disagrees with the hand reading in "Falsification criteria" above (F3 not
falsified; F1 pending) and is recorded, not adjudicated — the verdict stage runs once every
341875 arm has landed and the F2 reference deviation has a human ruling.

### Cross-references

The out-of-disk `ell_comps` population behind the crash is the same
mechanism Phase 4 Stage 3 measures as the PositionsLH degradation channel
(`../phase_04_positions/RESULTS.md` "Stage 3") and Phase 3 records as a
second reason p̂ is a lower bound
(`../phase_03_prodigy_reliability/RESULTS.md` "Non-physical ellipticity
lanes"). The library follow-up is PROGRAMME §9b W10.
