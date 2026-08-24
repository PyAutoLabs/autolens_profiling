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
- `coefficient_min = −5.2` in the A100 marginal band is the driver
  reporting the log-coefficient axis for transect draws; not a negative
  regularization.
