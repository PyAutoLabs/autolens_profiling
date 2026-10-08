# Linear-solver accuracy study — ledger (phases 1–3a, 2026-09 – 2026-10)

Campaign page: `wiki/campaigns/linear_solver_accuracy.md`. Package: `scripts/lens/solver/`.

## Pre-registered decision rule

Pre-registered decision rule — linear-solver accuracy study, phase 1 (written 2026-09-30, after the
8-system `slam_fixture_571` smoke run and BEFORE the deciding run on `slam48_hst` + `slam_spread_hst` +
`euclid_vis_lp`; committed at autolens_profiling commit `327f571`).

A candidate is **admissible** iff, on every system of every corpus group (CPU fp64, released library
2026.8.17.1 source checkouts as recorded in the summary provenance):

1. **Converged flag.** `converged == 1` on 100 % of systems, all 48 `slam48_hst` points included (the
   PyAutoArray#571 constraint). A candidate whose iterate is accurate but whose flag says unconverged is
   inadmissible: the flag is what production trusts (the fixture smoke run showed jaxnnls's own tolerance
   on the raw system is accurate to 2e-11 yet hits the 50-iteration cap on 8/8 systems — recorded as a
   caveat, and as evidence for a solution-based criterion in phase 2, not as a pass).
2. **Solution accuracy vs fnnls.** worst `amp_rel_max_sig` (columns with x_ref > 1e-6 · max x_ref) ≤ 1e-3
   AND worst |`flux_rel_source`| ≤ 1e-4 on every system. `amp_rel_max` over all x_ref > 0 columns is
   recorded but not gated: on these near-singular systems (cond(Q) ~ 1e11) it reaches 1e7 for iterates whose
   source flux agrees to 1e-5, because it is dominated by reference columns at the 1e-9 level (fixture smoke
   run, 2026-09-30).
3. **Euclid witness proxy.** On `euclid_vis_lp`, |`flux_rel_source`| ≤ 1e-4 — the eager NumPy path the
   euclid test compares against solves with fnnls, so this is the in-harness form of the rel-1e-3 latent
   check with 10x headroom.
4. **Optimality.** scaled KKT residual ≤ 10x `pdip_jacobi`'s on the systems where jacobi converges.

Among admissible candidates choose the lowest **median iteration count**; CPU wall is the tie-break; ties
go to the smallest library change (a constant change beats a new code path). If no candidate is
admissible the verdict is "no drop-in candidate; phase 2 needs a solution-based stop", and the study
names which metric the released stop is blind to.

A human may re-base this rule; the campaign page records the re-basing and its reason.

## Environment

From the result JSONs' `device.provenance` block:

- **Host:** `DESKTOP-H143S82` — the laptop (Intel i9-10885H, 8 cores, WSL2 kernel
  5.10.16.3-microsoft-standard-WSL2), CPU backend, fp64 (`x64: true`), 8 BLAS/OMP threads. No RAL
  job; `slurm` is null. Load average 1.9–2.1 at import and at write.
- **Libraries:** source checkouts stamped `2026.8.17.1` (the source-checkout version stamp, not a
  PyPI install): PyAutoArray `d4298445` (main after #591), PyAutoGalaxy `4900e680`, PyAutoLens
  `092897e4`, PyAutoFit `3fa8cf7f`, PyAutoNerves `1ec1c829`.
- **Dependencies:** jax / jaxlib 0.10.2, numpy 2.2.6, scipy 1.17.1, numba 0.62.1, nufftax 0.6.1.
- **Reference:** `autoarray.util.fnnls.fnnls_cholesky` from the dense-sign start, stored in the
  corpus when each group was added (never re-derived by a study).

## Corpus

| Group | Systems | n | cond(Q) | max abs(q) | Source |
|---|---|---|---|---|---|
| `slam_fixture_571` | 8 | 60 | 9.69e10 – 9.73e10 | 2.16e6 | the PyAutoArray#571 fixture, `scripts/imaging/hazards/mge_nnls_capture.py` |
| `slam48_hst` | 48 | 60 | 9.64e10 – 9.75e10 | 2.16e6 | all 48 #571 SLaM `source_lp[1]` vectors, `capture.py --source slam48` (profiling `373da5b`) |
| `slam_spread_hst` | 24 | 60 | 1.08e10 – 1.08e12 | 2.40e5 – 2.40e7 | 6 of those vectors at sigma_min x0.5 / x2 and noise x0.3 / x3, `capture.py --source slam_spread` |
| `euclid_vis_lp` | 1 | 60 | 5.56e11 | 8.49e6 | the euclid latent jit test's system, `capture.py --source euclid_vis_lp` (euclid pipeline `26e4385b`) |

81 systems, every one n = 60 with 20 source columns (the source galaxy's 20 linear Gaussians).

## Deciding run

Run at 2026-09-30 10:33 UTC on profiling `8adb629` — after the rule commit `327f571`; the
`scripts/` tree is identical between the two commits (`git diff 327f571 8adb629 -- scripts/` is
empty). Both cells over all four groups (`accuracy.py`, `early_stopping.py`, defaults).

**Reproduction.** Every per-row field except `wall_ms` (729 accuracy rows, 648 early-stopping
rows) and every aggregate except `median_wall_ms` is **bit-for-bit identical** to the 11:27 BST
run made before the rule was written (captured 10:27 UTC, profiling `373da5b`, same library
revisions). The committed `accuracy_summary_all` / `early_stopping_summary_all` JSONs are a third
run (10:35 UTC) made after the post-hoc metric `flux_inactive_rel` was added to the row schema;
their pre-registered fields are again bit-for-bit identical to the deciding run. The solves are
deterministic on this host; only wall times move.

| Candidate | Unconverged | Worst amp_rel_max_sig | Worst abs(flux_rel_source) | Worst objective gap | Worst KKT | Median iters | Max iters | Median wall ms |
|---|---|---|---|---|---|---|---|---|
| `fnnls` (reference) | 0/81 | 0 | 0 | 0 | 8.6e-16 | 9 | 20 | 1.37 |
| `pdip_jacobi` | 29/81 | 1.0e68 | 3.4e70 | 1.6e134 | 8.3e61 | 19 | 50 | 1.31 |
| `pdip_raw` (released) | 0/81 | 27.0 | 16.4 | 2.6e-12 | 2.7e-14 | 18 | 24 | 1.36 |
| `pdip_raw_tol_1e-1` | 0/81 | 71.5 | 43.0 | 1.7e-11 | 1.8e-13 | 16 | 23 | 1.20 |
| `pdip_raw_tol_1e-2` | 0/81 | 27.0 | 16.4 | 2.6e-12 | 2.7e-14 | 18 | 24 | 1.24 |
| `pdip_raw_tol_1e-3` | 0/81 | 10.0 | 6.15 | 3.9e-13 | 3.9e-15 | 19 | 25 | 1.42 |
| `pdip_raw_tol_jaxnnls` | 75/81 | 0.219 | 7.4e-5 | 8.2e-16 | 2.4e-16 | 50 | 50 | 3.05 |
| `pdip_raw_polish` | 0/81 | 0.219 | 4.96e-2 | 8.2e-16 | 3.2e-16 | 23 | 30 | 1.72 |
| `certified` | 48/81 | 119 | 7.5e-2 | 2.6e-4 | 1.8e-2 | 16 | 16 | 1.12 |

(`pdip_raw_tol_1e-2` reproduces `pdip_raw` exactly through `solve_nnls` directly, as designed.
(phase 1; no longer true after #595 — see [Phase 2](#phase-2-2026-09-30--library-fix-shipped-pyautoarray595):
`pdip_raw` is now polished, the `tol_1e-2` row is not, and the row now reproduces the *pre-fix*
`pdip_raw`.))

## Admissibility (the rule as written)

Criterion 4 compares each candidate's scaled KKT residual with `pdip_jacobi`'s system by system,
on the 52 systems where `pdip_jacobi` reports converged (1 fixture, 33 `slam48_hst`, 17
`slam_spread_hst`, the euclid system). `fnnls` is the reference, not a candidate; its row is shown
as a check on the criteria themselves.

| Candidate | 1. converged 100 % | 2. sig ≤ 1e-3 and abs(flux_rel_source) ≤ 1e-4 | 3. euclid abs(flux_rel_source) ≤ 1e-4 | 4. KKT ≤ 10x jacobi | Admissible |
|---|---|---|---|---|---|
| `pdip_jacobi` | FAIL 29/81 unconverged | FAIL sig 1.0e68, flux 3.4e70 | FAIL 6.9e-2 | pass (it is the yardstick) | no |
| `pdip_raw` (released) | pass | FAIL sig 27.0, flux 16.4 | FAIL 16.4 | FAIL 8/52 (worst 971x, euclid) | no |
| `pdip_raw_tol_1e-1` | pass | FAIL sig 71.5, flux 43.0 | FAIL 43.0 | FAIL 44/52 (worst 6.6e3x) | no |
| `pdip_raw_tol_1e-2` | pass | FAIL sig 27.0, flux 16.4 | FAIL 16.4 | FAIL 8/52 (worst 971x) | no |
| `pdip_raw_tol_1e-3` | pass | FAIL sig 10.0, flux 6.15 | FAIL 6.15 | FAIL 1/52 (143x, euclid) | no |
| `pdip_raw_tol_jaxnnls` | FAIL 75/81 unconverged | FAIL sig 0.219 (flux 7.4e-5 passes) | pass 7.5e-10 | FAIL 1/52 (32x) | no |
| `pdip_raw_polish` | pass | FAIL sig 0.219, flux 4.96e-2 | FAIL 4.96e-2 | FAIL 5/52 (worst 32x) | no |
| `certified` | FAIL 48/81 uncertified | FAIL sig 119, flux 7.5e-2 | pass 6.9e-10 | FAIL 37/52 (worst 3.1e14x) | no |
| *`fnnls` (reference, check)* | *pass* | *pass (0, 0)* | *pass (0)* | *FAIL 9/52 (worst 64x)* | — |

Notes on applying the rule, recorded rather than acted on (the rule is not re-based):

- **Criterion 4 is floor-dominated.** `pdip_jacobi`'s KKT residual on its converged systems is
  3.4e-18 – 2.4e-16, so "10x" is a comparison at the floating-point floor: the reference `fnnls`
  itself fails it on 9/52 systems with residuals ≤ 4.3e-16. It decides nothing here — every
  candidate already fails criterion 1 or 2 — but a phase-2 rule should give it an absolute floor.
- **One system caps criterion 2 for the two accurate candidates.** Both `pdip_raw_tol_jaxnnls` and
  `pdip_raw_polish` reach `amp_rel_max_sig` 0.219 on `slam_spread_hst/noise_x3_v32` with the same
  value to 13 digits, an objective gap of 0 / -1.4e-16 against fnnls and KKT ≤ 1.2e-16. Two
  independent tight solves agreeing with each other and matching the reference objective exactly
  point at a flat direction the 1e-6 · max significance floor does not exclude, not a solver
  error; their source-flux error there is 1.5e-5. They would fail criterion 2 or 3 elsewhere
  regardless (jaxnnls on criterion 1, the polish on flux 4.96e-2 at euclid).

## Verdict

**No drop-in candidate; phase 2 needs a solution-based stop.** No candidate is admissible under
the pre-registered rule. The released raw stop reports `converged` on 81/81 systems at an
objective gap ≤ 2.6e-12 and a scaled KKT residual ≤ 2.7e-14, yet on the euclid system it leaves
**11.5 % of the reference's total amplitude on columns the reference holds at zero**
(`flux_inactive_rel` 0.115; source-column amplitude-sum error `flux_rel_source` +16.4, i.e. +1640 %; euclid `total_source_flux` +5.76 %).
The metric the released stop is blind to is the **flux on reference-inactive columns**: its
tolerance is a threshold on the PDIP KKT residual (an objective-side test), and on these near-singular systems
(cond(Q) 1e10 – 1e12) mass can sit on an inactive column at an objective cost below 1e-12.
`amp_rel_max` and `amp_rel_max_sig` cannot see it either, because they score only reference-active
columns; `flux_inactive_rel` (post-hoc, below) is the metric that does.

Every tolerance change that keeps the flag truthful keeps the bias (`pdip_raw_tol_1e-3`: `flux_rel_source` 6.15 on
euclid), and every one tight enough to remove it runs to the cap (`pdip_raw_tol_jaxnnls`: 75/81).

## Early stopping (released raw PDIP vs iteration cap)

| Cap | Converged | Worst abs(flux_inactive_rel) | Median flux_inactive_rel | Worst abs(flux_rel_source) | Worst amp_rel_max_sig | Worst objective gap | Worst KKT |
|---|---|---|---|---|---|---|---|
| 8 | 0/81 | 447 | 9.3e-2 | 6.2e4 | 3.4e4 | 1.3e-3 | 3.2e-3 |
| 12 | 1/81 | 223 | 6.4e-3 | 3.1e4 | 9.0e3 | 3.5e-4 | 5.7e-4 |
| 16 | 15/81 | 190 | 1.2e-4 | 2.7e4 | 1.5e3 | 2.9e-5 | 2.5e-5 |
| 24 – 200 | 81/81 | 0.115 | 3.2e-5 | 16.4 | 27.0 | 2.6e-12 | 2.7e-14 |

The objective criterion stops every system between iteration 15 and 24 (median 18); from cap 24
up every row is identical, because the stop has fired. Where it fires, the objective is converged
to ≤ 2.6e-12 but the *solution* is not: the median system still carries 3.2e-5 of its mass on
inactive columns and the euclid system 0.115. The solution does converge within the 50-iteration
budget when the tolerance does not stop it — at cap 50 with a tolerance the loop never meets
(`pdip_raw_tol_1e-6`) the worst `flux_inactive_rel` is 8.1e-8 and the worst source-flux error
7.5e-10 — so the budget is not the constraint; the stopping test is. Caps below 24 are unsafe on
every metric (the cap-16 worst source-flux error is 2.7e4).

## Post-hoc exploration

**Post-hoc, not pre-registered; inputs to the phase-2 fix, not a verdict.** Added after the
deciding run: seven candidates (`_solvers.POSTHOC_CANDIDATES`, run by `accuracy.py --posthoc` into
their own artefact) and two metrics (`flux_inactive_rel`, `active_set_mismatch`, now in every
row and aggregate).

| Candidate | Unconverged | Worst abs(flux_inactive_rel) | Worst abs(flux_rel_source) | Worst amp_rel_max_sig | Median iters | Max iters | Median wall ms |
|---|---|---|---|---|---|---|---|
| `pdip_raw` (released, anchor) | 0/81 | 0.115 | 16.4 | 27.0 | 18 | 24 | 0.99 |
| `pdip_raw_polish` (anchor) | 0/81 | 3.3e-4 | 4.96e-2 | 0.219 | 23 | 30 | 1.20 |
| `pdip_raw_polish_cap20` | 0/81 | 3.3e-4 | 4.96e-2 | 0.219 | 23 | 30 | 1.39 |
| `pdip_raw_polish_cap50` | 0/81 | 3.3e-4 | 4.96e-2 | 0.219 | 23 | 30 | 1.26 |
| `pdip_raw_tol_jaxnnls` (anchor) | 75/81 | 6.2e-7 | 7.4e-5 | 0.219 | 50 | 50 | 2.20 |
| `pdip_raw_tol_jaxnnls_cap100` | 74/81 | 6.2e-7 | 7.4e-5 | 0.219 | 100 | 100 | 5.01 |
| `pdip_raw_tol_jaxnnls_cap200` | 74/81 | 6.2e-7 | 7.4e-5 | 0.219 | 200 | 200 | 7.54 |
| `pdip_raw_tol_1e-4` | 0/81 | 1.6e-2 | 2.30 | 3.52 | 20 | 26 | 1.18 |
| `pdip_raw_tol_1e-5` | 0/81 | 2.4e-3 | 0.337 | 0.227 | 21 | 28 | 1.19 |
| `pdip_raw_tol_1e-6` | 78/81 | 8.1e-8 | 7.5e-10 | 1.3e-10 | 50 | 50 | 2.36 |

What it says:

- **A larger polish cap buys nothing.** The polish converges within its 10 iterations and is kept
  on 81/81 systems, so caps 20 and 50 return the same iterates. Its limit is its own stop (jaxnnls's
  absolute KKT tolerance on the Jacobi system), which on euclid is met with 3.3e-4 of the mass
  still on inactive columns.
- **A larger cap does not rescue the tight tolerance.** jaxnnls's absolute tolerance on the raw
  system is unreachable in floating point on 74/81 systems even at 200 iterations, at 5–7.5x the
  released wall; the accuracy it has at cap 50 does not improve after it. On euclid alone it
  converges at 51 iterations.
- **The replace-the-constant family has no sweet spot.** 1e-5 keeps the flag truthful (0/81,
  median 21 iterations) with source-flux error ≤ 8.5e-4 on the SLaM groups but 0.337 on euclid; 1e-6
  is accurate everywhere but runs to the cap on 78/81. The flag and the accuracy part between 1e-5
  and 1e-6, and euclid needs more than 1e-5.
- **`active_set_mismatch` is not useful as defined.** An interior-point iterate keeps every column
  strictly positive, so the count (up to 26 for the released solve, 4 even for jaxnnls's tight
  iterate) mostly counts interior residue just above the 1e-6 · max threshold.
  `flux_inactive_rel` separates the candidates cleanly and is the metric to gate on.

### Euclid latent by candidate

`total_source_flux` of the euclid system from each candidate's reconstruction, computed by the
euclid pipeline's own latent code (`scripts/lens/solver/euclid_latent.py`: the reconstruction is
injected through `inversion_util.reconstruction_positive_only_from` into the eager NumPy fit, and
the value read from `LatentEuclid._source_flux_latents_on_uniform_grid`). **Validation passed:**
injecting `x_ref` gives 3.3198794 against the stored eager 3.3198795 (rel -3.3e-8) and injecting
`pdip_raw`'s `x` gives 3.5110933 against the stored jit 3.5110934 (rel -2.0e-8), both inside the
1e-6 gate. The -3e-8 offset is the eager fit solving its own NumPy-built system rather than the
captured JAX one.

| Candidate | total_source_flux | rel vs eager | Within the test's 1e-3 | Converged | Iterations |
|---|---|---|---|---|---|
| `fnnls` | 3.3198794 | -3.3e-8 | yes | yes | 8 |
| `pdip_jacobi` | 3.3202250 | +1.0e-4 | yes | yes | 19 |
| `pdip_raw` (released) | 3.5110933 | +5.76e-2 | **no** | yes | 24 |
| `pdip_raw_tol_1e-1` | 3.8766845 | +1.68e-1 | no | yes | 23 |
| `pdip_raw_tol_1e-3` | 3.3756725 | +1.68e-2 | no | yes | 25 |
| `pdip_raw_tol_jaxnnls` | 3.3198794 | -3.3e-8 | yes | **no** | 50 |
| `pdip_raw_polish` | 3.3201275 | +7.5e-5 | yes | yes | 30 |
| `certified` | 3.3198794 | -3.3e-8 | yes | yes | 8 |
| `pdip_raw_polish_cap20` / `_cap50` (post-hoc) | 3.3201275 | +7.5e-5 | yes | yes | 30 |
| `pdip_raw_tol_jaxnnls_cap100` / `_cap200` (post-hoc) | 3.3198794 | -3.3e-8 | yes | yes | 51 |
| `pdip_raw_tol_1e-4` (post-hoc) | 3.3333942 | +4.1e-3 | no | yes | 26 |
| `pdip_raw_tol_1e-5` (post-hoc) | 3.3215684 | +5.1e-4 | yes | yes | 28 |
| `pdip_raw_tol_1e-6` (post-hoc) | 3.3198794 | -3.3e-8 | yes | no | 50 |

(`pdip_raw_tol_1e-2` is identical to `pdip_raw`. (phase 1; no longer true after #595 — see Phase 2:
it still gives 3.5110933, the pre-fix value.)) Which candidates would turn the euclid test
green, taken on their own: the forward polish (+7.5e-5, 30 iterations), the tolerance at 1e-5
(+5.1e-4, 28), and jaxnnls's tolerance with a cap above 50 (exact, 51). The intensity-sum proxy of
criterion 3 is much stricter than the latent: the polish's source-flux proxy error is 4.96e-2 but
its latent error 7.5e-5, because the spurious amplitude sits on compact Gaussians whose unit
image-plane flux is small.

## What phase 2 should implement

Ranked by the evidence above. Iteration counts are medians / maxima over the 81 systems; walls
are laptop-CPU medians and only indicative.

1. **A solution-based stop (or finishing step)** — the only option the evidence supports on every
   system. The accurate iterate exists within the current 50-iteration budget (`pdip_raw_tol_1e-6`
   at cap 50: `flux_inactive_rel` ≤ 8.1e-8), and the certified active-set solve reaches the exact
   euclid answer in 8 passes, so what is missing is a *test* that recognises it: identify the
   active set from the iterate (complementarity, `x_j` against `z_j`), solve the equality system
   on the passive set by Cholesky, and accept when the zeros are dual-feasible and the passive
   solution positive (the certificate the certified solver already uses), otherwise keep
   iterating. Iteration cost **not measured** here: phase 2 must measure it (the prototype should
   log PDIP iterations until the certificate first holds, on this corpus).
2. **Forward polish** (`pdip_raw_polish`, PyAutoArray#573's backward rule on the forward value) —
   the cheapest measured candidate that turns the euclid test green: 0/81 unconverged, median 23
   iterations (+5 over the released 18), maximum 30, median wall 1.2–1.7 ms; euclid latent +7.5e-5.
   It fails the pre-registered accuracy gates (source-flux proxy 4.96e-2 on euclid,
   `flux_inactive_rel` 3.3e-4), and a larger polish cap does not help. A stop-gap, not a fix.
3. **Tight tolerance with a larger cap** — the most expensive and it does not repair the flag:
   jaxnnls's absolute tolerance still runs to the cap on 74/81 systems at 100 and 200 iterations
   (walls 5.0 / 7.5 ms). The data-scaled tolerance at 1e-5 is cheaper (median 21, maximum 28) and
   passes the euclid test at +5.1e-4, but leaves 0.337 on the source-flux proxy and 2.4e-3 on
   inactive columns there: it moves the edge, it does not remove it.

**Recommendation (the author's, not the rule's):** build option 1 in PyAutoArray and gate it with
a phase-2 rule on `flux_inactive_rel` plus the euclid latent directly, with an absolute floor on
the KKT criterion. If a stop-gap is wanted for the euclid test before that ships, option 2 is the
one the numbers support.

## Caveats

- **Wall times are indicative only.** A laptop under WSL2 at load ~2, five warm repeats per row;
  the same candidate's median wall moves 20–50 % between runs (`pdip_raw` 0.99–1.36 ms). The
  iteration counts are exact and deterministic; rank by them.
- **The noise variants are not pure rescalings.** `slam_spread_hst`'s noise x0.3 / x3 systems scale
  `Q` and `q`, but the 1e-3 diagonal add is unscaled, so they are genuinely different systems (cond
  1.08e10 – 1.08e12), not the same system in other units.
- **`amp_rel_max` is dominated by 1e-9 reference columns** (worst 6.2e10 for the released solve on
  systems whose significant-column error is 27); it is recorded, never gated, as the rule says.
- **The significance floor admits a flat direction** on `slam_spread_hst/noise_x3_v32` (see the
  admissibility notes).
- **The corpus has one euclid system.** The euclid findings rest on `euclid_vis_lp_k0` alone; the
  SLaM groups show the same mechanism at 1e-3 – 1e-2 (released source-flux error ≤ 4.0e-2), not the
  euclid magnitude.
- **The flux metrics are amplitude sums**, a proxy; only the euclid latent table is a photometric
  quantity.

## Artefacts

- Pre-registered cells: [`accuracy_summary_all_v2026.8.17.1.json`](../lens/solver/accuracy_summary_all_v2026.8.17.1.json)
  ([png](../lens/solver/accuracy_summary_all_v2026.8.17.1.png)),
  [`early_stopping_summary_all_v2026.8.17.1.json`](../lens/solver/early_stopping_summary_all_v2026.8.17.1.json)
  ([png](../lens/solver/early_stopping_summary_all_v2026.8.17.1.png)).
- Post-hoc: [`accuracy_posthoc_summary_all_v2026.8.17.1.json`](../lens/solver/accuracy_posthoc_summary_all_v2026.8.17.1.json)
  ([png](../lens/solver/accuracy_posthoc_summary_all_v2026.8.17.1.png)),
  [`euclid_latent_by_candidate_v2026.8.17.1.json`](../lens/solver/euclid_latent_by_candidate_v2026.8.17.1.json).
- Fixture smoke baseline: [`accuracy_summary_slam_fixture_571_v2026.8.17.1.json`](../lens/solver/accuracy_summary_slam_fixture_571_v2026.8.17.1.json),
  [`early_stopping_summary_slam_fixture_571_v2026.8.17.1.json`](../lens/solver/early_stopping_summary_slam_fixture_571_v2026.8.17.1.json).
- Phase 3a (released 2026.10.7.1, tag checkouts): CPU [`accuracy_summary_all_v2026.10.7.1.json`](../lens/solver/accuracy_summary_all_v2026.10.7.1.json)
  ([png](../lens/solver/accuracy_summary_all_v2026.10.7.1.png)), A100 [`accuracy_summary_all_gpu_v2026.10.7.1.json`](../lens/solver/accuracy_summary_all_gpu_v2026.10.7.1.json)
  ([png](../lens/solver/accuracy_summary_all_gpu_v2026.10.7.1.png)).
- Corpus: [`corpus/manifest.json`](../lens/solver/corpus/manifest.json).
- Package and how to re-run: [`scripts/lens/solver/README.md`](../../scripts/lens/solver/README.md).
- Campaign page: [`wiki/campaigns/linear_solver_accuracy.md`](../../wiki/campaigns/linear_solver_accuracy.md).

## Phase 2 (2026-09-30) — library fix shipped: PyAutoArray#595

**What changed in the library.** PyAutoArray#573 had added a *backward-pass* polish to the raw
mode: after the raw forward PDIP stop, at most `RAW_POLISH_MAX_ITER` (10) PDIP iterations on the
Jacobi-scaled system `(Q_pc, q_pc)` at jaxnnls's tight tolerance, warm-started from the mapped
iterate `(y, s/D, z·D)` and kept only if it converges to a finite, strictly interior point — but
used that point only for the gradient. The forward value stayed the raw stop, which is the value
phase 1 found blind to flux on reference-inactive columns. PyAutoArray#595 (issue #594; merged
2026-09-30 as merge `7a89e19a0`: red regression `7e62fa4d`, fix `31b1c2d7`, test-tolerance
follow-up `42c52358`) returns the polished iterate as the forward value too, via one shared
`_raw_forward_polished` that the custom-vjp primal and the forward rule both call, so plain,
jitted and differentiated calls return the same `y`. This is option 2 of "What phase 2 should
implement" (the forward polish), not option 1 (a solution-based stop). `converged` and the
returned iteration count are still the raw forward solve's; the polish is not counted.

**Re-run.** All five cells were re-run on 2026-09-30 16:54–16:57 UTC against the merged library
(PyAutoArray `7a89e19a09760a0daf22f40e8b111b5388767f73`, recorded in every JSON's
`device.provenance.library_revisions`; PyAutoFit `b13169e2`, PyAutoGalaxy `4c834ced`, PyAutoLens
`efd13c4c`, PyAutoNerves `1ec1c829`; profiling `44381ec`; same host, jax 0.10.2). Walls: `accuracy.py`
16 s, `--groups slam_fixture_571` 12 s, `--posthoc` 42 s, `early_stopping.py` 25 s,
`--groups slam_fixture_571` 19 s, `euclid_latent.py` 33 s. The numbers reproduce the verifier's
run on the pre-merge branch head to every quoted digit (the only library commit in between,
`42c52358`, is a test-file change). Every candidate other than `pdip_raw` (and the
`pdip_raw_cap_*` sweep, which calls the same library entry) is unchanged except wall time.

**`pdip_raw` before and after** (81 systems, CPU fp64):

| | Unconverged | Median / max iters | Worst abs(flux_inactive_rel) | Median flux_inactive_rel | Worst abs(flux_rel_all) | Worst abs(flux_rel_source) | Worst objective gap | Worst amp_rel_max_sig | Worst KKT |
|---|---|---|---|---|---|---|---|---|---|
| pre-#595 (phase 1, `d4298445`) | 0/81 | 18 / 24 | 0.115 | 3.2e-5 | 0.115 | 16.4 | 2.6e-12 | 27.0 | 2.7e-14 |
| post-#595 (`7a89e19a0`) | 0/81 (48/48 `slam48_hst`) | 18 / 24 † | 3.31e-4 | 1.7e-7 | 3.31e-4 | 4.96e-2 (euclid only) | 8.2e-16 | 0.219 | 3.2e-16 |

† Forward iterations only. The polish adds 1–7 more (5 on 46/81 systems, 6 on 21) and is
accepted on 81/81 (`backward_polish_converged` = 1 everywhere); counting them, as the phase-1
`pdip_raw_polish` candidate does, gives median 23 / max 30. Every metric field of every
`pdip_raw` row is **identical** to the phase-1 `pdip_raw_polish` row on the same system (81/81),
so the library now ships exactly the candidate phase 1 measured. On the 8 `slam_fixture_571`
systems the worst source-flux error is 4.9e-5 and the worst significant-column error 5.0e-6. The
0.219 `amp_rel_max_sig` is `slam_spread_hst/noise_x3_v32`, the flat-direction system every
accurate candidate shares (see the admissibility notes); its source-flux error is 1.5e-5.
Median warm wall 1.16 ms (pre-fix 0.99–1.36 ms across runs; indicative only).

**The pre-registered rule on the fixed `pdip_raw`.** Still **not admissible as written**, for the
rule weaknesses phase 1 already recorded; the library shipped on the flux-metric and latent
evidence, not on this rule (which is not re-based here).

| Criterion | Verdict | Why |
|---|---|---|
| 1. converged 100 % | **pass** | 81/81, all 48 `slam48_hst` included |
| 2. sig ≤ 1e-3 and abs(flux_rel_source) ≤ 1e-4 | FAIL on 2 systems | `euclid_vis_lp_k0` flux_rel_source 4.96e-2 (its sig is 5e-13); `slam_spread_hst/noise_x3_v32` sig 0.219 (the flat direction; flux 1.5e-5) |
| 3. euclid abs(flux_rel_source) ≤ 1e-4 | FAIL | 4.96e-2 — but the euclid latent itself is +7.47e-5 (below). The intensity-sum proxy is far stricter than the witness it stands for: euclid's single active source column carries ~0.4 % of the reference flux, so a small spurious amplitude on inactive source columns is a large fraction of the *source* sum, while those compact Gaussians contribute only 7.5e-5 to `total_source_flux` |
| 4. KKT ≤ 10x `pdip_jacobi` | FAIL 5/52 | all at the floating-point floor: residuals 1.1e-16 – 3.2e-16 against jacobi's 3.4e-18 – 2.7e-17 (worst 32x). Pre-fix 8/52, worst 971x |

**Euclid latent by candidate** (`euclid_latent.py`, euclid pipeline `26e4385b`). The validation
leg `pdip_raw -> jit` was re-based from the capture-time released jit 3.511093374 to the jitted
`LatentEuclid.variables` on library main, 3.320127604 (measured 2026-09-30 through the euclid
test's own code; `euclid_latent.py`'s `PDIP_RAW_JIT_EXPECTED`); the manifest keeps the old value as
the capture record. **Validation passed:** `x_ref` -> eager rel -3.26e-8, `pdip_raw` -> main jit
rel -2.0e-8 (the same -2e-8 eager-vs-JAX system offset as phase 1).

| Candidate | total_source_flux | rel vs eager 3.3198795 | Within the test's 1e-3 | Converged | Iterations |
|---|---|---|---|---|---|
| `fnnls` | 3.3198794 | -3.26e-8 | yes | yes | 8 |
| `pdip_raw` (post-#595) | 3.3201275 | +7.47e-5 | **yes** | yes | 24 (forward) |
| `pdip_raw` (pre-#595, phase 1) | 3.5110933 | +5.76e-2 | no | yes | 24 |
| `pdip_raw_polish` | 3.3201275 | +7.47e-5 | yes | yes | 30 |
| `pdip_jacobi` | 3.3202250 | +1.04e-4 | yes | yes | 19 |
| `pdip_raw_tol_1e-2` | 3.5110933 | +5.76e-2 | no | yes | 24 |
| `pdip_raw_tol_jaxnnls` | 3.3198794 | -3.26e-8 | yes | no | 50 |
| `certified` | 3.3198794 | -3.26e-8 | yes | yes | 8 |

The euclid pipeline's `tests/test_compute_latent_variable.py` passes 19/19 on library main
(`JIT_VS_EAGER_REL` 1e-3), measured by the phase-2 verifier; the jitted latent above is +7.47e-5.

**Correction: `pdip_raw_tol_1e-2` no longer reproduces `pdip_raw`.** It calls `solve_nnls`
directly at the released tolerance with no polish, so since #595 it reproduces the *pre-fix*
`pdip_raw` (worst source-flux 16.4, euclid latent +5.76e-2) rather than the current one. The
phase-1 sentences saying otherwise are annotated above; the candidate descriptions in
`_solvers.py` and the package README say so.

**Early stopping after the fix.** The cap sweep calls the library entry, so the polish now
follows every capped solve. From cap 24 up every row equals the fixed `pdip_raw` above (worst
`flux_inactive_rel` 3.31e-4, source-flux 4.96e-2). Below it the converged counts are unchanged
(0 / 1 / 15 of 81 at caps 8 / 12 / 16) and the worst-case errors are unchanged; only the cap-16 median `flux_inactive_rel` moves,
1.2e-4 -> 2.0e-7. Caps below 24 remain unsafe.

**Artefacts were overwritten in place.** The cells stamp `al.__version__`, which is `2026.8.17.1`
on a source checkout, so the re-run rewrote every `results/lens/solver/*_v2026.8.17.1.{json,png}`
(both `all` and `slam_fixture_571` summaries, the post-hoc summary and the euclid latent JSON).
That is deliberate: the artefacts are "current library main" by contract. The pre-fix versions
are at profiling commit **`3ad68af`** (git history), and their numbers are in the phase-1 tables
above.

**What phase 3 inherits.**

- The shipped forward value is the polished iterate: 0/81 unconverged, `flux_inactive_rel` ≤
  3.31e-4, euclid latent +7.47e-5 (inside the pipeline's 1e-3 with ~13x headroom), at 1–7 extra
  PDIP iterations that the reported count hides — a GPU/vmap campaign should time the call, not
  read the counter.
- Option 1 (a solution-based stop / active-set certificate) is still unbuilt; the polish's own
  stop is jaxnnls's absolute KKT tolerance, which on euclid leaves 3.3e-4 on inactive columns.
  Anything that needs better than that on euclid-like systems still needs option 1.
- The pre-registered rule needs re-basing before it can admit anything accurate: criterion 3
  should gate on the latent (or on `flux_inactive_rel`), not the intensity-sum proxy; criterion 4
  needs an absolute floor; criterion 2's significance floor admits the `noise_x3_v32` flat
  direction.
- The euclid cell's validation expectation is pinned to the library-main jit value; a future
  solver change must re-measure and re-base it the same way.

## Phase 3a (2026-10-07) — A100 parity of the corpus on the released 2026.10.7.1

**Question.** Does the released solver on the A100 reproduce the stored CPU fnnls references over
the 81-system corpus, as it does on CPU? Parity only; the GPU / `jit(vmap)` timing cell is phase
3b. Issue autolens_profiling#393.

**Provenance.** Both rows run the five libraries at tag **2026.10.7.1** (contains PyAutoArray#595):
PyAutoNerves `c5ade605`, PyAutoFit `710f4b34`, PyAutoArray `ccddfba6`, PyAutoGalaxy `b4946b8a`,
PyAutoLens `b6bf543c` (full SHAs in each JSON's `device.provenance.library_revisions`).

- **A100:** RAL job **397475** on `euclid-ral-gpu-1` (NVIDIA A100 80GB PCIe, driver 610.57.04),
  `--partition=gpu --gres=gpu:1`, COMPLETED 0:0 in 1:04 (21:36:11 – 21:37:15 BST). The libraries
  are a private clone at `/mnt/ral/jnightin/PyAuto_wt/linear-solver-p3/` (the shared mirror
  `/mnt/ral/jnightin/PyAuto` was not touched; only its venv was reused); the job's import guard
  confirmed all five `auto*` packages imported from there. jax / jaxlib 0.10.2, numpy 2.2.6,
  scipy 1.17.1, numba 0.65.1; fp64 (`JAX_ENABLE_X64=1`, backend `gpu`, `cuda:0`). Submit:
  `hpc/batch_gpu/submit_lens_solver_accuracy_a100_fp64` at profiling `10438ff`. Logs: 0
  tracebacks, 0 `RESOURCE_EXHAUSTED`, no float32 lines; the `.err` holds only four
  `SyntaxWarning`s from PyAutoGalaxy docstrings.
- **CPU:** the laptop (WSL2, i9-10885H), local `git worktree` checkouts of the same five tags,
  `autolens.__file__` confirmed in those checkouts; jax / jaxlib 0.10.2, numba 0.62.1.
- **Artefact names.** A source checkout at a release tag still stamps `al.__version__ =
  2026.8.17.1` (the release workflow writes the real version only into the built package), so
  both runs wrote to a scratch / `output/` directory and the pairs were copied to the tag's name:
  `accuracy_summary_all_v2026.10.7.1` (CPU) and `accuracy_summary_all_gpu_v2026.10.7.1` (A100).
  The `autolens_version` field inside each JSON reads 2026.8.17.1; the SHAs above are the
  provenance.
- **The candidate backends.** `fnnls` is NumPy/numba and runs on the host CPU inside the GPU job;
  every other candidate is a JAX solve on the A100.

**CPU at the tag reproduces phase 2.** Every per-row field except `wall_ms` of the 729-row CPU
run at 2026.10.7.1 is bit-for-bit identical to the phase-2 artefact (library main `7a89e19a0`,
2026-09-30). No solver change has landed between the two, and the table has not drifted.

**Parity table** (81 systems, fp64; scored against the stored CPU fnnls `x_ref`):

| Candidate | Device | Unconverged | Worst flux_inactive_rel | Median flux_inactive_rel | Worst amp_rel_max_sig | Worst abs(flux_rel_source) | Median / max iters |
|---|---|---|---|---|---|---|---|
| `pdip_raw` (released) | CPU | 0/81 | 3.31e-4 | 1.69e-7 | 0.219 | 4.96e-2 | 18 / 24 |
| `pdip_raw` (released) | A100 | 0/81 | 3.31e-4 | 1.69e-7 | 0.219 | 4.96e-2 | 18 / 24 |
| `pdip_raw_polish` | CPU | 0/81 | 3.31e-4 | 1.69e-7 | 0.219 | 4.96e-2 | 23 / 30 |
| `pdip_raw_polish` | A100 | 0/81 | 3.31e-4 | 1.69e-7 | 0.219 | 4.96e-2 | 23 / 30 |
| `pdip_raw_tol_1e-1` | CPU / A100 | 0/81 / 0/81 | 0.302 / 0.302 | 9.68e-5 / 9.68e-5 | 71.5 / 71.5 | 43.0 / 43.0 | 16 / 23 both |
| `pdip_raw_tol_1e-2` | CPU / A100 | 0/81 / 0/81 | 0.115 / 0.115 | 3.16e-5 / 3.16e-5 | 27.0 / 27.0 | 16.4 / 16.4 | 18 / 24 both |
| `pdip_raw_tol_1e-3` | CPU / A100 | 0/81 / 0/81 | 4.37e-2 / 4.37e-2 | 7.96e-6 / 7.96e-6 | 10.0 / 10.0 | 6.15 / 6.15 | 19 / 25 both |
| `pdip_raw_tol_jaxnnls` | CPU | 75/81 | 6.24e-7 | 4.5e-18 | 0.219 | 7.4e-5 | 50 / 50 |
| `pdip_raw_tol_jaxnnls` | A100 | **74/81** | 6.24e-7 | 4.5e-18 | 0.219 | 7.4e-5 | 50 / 50 |
| `pdip_jacobi` | CPU | 29/81 | 4.5e68 | 3.0e-7 | 1.0e68 | 3.4e70 | 19 / 50 |
| `pdip_jacobi` | A100 | **19/81** | 4.9e68 | 2.7e-7 | 7.2e73 | 1.8e70 | 19 / 50 |
| `certified` | CPU / A100 | 48/81 / 48/81 | 1.49e-2 / 1.49e-2 | 0 / 0 | 119 / 119 | 7.5e-2 / 7.5e-2 | 16 / 16 both |
| `fnnls` (reference, host CPU) | CPU / A100 job | 0/81 / 0/81 | 8.1e-8 / 8.1e-8 | 0 / 0 | 0 / 0 | 0 / 0 | 9 / 20 both |

**Per-system agreement, CPU vs A100** (max over the 81 systems):

| Candidate | max abs(Δ flux_inactive_rel) | max abs(Δ amp_rel_max_sig) | Systems whose iterations differ | Systems whose flag differs |
|---|---|---|---|---|
| `pdip_raw` / `pdip_raw_polish` | 5.4e-14 | 4.0e-10 | 0 | 0 |
| `pdip_raw_tol_1e-1` / `1e-2` / `1e-3` | 1.5e-10 / 2.3e-11 / 7.4e-12 | 2.1e-7 / 7.9e-8 / 2.9e-8 | 0 | 0 |
| `pdip_raw_tol_jaxnnls` | 1.7e-7 | 6.1e-11 | 1 | 1 |
| `certified` | 3.3e-13 | 1.1e-7 | 0 | 0 |
| `pdip_jacobi` | 4.9e68 (diverging) | 7.2e73 (diverging) | 17 | 14 |
| `fnnls` | 0 | 0 | 0 | 0 |

Context only (the cell records no logL): the largest per-system change in the NNLS objective
½xᵀQx − qᵀx between devices is 5.6e-9 absolute (relative 5.2e-16 for `pdip_raw`) against
objective magnitudes 1.1e5 – 1.1e7, i.e. at the fp64 floor.

**Where the devices disagree** (reported, not explained away):

- **`pdip_jacobi`** — the candidate already known to diverge (PyAutoArray#571). Unconverged on
  29/81 on CPU, 19/81 on the A100: 12 systems diverge on CPU only (`slam_fixture_571/k0`, `k3`,
  `k5`; `slam48_hst/v08`, `v28`, `v32`, `v37`, `v42`, `v47`; `slam_spread_hst/noise_x3_v08`,
  `sigma_min_x0p5_v08`, `sigma_min_x0p5_v32`) and 2 on the A100 only (`slam48_hst/v19`,
  `slam_spread_hst/noise_x0p3_v32`). On the systems both devices converge, `flux_inactive_rel`
  agrees to 8.9e-8. Which systems Jacobi diverges on is device-dependent; that it diverges is not.
- **`pdip_raw_tol_jaxnnls`** — on `euclid_vis_lp_k0` it reports converged at iteration 50 on the
  A100 and unconverged at the 50 cap on CPU (74/81 vs 75/81 unconverged); on
  `slam_spread_hst/noise_x3_v40` it takes 21 iterations on the A100 and 20 on CPU. Its accuracy
  metrics agree to 1.7e-7.

**Walls are context, not a result.** Median warm wall per call: `pdip_raw` 0.88 ms CPU and
3.96 ms A100; `fnnls` 0.82 / 0.90 ms (host CPU both times). These are single unbatched n = 60
solves, five warm calls each, after one compile (excluded); a single small solve is
launch-bound on the A100, so this says nothing about batched throughput, which is phase 3b's
`jit(vmap)` cell. The A100 run used the shared RAL JAX compile cache (`cache_fresh: false`,
204 autotune entries at start), which affects compile time only, not the solved values.

**Verdict against the pre-registered rule.** Parity holds for the released solver: on the A100,
`pdip_raw` at 2026.10.7.1 reproduces its CPU row on every system — 0/81 unconverged, identical
iteration counts on 81/81, worst `flux_inactive_rel` 3.31e-4 (euclid), median 1.69e-7, per-system
differences ≤ 5.4e-14 in `flux_inactive_rel` and ≤ 4.0e-10 in `amp_rel_max_sig`. The rule as
written gives the same answer on both devices: **not admissible**, failing criterion 2 on the
same two systems (`euclid_vis_lp_k0` source flux 4.96e-2, `slam_spread_hst/noise_x3_v32` sig
0.219), criterion 3 (4.96e-2) and criterion 4 at the floating-point floor (CPU 5/52, worst 32x;
A100 6/62, worst 64x — the A100 yardstick set is larger because Jacobi converges on 62 systems
there; the reference `fnnls` itself fails it on 9/52 and 7/62). The standing rule ("no drift in
the tables without a solver change") holds for the released solver and for every candidate
except the two flagged above. No baseline pin moves; no regression routes to /intake.

## Phase 3b (2026-10-08) — GPU/vmap timing of the solver corpus

**Question.** What does one positive-only solve cost per likelihood evaluation when a sampler
batches it, `jax.jit(jax.vmap(solve))` at B = 1 / 16 / 50, for the released raw PDIP
(`pdip_raw`, PyAutoArray#595) against the Jacobi PDIP (`pdip_jacobi`), fp64, on the laptop CPU and
an A100? Phase 3a's 3.96 ms A100 wall was one unbatched, launch-bound solve and said nothing about
batched throughput. Issue autolens_profiling#395. This is a timing; it is not an admissibility
result.

**Method.** New cell [`scripts/lens/solver/timing.py`](../../scripts/lens/solver/timing.py);
`_solvers.batched_kernel` vmaps each candidate's existing jitted body (`fn` / `kernel`, and so
`accuracy.py` / `early_stopping.py`, are unchanged). SLaM batches pool `slam_fixture_571` +
`slam48_hst` (56 distinct systems) and take the first B in manifest order, so lanes diverge;
`euclid_vis_lp` has one system, so its batches are B **tiled** copies (identical lanes). Stacked
inputs are placed on the device before timing. The first call of each (candidate, B) config is
recorded as `compile_s` (trace + compile + one run) and never enters a steady figure; steady cost
is 7 interleaved rounds (every config once per round, fixed order, blocking), reported as the
minimum and median batched-call wall divided by B. jit compiles once per (candidate, batch
shape): the euclid family has the SLaM shapes (n = 60), so its first calls are flagged
`compiled_here: false` and are not compiles. `fnnls` has no batched form; its rows are a host
Python loop over the B lanes, as context. Every lane is also solved unbatched in the same process
(the `accuracy.py` path) as a guard.

**Provenance.** Both rows run the five libraries at tag **2026.10.7.1**: PyAutoNerves `c5ade605`,
PyAutoFit `710f4b34`, PyAutoArray `ccddfba6`, PyAutoGalaxy `b4946b8a`, PyAutoLens `b6bf543c` (the
phase-3a SHAs; full SHAs in each JSON's `device.provenance.library_revisions`). Cell at profiling
`b8911d8`.

- **A100:** RAL job **398249** on `euclid-ral-gpu-2` (NVIDIA A100 80GB PCIe, driver 610.57.04,
  host AMD EPYC 7702), `--partition=gpu --gres=gpu:1`, COMPLETED 0:0 in 0:38 (09:16:04 – 09:16:42
  BST). Submit `hpc/batch_gpu/submit_lens_solver_timing_a100_fp64`; libraries from the phase-3a
  private clone `/mnt/ral/jnightin/PyAuto_wt/linear-solver-p3/` (import guard green, all five from
  there; the shared mirror was not touched). The phase-3a RAL worktree held its untracked 3a
  artefacts (byte-identical to the committed ones), which blocked a branch checkout, so the job ran
  from a sibling worktree `/mnt/ral/jnightin/autolens_profiling_wt/linear-solver-p3b`. jax / jaxlib
  0.10.2, numpy 2.2.6, numba 0.65.1; backend `gpu`, `cuda:0`. `.err` empty; post-check green.
  The shared RAL JAX compile cache was warm (`cache_fresh: false`, 204 autotune entries at start),
  so A100 compile walls are cache-assisted.
- **CPU:** the laptop (WSL2, i9-10885H, `OMP_NUM_THREADS=1`), local `git worktree` checkouts of
  the same five tags (`autolens.__file__` confirmed there); jax / jaxlib 0.10.2. Load average 1.9
  at import, with other sessions' test suites running beside it; interleaved minima absorb most of
  that, medians less. Laptop compile cache recorded `cache_fresh: true`.
- **Artefact names.** As in phase 3a, the source checkouts stamp `al.__version__ = 2026.8.17.1`,
  so both runs wrote to scratch / `output/` and the pairs were copied to
  `timing_summary_all_v2026.10.7.1` (CPU) and `timing_summary_all_gpu_v2026.10.7.1` (A100). The
  label is `all` because more than one group ran; the groups are the three above
  (`slam_spread_hst` is not in this cell's default set).

**Steady per-evaluation cost** (ms, min over 7 interleaved rounds of batched-call wall / B;
median in the JSON):

| Candidate | Family | CPU B=1 | CPU B=16 | CPU B=50 | A100 B=1 | A100 B=16 | A100 B=50 |
|---|---|---|---|---|---|---|---|
| `pdip_raw` (released) | SLaM (56 distinct) | 1.353 | 0.881 | 0.682 | 3.948 | 0.581 | **0.190** |
| `pdip_raw` (released) | euclid (tiled) | 1.611 | 0.969 | 0.798 | 4.942 | 0.689 | 0.226 |
| `pdip_jacobi` | SLaM (56 distinct) | 2.488 | 1.630 | 1.396 | 6.715 | 1.149 | 0.369 |
| `pdip_jacobi` | euclid (tiled) | 1.151 | 0.699 | 0.531 | 3.311 | 0.459 | 0.147 |
| `fnnls` (host loop) | SLaM | 1.697 | 1.312 | 1.306 | 1.021 | 0.867 | 0.878 |
| `fnnls` (host loop) | euclid (tiled) | 1.396 | 1.387 | 1.379 | 0.970 | 0.961 | 0.961 |

The A100 `fnnls` rows are the RAL host CPU (EPYC), not the GPU.

**Compile, kept separate** (s, first call per config, SLaM family; the euclid configs reuse these
executables):

| Candidate | CPU B=1 / 16 / 50 | A100 B=1 / 16 / 50 (warm persistent cache) |
|---|---|---|
| `pdip_raw` | 1.07 / 1.07 / 0.97 | 1.17 / 0.75 / 0.59 |
| `pdip_jacobi` | 0.50 / 0.70 / 0.67 | 0.32 / 0.36 / 0.38 |

At these walls one compile costs what 1–6 thousand steady A100 evaluations at B = 50 cost; it is
paid once per batch shape per process.

**Slowest lane.** A vmapped `while_loop` runs every lane until the slowest stops. Iterations per
lane (median / batch max):

| Candidate | Family | CPU B=1 | CPU B=16 | CPU B=50 | A100 B=1 | A100 B=16 | A100 B=50 |
|---|---|---|---|---|---|---|---|
| `pdip_raw` | SLaM | 18 / 18 | 18 / 19 | 17.5 / 19 | 18 / 18 | 18 / 19 | 17.5 / 19 |
| `pdip_raw` | euclid (tiled) | 24 / 24 | 24 / 24 | 24 / 24 | 24 / 24 | 24 / 24 | 24 / 24 |
| `pdip_jacobi` | SLaM | 50 / 50 | 50 / 50 | 19 / 50 | 41 / 41 | 19 / 50 | 19 / 50 |
| `pdip_jacobi` | euclid (tiled) | 19 / 19 | 19 / 19 | 19 / 19 | 19 / 19 | 19 / 19 | 19 / 19 |

The released raw solve's batch maximum is within one iteration of its median (19 vs 17.5 at
B = 50), so batching costs it at most ~9 % in idle lanes. The Jacobi batch pays its 50-iteration
cap whenever any lane diverges (median 19, max 50 at B = 16 and 50 on both devices): ~2.6x the
median lane's work, which is the pre-#595 mechanism the task asked to see. On the A100 at B = 50
SLaM, `pdip_jacobi` costs 1.94x `pdip_raw` per evaluation (0.369 / 0.190 ms) on the same 50 lanes.

**Guard: does the batched path solve what the unbatched cells solve?**

- `pdip_raw`: yes, on both devices. CPU: every lane's iterations and flag identical to the
  unbatched solve, max |Δ `flux_inactive_rel`| = 0. A100: iterations and flags identical on all
  lanes, max |Δ `flux_inactive_rel`| 1.4e-14 (euclid), 1.2e-15 (SLaM). Worst batched
  `flux_inactive_rel` 3.31e-4 (euclid) on both devices, the phase-3a value; 0 unconverged lanes.
- `pdip_jacobi` on CPU: identical (0 lanes differ).
- `pdip_jacobi` on the A100: **the batched trajectories differ** from the unbatched ones on 19 of
  the 50 SLaM lanes (iterations) and on 13 in the convergence flag: 4/50 unconverged batched
  against 13/50 for the unbatched A100 solve on the same lanes (the phase-3a A100 rows give the
  same 13; CPU, batched and unbatched alike: 19/50). On the lanes where Jacobi diverges somewhere, the difference reaches 4.9e68 in
  `flux_inactive_rel`. Phase 3a already found that which systems Jacobi diverges on depends on
  the device; under `vmap` on the A100 it depends on the batching too. Reported, not explained
  away; the released solver shows nothing of the kind.

**Caveats.**

- The euclid batches are tiled copies of one system: identical lanes cannot show lane divergence,
  and their B = 16 / 50 rows measure throughput only.
- Laptop walls are WSL2 under background load; the interleaved minimum is the figure to read.
  The A100 B = 1 `pdip_raw` wall (3.95 ms) agrees with phase 3a's unbatched 3.96 ms.
- Compile walls depend on cache state (A100 warm shared cache; laptop fresh) and are not
  comparable across devices as a compiler measurement.
- Per-eval figures are batched-call wall / B with inputs already on the device; host-to-device
  transfer of `(Q, q)` and anything else in a likelihood are outside them.
- The CPU runs batched JAX linear algebra on jax 0.10.2, the version Nerves at 2026.10.7.1
  excludes at install time for a CPU batched-LAPACK deadlock (Heart#274); none occurred here.

**Verdict.** A timing is not admissibility; no pin moves. Batched on the A100, the released raw
PDIP costs 0.190 ms per evaluation at B = 50 on distinct SLaM systems (0.226 ms on the tiled
euclid system), 21x below its own B = 1 wall and 3.6x below the laptop CPU's 0.682 ms at the
same batch; its lanes stop within one iteration of each other, and the batched path reproduces
its unbatched solve on every lane. Jacobi costs 1.9x as much at B = 50 on SLaM because a single
diverging lane pins the batch at the 50 cap. The phase-1/2 rule outcome for `pdip_raw` (not
admissible as written) is unchanged; nothing routes to /intake.

## Phase 4a (2026-10-08) — Jacobi batched vs unbatched on the A100

**Question.** Phase 3b found that on the A100 the Jacobi PDIP (`pdip_jacobi`, the library
`"jacobi"` mode and the Mapper default) gives different results inside `jit(vmap)` over 50
distinct SLaM systems than when each system is solved alone: 19/50 lanes differ in iterations
and 13/50 in the flag. CPU batched and unbatched are bit-identical, and `pdip_raw` matches
everywhere. Where does the difference come from? This phase is research only; issue
autolens_profiling#397. The write-up and decision table are in the
[research note](../../wiki/research/jacobi_a100_batched_divergence.md).

**Method.** A new probe,
[`scripts/lens/solver/batched_divergence.py`](../../scripts/lens/solver/batched_divergence.py),
calls only the library's solver entry points, composed the way `_solvers` composes them. It
tests `pdip_jacobi`, with `pdip_raw` and `certified` as controls, on the phase-3b batch (the
first 50 systems of `slam_fixture_571` + `slam48_hst`, n = 60, fp64). The probes are:

- determinism: the batched solve twice through a fresh `jit(vmap)` plus a repeat call, and the
  unbatched solve twice through a fresh `jit`;
- `jit(vmap)` at B = 1 vs `jit`;
- one system tiled at B = 2 / 8 / 50 (`euclid_vis_lp_k0` and `slam_fixture_571/k0`);
- the k = 0..50 trajectory, using the library `solve_nnls(Q_pc, q_pc, solver_tol=None,
  max_iter=k)` batched and unbatched, with a consistency check that k = 50 reproduces the
  candidate's `x` bit for bit (50/50 lanes on both devices, batched and unbatched);
- jit vs jit(vmap) of the primitives jaxnnls's PDIP is built from (`cho_factor`, `cho_solve`,
  matvec) on each lane's Jacobi system;
- a per-lane join with cond(Q), cond(Q_pc) and the phase-3a and phase-3b flags.

No timings are recorded.

**Provenance.** Both devices run the five libraries at tag **2026.10.7.1**: PyAutoNerves
`c5ade605`, PyAutoFit `710f4b34`, PyAutoArray `ccddfba6`, PyAutoGalaxy `b4946b8a`, PyAutoLens
`b6bf543c`. Full SHAs are in each JSON's `device.provenance.library_revisions`. The probe is at
profiling `6fb885e` on both devices.

- **A100:** RAL job **399050** on `euclid-ral-gpu-2` (NVIDIA A100 80GB PCIe, driver 610.57.04,
  host AMD EPYC 7702), `--partition=gpu --gres=gpu:1`. It COMPLETED 0:0 in 2:30 (10:14:11 –
  10:16:40 BST) and ran the probe twice: with the stack's XLA flags (`--xla_disable_hlo_passes=
  constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false`, set by
  PyAutoNerves at import), then with `--xla_gpu_deterministic_ops=true` added before jax was
  imported. Submit script: `hpc/batch_gpu/submit_lens_solver_batched_divergence_a100_fp64`. The
  libraries came from the phase-3a private clone `/mnt/ral/jnightin/PyAuto_wt/linear-solver-p3/`;
  the import guard was green and the shared mirror was not touched. jax / jaxlib 0.10.2, numpy
  2.2.6; backend `gpu`, `cuda:0`; the shared JAX compile cache was warm (`cache_fresh: false`).
  The `.err` was empty and the post-check green. The phase-3b RAL worktree held untracked 3b
  artefacts, which blocked a branch checkout, so the job ran from a new sibling worktree,
  `/mnt/ral/jnightin/autolens_profiling_wt/linear-solver-p4a`. Creating it took ~27 min on the
  RAL filesystem.
- **CPU:** the laptop (WSL2, i9-10885H, `OMP_NUM_THREADS=1`, load average 3–5 from other
  sessions; no timings are taken, so load does not affect the result), using local `git
  worktree` checkouts of the five tags with `autolens.__file__` confirmed there. jax / jaxlib
  0.10.2, numba 0.62.1.
- **Artefact names.** As in phases 3a and 3b, the source checkouts stamp `2026.8.17.1`, so the
  runs wrote to scratch and were copied to `batched_divergence_summary_all_v2026.10.7.1` (CPU),
  `_all_gpu_v2026.10.7.1` (A100) and `_all_gpu_det_v2026.10.7.1` (A100 with deterministic ops).
- **Known defect in the committed A100 JSONs.** Their per-element `max_ulp_*` fields were taken
  as a float64 difference of ~2^62-sized ordered integers, so they are rounded to multiples of
  ~512–1024 ulp, and values below ~512 read as 0. This was fixed after the run (exact integer
  difference). The relative norms (`rel_d*`, `rel_dx_by_k`) and every bitwise comparison are
  exact, and every number quoted below is one of those.

**Results** (number of lanes out of 50 that differ):

| Probe | CPU | A100 | A100, deterministic ops |
|---|---|---|---|
| batched run 1 vs run 2 / repeat call, unbatched run 1 vs run 2 (all candidates) | 0 | 0 | 0 |
| `jit(vmap)` B = 1 vs `jit` (all candidates) | 0 | 0 | 0 |
| tiled B = 2 / 8 / 50: lanes vs each other; lane vs unbatched | 0; 0 | 0; **differs at every B** | as A100 |
| batched vs unbatched `x` bits: `pdip_jacobi` / `pdip_raw` / `certified` | 0 / 0 / 0 | 50 / 50 / 50 | as A100 |
| batched vs unbatched iterations; flag: `pdip_jacobi` | 0; 0 | **19; 13** (4 vs 13 unconverged) | as A100 |
| batched vs unbatched iterations; flag: `pdip_raw`, `certified` | 0; 0 | 0; 0 | as A100 |
| primitives `cho_factor` / `cho_solve` / matvec / Jacobi `Q_pc` | 0 / 0 / 0 / 0 | **50 / 50** / 0 / 0 | as A100 |

The deterministic-ops run is identical to the default run in every per-lane comparison and every
trajectory difference. On the A100, the largest batched-vs-unbatched ‖Δx‖/‖x‖ is 1.3e-13 for
`pdip_raw`, 3.4e-12 for `certified` and 61 for `pdip_jacobi`. Lane by lane, both devices
reproduce the phase-3a unbatched flags and the phase-3b batched flags.

**Trajectory** (`pdip_jacobi`, A100). On every lane the first difference is at **k = 0**
(jaxnnls `initialize`), in all of x, s and z, on 55–60 of the 60 components, with
‖Δy‖/‖y‖ = 1.1e-15 – 2.1e-15. The lanes then split into two groups:

- **26 quiet lanes.** ‖Δx‖/‖x‖ stays ≤ 8.2e-13 for all 50 iterations, iterations are identical,
  and every one converges on CPU, A100 unbatched and A100 batched.
- **24 sensitive lanes.** These include every lane Jacobi fails to converge on anywhere: all 7
  CPU-only, all 12 both-device and the 1 A100-only member of the phase-3a set, plus 4 lanes that
  converge everywhere. Unconverged counts in this group are 19 on CPU, 13 on the A100 unbatched
  and 4 on the A100 batched. The difference reaches order one within one or two iterations (in
  the 19 lanes whose iterations differ, ‖Δx‖/‖x‖ > 1e-3 at k = 1 on 8 lanes and at k = 2 on 11).

cond(Q) does not separate the groups. It spans only 9.64e10 – 9.75e10 over the 50 lanes, and
cond(Q_pc) spans 1.44e11 – 1.55e11. The AUC for predicting "iterations differ" is 0.61 for
cond(Q) and 0.56 for cond(Q_pc); for "sensitive" it is 0.62 and 0.54. These AUCs and the
sensitive split (‖Δx‖/‖x‖ > 1e-6 at any k) were computed from the A100 JSON's per-lane `join`
and `trajectory` rows. The probe's own `join.statistics` AUC fields use "x differs", which is
true on all 50 A100 lanes, so they are `null` there.

**Supported explanation.** The difference is not run-to-run nondeterminism; it is bit-stable,
and the deterministic-ops flag changes nothing. It is batch-shape-dependent arithmetic in the
A100 Cholesky factorisation. The batched (B ≥ 2) `cho_factor` and `cho_solve` round differently
from the unbatched ones, while B = 1 through `vmap` lowers identically to `jit`. The PDIP state
therefore differs at rounding level from `initialize` onward. On CPU, batched and unbatched
Cholesky are bit-identical. Jacobi's unstable systems, the same ones that fail to converge on
some device, amplify the rounding-level difference to order one within two iterations, so their
convergence outcome follows the rounding. The released `pdip_raw` and `certified` see the same
Cholesky difference and stay at ≤ 3.4e-12 with identical iterations and flags.

**Not established.**

- Which GPU kernel each lowering uses (4b kernel-selection probe). The stack's autotune and
  Triton flags were not varied.
- Whether the phase-3a CPU-vs-A100 set difference has the same cause.
- Whether Mapper (pixelized) systems fall in the sensitive group. The corpus here is SLaM MGE
  systems.
- Why 4 always-convergent lanes are sensitive.

**Verdict.** Research only. No default, pin or tolerance changed, and nothing routes to /intake.
The decision (keep Jacobi and document it, a determinism flag, a tolerance or cap change, or
moving the Mapper default) is laid out with costs in the
[research note](../../wiki/research/jacobi_a100_batched_divergence.md#decision-table-for-the-human-nothing-here-is-decided).
The determinism flag is ruled out by this evidence.
