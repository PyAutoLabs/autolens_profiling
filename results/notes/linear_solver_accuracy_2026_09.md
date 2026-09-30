# Linear-solver accuracy study — ledger (phase 1, 2026-09)

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

(`pdip_raw_tol_1e-2` reproduces `pdip_raw` exactly through `solve_nnls` directly, as designed.)

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

(`pdip_raw_tol_1e-2` is identical to `pdip_raw`.) Which candidates would turn the euclid test
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
- Corpus: [`corpus/manifest.json`](../lens/solver/corpus/manifest.json).
- Package and how to re-run: [`scripts/lens/solver/README.md`](../../scripts/lens/solver/README.md).
- Campaign page: [`wiki/campaigns/linear_solver_accuracy.md`](../../wiki/campaigns/linear_solver_accuracy.md).
