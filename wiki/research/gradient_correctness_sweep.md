# Gradient-correctness sweep: triage of every open JAX-gradient bug and hazard

Research note for autolens_profiling#517, started 2026-11-02. This is the umbrella that records a verdict for each of 14 items. Library fixes ship as their own issues and branches; this page links them.

**Status:** order approved by the human 2026-11-02. Quick wins (10, 11, 12) done; item 1 measured and re-filed; item 2 next.

## Method used for every claim

- Compare autodiff with the stored central FD at the smallest steps (1e-8, 1e-7, 1e-6), as well as at the plateau. A gap that is the same at every small step on a non-tiny gradient is a wrong derivative; FD noise or roughness is not.
- Check where the point is evaluated ([PyAutoPulse `knowledge/gradient_witness_evaluation_point.md`](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/knowledge/gradient_witness_evaluation_point.md)).
- Bisect stage by stage: eager vs `jit`, `jvp` vs `jacrev` (reverse only where NNLS is involved, because the NNLS `custom_vjp` has no forward mode).
- When a fix leaves values unchanged, do a before/after on the same cell. The stored FD values stay valid.

Tools: `Science/scratch/gradient_campaign_impact_2026-11-02/` (`small_step_check.py`, `ab_cell.sh`, `ab_compare.py`) and `Science/scratch/rectangular_gradient_rtu_sparse_2026-11-02/` (`repro.py`, `stages.py`, `stages2.py`).

## What the triage changed

Re-running `small_step_check.py` over the 73 committed `results/**/gradient.json` records gave two new framings:

1. **Item 1 (MGE) is specific to the imaging/mge cell, not to MGE in general.**
   - Every `imaging/mge/{ao,euclid,hst,jwst}` cell has a flat gap of 1e-4 to 4e-3 at all three small steps, the same on CPU and A100.
   - But `group/mge/euclid` and `multi_galaxy/mge/hst` (also MGE, also NNLS) agree to about 1e-6.
   - The imaging/mge cell is built differently, by `scripts/imaging/mge/gradient_breakdown.py` with `cell_builders/imaging_mge.py`, and differs on two counts:
     - **Evaluation point:** it is taken at the true prior medians (its `prior_medians` label is correct), with the lens light centre at (0, 0) and radial-bin over-sampling centred there. The cells that agree are taken at iid draw 0.
     - **NNLS warm-start memo:** `build_analysis` uses default `Settings`, so the memo is at its library default (on). `standard_jax_cell` and `galaxy_group_standard.py` turn it off.
   - **Measured 2026-11-02 (local CPU, `imaging/mge/hst`; scripts and JSON in `Science/scratch/gradient_correctness_sweep_2026-11-02/`):** see "Item 1 result" below. The memo is not the cause; the evaluation point is.
2. **Item 2 (Delaunay) is partly CPU-vs-GPU on the same path, not only sparse-vs-dense.**
   - `interferometer/delaunay/jvla` runs `jax_*_sparse_fp64_wtilde` on **both** devices. CPU gives a flat 1e-2 gap at every small step; A100 gives 2e-6. Same likelihood path, different backend.
   - `group/delaunay/euclid` direction 1 has a flat 0.2 gap on CPU-sparse (a plateau, so a wrong derivative, not roughness) against 2e-4 on GPU-dense.
   - This points at XLA:CPU compilation, such as a `jit`-unstable reduction like the one #672 fixed, or the constant-folding pass that the real HST cell already needs disabled. It points less at the sparse algebra itself.

## Triage table

| # | Item | Status on `main` | Suspected mechanism | Repo | Cost | Order |
|---|---|---|---|---|---|---|
| 10 | `gradient.json` `point` mislabel | **fixed on this branch** | label default `prior_medians` written for iid draw 0 (and for the weak cell's simulator truth) | autolens_profiling | local, minutes | 1 |
| 11 | #515 note row for image-plane cells | **fixed on this branch** | it is the FD that is zero (mechanism E) | autolens_profiling | local, minutes | 1 |
| 12 | Rectangular mapper row weights | **already fixed**: PyAutoArray#490 (72fb01d1, 2026-08-26) | — | PyAutoArray | none | 1 (retire) |
| 1 | MGE plateau off autodiff (mechanism B) | **measured**: re-filed narrower | evaluation exactly at `ell_comps` = 0 (the prior median); not NNLS, not the memo | PyAutoGalaxy | local, HST cell | 2 (done) |
| 2 | CPU vs GPU Delaunay gradient gap | open | XLA:CPU-specific: a reduction tie or constant folding; sparse curvature path second | PyAutoArray | local CPU; GPU leg needs RAL go | 3 |
| 14 | Sparse vs dense log L, non-linear lens light | open (not chased) | sparse path drops blurring-region light (`convolver.py:1053` warning) | PyAutoArray | local | 3 (with 2) |
| 3 | MGE deflection reverse-mode NaN at a grid centre | open, unverified since 2026-09-03 | double-`where` trap in `zeta_from` (`xp.abs(y)`, conjugate select) | PyAutoGalaxy | local, seconds | 4 |
| 5 | Interferometer sparse d/d(coefficient) 1.3 % | open, unverified | probably the FD step (single eps = 1e-5) | PyAutoArray | local, seconds | 4 |
| 6 | Solver branch flips, PyAutoArray#377 | open | active-set non-smoothness, or a tie / `jit` artefact | PyAutoArray | local | 5 |
| 4 | q-clamp and bare-`sqrt` ellipticity sites | **open**: all sites still present; mass Chameleon clamp is now a traceable `xp.where` | NaN grad at `ell_comps = 0`, zero grad past clamps | PyAutoGalaxy | local | 6 (PyAutoGalaxy claimed by `magnification-maps-uncertainty`) |
| 13 | Critical-curve / caustic `jit` / `grad` parity | **open**: no test covers `_zero_contour.py` | — (test gap) | autolens_workspace_test | local | 7 |
| 8 | Noisy interferometer rectangular log L (mechanism D) | open | solver tolerance, W~ precision or float accumulation | PyAutoArray | local (`ab_interferometer_rectangular_sma` exists) | 8 |
| 9 | Piecewise-constant image-plane point likelihoods (mechanism E) | open | discrete image matching | PyAutoLens | none (decision) | 8 |
| 7 | Densely non-smooth pixelized likelihoods (mechanism C) | open (cause ruled intrinsic 2026-11-02) | triangle flips; research question for gradient samplers | autolens_profiling | none (decision) | 9 |

Duplicates folded in: items 2 and 3 of `unexplained_gradient_gaps_cpu_sparse_delaunay_and_mge.md` are item 1 above; its item 1 is item 2 above.

## Item 1 result: the gap lives at `ell_comps` = 0

The probe (`item1_mge_probe.py`) rebuilds the exact cell and compares `grad . v` with the central FD along the harness's own seeded directions.

| Variant (`imaging/mge/hst`) | Relative gap, h = 1e-4 … 1e-8 |
|---|---|
| As stored (prior medians, memo on) | 1.8e-3 / 2.8e-3 / 8.0e-4, flat in h (matches the stored record) |
| Memo off (`AUTOARRAY_NNLS_WARM_START=0`) | identical to stored |
| All 8 free parameters nudged by ~1e-3 | 1e-10 to 1e-7 |
| Only the 4 `ell_comps` nudged by ~1e-3 | 1e-11 to 7e-8 |
| Only the 4 centres nudged | 6e-4 to 2.2e-3, flat (gap stays) |
| `ell_comps` nudged by ~1e-6 | agrees at h ≤ 1e-7 (2e-8); FD wrong at h ≥ 1e-6, where the stencil crosses the origin |

What this establishes:

- **It is the evaluation point, not the NNLS solve.** All 8 free-parameter prior medians are exactly 0 (both centres, both `ell_comps`), and the gap needs `ell_comps` = 0. Autodiff is correct (to about 1e-9) whenever |e| ≳ 1e-6 and the FD stencil stays off the origin.
- **The likelihood is smooth through e = 0 at scales 1e-4 to 1e-2** (`item1_cone_test.py`: along a line through the origin, the even part halves-ratio is 4.0, i.e. quadratic, and the odd slope converges). So it is not a genuine kink, and the autodiff value at exactly 0 is wrong.
- **At exactly 0 the value is offset too.** One-sided FD at the stored point shows f(θ₀) about 2.5e-4 below its neighbours (forward − backward ∝ 1/h). That matches `convert._nudge_off_origin`, which moves e₁ by 1e-8 at exactly (0, 0) for traced values; its docstring's "the gradient at the nudged point is exact" holds for shear and multipoles (linear in their components), not obviously for elliptical light.
- **Unresolved:** inside |e| ≲ 1e-5 the local slope (8329.3 along direction 0) differs from the slope across the origin (8336), and the cone test's even part stops scaling below t ≈ 1e-4. There is small-scale structure near round in the light-profile ellipticity path that this probe did not pin down. No near-round clamp exists in `profiles/light/` (the 0.9999 / 0.99999 clamps are all mass-side).
- **Side note:** the stored record's `n_parameters: 160` counts pytree leaves (centre and `ell_comps` of each of the 20 Gaussians × 2 galaxies, untied), not the model's 8 free parameters. Harmless for a correctness check, but the FD directions move each Gaussian independently.

Verdict: **re-filed narrower** into PyAutoGalaxy with item 4 (near-round light-profile ellipticity gives a value and gradient inconsistent with the smooth surrounding likelihood; the MGE prior median `ell_comps` = 0 sits on it). `interferometer/mge/sma` (74 % in one direction) is presumably the same point; confirm it under the re-file.

## Verdicts

| # | Verdict | Evidence |
|---|---|---|
| 10 | fixed | `harness.run_gradient` now requires `point=` (constants `POINT_PRIOR_MEDIANS`, `POINT_IID_DRAW_0`); all 13 drivers pass their real point; 43 stored records relabelled in place (41 standard cells → iid draw 0, 2 weak → simulator truth; label only, values unchanged; catalogue `setups.json` rebuilt, 43 hash changes, nothing else) |
| 11 | fixed | `gradient_campaign_impact_border_tie.md` row now says FD is zero and autodiff nonzero, with the stored values (cluster −7.6 / −4.07 / −0.85, point_source_image 415 / −622 / 1.4e3) |
| 1 | re-filed narrower (PyAutoGalaxy, with item 4) | "Item 1 result" above; not NNLS, not the memo; autodiff correct off e = 0 |
| 12 | already-fixed | PyAutoArray#490, commit 72fb01d1 (2026-08-26); verified at `rectangular.py:723-766`; Mind record `complete/2026/11/rectangular-mapper-bilinear-row-weights.md` |

(Other items are filled in as each one closes.)
