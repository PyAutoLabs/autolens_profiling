# `scripts/lens/solver/` — the linear-solver programme

A standing, **accumulating** home for positive-only linear-solver accuracy and tolerance
studies. Every PyAutoLens likelihood with linear light profiles or a pixelization ends in a
non-negative least-squares solve of `(Q, q)` = (`curvature_reg_matrix`, `data_vector`); the
solver that does it (fnnls on NumPy, a primal-dual interior-point method or a certified
active-set method on JAX) decides whether the reconstructed amplitudes — and therefore the
log likelihood — are right. This section answers, release after release:

> *On the systems real models produce, does each positive-only solver return the reference
> amplitudes, does its convergence flag tell the truth, and what does each digit of accuracy
> cost in iterations?*

It is dataset-free in the `scripts/lens/` sense: the input is a frozen **corpus** of captured
systems, not a dataset. A system never moves when the library's likelihood code does, so a
changed metric here is always a changed *solver*. The corpus only grows — new captures are added
as groups, old groups are never rewritten — and every release re-runs the cells over all of it.

## Layout

| File | Role |
|------|------|
| [`_corpus.py`](./_corpus.py) | The corpus: `load_corpus()`, `iter_systems()`, `add_group()`, and the first ingest `import_571_fixture()`. `python scripts/lens/solver/_corpus.py` (re)builds group `slam_fixture_571`. |
| [`_solvers.py`](./_solvers.py) | The candidate registry `CANDIDATES` — each a composition of library primitives only, `jax.jit`-compiled once per candidate. |
| [`_metrics.py`](./_metrics.py) | `metrics(x, system)` against the stored fnnls reference, and `timing()` (median warm wall). |
| [`_driver.py`](./_driver.py) | Shared CLI, per-(system, candidate) evaluation, artefact naming and JSON write. |
| [`capture.py`](./capture.py) | Adds a captured group to the corpus: `--source slam48` (group `slam48_hst`, all 48 #571 SLaM vectors, fixture reproduction recorded), `--source slam_spread` (`slam_spread_hst`, sigma_min x0.5/x2 and noise x0.3/x3 at 6 vectors) and `--source euclid_vis_lp` (`euclid_vis_lp`, the euclid latent jit test's system plus its eager/jitted `total_source_flux` as `latent_reference`). Phase 5 (Mapper corpus, #399): `--source delaunay_hst` / `rectangular_hst` (Mapper-only inversions with the HST presets' Delaunay / rectangular sources, lens light fixed) and `--source slam_mixed_hst` (lens 2x20 MGE + Delaunay, the SLaM `source_pix` case), 8 systems each, every capture asserted to be the positive-only Mapper solve in `"jacobi"` mode. |
| [`accuracy.py`](./accuracy.py) | Every candidate on every system → `results/lens/solver/accuracy_summary_<corpus>_v<version>.{json,png}`. |
| [`early_stopping.py`](./early_stopping.py) | Released raw PDIP at each iteration cap → `results/lens/solver/early_stopping_summary_<corpus>_v<version>.{json,png}`. |
| [`timing.py`](./timing.py) | Batched cost: `jax.jit(jax.vmap(solve))` per-evaluation wall at B = 1 / 16 / 50 (`pdip_raw`, `pdip_jacobi`; `fnnls` as a host loop), compile wall kept separate, per-lane iterations and an unbatched-guard → `results/lens/solver/timing_summary_<corpus>[_gpu]_v<version>.{json,png}`. |
| [`batched_divergence.py`](./batched_divergence.py) | **Research (phase 4a).** Does `jit(vmap)` solve what `jit` solves, lane by lane? Determinism (fresh-jit reruns; `--deterministic` sets `--xla_gpu_deterministic_ops=true`), `jit(vmap)` at B = 1 vs `jit`, tiled batches, the k = 0..50 first-difference PDIP trajectory of `pdip_jacobi` via the library `solve_nnls`, jit vs jit(vmap) of the Cholesky / matvec primitives, joined per lane with cond(Q) and the phase-3a/3b flags; no timings → `results/lens/solver/batched_divergence_summary_<corpus>[_gpu][_det]_v<version>.{json,png}`. Write-up: [`wiki/research/jacobi_a100_batched_divergence.md`](../../../wiki/research/jacobi_a100_batched_divergence.md). |
| [`euclid_latent.py`](./euclid_latent.py) | **Post-hoc.** Every candidate's reconstruction of `euclid_vis_lp_k0` pushed through the euclid pipeline's own `total_source_flux` latent code (validated against the stored eager / jit values) → `results/lens/solver/euclid_latent_by_candidate_v<version>.json`. |

The corpus itself lives in `results/lens/solver/corpus/`: `manifest.json` (per-system metadata)
plus one compressed `<group>.npz` per group holding `Q_<name>`, `q_<name>` and `x_ref_<name>` —
the fnnls reference, computed once when the group is added and stored, so a study never
re-derives its truth. The n ~ 1500 Mapper groups store `Q` with the lossless `sym_tri_xor`
encoding (upper triangle + bitwise XOR of the lower; the group's manifest entry says
`"encoding"`), because a dense 8-system group exceeds GitHub's 100 MB file limit;
`load_corpus()` rebuilds `Q` bit for bit. Cells over the Mapper groups run one group at a time:
`timing.py` pools a run's multi-system groups into one batch, which needs equal n, and a
multi-group run is labelled `all`, which would overwrite the phase-3/4 `all` artefacts.

## Corpus

<!-- BEGIN auto-table:solver-corpus -->
| Group | Systems | n | cond(Q) | max abs(q) | Source columns | Captured by | Model |
|-------|---------|---|---------|------------|----------------|-------------|-------|
| `slam_fixture_571` | 8 | 60 | 9.69e+10 – 9.73e+10 | 2.16e+06 – 2.16e+06 | 20 | `scripts/imaging/hazards/mge_nnls_capture.py` | SLaM source_lp[1]: lens 2x20 MGE (sigma_min=pixel_scale/10) + source 20 MGE, free Isothermal + ExternalShear |
| `slam48_hst` | 48 | 60 | 9.64e+10 – 9.75e+10 | 2.16e+06 – 2.16e+06 | 20 | `scripts/lens/solver/capture.py` | SLaM source_lp[1]: lens 2x20 MGE (sigma_min=pixel_scale/10) + source 20 MGE, free Isothermal + ExternalShear |
| `slam_spread_hst` | 24 | 60 | 1.08e+10 – 1.08e+12 | 2.40e+05 – 2.40e+07 | 20 | `scripts/lens/solver/capture.py` | SLaM source_lp[1]: lens 2x20 MGE (sigma_min=pixel_scale/10) + source 20 MGE, free Isothermal + ExternalShear |
| `euclid_vis_lp` | 1 | 60 | 5.56e+11 – 5.56e+11 | 8.49e+06 – 8.49e+06 | 20 | `scripts/lens/solver/capture.py` | euclid vis_lp (initial_lens_model.vis_lp_model_from) at _ordered_median_vector, dataset simulated/euclid_dr1_like |
| `delaunay_hst` | 8 | 1500 | 6.07e+05 – 2.42e+07 | 1.91e+03 – 1.92e+05 | 1500 | `scripts/lens/solver/capture.py` | Mapper only: fixed truth Sersic lens light (no columns) + free Isothermal + ExternalShear; source Delaunay (Hilbert 1500, AdaptSplit), the HST Delaunay preset's source |
| `rectangular_hst` | 8 | 1369 | 1.30e+06 – 5.42e+07 | 1.74e+03 – 1.75e+05 | 1369 | `scripts/lens/solver/capture.py` | Mapper only: fixed truth Sersic lens light (no columns) + free Isothermal + ExternalShear; source RectangularBilinearAdaptImage 39x39 + Constant(1.0), the HST rectangular preset's source |
| `slam_mixed_hst` | 8 | 1540 | 5.54e+09 – 5.56e+11 | 2.11e+05 – 2.11e+07 | 1500 | `scripts/lens/solver/capture.py` | Mixed (SLaM source_pix): lens 2x20 MGE (sigma_min=pixel_scale/10, linear) + free Isothermal + ExternalShear; source Delaunay (Hilbert 1500, AdaptSplit) |
<!-- END auto-table:solver-corpus -->

## Candidates

Hand-written from [`_solvers.py`](./_solvers.py); the library function each one composes is in
its registry entry. All return the reconstruction in raw coordinates exactly as
`inversion_util.reconstruction_positive_only_from` does.

| Candidate | What it is | Library primitive(s) |
|-----------|------------|----------------------|
| `fnnls` | **Reference.** NumPy fnnls with incremental Cholesky from the dense-sign start — the NumPy production path (memo off). Defines `x_ref`; its row is a self-check. | `autoarray.util.fnnls.fnnls_cholesky` |
| `pdip_jacobi` | Jacobi-preconditioned PDIP at jaxnnls's own tolerance, cap 50 — the library `"jacobi"` mode (default with a Mapper; the MGE default before PyAutoArray#571). | `jax_nnls.solve_nnls_primal_with_status` |
| `pdip_raw` | **Library default for MGE (linear-object-only) inversions**: forward PDIP on the raw system at `data_scaled_solver_tol(q)`, cap 50, mapped to the Jacobi system and — since PyAutoArray#595 — polished there by the #573 rule before being returned as `y * D`. `converged` / `iterations` are the forward solve's (the 1–7 polish iterations are not counted). Extras record the polish / relaxed-KKT status. | `jax_nnls.solve_nnls_primal_raw_forward` (+ `raw_forward_backward_status`) |
| `pdip_raw_tol_1e-1` / `_1e-2` / `_1e-3` | Raw forward PDIP with the data-scaled tolerance's factor set to 1e-1 / 1e-2 / 1e-3 in place of `DATA_SCALED_TOL_FACTOR` (1e-2 = released), no polish — so since PyAutoArray#595 the 1e-2 row is the *pre-#595* `pdip_raw`, not the current one. | `jax_nnls.solve_nnls` + `data_scaled_solver_tol` |
| `pdip_raw_tol_jaxnnls` | Raw forward PDIP at jaxnnls's absolute tolerance `min(n·eps·5e3, 1e-2)` — unreachable on an unscaled system, so it runs to the cap. | `jax_nnls.solve_nnls(solver_tol=None)` |
| `pdip_raw_polish` | Raw forward, then ≤ `RAW_BACKWARD_POLISH_MAX_ITER` tight PDIP iterations on the Jacobi system warm-started from the mapped iterate, kept iff converged, finite and `s, z > 0` — PyAutoArray#573's backward rule applied to the *forward* value. | `jax_nnls.solve_nnls` (the `_raw_forward_backward_point` rule) |
| `pdip_raw_cap_{8,12,16,24,32,50,100,200}` | `pdip_raw` at that `max_iter` (the `early_stopping.py` sweep). | `jax_nnls.solve_nnls_primal_raw_forward` |
| `certified` | Certified active-set solve on the Jacobi system, budget / `tau_rel` from `Settings()`, **no** PDIP fallback (an uncertified iterate is scored as-is). Context only — the library never routes MGE systems here. | `jax_active_set.solve_certified` |

**Post-hoc candidates** (added 2026-09-30 *after* the phase-1 deciding run; not pre-registered —
inputs to the phase-2 fix, not a verdict). `accuracy.py --posthoc` runs them, beside the anchors
`fnnls`, `pdip_raw`, `pdip_raw_polish` and `pdip_raw_tol_jaxnnls`, into a separate
`accuracy_posthoc_summary_*` artefact:

| Candidate | What it is | Library primitive(s) |
|-----------|------------|----------------------|
| `pdip_raw_polish_cap20` / `_cap50` | `pdip_raw_polish` with the polish capped at 20 / 50 iterations instead of `RAW_BACKWARD_POLISH_MAX_ITER` (10). | `jax_nnls.solve_nnls` |
| `pdip_raw_tol_jaxnnls_cap100` / `_cap200` | `pdip_raw_tol_jaxnnls` with `max_iter` 100 / 200 instead of 50. | `jax_nnls.solve_nnls(solver_tol=None)` |
| `pdip_raw_tol_1e-4` / `_1e-5` / `_1e-6` | The replace-the-constant family tightened past 1e-3, cap 50. | `jax_nnls.solve_nnls` + `data_scaled_solver_tol` |

## Metrics

All against the stored fnnls reference `x_ref` (definitions in [`_metrics.py`](./_metrics.py)):
`amp_rel_l2`; `amp_rel_max` (worst relative error over columns with `x_ref > 0` — dominated by
barely-active reference columns, so read it with `amp_rel_max_sig`, the same over
`x_ref > 1e-6·max x_ref`); `flux_rel_all` / `flux_rel_source` (signed summed-amplitude error over
all / source columns — an intensity-sum flux proxy; `null` when the source columns are unknown);
`flux_inactive_rel` (Σ x over the columns the reference leaves at `x_ref ≤ 1e-6·max x_ref`,
divided by Σ x_ref — spurious mass on reference-*inactive* columns, which `amp_rel_max*` cannot
see because they score reference-active columns only; added 2026-09-30) and
`active_set_mismatch` (columns whose activity `v > 1e-6·max v` differs from the reference);
`objective_gap` (relative to the reference objective); `kkt_residual_scaled` (the
`scripts/imaging/rectangular/hazards.py` scale-normalised KKT residual: max of primal violation,
dual violation and complementarity). Rows also carry the solver's own `converged` / `iterations`
and the median warm wall-clock (`wall_ms`, 5 repeats after one compile call).

## Accuracy (latest run per corpus)

<!-- BEGIN auto-table:solver-accuracy -->
| Corpus | Candidate | Unconverged | Non-finite | Worst amp_rel_max | Worst amp_rel_max (sig) | Worst abs(flux_rel_source) | Worst abs(flux_inactive_rel) | Worst objective gap | Worst KKT | Median iters | Median wall ms | Version |
|--------|-----------|-------------|------------|-------------------|-------------------------|----------------------------|------------------------------|---------------------|-----------|--------------|----------------|---------|
| `all` | `fnnls` | 0/81 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 8.05e-08 | 0.00e+00 | 8.63e-16 | 9.0 | 0.817 | v2026.10.7.1 |
| `all` | `pdip_jacobi` | 29/81 | 0 | 2.73e+68 | 1.03e+68 | 3.36e+70 | 4.50e+68 | 1.56e+134 | 8.25e+61 | 19.0 | 0.804 | v2026.10.7.1 |
| `all` | `pdip_raw` | 0/81 | 0 | 5.03e+08 | 2.19e-01 | 4.96e-02 | 3.31e-04 | 8.18e-16 | 3.24e-16 | 18.0 | 0.882 | v2026.10.7.1 |
| `all` | `pdip_raw_tol_1e-1` | 0/81 | 0 | 4.28e+11 | 7.15e+01 | 4.30e+01 | 3.02e-01 | 1.75e-11 | 1.80e-13 | 16.0 | 0.634 | v2026.10.7.1 |
| `all` | `pdip_raw_tol_1e-2` | 0/81 | 0 | 6.23e+10 | 2.70e+01 | 1.64e+01 | 1.15e-01 | 2.61e-12 | 2.66e-14 | 18.0 | 0.671 | v2026.10.7.1 |
| `all` | `pdip_raw_tol_1e-3` | 0/81 | 0 | 2.38e+10 | 1.00e+01 | 6.15e+00 | 4.37e-02 | 3.87e-13 | 3.93e-15 | 19.0 | 0.797 | v2026.10.7.1 |
| `all` | `pdip_raw_tol_jaxnnls` | 75/81 | 0 | 4.30e+04 | 2.19e-01 | 7.40e-05 | 6.24e-07 | 8.18e-16 | 2.43e-16 | 50.0 | 1.584 | v2026.10.7.1 |
| `all` | `pdip_raw_polish` | 0/81 | 0 | 5.03e+08 | 2.19e-01 | 4.96e-02 | 3.31e-04 | 8.18e-16 | 3.24e-16 | 23.0 | 0.769 | v2026.10.7.1 |
| `all` | `certified` | 48/81 | 0 | 1.19e+02 | 1.19e+02 | 7.47e-02 | 1.49e-02 | 2.57e-04 | 1.81e-02 | 16.0 | 0.571 | v2026.10.7.1 |
| `all_gpu` | `fnnls` | 0/81 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 8.05e-08 | 0.00e+00 | 8.63e-16 | 9.0 | 0.899 | v2026.10.7.1 |
| `all_gpu` | `pdip_jacobi` | 19/81 | 0 | 7.20e+73 | 7.20e+73 | 1.83e+70 | 4.88e+68 | 1.40e+135 | 1.36e+62 | 19.0 | 3.444 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw` | 0/81 | 0 | 5.03e+08 | 2.19e-01 | 4.96e-02 | 3.31e-04 | 6.04e-16 | 3.24e-16 | 18.0 | 3.958 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw_tol_1e-1` | 0/81 | 0 | 4.28e+11 | 7.15e+01 | 4.30e+01 | 3.02e-01 | 1.75e-11 | 1.80e-13 | 16.0 | 2.841 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw_tol_1e-2` | 0/81 | 0 | 6.23e+10 | 2.70e+01 | 1.64e+01 | 1.15e-01 | 2.61e-12 | 2.66e-14 | 18.0 | 3.130 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw_tol_1e-3` | 0/81 | 0 | 2.38e+10 | 1.00e+01 | 6.15e+00 | 4.37e-02 | 3.87e-13 | 3.93e-15 | 19.0 | 4.095 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw_tol_jaxnnls` | 74/81 | 0 | 4.30e+04 | 2.19e-01 | 7.40e-05 | 6.24e-07 | 6.03e-16 | 2.16e-16 | 50.0 | 10.242 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw_polish` | 0/81 | 0 | 5.03e+08 | 2.19e-01 | 4.96e-02 | 3.31e-04 | 6.04e-16 | 3.24e-16 | 23.0 | 3.994 | v2026.10.7.1 |
| `all_gpu` | `certified` | 48/81 | 0 | 1.19e+02 | 1.19e+02 | 7.47e-02 | 1.49e-02 | 2.57e-04 | 1.81e-02 | 16.0 | 2.111 | v2026.10.7.1 |
| `delaunay_hst` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 4.60e-14 | 2.5 | 375.083 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_jacobi` | 0/8 | 0 | 6.20e-08 | 6.20e-08 | 1.05e-13 | 1.01e-14 | 5.47e-15 | 4.98e-14 | 18.0 | 1509.681 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw` | 0/8 | 0 | 2.36e-08 | 2.36e-08 | 9.97e-14 | 1.17e-14 | 6.13e-15 | 4.22e-14 | 13.0 | 1014.056 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 2.36e-04 | 2.36e-04 | 2.90e-09 | 3.55e-09 | 7.00e-13 | 6.71e-13 | 12.5 | 853.400 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 2.36e-06 | 2.36e-06 | 3.27e-11 | 4.04e-11 | 3.99e-14 | 1.96e-14 | 13.0 | 850.130 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 2.36e-06 | 2.36e-06 | 5.33e-13 | 4.04e-13 | 7.90e-15 | 1.51e-14 | 13.5 | 943.698 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw_tol_jaxnnls` | 0/8 | 0 | 2.36e-08 | 2.36e-08 | 9.23e-14 | 1.17e-14 | 7.90e-15 | 1.51e-14 | 14.0 | 886.766 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw_polish` | 0/8 | 0 | 2.36e-08 | 2.36e-08 | 9.97e-14 | 1.17e-14 | 6.13e-15 | 4.22e-14 | 14.0 | 999.805 | v2026.10.7.1 |
| `delaunay_hst` | `certified` | 0/8 | 0 | 8.10e-11 | 8.10e-11 | 3.61e-14 | 0.00e+00 | 7.20e-15 | 5.34e-14 | 2.0 | 266.887 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 4.60e-14 | 2.5 | 124.239 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_jacobi` | 0/8 | 0 | 6.20e-08 | 6.20e-08 | 5.58e-14 | 1.01e-14 | 8.81e-15 | 3.07e-14 | 18.0 | 29.896 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw` | 0/8 | 0 | 2.36e-08 | 2.36e-08 | 5.88e-14 | 1.17e-14 | 1.20e-14 | 3.51e-14 | 13.0 | 23.581 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 2.36e-04 | 2.36e-04 | 2.90e-09 | 3.55e-09 | 7.00e-13 | 6.71e-13 | 12.5 | 21.203 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 2.36e-06 | 2.36e-06 | 3.27e-11 | 4.04e-11 | 3.77e-14 | 3.45e-14 | 13.0 | 21.972 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 2.36e-06 | 2.36e-06 | 5.23e-13 | 4.04e-13 | 5.36e-15 | 3.45e-14 | 13.5 | 22.769 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw_tol_jaxnnls` | 0/8 | 0 | 2.36e-08 | 2.36e-08 | 3.00e-14 | 1.17e-14 | 5.36e-15 | 3.45e-14 | 14.0 | 23.600 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw_polish` | 0/8 | 0 | 2.36e-08 | 2.36e-08 | 5.88e-14 | 1.17e-14 | 1.20e-14 | 3.51e-14 | 14.0 | 23.630 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `certified` | 0/8 | 0 | 9.33e-11 | 9.33e-11 | 1.43e-13 | 0.00e+00 | 7.51e-15 | 7.64e-14 | 2.0 | 5.740 | v2026.10.7.1 |
| `rectangular_hst` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 6.25e-09 | 0.00e+00 | 4.33e-16 | 45.0 | 733.105 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_jacobi` | 0/8 | 0 | 2.30e+07 | 1.05e+00 | 1.45e-07 | 1.43e-07 | 1.11e-13 | 2.75e-14 | 15.0 | 1410.434 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw` | 0/8 | 0 | 2.20e+07 | 9.56e-01 | 1.18e-07 | 1.15e-07 | 7.47e-14 | 1.78e-14 | 18.0 | 2297.569 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 2.72e+09 | 9.19e+01 | 1.54e-05 | 1.26e-05 | 1.35e-11 | 2.51e-12 | 17.0 | 2089.161 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 1.04e+09 | 3.46e+01 | 5.78e-06 | 4.81e-06 | 1.84e-12 | 3.65e-13 | 18.0 | 1827.932 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 3.96e+08 | 1.27e+01 | 2.14e-06 | 1.83e-06 | 2.58e-13 | 4.24e-14 | 19.0 | 2293.357 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw_tol_jaxnnls` | 0/8 | 0 | 2.20e+07 | 9.56e-01 | 1.18e-07 | 1.15e-07 | 7.47e-14 | 1.78e-14 | 21.0 | 1883.948 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw_polish` | 0/8 | 0 | 2.20e+07 | 9.56e-01 | 1.18e-07 | 1.15e-07 | 7.47e-14 | 1.78e-14 | 21.0 | 1875.534 | v2026.10.7.1 |
| `rectangular_hst` | `certified` | 0/8 | 0 | 1.00e+00 | 3.70e-04 | 3.06e-10 | 6.20e-09 | 5.24e-16 | 4.08e-11 | 8.5 | 1183.011 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 6.25e-09 | 0.00e+00 | 4.33e-16 | 45.0 | 357.934 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_jacobi` | 0/8 | 0 | 2.30e+07 | 1.05e+00 | 1.45e-07 | 1.43e-07 | 1.11e-13 | 2.75e-14 | 15.0 | 23.537 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw` | 0/8 | 0 | 2.20e+07 | 9.56e-01 | 1.18e-07 | 1.15e-07 | 7.47e-14 | 1.78e-14 | 18.0 | 32.527 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 2.72e+09 | 9.19e+01 | 1.54e-05 | 1.26e-05 | 1.35e-11 | 2.51e-12 | 17.0 | 26.331 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 1.04e+09 | 3.46e+01 | 5.78e-06 | 4.81e-06 | 1.84e-12 | 3.65e-13 | 18.0 | 27.813 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 3.96e+08 | 1.27e+01 | 2.14e-06 | 1.83e-06 | 2.58e-13 | 4.24e-14 | 19.0 | 29.277 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw_tol_jaxnnls` | 0/8 | 0 | 2.20e+07 | 9.56e-01 | 1.18e-07 | 1.15e-07 | 7.47e-14 | 1.78e-14 | 21.0 | 32.143 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw_polish` | 0/8 | 0 | 2.20e+07 | 9.56e-01 | 1.18e-07 | 1.15e-07 | 7.47e-14 | 1.78e-14 | 21.0 | 32.376 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `certified` | 0/8 | 0 | 1.00e+00 | 3.70e-04 | 3.06e-10 | 6.20e-09 | 8.98e-16 | 4.08e-11 | 8.5 | 13.187 | v2026.10.7.1 |
| `slam_fixture_571` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 7.68e-08 | 0.00e+00 | 3.24e-16 | 7.0 | 1.066 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_jacobi` | 7/8 | 0 | 2.73e+68 | 1.03e+68 | 3.36e+70 | 4.50e+68 | 1.56e+134 | 8.25e+61 | 50.0 | 2.482 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw` | 0/8 | 0 | 6.91e+04 | 4.95e-06 | 4.93e-05 | 3.44e-07 | 3.60e-16 | 2.16e-16 | 18.0 | 1.366 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 5.87e+07 | 9.08e+00 | 1.69e-02 | 1.45e-04 | 2.73e-12 | 3.60e-15 | 17.0 | 2.374 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 2.24e+07 | 3.18e+00 | 6.44e-03 | 5.36e-05 | 4.14e-13 | 5.24e-16 | 18.0 | 1.378 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 3.26e+06 | 9.54e-01 | 2.46e-03 | 1.52e-05 | 3.55e-14 | 4.20e-17 | 19.0 | 1.324 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_jaxnnls` | 8/8 | 0 | 2.09e-11 | 2.09e-11 | 3.05e-13 | 7.68e-08 | 6.01e-16 | 1.08e-16 | 50.0 | 2.471 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_polish` | 0/8 | 0 | 6.91e+04 | 4.95e-06 | 4.93e-05 | 3.44e-07 | 3.60e-16 | 2.16e-16 | 23.0 | 2.693 | v2026.8.17.1 |
| `slam_fixture_571` | `certified` | 5/8 | 0 | 8.76e+01 | 8.76e+01 | 1.12e-02 | 3.53e-03 | 1.61e-05 | 1.81e-02 | 16.0 | 1.008 | v2026.8.17.1 |
| `slam_mixed_hst` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 6.82e-08 | 0.00e+00 | 8.59e-16 | 18.0 | 591.318 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_jacobi` | 0/8 | 0 | 7.34e-08 | 7.34e-08 | 1.01e-13 | 6.82e-08 | 7.18e-16 | 4.91e-16 | 18.0 | 1632.813 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw` | 0/8 | 0 | 6.69e-08 | 6.69e-08 | 4.69e-13 | 6.82e-08 | 1.76e-15 | 9.82e-16 | 16.5 | 2124.239 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 4.50e-01 | 4.50e-01 | 7.47e-07 | 7.55e-08 | 4.42e-11 | 6.16e-13 | 15.5 | 1904.345 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 4.86e-02 | 4.86e-02 | 3.98e-08 | 6.83e-08 | 3.28e-12 | 5.73e-15 | 16.5 | 2331.484 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 6.71e-04 | 6.71e-04 | 4.39e-10 | 6.82e-08 | 3.24e-14 | 4.30e-16 | 17.5 | 2241.597 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw_tol_jaxnnls` | 2/8 | 0 | 6.68e-08 | 6.68e-08 | 4.86e-13 | 6.82e-08 | 2.70e-15 | 4.91e-16 | 19.0 | 1919.127 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw_polish` | 0/8 | 0 | 6.69e-08 | 6.69e-08 | 4.69e-13 | 6.82e-08 | 1.76e-15 | 9.82e-16 | 18.5 | 2401.458 | v2026.10.7.1 |
| `slam_mixed_hst` | `certified` | 1/8 | 0 | 3.07e+01 | 3.07e+01 | 7.83e-03 | 2.49e-04 | 7.06e-05 | 3.26e-02 | 13.0 | 1647.016 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 6.82e-08 | 0.00e+00 | 8.59e-16 | 18.0 | 284.574 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_jacobi` | 0/8 | 0 | 7.32e-08 | 7.32e-08 | 1.20e-13 | 6.82e-08 | 1.35e-15 | 4.91e-16 | 18.0 | 30.967 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw` | 0/8 | 0 | 6.70e-08 | 6.70e-08 | 3.99e-13 | 6.82e-08 | 2.43e-15 | 4.91e-16 | 16.5 | 31.961 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 4.50e-01 | 4.50e-01 | 7.47e-07 | 7.55e-08 | 4.42e-11 | 6.16e-13 | 15.5 | 26.928 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 4.86e-02 | 4.86e-02 | 3.98e-08 | 6.83e-08 | 3.28e-12 | 5.73e-15 | 16.5 | 28.529 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 6.71e-04 | 6.71e-04 | 4.39e-10 | 6.82e-08 | 3.26e-14 | 2.15e-16 | 17.5 | 30.067 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw_tol_jaxnnls` | 1/8 | 0 | 6.69e-08 | 6.69e-08 | 3.97e-13 | 6.82e-08 | 8.60e-16 | 4.30e-16 | 18.5 | 31.793 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw_polish` | 0/8 | 0 | 6.70e-08 | 6.70e-08 | 3.99e-13 | 6.82e-08 | 2.43e-15 | 4.91e-16 | 18.5 | 32.071 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `certified` | 1/8 | 0 | 3.07e+01 | 3.07e+01 | 7.83e-03 | 2.49e-04 | 7.06e-05 | 3.26e-02 | 13.0 | 20.735 | v2026.10.7.1 |
<!-- END auto-table:solver-accuracy -->

## Early stopping (latest run per corpus)

<!-- BEGIN auto-table:solver-early-stopping -->
| Corpus | Cap | Converged | Worst amp_rel_max | Worst amp_rel_max (sig) | Median amp_rel_max | Worst abs(flux_rel_source) | Worst abs(flux_inactive_rel) | Worst KKT | Median wall ms | Version |
|--------|-----|-----------|-------------------|-------------------------|--------------------|----------------------------|------------------------------|-----------|----------------|---------|
| `all` | 8 | 0/81 | 2.02e+14 | 3.43e+04 | 2.89e+01 | 6.18e+04 | 4.47e+02 | 3.19e-03 | 1.309 | v2026.8.17.1 |
| `all` | 12 | 1/81 | 3.48e+13 | 8.96e+03 | 2.82e-01 | 3.14e+04 | 2.23e+02 | 5.71e-04 | 1.603 | v2026.8.17.1 |
| `all` | 16 | 15/81 | 2.97e+13 | 1.49e+03 | 6.65e-11 | 2.69e+04 | 1.90e+02 | 2.47e-05 | 1.890 | v2026.8.17.1 |
| `all` | 24 | 81/81 | 5.03e+08 | 2.19e-01 | 5.29e-11 | 4.96e-02 | 3.31e-04 | 3.24e-16 | 2.044 | v2026.8.17.1 |
| `all` | 32 | 81/81 | 5.03e+08 | 2.19e-01 | 5.29e-11 | 4.96e-02 | 3.31e-04 | 3.24e-16 | 2.249 | v2026.8.17.1 |
| `all` | 50 | 81/81 | 5.03e+08 | 2.19e-01 | 5.29e-11 | 4.96e-02 | 3.31e-04 | 3.24e-16 | 3.875 | v2026.8.17.1 |
| `all` | 100 | 81/81 | 5.03e+08 | 2.19e-01 | 5.29e-11 | 4.96e-02 | 3.31e-04 | 3.24e-16 | 1.975 | v2026.8.17.1 |
| `all` | 200 | 81/81 | 5.03e+08 | 2.19e-01 | 5.29e-11 | 4.96e-02 | 3.31e-04 | 3.24e-16 | 1.776 | v2026.8.17.1 |
| `slam_fixture_571` | 8 | 0/8 | 3.00e+10 | 5.85e+03 | 5.14e+05 | 2.56e+01 | 1.51e-01 | 3.02e-04 | 1.951 | v2026.8.17.1 |
| `slam_fixture_571` | 12 | 0/8 | 2.76e+09 | 1.18e+03 | 1.24e+05 | 6.71e+00 | 3.39e-02 | 4.14e-07 | 2.409 | v2026.8.17.1 |
| `slam_fixture_571` | 16 | 1/8 | 6.91e+04 | 4.95e-06 | 1.04e+00 | 4.93e-05 | 3.44e-07 | 1.08e-16 | 2.490 | v2026.8.17.1 |
| `slam_fixture_571` | 24 | 8/8 | 6.91e+04 | 4.95e-06 | 1.04e+00 | 4.93e-05 | 3.44e-07 | 2.16e-16 | 2.251 | v2026.8.17.1 |
| `slam_fixture_571` | 32 | 8/8 | 6.91e+04 | 4.95e-06 | 1.04e+00 | 4.93e-05 | 3.44e-07 | 2.16e-16 | 2.040 | v2026.8.17.1 |
| `slam_fixture_571` | 50 | 8/8 | 6.91e+04 | 4.95e-06 | 1.04e+00 | 4.93e-05 | 3.44e-07 | 2.16e-16 | 2.162 | v2026.8.17.1 |
| `slam_fixture_571` | 100 | 8/8 | 6.91e+04 | 4.95e-06 | 1.04e+00 | 4.93e-05 | 3.44e-07 | 2.16e-16 | 2.238 | v2026.8.17.1 |
| `slam_fixture_571` | 200 | 8/8 | 6.91e+04 | 4.95e-06 | 1.04e+00 | 4.93e-05 | 3.44e-07 | 2.16e-16 | 2.562 | v2026.8.17.1 |
<!-- END auto-table:solver-early-stopping -->

## Post-hoc exploration (latest run per corpus)

**Post-hoc, not pre-registered — inputs to the phase-2 fix, not a verdict.** The pre-registered
table is "Accuracy" above; the verdict it gave is in the
[ledger](../../../results/notes/linear_solver_accuracy_2026_09.md).

<!-- BEGIN auto-table:solver-accuracy-posthoc -->
| Corpus | Candidate | Unconverged | Worst abs(flux_inactive_rel) | Worst abs(flux_rel_source) | Worst amp_rel_max (sig) | Max active-set mismatch | Median iters | Max iters | Median wall ms | Version |
|--------|-----------|-------------|------------------------------|----------------------------|-------------------------|-------------------------|--------------|-----------|----------------|---------|
| `all` | `fnnls` | 0/81 | 8.05e-08 | 0.00e+00 | 0.00e+00 | 0 | 9.0 | 20 | 2.616 | v2026.8.17.1 |
| `all` | `pdip_raw` | 0/81 | 3.31e-04 | 4.96e-02 | 2.19e-01 | 23 | 18.0 | 24 | 1.542 | v2026.8.17.1 |
| `all` | `pdip_raw_polish` | 0/81 | 3.31e-04 | 4.96e-02 | 2.19e-01 | 23 | 23.0 | 30 | 1.860 | v2026.8.17.1 |
| `all` | `pdip_raw_tol_jaxnnls` | 75/81 | 6.24e-07 | 7.40e-05 | 2.19e-01 | 4 | 50.0 | 50 | 3.176 | v2026.8.17.1 |
| `all` | `pdip_raw_polish_cap20` | 0/81 | 3.31e-04 | 4.96e-02 | 2.19e-01 | 23 | 23.0 | 30 | 2.199 | v2026.8.17.1 |
| `all` | `pdip_raw_polish_cap50` | 0/81 | 3.31e-04 | 4.96e-02 | 2.19e-01 | 23 | 23.0 | 30 | 2.038 | v2026.8.17.1 |
| `all` | `pdip_raw_tol_jaxnnls_cap100` | 74/81 | 6.24e-07 | 7.40e-05 | 2.19e-01 | 4 | 100.0 | 100 | 5.826 | v2026.8.17.1 |
| `all` | `pdip_raw_tol_jaxnnls_cap200` | 74/81 | 6.24e-07 | 7.40e-05 | 2.19e-01 | 4 | 200.0 | 200 | 11.559 | v2026.8.17.1 |
| `all` | `pdip_raw_tol_1e-4` | 0/81 | 1.65e-02 | 2.30e+00 | 3.52e+00 | 26 | 20.0 | 26 | 1.889 | v2026.8.17.1 |
| `all` | `pdip_raw_tol_1e-5` | 0/81 | 2.36e-03 | 3.37e-01 | 2.27e-01 | 25 | 21.0 | 28 | 1.605 | v2026.8.17.1 |
| `all` | `pdip_raw_tol_1e-6` | 78/81 | 8.05e-08 | 7.49e-10 | 1.25e-10 | 0 | 50.0 | 50 | 3.681 | v2026.8.17.1 |
<!-- END auto-table:solver-accuracy-posthoc -->

## Timing (latest run per corpus)

Batched per-evaluation cost from [`timing.py`](./timing.py) (phase 3b). The SLaM family pools
`slam_fixture_571` + `slam48_hst` (56 distinct systems; a batch of B takes the first B in
manifest order, so lanes diverge); `euclid_vis_lp` has one system and is **tiled** B times, so
its lanes are identical. Per-eval = batched-call wall / B: min and median over 7 interleaved
rounds, inputs already on the device. *Compile s* is the first call of that config (trace +
compile + one run), never part of a steady figure; `(cached)` marks a config whose batch shape
an earlier family already compiled. A vmapped `while_loop` runs to the batch's slowest lane, so
the wall follows *max iters*, not the median. `fnnls` has no batched form: its rows are a host
Python loop over the lanes (context only). A timing is not an admissibility result — see the
[ledger](../../../results/notes/linear_solver_accuracy_2026_09.md).

<!-- BEGIN auto-table:solver-timing -->
| Corpus | Candidate | Batch family | B | Per-eval min ms | Per-eval median ms | Batch wall min ms | Compile s | Median / max iters | Unconverged | Worst abs(flux_inactive_rel) | Version |
|--------|-----------|--------------|---|-----------------|--------------------|-------------------|-----------|--------------------|-------------|------------------------------|---------|
| `all` | `pdip_raw` | slam_fixture_571+slam48_hst | 1 | 1.3532 | 1.4973 | 1.353 | 1.07 | 18 / 18 | 0/1 | 2.28e-07 | v2026.10.7.1 |
| `all` | `pdip_raw` | slam_fixture_571+slam48_hst | 16 | 0.8809 | 0.9604 | 14.094 | 1.07 | 18 / 19 | 0/16 | 3.44e-07 | v2026.10.7.1 |
| `all` | `pdip_raw` | slam_fixture_571+slam48_hst | 50 | 0.6821 | 0.7832 | 34.106 | 0.97 | 17.5 / 19 | 0/50 | 4.32e-07 | v2026.10.7.1 |
| `all` | `pdip_raw` | euclid_vis_lp (tiled) | 1 | 1.6109 | 1.7856 | 1.611 | 0.00 (cached) | 24 / 24 | 0/1 | 3.31e-04 | v2026.10.7.1 |
| `all` | `pdip_raw` | euclid_vis_lp (tiled) | 16 | 0.9688 | 1.1692 | 15.500 | 0.02 (cached) | 24 / 24 | 0/16 | 3.31e-04 | v2026.10.7.1 |
| `all` | `pdip_raw` | euclid_vis_lp (tiled) | 50 | 0.7980 | 0.9195 | 39.899 | 0.05 (cached) | 24 / 24 | 0/50 | 3.31e-04 | v2026.10.7.1 |
| `all` | `pdip_jacobi` | slam_fixture_571+slam48_hst | 1 | 2.4881 | 2.5993 | 2.488 | 0.50 | 50 / 50 | 1/1 | 1.85e+01 | v2026.10.7.1 |
| `all` | `pdip_jacobi` | slam_fixture_571+slam48_hst | 16 | 1.6299 | 1.8678 | 26.079 | 0.69 | 50 / 50 | 9/16 | 4.50e+68 | v2026.10.7.1 |
| `all` | `pdip_jacobi` | slam_fixture_571+slam48_hst | 50 | 1.3960 | 1.5136 | 69.801 | 0.67 | 19 / 50 | 19/50 | 4.50e+68 | v2026.10.7.1 |
| `all` | `pdip_jacobi` | euclid_vis_lp (tiled) | 1 | 1.1509 | 1.1850 | 1.151 | 0.00 (cached) | 19 / 19 | 0/1 | 4.62e-04 | v2026.10.7.1 |
| `all` | `pdip_jacobi` | euclid_vis_lp (tiled) | 16 | 0.6991 | 0.7358 | 11.185 | 0.01 (cached) | 19 / 19 | 0/16 | 4.62e-04 | v2026.10.7.1 |
| `all` | `pdip_jacobi` | euclid_vis_lp (tiled) | 50 | 0.5307 | 0.5598 | 26.537 | 0.03 (cached) | 19 / 19 | 0/50 | 4.62e-04 | v2026.10.7.1 |
| `all` | `fnnls` | slam_fixture_571+slam48_hst (host loop) | 1 | 1.6969 | 1.8096 | 1.697 | 0.43 | 11 / 11 | 0/1 | 4.17e-12 | v2026.10.7.1 |
| `all` | `fnnls` | slam_fixture_571+slam48_hst (host loop) | 16 | 1.3121 | 1.4275 | 20.994 | 0.03 | 8.5 / 13 | 0/16 | 8.05e-08 | v2026.10.7.1 |
| `all` | `fnnls` | slam_fixture_571+slam48_hst (host loop) | 50 | 1.3057 | 1.4503 | 65.284 | 0.07 | 8.5 / 20 | 0/50 | 8.05e-08 | v2026.10.7.1 |
| `all` | `fnnls` | euclid_vis_lp (tiled) (host loop) | 1 | 1.3956 | 1.5349 | 1.396 | 0.00 | 8 / 8 | 0/1 | 1.26e-09 | v2026.10.7.1 |
| `all` | `fnnls` | euclid_vis_lp (tiled) (host loop) | 16 | 1.3869 | 1.5370 | 22.191 | 0.02 | 8 / 8 | 0/16 | 1.26e-09 | v2026.10.7.1 |
| `all` | `fnnls` | euclid_vis_lp (tiled) (host loop) | 50 | 1.3791 | 1.5259 | 68.953 | 0.10 | 8 / 8 | 0/50 | 1.26e-09 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw` | slam_fixture_571+slam48_hst | 1 | 3.9475 | 4.0301 | 3.947 | 1.17 | 18 / 18 | 0/1 | 2.28e-07 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw` | slam_fixture_571+slam48_hst | 16 | 0.5813 | 0.5858 | 9.301 | 0.75 | 18 / 19 | 0/16 | 3.44e-07 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw` | slam_fixture_571+slam48_hst | 50 | 0.1900 | 0.1919 | 9.501 | 0.59 | 17.5 / 19 | 0/50 | 4.32e-07 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw` | euclid_vis_lp (tiled) | 1 | 4.9423 | 4.9585 | 4.942 | 0.01 (cached) | 24 / 24 | 0/1 | 3.31e-04 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw` | euclid_vis_lp (tiled) | 16 | 0.6885 | 0.6924 | 11.016 | 0.01 (cached) | 24 / 24 | 0/16 | 3.31e-04 | v2026.10.7.1 |
| `all_gpu` | `pdip_raw` | euclid_vis_lp (tiled) | 50 | 0.2259 | 0.2269 | 11.297 | 0.01 (cached) | 24 / 24 | 0/50 | 3.31e-04 | v2026.10.7.1 |
| `all_gpu` | `pdip_jacobi` | slam_fixture_571+slam48_hst | 1 | 6.7147 | 6.7954 | 6.715 | 0.32 | 41 / 41 | 0/1 | 1.60e-07 | v2026.10.7.1 |
| `all_gpu` | `pdip_jacobi` | slam_fixture_571+slam48_hst | 16 | 1.1494 | 1.1540 | 18.390 | 0.36 | 19 / 50 | 1/16 | 2.90e+01 | v2026.10.7.1 |
| `all_gpu` | `pdip_jacobi` | slam_fixture_571+slam48_hst | 50 | 0.3690 | 0.3705 | 18.452 | 0.38 | 19 / 50 | 4/50 | 2.90e+01 | v2026.10.7.1 |
| `all_gpu` | `pdip_jacobi` | euclid_vis_lp (tiled) | 1 | 3.3114 | 3.3627 | 3.311 | 0.00 (cached) | 19 / 19 | 0/1 | 4.62e-04 | v2026.10.7.1 |
| `all_gpu` | `pdip_jacobi` | euclid_vis_lp (tiled) | 16 | 0.4587 | 0.4604 | 7.339 | 0.01 (cached) | 19 / 19 | 0/16 | 4.62e-04 | v2026.10.7.1 |
| `all_gpu` | `pdip_jacobi` | euclid_vis_lp (tiled) | 50 | 0.1470 | 0.1475 | 7.352 | 0.01 (cached) | 19 / 19 | 0/50 | 4.62e-04 | v2026.10.7.1 |
| `all_gpu` | `fnnls` | slam_fixture_571+slam48_hst (host loop) | 1 | 1.0209 | 1.0395 | 1.021 | 5.02 | 11 / 11 | 0/1 | 4.17e-12 | v2026.10.7.1 |
| `all_gpu` | `fnnls` | slam_fixture_571+slam48_hst (host loop) | 16 | 0.8667 | 0.8774 | 13.868 | 0.47 | 8.5 / 13 | 0/16 | 8.05e-08 | v2026.10.7.1 |
| `all_gpu` | `fnnls` | slam_fixture_571+slam48_hst (host loop) | 50 | 0.8781 | 0.8848 | 43.904 | 0.05 | 8.5 / 20 | 0/50 | 8.05e-08 | v2026.10.7.1 |
| `all_gpu` | `fnnls` | euclid_vis_lp (tiled) (host loop) | 1 | 0.9701 | 0.9784 | 0.970 | 0.00 | 8 / 8 | 0/1 | 1.26e-09 | v2026.10.7.1 |
| `all_gpu` | `fnnls` | euclid_vis_lp (tiled) (host loop) | 16 | 0.9605 | 0.9668 | 15.367 | 0.02 | 8 / 8 | 0/16 | 1.26e-09 | v2026.10.7.1 |
| `all_gpu` | `fnnls` | euclid_vis_lp (tiled) (host loop) | 50 | 0.9612 | 0.9656 | 48.060 | 0.05 | 8 / 8 | 0/50 | 1.26e-09 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw` | delaunay_hst | 1 | 969.1200 | 1034.9559 | 969.120 | 1.85 | 13 / 13 | 0/1 | 1.17e-14 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw` | delaunay_hst | 8 | 659.1268 | 708.7968 | 5273.014 | 5.54 | 13 / 18 | 0/8 | 1.17e-14 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_raw` | delaunay_hst (tiled) | 16 | 752.3292 | 788.1731 | 12037.267 | 12.90 | 13 / 18 | 0/16 | 1.17e-14 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_jacobi` | delaunay_hst | 1 | 1235.0293 | 1497.9067 | 1235.029 | 1.90 | 18 / 18 | 0/1 | 1.02e-15 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_jacobi` | delaunay_hst | 8 | 652.5491 | 753.8698 | 5220.393 | 6.09 | 18 / 19 | 0/8 | 1.01e-14 | v2026.10.7.1 |
| `delaunay_hst` | `pdip_jacobi` | delaunay_hst (tiled) | 16 | 701.3433 | 703.2906 | 11221.493 | 11.40 | 18 / 19 | 0/16 | 1.01e-14 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw` | delaunay_hst | 1 | 23.7149 | 23.8151 | 23.715 | 0.77 | 13 / 13 | 0/1 | 1.17e-14 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw` | delaunay_hst | 8 | 30.8918 | 30.9001 | 247.134 | 1.07 | 13 / 18 | 0/8 | 1.17e-14 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_raw` | delaunay_hst (tiled) | 16 | 18.8253 | 18.8375 | 301.205 | 0.92 | 13 / 18 | 0/16 | 1.17e-14 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_jacobi` | delaunay_hst | 1 | 29.7595 | 29.9257 | 29.759 | 0.37 | 18 / 18 | 0/1 | 1.02e-15 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_jacobi` | delaunay_hst | 8 | 29.4329 | 29.4356 | 235.463 | 0.59 | 18 / 19 | 0/8 | 1.01e-14 | v2026.10.7.1 |
| `delaunay_hst_gpu` | `pdip_jacobi` | delaunay_hst (tiled) | 16 | 17.9334 | 17.9438 | 286.935 | 0.65 | 18 / 19 | 0/16 | 1.01e-14 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw` | rectangular_hst | 1 | 2251.0810 | 2800.3106 | 2251.081 | 5.31 | 18 / 18 | 0/1 | 9.97e-08 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw` | rectangular_hst | 8 | 1634.3425 | 1922.4457 | 13074.740 | 19.76 | 18 / 24 | 0/8 | 1.15e-07 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_raw` | rectangular_hst (tiled) | 16 | 1709.4489 | 1857.2531 | 27351.183 | 43.19 | 18 / 24 | 0/16 | 1.15e-07 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_jacobi` | rectangular_hst | 1 | 1594.0598 | 1777.9394 | 1594.060 | 3.82 | 15 / 15 | 0/1 | 1.22e-07 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_jacobi` | rectangular_hst | 8 | 921.5699 | 1318.4088 | 7372.559 | 12.91 | 15 / 17 | 0/8 | 1.43e-07 | v2026.10.7.1 |
| `rectangular_hst` | `pdip_jacobi` | rectangular_hst (tiled) | 16 | 996.2502 | 1210.1622 | 15940.004 | 28.28 | 15 / 17 | 0/16 | 1.43e-07 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw` | rectangular_hst | 1 | 32.4293 | 32.5622 | 32.429 | 0.82 | 18 / 18 | 0/1 | 9.97e-08 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw` | rectangular_hst | 8 | 37.8297 | 37.8418 | 302.638 | 1.01 | 18 / 24 | 0/8 | 1.15e-07 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_raw` | rectangular_hst (tiled) | 16 | 22.8381 | 22.8495 | 365.410 | 1.00 | 18 / 24 | 0/16 | 1.15e-07 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_jacobi` | rectangular_hst | 1 | 23.6531 | 23.6719 | 23.653 | 0.38 | 15 / 15 | 0/1 | 1.22e-07 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_jacobi` | rectangular_hst | 8 | 23.3749 | 23.3774 | 186.999 | 0.61 | 15 / 17 | 0/8 | 1.43e-07 | v2026.10.7.1 |
| `rectangular_hst_gpu` | `pdip_jacobi` | rectangular_hst (tiled) | 16 | 14.0895 | 14.0917 | 225.431 | 0.61 | 15 / 17 | 0/16 | 1.43e-07 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw` | slam_mixed_hst | 1 | 2240.2285 | 2401.4493 | 2240.228 | 3.85 | 16 / 16 | 0/1 | 3.70e-16 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw` | slam_mixed_hst | 8 | 1330.6747 | 1330.9647 | 10645.398 | 11.95 | 16.5 / 19 | 0/8 | 6.82e-08 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_raw` | slam_mixed_hst (tiled) | 16 | 1303.6391 | 1465.4892 | 20858.225 | 20.29 | 16.5 / 19 | 0/16 | 6.82e-08 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_jacobi` | slam_mixed_hst | 1 | 1995.7634 | 2154.0752 | 1995.763 | 2.57 | 19 / 19 | 0/1 | 6.25e-17 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_jacobi` | slam_mixed_hst | 8 | 1122.8004 | 1159.2826 | 8982.403 | 8.59 | 18 / 20 | 0/8 | 6.82e-08 | v2026.10.7.1 |
| `slam_mixed_hst` | `pdip_jacobi` | slam_mixed_hst (tiled) | 16 | 1175.5738 | 1209.2472 | 18809.181 | 18.76 | 18 / 20 | 0/16 | 6.82e-08 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw` | slam_mixed_hst | 1 | 31.1473 | 31.3423 | 31.147 | 0.80 | 16 / 16 | 0/1 | 3.70e-16 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw` | slam_mixed_hst | 8 | 37.1343 | 37.1490 | 297.074 | 0.99 | 16.5 / 19 | 0/8 | 6.82e-08 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_raw` | slam_mixed_hst (tiled) | 16 | 22.7089 | 22.7140 | 363.342 | 0.99 | 16.5 / 19 | 0/16 | 6.82e-08 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_jacobi` | slam_mixed_hst | 1 | 32.5388 | 32.6974 | 32.539 | 0.41 | 19 / 19 | 0/1 | 6.25e-17 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_jacobi` | slam_mixed_hst | 8 | 32.4284 | 32.4373 | 259.427 | 0.69 | 18 / 20 | 0/8 | 6.82e-08 | v2026.10.7.1 |
| `slam_mixed_hst_gpu` | `pdip_jacobi` | slam_mixed_hst (tiled) | 16 | 19.8463 | 19.8541 | 317.541 | 0.68 | 18 / 20 | 0/16 | 6.82e-08 | v2026.10.7.1 |
<!-- END auto-table:solver-timing -->

## Euclid latent by candidate (post-hoc)

`total_source_flux` of the euclid latent jit test's system from each candidate's reconstruction,
computed by the euclid pipeline's own latent code ([`euclid_latent.py`](./euclid_latent.py)). The
test passes at rel 1e-3 against the eager NumPy value.

<!-- BEGIN auto-table:solver-euclid-latent -->
_System `euclid_vis_lp/euclid_vis_lp_k0`; eager NumPy `total_source_flux` 3.3198794874035915, released jit 3.511093374207152, library-main jit 3.320127603567922; validation passed (x_ref rel -3.26e-08, pdip_raw rel -2.00e-08); the euclid test's rtol is 0.001._

| Candidate | Post-hoc | total_source_flux | rel vs eager | Within test rtol | Converged | Iterations | Version |
|-----------|----------|-------------------|--------------|------------------|-----------|------------|---------|
| `fnnls` | no | 3.3198794 | -3.26e-08 | yes | True | 8 | v2026.8.17.1 |
| `pdip_jacobi` | no | 3.3202250 | +1.04e-04 | yes | True | 19 | v2026.8.17.1 |
| `pdip_raw` | no | 3.3201275 | +7.47e-05 | yes | True | 24 | v2026.8.17.1 |
| `pdip_raw_tol_1e-1` | no | 3.8766845 | +1.68e-01 | no | True | 23 | v2026.8.17.1 |
| `pdip_raw_tol_1e-2` | no | 3.5110933 | +5.76e-02 | no | True | 24 | v2026.8.17.1 |
| `pdip_raw_tol_1e-3` | no | 3.3756725 | +1.68e-02 | no | True | 25 | v2026.8.17.1 |
| `pdip_raw_tol_jaxnnls` | no | 3.3198794 | -3.26e-08 | yes | False | 50 | v2026.8.17.1 |
| `pdip_raw_polish` | no | 3.3201275 | +7.47e-05 | yes | True | 30 | v2026.8.17.1 |
| `certified` | no | 3.3198794 | -3.26e-08 | yes | True | 8 | v2026.8.17.1 |
| `pdip_raw_polish_cap20` | yes | 3.3201275 | +7.47e-05 | yes | True | 30 | v2026.8.17.1 |
| `pdip_raw_polish_cap50` | yes | 3.3201275 | +7.47e-05 | yes | True | 30 | v2026.8.17.1 |
| `pdip_raw_tol_jaxnnls_cap100` | yes | 3.3198794 | -3.26e-08 | yes | True | 51 | v2026.8.17.1 |
| `pdip_raw_tol_jaxnnls_cap200` | yes | 3.3198794 | -3.26e-08 | yes | True | 51 | v2026.8.17.1 |
| `pdip_raw_tol_1e-4` | yes | 3.3333942 | +4.07e-03 | no | True | 26 | v2026.8.17.1 |
| `pdip_raw_tol_1e-5` | yes | 3.3215684 | +5.09e-04 | yes | True | 28 | v2026.8.17.1 |
| `pdip_raw_tol_1e-6` | yes | 3.3198794 | -3.26e-08 | yes | False | 50 | v2026.8.17.1 |
<!-- END auto-table:solver-euclid-latent -->

## How to add a system

1. Add a `--source` runner to [`capture.py`](./capture.py), or capture `(curvature_reg_matrix, data_vector)` as the JAX likelihood hands them to
   `reconstruction_positive_only_from` — wrap that function the way
   [`mge_nnls_capture.py`](../../imaging/mge/hazards_nnls_capture.py) does (`_recording_solver`).
2. Call `_corpus.add_group(name, systems, source)` with one dict per system (`name`, `Q`, `q`,
   and whatever of `model`, `source_column_index_list`, `no_regularization_index_list`,
   `library_versions` you actually know — unknown fields stay `null`, never a guess) and
   `source = {"script", "args", "git_sha"}`. The fnnls reference is computed and stored for you.
3. New captures go in a **new group**; never rewrite an existing group's systems.
4. Re-run both cells and `python scripts/misc/tooling/build_readme.py`.

## How to add a candidate

1. Write a builder in [`_solvers.py`](./_solvers.py) from library calls only (no transcribed
   internals), wrap it with `_jax_candidate(...)` in `_build_registry()`, and name the composed
   library function in its description.
2. Add its row to the Candidates table above.
3. Re-run `accuracy.py` (it picks up every non-cap candidate by default).

## How to re-run on a release

From the repo root, with the release's libraries on `PYTHONPATH` (CPU, fp64):

```bash
python scripts/lens/solver/accuracy.py
python scripts/lens/solver/early_stopping.py
python scripts/lens/solver/accuracy.py --posthoc      # exploratory set (separate artefact)
python scripts/lens/solver/euclid_latent.py           # needs the euclid pipeline checkout
python scripts/lens/solver/timing.py                  # batched timing (add --device gpu on an A100)
python scripts/lens/solver/batched_divergence.py      # batched-vs-unbatched probe (--device gpu [--deterministic])
python scripts/misc/tooling/build_readme.py
```

Each writes a new `_v<version>` artefact pair beside the previous release's, so the trend is
the file list. `--groups`, `--candidates`, `--device gpu` (writes a `_gpu` artefact) and
`--output-dir` are available on every cell; `timing.py` adds `--batch-sizes` and `--rounds`.

## Related

- Campaign page: [`wiki/campaigns/linear_solver_accuracy.md`](../../../wiki/campaigns/linear_solver_accuracy.md)
- Solver ledger: [`results/notes/nnls_solver_ledger.md`](../../../results/notes/nnls_solver_ledger.md)
- Certified solver campaign: [`wiki/campaigns/certified_positive_solver.md`](../../../wiki/campaigns/certified_positive_solver.md)
- NNLS warm-start memo: [`results/nnls_warm_start/nnls_warm_start_memo.md`](../../../results/nnls_warm_start/nnls_warm_start_memo.md)
