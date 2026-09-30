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
| [`accuracy.py`](./accuracy.py) | Every candidate on every system → `results/lens/solver/accuracy_summary_<corpus>_v<version>.{json,png}`. |
| [`early_stopping.py`](./early_stopping.py) | Released raw PDIP at each iteration cap → `results/lens/solver/early_stopping_summary_<corpus>_v<version>.{json,png}`. |

The corpus itself lives in `results/lens/solver/corpus/`: `manifest.json` (per-system metadata)
plus one compressed `<group>.npz` per group holding `Q_<name>`, `q_<name>` and `x_ref_<name>` —
the fnnls reference, computed once when the group is added and stored, so a study never
re-derives its truth.

## Corpus

<!-- BEGIN auto-table:solver-corpus -->
| Group | Systems | n | cond(Q) | max abs(q) | Source columns | Captured by | Model |
|-------|---------|---|---------|------------|----------------|-------------|-------|
| `slam_fixture_571` | 8 | 60 | 9.69e+10 – 9.73e+10 | 2.16e+06 – 2.16e+06 | 20 | `scripts/imaging/hazards/mge_nnls_capture.py` | SLaM source_lp[1]: lens 2x20 MGE (sigma_min=pixel_scale/10) + source 20 MGE, free Isothermal + ExternalShear |
<!-- END auto-table:solver-corpus -->

## Candidates

Hand-written from [`_solvers.py`](./_solvers.py); the library function each one composes is in
its registry entry. All return the reconstruction in raw coordinates exactly as
`inversion_util.reconstruction_positive_only_from` does.

| Candidate | What it is | Library primitive(s) |
|-----------|------------|----------------------|
| `fnnls` | **Reference.** NumPy fnnls with incremental Cholesky from the dense-sign start — the NumPy production path (memo off). Defines `x_ref`; its row is a self-check. | `autoarray.util.fnnls.fnnls_cholesky` |
| `pdip_jacobi` | Jacobi-preconditioned PDIP at jaxnnls's own tolerance, cap 50 — the library `"jacobi"` mode (default with a Mapper; the MGE default before PyAutoArray#571). | `jax_nnls.solve_nnls_primal_with_status` |
| `pdip_raw` | **Released default for MGE (linear-object-only) inversions**: forward PDIP on the raw system at `data_scaled_solver_tol(q)`, cap 50, returned as `y * D`. Extras record the backward-pass polish / relaxed-KKT status. | `jax_nnls.solve_nnls_primal_raw_forward` (+ `raw_forward_backward_status`) |
| `pdip_raw_tol_1e-1` / `_1e-2` / `_1e-3` | Raw forward PDIP with the data-scaled tolerance's factor set to 1e-1 / 1e-2 / 1e-3 in place of `DATA_SCALED_TOL_FACTOR` (1e-2 = released). | `jax_nnls.solve_nnls` + `data_scaled_solver_tol` |
| `pdip_raw_tol_jaxnnls` | Raw forward PDIP at jaxnnls's absolute tolerance `min(n·eps·5e3, 1e-2)` — unreachable on an unscaled system, so it runs to the cap. | `jax_nnls.solve_nnls(solver_tol=None)` |
| `pdip_raw_polish` | Raw forward, then ≤ `RAW_BACKWARD_POLISH_MAX_ITER` tight PDIP iterations on the Jacobi system warm-started from the mapped iterate, kept iff converged, finite and `s, z > 0` — PyAutoArray#573's backward rule applied to the *forward* value. | `jax_nnls.solve_nnls` (the `_raw_forward_backward_point` rule) |
| `pdip_raw_cap_{8,12,16,24,32,50,100,200}` | `pdip_raw` at that `max_iter` (the `early_stopping.py` sweep). | `jax_nnls.solve_nnls_primal_raw_forward` |
| `certified` | Certified active-set solve on the Jacobi system, budget / `tau_rel` from `Settings()`, **no** PDIP fallback (an uncertified iterate is scored as-is). Context only — the library never routes MGE systems here. | `jax_active_set.solve_certified` |

## Metrics

All against the stored fnnls reference `x_ref` (definitions in [`_metrics.py`](./_metrics.py)):
`amp_rel_l2`; `amp_rel_max` (worst relative error over columns with `x_ref > 0` — dominated by
barely-active reference columns, so read it with `amp_rel_max_sig`, the same over
`x_ref > 1e-6·max x_ref`); `flux_rel_all` / `flux_rel_source` (signed summed-amplitude error over
all / source columns — an intensity-sum flux proxy; `null` when the source columns are unknown);
`objective_gap` (relative to the reference objective); `kkt_residual_scaled` (the
`scripts/imaging/hazards/pixelization.py` scale-normalised KKT residual: max of primal violation,
dual violation and complementarity). Rows also carry the solver's own `converged` / `iterations`
and the median warm wall-clock (`wall_ms`, 5 repeats after one compile call).

## Accuracy (latest run per corpus)

<!-- BEGIN auto-table:solver-accuracy -->
| Corpus | Candidate | Unconverged | Non-finite | Worst amp_rel_max | Worst amp_rel_max (sig) | Worst abs(flux_rel_source) | Worst objective gap | Worst KKT | Median iters | Median wall ms | Version |
|--------|-----------|-------------|------------|-------------------|-------------------------|----------------------------|---------------------|-----------|--------------|----------------|---------|
| `slam_fixture_571` | `fnnls` | 0/8 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 3.24e-16 | 7.0 | 0.932 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_jacobi` | 7/8 | 0 | 2.73e+68 | 1.03e+68 | 3.36e+70 | 1.56e+134 | 8.25e+61 | 50.0 | 2.490 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw` | 0/8 | 0 | 2.24e+07 | 3.18e+00 | 6.44e-03 | 4.14e-13 | 5.24e-16 | 18.0 | 1.205 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_1e-1` | 0/8 | 0 | 5.87e+07 | 9.08e+00 | 1.69e-02 | 2.73e-12 | 3.60e-15 | 17.0 | 1.275 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_1e-2` | 0/8 | 0 | 2.24e+07 | 3.18e+00 | 6.44e-03 | 4.14e-13 | 5.24e-16 | 18.0 | 1.445 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_1e-3` | 0/8 | 0 | 3.26e+06 | 9.54e-01 | 2.46e-03 | 3.55e-14 | 4.20e-17 | 19.0 | 1.314 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_tol_jaxnnls` | 8/8 | 0 | 2.09e-11 | 2.09e-11 | 3.05e-13 | 6.01e-16 | 1.08e-16 | 50.0 | 2.593 | v2026.8.17.1 |
| `slam_fixture_571` | `pdip_raw_polish` | 0/8 | 0 | 6.91e+04 | 4.95e-06 | 4.93e-05 | 3.60e-16 | 2.16e-16 | 23.0 | 1.376 | v2026.8.17.1 |
| `slam_fixture_571` | `certified` | 5/8 | 0 | 8.76e+01 | 8.76e+01 | 1.12e-02 | 1.61e-05 | 1.81e-02 | 16.0 | 0.909 | v2026.8.17.1 |
<!-- END auto-table:solver-accuracy -->

## Early stopping (latest run per corpus)

<!-- BEGIN auto-table:solver-early-stopping -->
| Corpus | Cap | Converged | Worst amp_rel_max | Worst amp_rel_max (sig) | Median amp_rel_max | Worst abs(flux_rel_source) | Worst KKT | Median wall ms | Version |
|--------|-----|-----------|-------------------|-------------------------|--------------------|----------------------------|-----------|----------------|---------|
| `slam_fixture_571` | 8 | 0/8 | 3.00e+10 | 5.85e+03 | 5.14e+05 | 2.56e+01 | 3.02e-04 | 0.726 | v2026.8.17.1 |
| `slam_fixture_571` | 12 | 0/8 | 2.76e+09 | 1.18e+03 | 1.24e+05 | 6.71e+00 | 4.14e-07 | 0.940 | v2026.8.17.1 |
| `slam_fixture_571` | 16 | 1/8 | 5.87e+07 | 2.46e+01 | 2.67e+03 | 1.25e-01 | 6.11e-14 | 1.046 | v2026.8.17.1 |
| `slam_fixture_571` | 24 | 8/8 | 2.24e+07 | 3.18e+00 | 1.82e+02 | 6.44e-03 | 5.24e-16 | 1.162 | v2026.8.17.1 |
| `slam_fixture_571` | 32 | 8/8 | 2.24e+07 | 3.18e+00 | 1.82e+02 | 6.44e-03 | 5.24e-16 | 1.124 | v2026.8.17.1 |
| `slam_fixture_571` | 50 | 8/8 | 2.24e+07 | 3.18e+00 | 1.82e+02 | 6.44e-03 | 5.24e-16 | 1.207 | v2026.8.17.1 |
| `slam_fixture_571` | 100 | 8/8 | 2.24e+07 | 3.18e+00 | 1.82e+02 | 6.44e-03 | 5.24e-16 | 1.157 | v2026.8.17.1 |
| `slam_fixture_571` | 200 | 8/8 | 2.24e+07 | 3.18e+00 | 1.82e+02 | 6.44e-03 | 5.24e-16 | 1.238 | v2026.8.17.1 |
<!-- END auto-table:solver-early-stopping -->

## How to add a system

1. Capture `(curvature_reg_matrix, data_vector)` as the JAX likelihood hands them to
   `reconstruction_positive_only_from` — wrap that function the way
   [`mge_nnls_capture.py`](../../imaging/hazards/mge_nnls_capture.py) does (`_recording_solver`).
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
python scripts/misc/tooling/build_readme.py
```

Each writes a new `_v<version>` artefact pair beside the previous release's, so the trend is
the file list. `--groups`, `--candidates`, `--device gpu` (writes a `_gpu` artefact) and
`--output-dir` are available on both cells.

## Related

- Campaign page: [`wiki/campaigns/linear_solver_accuracy.md`](../../../wiki/campaigns/linear_solver_accuracy.md)
- Solver ledger: [`results/notes/nnls_solver_ledger.md`](../../../results/notes/nnls_solver_ledger.md)
- Certified solver campaign: [`wiki/campaigns/certified_positive_solver.md`](../../../wiki/campaigns/certified_positive_solver.md)
- NNLS warm-start memo: [`results/nnls_warm_start/nnls_warm_start_memo.md`](../../../results/nnls_warm_start/nnls_warm_start_memo.md)
