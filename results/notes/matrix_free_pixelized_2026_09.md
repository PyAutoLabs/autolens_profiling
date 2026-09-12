# Matrix-free pixelized likelihood — PCG + SLQ reference and the N_src sweep (2026-09-12)

autolens_profiling issue [#247](https://github.com/PyAutoLabs/autolens_profiling/issues/247),
branch `feature/matrix-free-pixelized-likelihood`, harvest commit `d2a3848`. Twenty-nine
A100 jobs: three fiducial matrix-free legs (342664–342666, one per mesh) and a reduced
N_src sweep of twenty-six legs (342669–342694, dense / sparse / matrix-free at
N_src = 3000 / 5000 / 8000 / 12000). **The verdict is no-go, on both axes.** Matrix-free
— best measured PCG plus SLQ at m = 320 for both log-dets — is ~350× the exact path
(Cholesky solve plus two log-det Choleskys) at n ≈ 1500 and still ~16× at n = 12000, with
no crossover at any measured N (1500 / 3000 / 5000 / 12000, rectangular and Delaunay) and
no extrapolation beyond 12000. SLQ never reaches the 0.5-nat go bar at the fiducial N on
any mesh, even at p = 32 probes / m = 640 Lanczos steps (rectangular −4.98, Delaunay
−1.09, DelaunayNN +0.55 nats), and that setting costs ~720 ms — itself ~14× the whole
exact likelihood. The error is dominated by `log det(F+λH)`, not `log det(H)`. The
unconstrained-PCG log-evidence "error" (14 / −68 nats) is a *different minimiser*, not a
solver error — do not quote it as accuracy.

## Scope — read this before quoting a number

- **What was measured.** Three fiducial legs at n ≈ 1500 (342664 rectangular, 342665
  Delaunay, 342666 DelaunayNN) and twenty-six reduced-sweep legs (342669–342694) over
  N_src ∈ {3000, 5000, 8000, 12000} × {dense, sparse, matrix-free} × {rectangular,
  Delaunay, DelaunayNN}, all on `euclid-ral-gpu-2` (NVIDIA A100 80GB PCIe), fp64. The
  fiducial tier is the one the [A100 baseline note](./a100_pixelized_baseline_2026_09.md)
  defines (HST, mask 3.5″, MGE-60 lens light, pixelization over-sampling 1); the sweep
  legs change only N_src.
- **What "matrix-free" means here.** Preconditioned conjugate gradients on
  `(F + λH) x = D` with the matrix-vector product taken through the sparse operator's FFT
  apply (never forming `F`); stochastic Lanczos quadrature (SLQ) log-dets with a fixed
  Rademacher probe set; and a matrix-free primal-dual interior point (PDIP) comparator for
  the positive reconstruction. All three live in
  `scripts/misc/likelihood_breakdown/matrix_free_steps.py` and are driven by the
  `scripts/imaging/likelihood_breakdown/matrix_free.py` cell.
- **What was NOT measured.** There is no matrix-free sweep leg for `delaunay_nn` at any N,
  and none for any mesh at n = 8000. Matrix-free PDIP was disabled (`--no-pdip`) on all
  six matrix-free sweep legs, so there is no PDIP measurement above n ≈ 1500. No PyAutoArray
  phase was run — no library code changed. **Nothing here is extrapolated**: every "no
  crossover" statement is a statement about the measured grid only.
- **Regularization** is `Constant(1.0)` on the rectangular cell and `adapt_split`
  (0.1 / 10.0 / 0.1) on the Delaunay family, matching each cell's shipped default. Compare
  log-det terms within a mesh, never across.

## Provenance and gate

| Job | Mesh | Path | N_src (requested) | Exit | cache_fresh | Wall | Runtime pins |
|---|---|---|---:|---|---|---:|---|
| 342664 | rectangular | matrix_free | fiducial | — (no sweep wrapper) | fresh | 0:05:18 | PASSED / PASSED |
| 342665 | delaunay | matrix_free | fiducial | — (no sweep wrapper) | fresh | 0:07:38 | PASSED / PASSED |
| 342666 | delaunay_nn | matrix_free | fiducial | — (no sweep wrapper) | fresh | 0:08:23 | PASSED / PASSED |
| 342669–342674 | rect / delaunay | matrix_free | 3000 / 5000 / 12000 | 0 | fresh | 0:03:00–0:06:49 | SKIPPED (non-fiducial N) |
| 342675–342684 | rect / delaunay / delaunay_nn | sparse | 3000 / 5000 / 8000 / 12000 | 0 | fresh | 0:02:00–0:06:40 | — |
| 342685–342694 | rect / delaunay / delaunay_nn | dense | 3000 / 5000 / 8000 / 12000 | 0 | fresh | 0:01:53–0:06:14 | — |

**Counts: legs = 29; result JSONs written = 29; off-node = none; `cache_fresh: false` =
none; non-zero exit = none; runtime pin FAILED = none.** Every leg reports `hostname` in
`['euclid-ral-gpu-2']`, `autotune_cache_entries_at_start: 0` and `AUTOTUNE_ENTRIES
count=0` at exit, so no leg inherited a seeded autotune cache. No NaN or Infinity appears
in any of the twenty-nine JSONs.

The runtime log-det pins PASSED on all three fiducial legs. The twenty-six sweep legs
print `Pinned log-det check SKIPPED` / `Eager regression assertion SKIPPED` **by design**:
a non-fiducial N builds a different mesh from the one the pins describe, so the pin is not
applicable rather than bypassed.

Branch history behind the harvest: `522328c` (matrix-free kernels + tests),
`79a91f1` (`--source-pixels N` on the breakdown CLI, pins gated on the fiducial N),
`55bb012` (the `matrix_free` cell), `b959cd6` + `7e8f66c` (the reduced sweep: templated
leg, launcher, aggregator, manifest), `d2a3848` (this harvest — 29 JSON + 29 PNG).

## Fiducial n ≈ 1500 — matrix-free rows vs the exact `_recon_split` rows

Per-call milliseconds. The matrix-free cell measures its own `nnls_pdip` and
`cholesky solve` reference rows **in the same process**; the `_recon_split` column is the
independent exact leg (jobs 342643–342645, already on `main`, documented in the
[A100 baseline note](./a100_pixelized_baseline_2026_09.md#reconstruction-split-2026-09-11)).
**These are overlapping comparators — never a partition of a total, and never to be
summed.** They exist to answer "what would this piece cost on its own?".

### rectangular (N_src = 1521, 1581 total parameters)

| Row | matrix-free leg (ms) | `_recon_split` exact leg (ms) |
|---|---:|---:|
| NNLS PDIP (cell-driven, max_iter 50) | 36.836 | 37.124 |
| Cholesky solve (unconstrained) | 1.592 | 1.556 |
| Cholesky (F+λH) | n/a | 1.263 |
| Log det Cholesky (F+λH reduced) | n/a | 1.271 |
| Log det Cholesky (H reduced) | n/a | 1.191 |
| Blurred mapping matrix (PSF) | not measured in this cell | 7.679 |
| Curvature matrix (F) | not measured in this cell | 4.821 |
| Regularized reconstruction (step 12) | not measured in this cell | 36.962 |
| Mapped recon + log evidence (step 13) | not measured in this cell | 2.412 |
| **TOTAL step-by-step** | not measured in this cell | **59.889** |
| context build (eager, one-off, NOT per-call) | 1925.992 | — |
| `curvature_reg_matvec` ×1 | 0.406 | — |
| `curvature_reg_matvec` ×32 | 1.263 (0.039 / column) | — |

PCG on the unconstrained system (`cg_tol` 1e-10, `cg_maxiter` 15810); CG iterations equal
matvecs against `F + λH`, one per iteration:

| Variant | ms | CG iterations | rel. residual | converged | max abs diff vs Cholesky solve | max rel diff |
|---|---:|---:|---:|---|---:|---:|
| Jacobi est. | 1179.947 | 5578 | 7.941e-11 | True | 2.534e-03 | 1.748e-04 |
| exact diag | 1152.970 | 5263 | 9.084e-11 | True | 2.716e-03 | 1.873e-04 |
| no precond | 3135.294 | 14675 | 9.502e-11 | True | 1.190e-03 | 8.211e-05 |

Matrix-free PDIP: 8232.383 ms, 36 PDIP iterations, 31451 total CG iterations, converged
True; agreement with the exact `nnls_pdip` to 3.428e-04 max abs / 2.421e-05 max rel;
**cost ratio 223.49×** (`pdip_cg_tol` 1e-08, `pdip_cg_maxiter` 1581) against the exact
`nnls_pdip` in the same process (36.836 ms, 21 iterations, 1.754 ms/iteration).

Log-evidence at the cell's chosen SLQ setting (p = 16, m = 40): error **−324.549 nats**,
go bar 0.5, `within_go_bar` **False**. Term errors: χ² 0.000e+00, regularization term
5.684e-14, `log det(F+λH)` 649.727, `log det(H)` 0.630, against exact log-dets
3888.258 and 1692.787. `cond(F+λH)` = 4.117e+10; 152 edge-zeroed pixels;
`add_to_diag` 0.001. Unconstrained-PCG log-evidence difference (a different minimiser, see
the JSON note): 14.440 nats. Peak bytes 216109824 after setup and after the dense
comparators, 922273792 after the matrix-free rows.

### delaunay (N_src = 1500, 1560 total parameters)

| Row | matrix-free leg (ms) | `_recon_split` exact leg (ms) |
|---|---:|---:|
| NNLS PDIP (cell-driven, max_iter 50) | 37.541 | 37.316 |
| Cholesky solve (unconstrained) | 1.576 | 1.576 |
| Cholesky (F+λH) | n/a | 1.194 |
| Log det Cholesky (F+λH reduced) | n/a | 1.156 |
| Log det Cholesky (H reduced) | n/a | 1.147 |
| Blurred mapping matrix (PSF) | not measured in this cell | 8.246 |
| Curvature matrix (F) | not measured in this cell | 4.860 |
| Regularized reconstruction (step 12) | not measured in this cell | 37.386 |
| Mapped recon + log evidence (step 13) | not measured in this cell | 2.233 |
| **TOTAL step-by-step** | not measured in this cell | **67.157** |
| context build (eager, one-off, NOT per-call) | 2058.253 | — |
| `curvature_reg_matvec` ×1 | 0.340 | — |
| `curvature_reg_matvec` ×32 | 1.161 (0.036 / column) | — |

PCG (`cg_tol` 1e-10, `cg_maxiter` 15600):

| Variant | ms | CG iterations | rel. residual | converged | max abs diff vs Cholesky solve | max rel diff |
|---|---:|---:|---:|---|---:|---:|
| Jacobi est. | 2385.656 | 11512 | 9.941e-11 | True | 3.106e-03 | 1.052e-04 |
| exact diag | 1678.068 | 8247 | 9.251e-11 | True | 7.677e-04 | 2.601e-05 |
| no precond | 2909.545 | 14152 | 6.924e-11 | True | 1.617e-03 | 5.478e-05 |

Matrix-free PDIP: 15519.929 ms, **50 PDIP iterations = `max_iter`, converged False**,
79824 total CG iterations; agreement 4.998e-04 max abs / 3.487e-05 max rel; **cost ratio
413.41×** against the exact `nnls_pdip` (37.541 ms, 22 iterations, 1.706 ms/iteration).
This is the one non-converged solver flag in the whole harvest.

Log-evidence at p = 16, m = 40: error **−95.487 nats**, `within_go_bar` **False**. Term
errors: χ² 0.000e+00, regularization term −1.253e-10, `log det(F+λH)` 381.610,
`log det(H)` 190.637, against exact 8360.402 and 7756.615. `cond(F+λH)` = 4.107e+10; 0
edge-zeroed pixels; `add_to_diag` 0.001. Unconstrained-PCG log-evidence difference
−68.045 nats. Peak bytes 215430656 / 215430656 / 921959680.

### delaunay_nn (N_src = 1500, 1560 total parameters)

| Row | matrix-free leg (ms) | `_recon_split` exact leg (ms) |
|---|---:|---:|
| NNLS PDIP (cell-driven, max_iter 50) | 37.369 | 37.304 |
| Cholesky solve (unconstrained) | 1.548 | 1.522 |
| Cholesky (F+λH) | n/a | 1.235 |
| Log det Cholesky (F+λH reduced) | n/a | 1.164 |
| Log det Cholesky (H reduced) | n/a | 1.184 |
| Blurred mapping matrix (PSF) | not measured in this cell | 8.217 |
| Curvature matrix (F) | not measured in this cell | 4.820 |
| Regularized reconstruction (step 12) | not measured in this cell | 37.569 |
| Mapped recon + log evidence (step 13) | not measured in this cell | 2.261 |
| **TOTAL step-by-step** | not measured in this cell | **80.436** |
| context build (eager, one-off, NOT per-call) | 1991.596 | — |
| `curvature_reg_matvec` ×1 | 0.442 | — |
| `curvature_reg_matvec` ×32 | 1.668 (0.052 / column) | — |

PCG (`cg_tol` 1e-10, `cg_maxiter` 15600):

| Variant | ms | CG iterations | rel. residual | converged | max abs diff vs Cholesky solve | max rel diff |
|---|---:|---:|---:|---|---:|---:|
| Jacobi est. | 1913.830 | 7716 | 7.072e-11 | True | 2.416e-03 | 7.346e-05 |
| exact diag | 1353.256 | 5377 | 7.256e-11 | True | 1.585e-03 | 4.820e-05 |
| no precond | 2704.949 | 10909 | 8.907e-11 | True | 1.766e-03 | 5.369e-05 |

Matrix-free PDIP: 16981.613 ms, 34 PDIP iterations, 37049 total CG iterations, converged
True; agreement 3.462e-04 max abs / 2.434e-05 max rel; **cost ratio 454.43×** against the
exact `nnls_pdip` (37.369 ms, 22 iterations, 1.699 ms/iteration).

Log-evidence at p = 16, m = 40: error **−103.986 nats**, `within_go_bar` **False**. Term
errors: χ² 0.000e+00, regularization term −5.832e-11, `log det(F+λH)` 333.577,
`log det(H)` 125.605, against exact 7224.569 and 6690.184. `cond(F+λH)` = 4.113e+10; 0
edge-zeroed pixels; `add_to_diag` 0.001. Unconstrained-PCG log-evidence difference
−70.806 nats. Peak bytes 916542720 on all three marks.

## SLQ step curve — the accuracy no-go

`log_evidence_error = −0.5 × err[log det(F+λH)] + 0.5 × err[log det(H)]`, verified against
the cell's own `term_errors` at its chosen setting. `ms total` is the sum of the two SLQ
rows at that `(probes, steps)`; errors are estimate minus exact, in nats.

### rectangular — best measured p = 32 / m = 640 at −4.977 nats (720.459 ms)

| probes | steps | log det(F+λH) est. | err | log det(H) est. | err | ms (F+λH) | ms (H) | ms total | log-ev err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 20 | 5737.840 | 1849.582 | 1705.978 | 13.191 | 10.605 | 1.021 | 11.627 | −918.195 |
| 4 | 40 | 4545.136 | 656.878 | 1702.649 | 9.862 | 20.922 | 1.721 | 22.643 | −323.508 |
| 4 | 80 | 4094.570 | 206.312 | 1698.328 | 5.541 | 41.023 | 3.220 | 44.243 | −100.386 |
| 4 | 160 | 3945.340 | 57.082 | 1686.569 | −6.218 | 86.407 | 10.894 | 97.301 | −31.650 |
| 4 | 320 | 3907.278 | 19.019 | 1686.569 | −6.218 | 173.762 | 22.575 | 196.337 | −12.619 |
| 4 | 640 | 3901.609 | 13.351 | 1686.569 | −6.218 | 350.683 | 48.712 | 399.394 | −9.785 |
| 8 | 20 | 5698.713 | 1810.455 | 1692.479 | −0.308 | 11.725 | 1.079 | 12.804 | −905.382 |
| 8 | 40 | 4524.177 | 635.919 | 1688.040 | −4.747 | 23.121 | 1.887 | 25.007 | −320.333 |
| 8 | 80 | 4075.137 | 186.879 | 1681.427 | −11.359 | 45.946 | 3.526 | 49.471 | −99.119 |
| 8 | 160 | 3927.527 | 39.269 | 1666.181 | −26.606 | 96.184 | 11.585 | 107.769 | −32.937 |
| 8 | 320 | 3887.544 | −0.714 | 1666.181 | −26.606 | 193.606 | 24.812 | 218.418 | −12.946 |
| 8 | 640 | 3882.046 | −6.212 | 1666.181 | −26.606 | 394.597 | 54.780 | 449.377 | −10.197 |
| 16 | 20 | 5693.053 | 1804.795 | 1696.236 | 3.449 | 16.418 | 1.310 | 17.728 | −900.673 |
| 16 | 40 | 4537.985 | 649.727 | 1693.417 | 0.630 | 33.443 | 2.216 | 35.659 | −324.549 |
| 16 | 80 | 4091.444 | 203.186 | 1689.636 | −3.151 | 63.277 | 4.135 | 67.412 | −103.169 |
| 16 | 160 | 3940.794 | 52.536 | 1680.194 | −12.593 | 129.697 | 13.178 | 142.875 | −32.564 |
| 16 | 320 | 3901.235 | 12.977 | 1680.194 | −12.593 | 253.708 | 28.630 | 282.338 | −12.785 |
| 16 | 640 | 3895.831 | 7.573 | 1680.194 | −12.593 | 518.657 | 68.969 | 587.626 | −10.083 |
| 32 | 20 | 5721.256 | 1832.998 | 1698.540 | 5.753 | 18.031 | 1.563 | 19.593 | −913.622 |
| 32 | 40 | 4530.665 | 642.407 | 1695.524 | 2.737 | 37.213 | 2.992 | 40.205 | −319.835 |
| 32 | 80 | 4086.920 | 198.662 | 1691.614 | −1.173 | 71.257 | 5.738 | 76.995 | −99.918 |
| 32 | 160 | 3934.645 | 46.387 | 1683.129 | −9.658 | 147.781 | 16.863 | 164.644 | −28.022 |
| 32 | 320 | 3894.387 | 6.129 | 1683.129 | −9.658 | 298.619 | 38.351 | 336.971 | −7.893 |
| 32 | 640 | 3888.554 | 0.296 | 1683.129 | −9.658 | 618.357 | 102.102 | 720.459 | **−4.977** |

### delaunay — best measured p = 32 / m = 160 at −1.087 nats (160.312 ms)

| probes | steps | log det(F+λH) est. | err | log det(H) est. | err | ms (F+λH) | ms (H) | ms total | log-ev err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 20 | 9320.855 | 960.453 | 8097.908 | 341.293 | 9.980 | 1.009 | 10.989 | −309.580 |
| 4 | 40 | 8682.827 | 322.425 | 7854.357 | 97.742 | 19.478 | 1.767 | 21.245 | −112.342 |
| 4 | 80 | 8433.417 | 73.015 | 7757.412 | 0.797 | 38.717 | 3.371 | 42.088 | −36.109 |
| 4 | 160 | 8344.843 | −15.558 | 7727.653 | −28.962 | 81.901 | 11.400 | 93.301 | −6.702 |
| 4 | 320 | 8328.860 | −31.542 | 7720.293 | −36.322 | 163.627 | 22.792 | 186.419 | −2.390 |
| 4 | 640 | 8327.625 | −32.776 | 7717.259 | −39.356 | 330.191 | 51.974 | 382.165 | −3.290 |
| 8 | 20 | 9344.614 | 984.212 | 8152.989 | 396.374 | 11.168 | 1.062 | 12.231 | −293.919 |
| 8 | 40 | 8714.863 | 354.461 | 7907.355 | 150.740 | 22.188 | 1.992 | 24.180 | −101.861 |
| 8 | 80 | 8465.091 | 104.689 | 7809.892 | 53.277 | 43.967 | 5.032 | 48.999 | −25.706 |
| 8 | 160 | 8377.094 | 16.693 | 7776.936 | 20.321 | 92.443 | 11.845 | 104.288 | 1.814 |
| 8 | 320 | 8359.824 | −0.578 | 7767.515 | 10.900 | 185.020 | 24.743 | 209.763 | 5.739 |
| 8 | 640 | 8358.412 | −1.990 | 7762.966 | 6.351 | 378.599 | 55.371 | 433.970 | 4.171 |
| 16 | 20 | 9374.052 | 1013.650 | 8185.465 | 428.850 | 13.765 | 1.253 | 15.018 | −292.400 |
| 16 | 40 | 8742.012 | 381.610 | 7947.252 | 190.637 | 27.511 | 2.330 | 29.842 | −95.487 |
| 16 | 80 | 8491.660 | 131.258 | 7848.441 | 91.827 | 54.590 | 4.318 | 58.908 | −19.716 |
| 16 | 160 | 8401.791 | 41.389 | 7815.338 | 58.723 | 112.775 | 13.524 | 126.299 | 8.667 |
| 16 | 320 | 8384.559 | 24.158 | 7806.583 | 49.968 | 229.664 | 28.936 | 258.600 | 12.905 |
| 16 | 640 | 8383.057 | 22.655 | 7804.017 | 47.402 | 470.728 | 69.218 | 539.946 | 12.374 |
| 32 | 20 | 9344.546 | 984.145 | 8142.991 | 386.376 | 17.463 | 1.674 | 19.137 | −298.884 |
| 32 | 40 | 8715.290 | 354.889 | 7901.019 | 144.404 | 34.161 | 3.292 | 37.453 | −105.242 |
| 32 | 80 | 8467.337 | 106.935 | 7802.781 | 46.166 | 68.973 | 6.130 | 75.103 | −30.385 |
| 32 | 160 | 8377.058 | 16.656 | 7771.096 | 14.481 | 142.594 | 17.718 | 160.312 | **−1.087** |
| 32 | 320 | 8360.589 | 0.187 | 7762.132 | 5.517 | 289.161 | 38.804 | 327.965 | 2.665 |
| 32 | 640 | 8359.323 | −1.079 | 7759.026 | 2.411 | 608.877 | 103.223 | 712.100 | 1.745 |

### delaunay_nn — best measured p = 32 / m = 160 at +0.547 nats (234.490 ms)

| probes | steps | log det(F+λH) est. | err | log det(H) est. | err | ms (F+λH) | ms (H) | ms total | log-ev err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 20 | 8002.243 | 777.675 | 6941.692 | 251.508 | 12.121 | 1.074 | 13.195 | −263.083 |
| 4 | 40 | 7524.933 | 300.364 | 6760.521 | 70.337 | 23.466 | 1.891 | 25.356 | −115.013 |
| 4 | 80 | 7298.723 | 74.154 | 6698.486 | 8.303 | 46.648 | 3.481 | 50.128 | −32.926 |
| 4 | 160 | 7229.049 | 4.481 | 6685.270 | −4.914 | 97.318 | 11.288 | 108.606 | −4.697 |
| 4 | 320 | 7218.733 | −5.835 | 6681.580 | −8.604 | 193.937 | 23.102 | 217.039 | −1.384 |
| 4 | 640 | 7218.311 | −6.258 | 6677.338 | −12.846 | 385.739 | 49.501 | 435.240 | −3.294 |
| 8 | 20 | 8002.181 | 777.612 | 6966.730 | 276.546 | 14.434 | 1.323 | 15.757 | −250.533 |
| 8 | 40 | 7525.542 | 300.973 | 6781.097 | 90.913 | 28.453 | 2.549 | 31.002 | −105.030 |
| 8 | 80 | 7307.467 | 82.898 | 6713.983 | 23.799 | 56.595 | 4.762 | 61.357 | −29.549 |
| 8 | 160 | 7235.136 | 10.567 | 6698.945 | 8.761 | 117.042 | 13.797 | 130.840 | −0.903 |
| 8 | 320 | 7224.001 | −0.568 | 6693.919 | 3.735 | 229.890 | 28.594 | 258.484 | 2.152 |
| 8 | 640 | 7223.478 | −1.091 | 6687.428 | −2.755 | 462.317 | 61.652 | 523.969 | −0.832 |
| 16 | 20 | 8017.287 | 792.718 | 6999.644 | 309.460 | 18.780 | 1.548 | 20.328 | −241.629 |
| 16 | 40 | 7558.146 | 333.577 | 6815.789 | 125.605 | 36.753 | 3.086 | 39.839 | −103.986 |
| 16 | 80 | 7328.825 | 104.257 | 6749.833 | 59.649 | 73.425 | 5.638 | 79.062 | −22.304 |
| 16 | 160 | 7255.572 | 31.003 | 6735.213 | 45.029 | 150.561 | 15.994 | 166.555 | 7.013 |
| 16 | 320 | 7245.084 | 20.515 | 6731.591 | 41.407 | 311.267 | 33.719 | 344.985 | 10.446 |
| 16 | 640 | 7244.627 | 20.059 | 6728.166 | 37.982 | 629.419 | 79.617 | 709.036 | 8.962 |
| 32 | 20 | 8006.029 | 781.460 | 6970.014 | 279.830 | 26.618 | 2.071 | 28.689 | −250.815 |
| 32 | 40 | 7540.955 | 316.386 | 6786.960 | 96.776 | 52.450 | 3.920 | 56.371 | −109.805 |
| 32 | 80 | 7314.034 | 89.466 | 6722.330 | 32.146 | 105.040 | 7.615 | 112.655 | −28.660 |
| 32 | 160 | 7240.930 | 16.361 | 6707.638 | 17.455 | 214.123 | 20.366 | 234.490 | **0.547** |
| 32 | 320 | 7230.298 | 5.729 | 6703.506 | 13.322 | 432.681 | 44.047 | 476.728 | 3.796 |
| 32 | 640 | 7229.821 | 5.252 | 6699.665 | 9.482 | 897.287 | 113.368 | 1010.655 | 2.115 |

**No measured `(probes, steps)` reaches the 0.5-nat bar on any mesh**, and the settings
that come closest cost 160–720 ms — 2.4× to ~12× the *whole* exact likelihood. The error
is dominated by `log det(F+λH)`: at rectangular p = 16 / m = 40 the cell shipped 4537.985
against the exact 3888.258 (+649.727 nats) while `log det(H)` was within 0.630 nats.

**The rectangular λH curve saturates at m = 160.** Its `log det(H)` estimate is identical
to twelve decimal places at m = 160, 320 and 640 for every probe count (p = 4: −6.218 nats
at all three; p = 8: −26.606; p = 16: −12.593; p = 32: −9.658) — more Lanczos steps buy
nothing. The Delaunay family does **not** do this: its λH error keeps moving with m
(delaunay p = 32: 14.481 / 5.517 / 2.411 at m = 160 / 320 / 640; delaunay_nn p = 32:
17.455 / 13.322 / 9.482). *Interpretation, clearly labelled as such:* rectangular uses
`Constant(1.0)` regularization, whose H has a tightly clustered spectrum, so Lanczos has
already converged at m = 160 and the residual is the Hutchinson **probe** variance, which
only more probes reduce; the Delaunay cells' `adapt_split` H has a spread spectrum that
keeps rewarding more steps. This is an interpretation of the pattern, not a measured
decomposition.

## Conditioning — why PCG cannot win

`cond(F+λH)` ≈ 4.1e+10 on **every** leg and **every** N (4.117e+10 / 4.107e+10 /
4.113e+10 at the fiducial; 4.1e+10, 4.08e+10, 4.06e+10 on the rectangular sweep at
3000 / 5000 / 12000; 4.08e+10, 4.1e+10, 4.06e+10 on the Delaunay sweep). PCG converges on
every leg and variant (relative residual < 1e-10, well inside `cg_maxiter`) but needs
5.2k–14.7k matvecs at n ≈ 1500 and 5.7k–18.3k at n = 3000; `no precond` costs ~2.6× the
iterations of `exact diag`.

**The matvec is cheap; the count is the problem.** `curvature_reg_matvec` is 0.036–0.052
ms per column at ×32 batching — the FFT apply is doing its job. What sets the count is the
spectrum, and the spectrum is set by the 60 unregularised linear-MGE columns (λmin is the
1e-3 `add_to_diag` floor), not by the mapper block, whose own condition number is
1.2e7 / 2.1e6 source-only (CPU probe, 2026-09-11). No preconditioner tried here touches
that: Jacobi and exact-diagonal both leave four to five orders of magnitude of spread.

## The sweep — fits and per-call cost

### Fits: does the path still run at this mesh size?

Every one of the twenty-six sweep legs and three fiducial legs wrote its JSON; **no
submitted leg died.**

| Mesh | N | dense | sparse | matrix-free |
|---|---:|---|---|---|
| rectangular | 3000 | ✓ | ✓ | ✓ 0.6 GB |
| delaunay | 3000 | ✓ | ✓ | ✓ 0.6 GB |
| delaunay_nn | 3000 | ✓ | ✓ | — (not submitted) |
| rectangular | 5000 | ✓ (vmap-16 row OOM) | ✓ | ✓ 1.6 GB |
| delaunay | 5000 | ✓ (vmap-16 row OOM) | ✓ | ✓ 1.6 GB |
| delaunay_nn | 5000 | ✓ (vmap-16 row OOM) | ✓ | — (not submitted) |
| rectangular | 8000 | ✓ (vmap-16 row OOM) | ✓ | — (not submitted) |
| delaunay | 8000 | ✓ (vmap-16 row OOM) | ✓ | — (not submitted) |
| rectangular | 12000 | ✓ (vmap-16 row OOM) | ✓ | ✓ 7.7 GB |
| delaunay | 12000 | ✓ (vmap-16 row OOM) | ✓ | ✓ 7.6 GB |

**Every dense and sparse single-call leg fits an 80 GB A100 up to n = 12000**, on both the
rectangular and the Delaunay mesh. The only lost measurement is the *optional* `vmap batch
16` batched-NNLS comparator, which OOM'd on the seven dense legs 342688–342694 (n ≥ 5000)
with `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED` for allocations up to **93.73 GiB**
(`jit_fn` / `jit__nnls_pdip_batched`; `hlo_rematerialization` could not get 342694 below
57.58 GiB). The traceback is captured in each JSON, the unbatched step rows are unaffected
and every leg still exited 0. Sparse legs 342683/342684 and dense 342685–342687 logged
soft `bfc_allocator` retry warnings only.

### Per-call cost, dense and sparse legs (ms)

| Mesh | N | Path | Job | TOTAL step-by-step | Regularized reconstruction | NNLS PDIP (cell) | PDIP iters | ms/iter | Cholesky (F+λH) | Cholesky solve | Log det Chol (F+λH) | Log det Chol (H) | Blurred mapping matrix | Curvature F |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rectangular | 3000 | sparse | 342675 | 137.414 | 82.996 | 83.267 | 24 | 3.469 | 2.407 | 3.038 | 2.226 | 2.240 | n/a | n/a |
| delaunay | 3000 | sparse | 342676 | 139.663 | 76.632 | 75.978 | 23 | 3.303 | 2.246 | 2.935 | 2.236 | 2.286 | n/a | n/a |
| delaunay_nn | 3000 | sparse | 342677 | 159.223 | 82.789 | 82.754 | 25 | 3.310 | 2.249 | 2.908 | 2.110 | 2.130 | n/a | n/a |
| rectangular | 5000 | sparse | 342678 | 244.210 | 163.940 | 163.858 | 21 | 7.803 | 5.937 | 6.871 | 5.464 | 5.489 | n/a | n/a |
| delaunay | 5000 | sparse | 342679 | 295.331 | 171.234 | 171.423 | 23 | 7.453 | 5.561 | 6.769 | 5.146 | 5.181 | n/a | n/a |
| delaunay_nn | 5000 | sparse | 342680 | 275.969 | 171.256 | 171.082 | 23 | 7.438 | 5.500 | 6.611 | 5.169 | 5.193 | n/a | n/a |
| rectangular | 8000 | sparse | 342681 | 631.817 | 485.425 | 485.751 | 24 | 20.240 | 16.565 | 18.247 | 15.505 | 15.295 | n/a | n/a |
| delaunay | 8000 | sparse | 342682 | 791.309 | 605.943 | 606.932 | 31 | 19.578 | 15.990 | 17.715 | 15.211 | 14.970 | n/a | n/a |
| rectangular | 12000 | sparse | 342683 | 1684.022 | 1429.198 | 1429.137 | 28 | 51.041 | 44.180 | 47.217 | 43.266 | 42.674 | n/a | n/a |
| delaunay | 12000 | sparse | 342684 | 1844.219 | 1552.141 | 1562.060 | 30 | 52.069 | 45.347 | 48.054 | 42.491 | 41.291 | n/a | n/a |
| rectangular | 3000 | dense | 342685 | 132.343 | 82.503 | 83.046 | 24 | 3.460 | 2.371 | 3.017 | 2.238 | 2.251 | 15.597 | 21.243 |
| delaunay | 3000 | dense | 342686 | 137.240 | 77.114 | 76.299 | 23 | 3.317 | 2.249 | 2.920 | 2.162 | 2.142 | 17.222 | 19.463 |
| delaunay_nn | 3000 | dense | 342687 | 161.281 | 83.557 | 82.694 | 25 | 3.308 | 2.258 | 2.922 | 2.123 | 2.149 | 15.755 | 19.566 |
| rectangular | 5000 | dense | 342688 | 256.486 | 164.012 | 162.956 | 21 | 7.760 | 5.799 | 6.900 | 5.477 | 5.410 | 25.523 | 47.209 |
| delaunay | 5000 | dense | 342689 | 273.683 | 171.330 | 171.487 | 23 | 7.456 | 5.587 | 6.629 | 5.156 | 5.147 | 30.181 | 46.491 |
| delaunay_nn | 5000 | dense | 342690 | 308.057 | 171.423 | 171.685 | 23 | 7.465 | 5.541 | 6.661 | 5.195 | 5.198 | 27.592 | 48.400 |
| rectangular | 8000 | dense | 342691 | 679.077 | 485.135 | 483.899 | 24 | 20.162 | 16.490 | 18.163 | 16.182 | 15.251 | 41.116 | 111.748 |
| delaunay | 8000 | dense | 342692 | 839.140 | 599.625 | 600.335 | 31 | 19.366 | 15.853 | 17.512 | 14.956 | 14.955 | 28.928 | 111.160 |
| rectangular | 12000 | dense | 342693 | 1824.739 | 1413.762 | 1421.665 | 28 | 50.774 | 43.868 | 46.853 | 42.795 | 42.728 | 67.365 | 247.899 |
| delaunay | 12000 | dense | 342694 | 2011.462 | 1538.514 | 1543.567 | 30 | 51.452 | 44.930 | 47.505 | 41.714 | 41.406 | 89.303 | 244.625 |

**The NNLS/PDIP reconstruction row dominates the per-call cost at every N** — ~60–85 % of
the step-by-step total: n = 3000 ≈ 83 ms (24 iterations × 3.5 ms), 5000 ≈ 164 ms (21 ×
7.8), 8000 ≈ 485–606 ms (24–31 × 20), 12000 ≈ 1.41–1.55 s (28–30 × 51 ms). The exact
unconstrained Cholesky solve of the same system at 12000 is **47 ms**: positivity costs
~30× the linear solve at every N, the same ratio the fiducial grid found at n ≈ 1500.

**Dense and sparse per-call totals are within ~10 % of each other at every N** (sparse
wins slightly at 5000 / 8000 / 12000 rectangular), and the F build scales as expected —
dense `Curvature matrix (F)` 21.2 ms at 3000 to 247.9 ms at 12000.

### Per-call cost, matrix-free sweep legs

| Mesh | N_src (built) | Job | Cholesky solve ref (ms) | PCG Jacobi ms / iters / resid | PCG exact-diag ms / iters | PCG no-precond ms / iters | SLQ p16 m20 err (F+λH / H) | m80 err | m320 err | SLQ m320 ms (F+λH / H) | log-ev err @cell setting |
|---|---:|---|---:|---|---|---|---|---|---|---|---:|
| rectangular | 3025 | 342669 | 3.056 | 1389.980 / 6429 / 9.97e-11 | 1296.687 / 5728 | 3969.857 / 18273 | 3558.089 / 10.598 | 387.168 / 3.025 | 0.824 / −8.478 | 256.198 / 34.354 | −4.651 |
| delaunay | 3000 | 342670 | 2.910 | 2774.956 / 13543 / 8.63e-11 | 1897.106 / 9268 | 2873.210 / 14072 | 1196.243 / 452.453 | 132.941 / 75.823 | 36.600 / 43.598 | 232.465 / 35.240 | 3.499 |
| rectangular | 5041 | 342671 | 6.885 | 1582.548 / 7221 / 8.81e-11 | 1440.921 / 6585 | 4898.502 / 22681 | 5122.831 / −10.551 | 652.593 / −20.904 | −6.511 / −34.240 | 262.883 / 29.059 | −13.865 |
| delaunay | 5000 | 342672 | 6.608 | 2619.359 / 12366 / 9.00e-11 | 1787.044 / 8429 | 2392.439 / 11098 | 1063.673 / 244.901 | 20.358 / 3.179 | −25.468 / −14.384 | 235.295 / 32.318 | 5.542 |
| rectangular | 12100 | 342673 | 46.959 | 1934.678 / 8984 / 8.86e-11 | 1758.986 / 7883 | 6613.314 / 30744 | 9596.265 / 19.249 | 1740.865 / 0.446 | 121.406 / −12.785 | 248.838 / 34.925 | −67.096 |
| delaunay | 12000 | 342674 | 47.692 | 2831.271 / 13320 / 8.41e-11 | 1902.817 / 8813 | 2618.792 / 12213 | 1340.462 / 385.204 | 130.340 / 111.447 | 83.986 / 93.127 | 236.114 / 39.219 | 4.571 |

CG iterations equal matvecs against `F + λH`, one per iteration. SLQ error columns are
estimate minus exact, in nats, at p = 16.

## Crossover — measured, never extrapolated

Matrix-free per-call = unconstrained PCG (best preconditioner measured on that leg) plus
SLQ at m = 320 for both log-dets. The dense/sparse exact per-call is that leg's own
`Cholesky solve (unconstrained)` plus its two `Log det Cholesky` rows — the like-for-like
comparison, not the whole-likelihood total.

| Mesh | N_src | matrix-free (PCG best + SLQ m320) ms | dense exact ms | sparse exact ms | mf ≤ dense? | mf ≤ sparse? |
|---|---:|---:|---:|---:|---|---|
| rectangular | 1500 | 1435.308 | 4.019 | n/a (2026-09-10 fiducial sparse legs carry no reconstruction sub-rows) | no | n/a |
| rectangular | 3000 | 1587.238 | 7.507 | 7.504 | no | no |
| rectangular | 5000 | 1732.863 | 17.786 | 17.824 | no | no |
| rectangular | 8000 | not measured (leg not submitted) | 49.596 | 49.047 | n/a | n/a |
| rectangular | 12000 | 2042.748 | 132.376 | 133.157 | no | no |
| delaunay | 1500 | 1936.668 | 3.880 | n/a (as above) | no | n/a |
| delaunay | 3000 | 2164.811 | 7.224 | 7.457 | no | no |
| delaunay | 5000 | 2054.657 | 16.932 | 17.096 | no | no |
| delaunay | 8000 | not measured (leg not submitted) | 47.424 | 47.896 | n/a | n/a |
| delaunay | 12000 | 2178.150 | 130.625 | 131.837 | no | no |
| delaunay_nn | 1500 | 1698.242 | 3.870 | n/a (as above) | no | n/a |
| delaunay_nn | 3000 | not measured (leg not submitted) | 7.195 | 7.148 | n/a | n/a |
| delaunay_nn | 5000 | not measured (leg not submitted) | 17.054 | 16.973 | n/a | n/a |
| delaunay_nn | 8000 | not measured (leg not submitted) | n/a | n/a | n/a | n/a |
| delaunay_nn | 12000 | not measured (leg not submitted) | n/a | n/a | n/a | n/a |

**No crossover is reached at any measured N, against dense or sparse, for any mesh.** The
largest N measured is 12000 (rectangular, Delaunay); DelaunayNN was swept only to 5000 and
has no matrix-free sweep leg at all, only its fiducial. **A crossover above the largest
measured N is not measured and is not extrapolated here.**

*Methodology note.* The committed aggregator
([`results/sweep/size_sweep_tables.md`](../sweep/size_sweep_tables.md)) builds its
matrix-free per-call figure from the **Jacobi-estimate** PCG row; the table above uses the
**fastest measured** PCG row (usually `exact diag`). Rectangular fiducial: 1462.284 ms
(aggregator) against 1435.308 ms (here). Both lose to the exact path by ~350×; the choice
changes no verdict. The aggregator also reports a whole-likelihood `per-call ms` column
for the exact paths, which is a *different quantity* from the three-row exact comparator
used here — read its `components` column for the like-for-like figure.

## What this says next

The lever is the positivity/NNLS row and the MGE conditioning, at every N — not the linear
solve, and not the log-dets.

1. **Fixed lens light after SLaM `light[1]` (a source-only inversion) is the next lever**,
   filed as `PyAutoMind/draft/research/autolens_profiling/fixed_lens_light_source_only_inversion.md`.
   A CPU falsifier run this session (session probe, uncommitted): with the MGE converted to
   regular light profiles, a certified active-set positive solve reaches the NNLS solution
   in **3 factorisations** on Delaunay against 17 PDIP iterations, and the evidence
   converges within 4 factorisations on rectangular (certification at 13). With the MGE
   still in the system the scheme never certifies.
2. **The PyAutoArray matrix-free phase is NOT filed.** This is a no-go, and the note is the
   record of why. A structure-aware preconditioner — rectangular λH is a grid Laplacian, so
   a DCT preconditioner is available — remains a possible research prompt, but it cannot fix
   the MGE-driven conditioning floor, which is what actually sets the CG iteration count.
3. **The w-tilde weight-map bug with regular light profiles is filed separately** as
   `PyAutoMind/draft/bug/autoarray/sparse_inversion_ignores_profile_subtracted_image.md`.

## Artifacts

```
results/breakdown/imaging/matrix_free_{rectangular,delaunay,delaunay_nn}_hpc_a100_fp64_matrix_free.{json,png}
results/breakdown/imaging/matrix_free_{rectangular,delaunay}_hpc_a100_fp64_matrix_free_n{3000,5000,12000}.{json,png}
results/breakdown/imaging/{pixelization,delaunay,delaunay_nn}_hpc_a100_fp64_n{3000,5000,8000,12000}{,_sparse}.{json,png}
results/sweep/size_sweep_manifest.jsonl        # one line per submitted leg
results/sweep/size_sweep_tables.md             # rendered aggregator output at d2a3848
```

Code:

```
scripts/misc/likelihood_breakdown/matrix_free_steps.py   # matvec, PCG, SLQ, matrix-free PDIP
scripts/imaging/likelihood_breakdown/matrix_free.py      # the fiducial + sweep cell
scripts/misc/likelihood_breakdown/size_sweep_table.py    # the aggregator
scripts/misc/likelihood_breakdown/_profile_cli.py        # --source-pixels N, fiducial-gated pins
hpc/batch_gpu/submit_breakdown_imaging_matrix_free_{pixelization,delaunay,delaunay_nn}_a100_hst_fp64
hpc/batch_gpu/submit_matrix_free_fiducial.sh             # the three fiducial legs
hpc/batch_gpu/submit_size_sweep_leg, hpc/batch_gpu/submit_size_sweep.sh
scripts/misc/test/test_matrix_free_steps.py
scripts/misc/test/test_size_sweep_table.py
```

Matrix-free JSON keys: `matrix_free.evidence` (`exact_terms`, `matrix_free_terms`,
`term_errors`, `log_evidence_error`, `go_bar_nats`, `within_go_bar`), `exact_log_dets`,
`cond_curvature_reg`, `peak_bytes`, `jit_phases`, `device.cache_fresh`; the three fiducial
legs additionally carry `unconstrained_pcg_terms`. The matrix-free cell does **not** emit
the full step-by-step pipeline rows (no `steps`, `setup_split`, `total_step_by_step`),
which is why the fiducial tables above read blurred-mapping-matrix / F / total from the
`_recon_split` legs.
