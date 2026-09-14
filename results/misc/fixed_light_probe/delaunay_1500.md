# Fixed-lens-light source-only probe — mesh delaunay, 1500 source pixels

`delaunay_1500.json` — autolens 2026.8.17.1, instrument hst, reg adapt_split, tau_rel = 1e-09, pass cap 40. CPU, fp64, dense.

`d log_ev` is nats against the reference for that convention — the library's own edge-zeroed reconstruction for `edge_zeroed`, the PDIP solution for `pure`. A **positive** value means an infeasible early iterate scoring above the constrained optimum, which is not a usable evidence.

| quantity | S0_current_mge60 | S3_mge_converted_to_regular | S3_mge_converted_to_regular__er_plus_1pct | S3_mge_converted_to_regular__er_minus_1pct | S3_mge_converted_to_regular__ell0_plus_0.01 |
|---|---|---|---|---|---|
| n_params | 1560 | 1500 | 1500 | 1500 | 1500 |
| n_funcs (MGE columns) | 60 | 0 | 0 | 0 | 0 |
| edge-zeroed pixels | 0 | 0 | 0 | 0 | 0 |
| cond(F+lH) | 4.107e+10 | 2.051e+06 | - | - | - |
| PDIP iterations | 22 | 17 | 17 | 17 | 17 |
| n(A*) tol 1e-08 | 17 | 5 | 11 | 2 | 5 |
| n(A*) tol 1e-06 | 17 | 5 | 11 | 2 | 5 |
| n(A*) tol 0.0001 | 36 | 6 | 11 | 3 | 6 |
| n(N) = #(x_unc < 0) | 24 | 4 | 8 | 2 | 4 |
| Jaccard(N, A* 1e-6) | 0.4138 | 0.8 | 0.7273 | 1 | 0.8 |
| log_ev(PDIP) | 2.914e+04 | 2.914e+04 | 2.774e+04 | 2.78e+04 | 2.862e+04 |
| log_ev(library, edge-zeroed) | 2.914e+04 | 2.914e+04 | 2.774e+04 | 2.78e+04 | 2.862e+04 |
| d log_ev unconstrained - PDIP | 27.44 | 6.404 | 18.42 | 2.478 | 7.093 |
| **edge_zeroed__free_all** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 2 | 2 | 1 | 2 |
|   factorisations | 41 | 3 | 3 | 2 | 3 |
|   max dev / max x_ref | 0.4885 | 6.663e-13 | 1.983e-13 | 2.468e-13 | 3.959e-13 |
|   d log_ev final | 6.74 | 2.547e-11 | 2.401e-10 | -3.274e-10 | 4.366e-11 |
|   Jaccard(Z_final, A* 1e-6) | 0.4 | 1 | 1 | 1 | 1 |
|   d log_ev at 1 pass(es) [raw] | -28.68 | 0.01472 | 0.3537 | -3.274e-10 | 0.04554 |
|   d log_ev at 1 pass(es) [clipped] | -1.618e+08 | -0.01051 | -0.1084 | -3.274e-10 | -0.01828 |
|   d log_ev at 2 pass(es) [raw] | -6.818 | 2.547e-11 | 2.401e-10 | - | 4.366e-11 |
|   d log_ev at 2 pass(es) [clipped] | -1.396e+06 | 2.547e-11 | 2.401e-10 | - | 4.366e-11 |
|   d log_ev at 3 pass(es) [raw] | 16.49 | - | - | - | - |
|   d log_ev at 3 pass(es) [clipped] | -5.872e+08 | - | - | - | - |
|   d log_ev at 4 pass(es) [raw] | -2.434 | - | - | - | - |
|   d log_ev at 4 pass(es) [clipped] | -5.761e+08 | - | - | - | - |
|   d log_ev at 6 pass(es) [raw] | 4.32 | - | - | - | - |
|   d log_ev at 6 pass(es) [clipped] | -3.261e+05 | - | - | - | - |
|   d log_ev at 8 pass(es) [raw] | -1.955 | - | - | - | - |
|   d log_ev at 8 pass(es) [clipped] | -1.274e+08 | - | - | - | - |
|   d log_ev at 12 pass(es) [raw] | -6.211 | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | -5.837e+06 | - | - | - | - |
| **edge_zeroed__free_all_warm_start** | | | | | |
|   certified | - | - | yes | yes | yes |
|   passes to certification | - | - | 3 | 2 | 1 |
|   factorisations | - | - | 3 | 2 | 1 |
|   max dev / max x_ref | - | - | 1.983e-13 | 2.468e-13 | 3.959e-13 |
|   d log_ev final | - | - | 2.401e-10 | -3.274e-10 | 4.366e-11 |
|   Jaccard(Z_final, A* 1e-6) | - | - | - | - | - |
|   d log_ev at 1 pass(es) [raw] | - | - | 3.411 | -2.151 | 4.366e-11 |
|   d log_ev at 1 pass(es) [clipped] | - | - | -0.5151 | -4.255 | 4.366e-11 |
|   d log_ev at 2 pass(es) [raw] | - | - | 0.04483 | -3.274e-10 | - |
|   d log_ev at 2 pass(es) [clipped] | - | - | -0.001057 | -3.274e-10 | - |
|   d log_ev at 3 pass(es) [raw] | - | - | 2.401e-10 | - | - |
|   d log_ev at 3 pass(es) [clipped] | - | - | 2.401e-10 | - | - |
|   d log_ev at 4 pass(es) [raw] | - | - | - | - | - |
|   d log_ev at 4 pass(es) [clipped] | - | - | - | - | - |
|   d log_ev at 6 pass(es) [raw] | - | - | - | - | - |
|   d log_ev at 6 pass(es) [clipped] | - | - | - | - | - |
|   d log_ev at 8 pass(es) [raw] | - | - | - | - | - |
|   d log_ev at 8 pass(es) [clipped] | - | - | - | - | - |
|   d log_ev at 12 pass(es) [raw] | - | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | - | - | - | - | - |
| **edge_zeroed__free_one** | | | | | |
|   certified | yes | yes | yes | yes | yes |
|   passes to certification | 32 | 2 | 2 | 1 | 2 |
|   factorisations | 33 | 3 | 3 | 2 | 3 |
|   max dev / max x_ref | 7.168e-11 | 6.663e-13 | 1.983e-13 | 2.468e-13 | 3.959e-13 |
|   d log_ev final | -8.731e-11 | 2.547e-11 | 2.401e-10 | -3.274e-10 | 4.366e-11 |
|   Jaccard(Z_final, A* 1e-6) | 1 | 1 | 1 | 1 | 1 |
|   d log_ev at 1 pass(es) [raw] | -28.68 | 0.01472 | 0.3537 | -3.274e-10 | 0.04554 |
|   d log_ev at 1 pass(es) [clipped] | -1.618e+08 | -0.01051 | -0.1084 | -3.274e-10 | -0.01828 |
|   d log_ev at 2 pass(es) [raw] | -24.7 | 2.547e-11 | 2.401e-10 | - | 4.366e-11 |
|   d log_ev at 2 pass(es) [clipped] | -2.021e+04 | 2.547e-11 | 2.401e-10 | - | 4.366e-11 |
|   d log_ev at 3 pass(es) [raw] | -17.76 | - | - | - | - |
|   d log_ev at 3 pass(es) [clipped] | -75.28 | - | - | - | - |
|   d log_ev at 4 pass(es) [raw] | -12.77 | - | - | - | - |
|   d log_ev at 4 pass(es) [clipped] | -13.13 | - | - | - | - |
|   d log_ev at 6 pass(es) [raw] | -10.78 | - | - | - | - |
|   d log_ev at 6 pass(es) [clipped] | -10.78 | - | - | - | - |
|   d log_ev at 8 pass(es) [raw] | -10.4 | - | - | - | - |
|   d log_ev at 8 pass(es) [clipped] | -10.4 | - | - | - | - |
|   d log_ev at 12 pass(es) [raw] | -6.554 | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | -6.554 | - | - | - | - |
| **pure__free_all** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 2 | 2 | 1 | 2 |
|   factorisations | 41 | 3 | 3 | 2 | 3 |
|   max dev / max x_ref | 0.4885 | 4.982e-13 | 2.2e-13 | 2.948e-13 | 3.23e-13 |
|   d log_ev final | 6.74 | 1.819e-10 | -7.276e-12 | -1.091e-10 | 3.238e-10 |
|   Jaccard(Z_final, A* 1e-6) | 0.4 | 1 | 1 | 1 | 1 |
|   d log_ev at 1 pass(es) [raw] | -28.68 | 0.01472 | 0.3537 | -1.091e-10 | 0.04554 |
|   d log_ev at 1 pass(es) [clipped] | -1.618e+08 | -0.01051 | -0.1084 | -1.091e-10 | -0.01828 |
|   d log_ev at 2 pass(es) [raw] | -6.818 | 1.819e-10 | -7.276e-12 | - | 3.238e-10 |
|   d log_ev at 2 pass(es) [clipped] | -1.396e+06 | 1.819e-10 | -7.276e-12 | - | 3.238e-10 |
|   d log_ev at 3 pass(es) [raw] | 16.49 | - | - | - | - |
|   d log_ev at 3 pass(es) [clipped] | -5.872e+08 | - | - | - | - |
|   d log_ev at 4 pass(es) [raw] | -2.434 | - | - | - | - |
|   d log_ev at 4 pass(es) [clipped] | -5.761e+08 | - | - | - | - |
|   d log_ev at 6 pass(es) [raw] | 4.32 | - | - | - | - |
|   d log_ev at 6 pass(es) [clipped] | -3.261e+05 | - | - | - | - |
|   d log_ev at 8 pass(es) [raw] | -1.955 | - | - | - | - |
|   d log_ev at 8 pass(es) [clipped] | -1.274e+08 | - | - | - | - |
|   d log_ev at 12 pass(es) [raw] | -6.211 | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | -5.837e+06 | - | - | - | - |
| **pure__free_one** | | | | | |
|   certified | yes | yes | yes | yes | yes |
|   passes to certification | 32 | 2 | 2 | 1 | 2 |
|   factorisations | 33 | 3 | 3 | 2 | 3 |
|   max dev / max x_ref | 3.487e-05 | 4.982e-13 | 2.2e-13 | 2.948e-13 | 3.23e-13 |
|   d log_ev final | 1.819e-11 | 1.819e-10 | -7.276e-12 | -1.091e-10 | 3.238e-10 |
|   Jaccard(Z_final, A* 1e-6) | 1 | 1 | 1 | 1 | 1 |
|   d log_ev at 1 pass(es) [raw] | -28.68 | 0.01472 | 0.3537 | -1.091e-10 | 0.04554 |
|   d log_ev at 1 pass(es) [clipped] | -1.618e+08 | -0.01051 | -0.1084 | -1.091e-10 | -0.01828 |
|   d log_ev at 2 pass(es) [raw] | -24.7 | 1.819e-10 | -7.276e-12 | - | 3.238e-10 |
|   d log_ev at 2 pass(es) [clipped] | -2.021e+04 | 1.819e-10 | -7.276e-12 | - | 3.238e-10 |
|   d log_ev at 3 pass(es) [raw] | -17.76 | - | - | - | - |
|   d log_ev at 3 pass(es) [clipped] | -75.28 | - | - | - | - |
|   d log_ev at 4 pass(es) [raw] | -12.77 | - | - | - | - |
|   d log_ev at 4 pass(es) [clipped] | -13.13 | - | - | - | - |
|   d log_ev at 6 pass(es) [raw] | -10.78 | - | - | - | - |
|   d log_ev at 6 pass(es) [clipped] | -10.78 | - | - | - | - |
|   d log_ev at 8 pass(es) [raw] | -10.4 | - | - | - | - |
|   d log_ev at 8 pass(es) [clipped] | -10.4 | - | - | - | - |
|   d log_ev at 12 pass(es) [raw] | -6.554 | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | -6.554 | - | - | - | - |
| **masked JAX (edge-zeroed, free_all)** | | | | | |
|   pass budget | 6 | 2 | 2 | 1 | 2 |
|   certified at pass | - | 2 | 2 | 1 | 2 |
|   max dev vs library / max x | 0.8873 | 6.644e-13 | 2.023e-13 | 3.193e-13 | 3.511e-13 |
|   d log_ev | 4.32 | -1.819e-11 | -3.274e-11 | -5.093e-11 | -2.474e-10 |

Per-pass fixed-set history, `edge_zeroed__free_all` (pass: |Z|, primal viol., dual viol.):

- `S0_current_mge60` — 1:(24,3,6), 2:(21,5,12), 3:(14,8,6), 4:(16,5,10), 5:(11,8,2), 6:(17,3,8), 7:(12,7,4), 8:(15,5,10), 9:(10,8,2), 10:(16,4,7), 11:(13,6,3), 12:(16,7,5), 13:(18,5,8), 14:(15,4,6), 15:(13,6,6), 16:(13,5,5), 17:(13,7,6), 18:(14,5,2), 19:(17,4,10), 20:(11,7,3) ...
- `S3_mge_converted_to_regular` — 1:(4,1,0), 2:(5,0,0)
- `S3_mge_converted_to_regular__er_plus_1pct` — 1:(8,3,0), 2:(11,0,0)
- `S3_mge_converted_to_regular__er_minus_1pct` — 1:(2,0,0)
- `S3_mge_converted_to_regular__ell0_plus_0.01` — 1:(4,1,0), 2:(5,0,0)

Cross-draw Jaccard of A* (tol 1e-6):

| pair | Jaccard |
|---|---|
| er_plus_1pct vs fiducial | 0.455 |
| er_minus_1pct vs fiducial | 0.167 |
| er_minus_1pct vs er_plus_1pct | 0.182 |
| ell0_plus_0.01 vs fiducial | 1.000 |
| ell0_plus_0.01 vs er_plus_1pct | 0.455 |
| ell0_plus_0.01 vs er_minus_1pct | 0.167 |
