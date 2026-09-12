# Fixed-lens-light source-only probe — mesh rectangular, 1521 source pixels

`rectangular_1521.json` — autolens 2026.8.17.1, instrument hst, reg constant, tau_rel = 1e-09, pass cap 40. CPU, fp64, dense.

`d log_ev` is nats against the reference for that convention — the library's own edge-zeroed reconstruction for `edge_zeroed`, the PDIP solution for `pure`. A **positive** value means an infeasible early iterate scoring above the constrained optimum, which is not a usable evidence.

| quantity | S0_current_mge60 | S3_mge_converted_to_regular | S3_mge_converted_to_regular__er_plus_1pct | S3_mge_converted_to_regular__er_minus_1pct | S3_mge_converted_to_regular__ell0_plus_0.01 |
|---|---|---|---|---|---|
| n_params | 1581 | 1521 | 1521 | 1521 | 1521 |
| n_funcs (MGE columns) | 60 | 0 | 0 | 0 | 0 |
| edge-zeroed pixels | 152 | 152 | 152 | 152 | 152 |
| cond(F+lH) | 4.117e+10 | 1.208e+07 | - | - | - |
| PDIP iterations | 21 | 15 | 15 | 15 | 15 |
| n(A*) tol 1e-08 | 73 | 103 | 130 | 124 | 103 |
| n(A*) tol 1e-06 | 73 | 103 | 130 | 124 | 103 |
| n(A*) tol 0.0001 | 93 | 138 | 157 | 158 | 143 |
| n(N) = #(x_unc < 0) | 171 | 136 | 155 | 137 | 127 |
| Jaccard(N, A* 1e-6) | 0.3407 | 0.4398 | 0.4179 | 0.4581 | 0.4024 |
| log_ev(PDIP) | 2.893e+04 | 2.884e+04 | 2.829e+04 | 2.823e+04 | 2.87e+04 |
| log_ev(library, edge-zeroed) | 2.862e+04 | 2.862e+04 | 2.808e+04 | 2.802e+04 | 2.848e+04 |
| d log_ev unconstrained - PDIP | 34.98 | 119.7 | 125.2 | 141.4 | 120.2 |
| **edge_zeroed__free_all** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 7 | 7 | 7 | 8 |
|   factorisations | 41 | 8 | 8 | 8 | 9 |
|   max dev / max x_ref | 0.2048 | 6.432e-14 | 9.618e-14 | 4.424e-14 | 6.714e-14 |
|   d log_ev final | 1.076 | 0 | 0 | 0 | 0 |
|   Jaccard(Z_final, A* 1e-6) | 0.2823 | 0.434 | 0.4962 | 0.4788 | 0.4328 |
|   d log_ev at 1 pass(es) [raw] | 17.55 | 16.08 | 14.06 | 13.76 | 10.83 |
|   d log_ev at 1 pass(es) [clipped] | -4.07e+05 | -12.73 | -7.882 | -10.75 | -7.229 |
|   d log_ev at 2 pass(es) [raw] | 6.472 | 5.581 | 3.035 | 4.278 | 4.618 |
|   d log_ev at 2 pass(es) [clipped] | -1.177e+08 | -2.029 | -0.4546 | -1.011 | -2.266 |
|   d log_ev at 3 pass(es) [raw] | -14.85 | 0.3449 | -0.0003824 | 0.04531 | 0.1008 |
|   d log_ev at 3 pass(es) [clipped] | -15.59 | -0.5957 | -0.001202 | -0.04205 | -0.03494 |
|   d log_ev at 4 pass(es) [raw] | 2.65 | -4.431e-09 | -3.667e-09 | -3.471e-09 | 9.518e-05 |
|   d log_ev at 4 pass(es) [clipped] | -9.302e+07 | -4.431e-09 | -3.667e-09 | -3.471e-09 | -5.623e-05 |
|   d log_ev at 6 pass(es) [raw] | 3.328 | -2.91e-11 | -2.547e-11 | -2.183e-11 | -3.638e-11 |
|   d log_ev at 6 pass(es) [clipped] | -1.264e+08 | -2.91e-11 | -2.547e-11 | -2.183e-11 | -3.638e-11 |
|   d log_ev at 8 pass(es) [raw] | -12.21 | - | - | - | 0 |
|   d log_ev at 8 pass(es) [clipped] | -3.135e+04 | - | - | - | 0 |
|   d log_ev at 12 pass(es) [raw] | 1.537 | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | -2.693e+05 | - | - | - | - |
| **edge_zeroed__free_all_warm_start** | | | | | |
|   certified | - | - | yes | yes | yes |
|   passes to certification | - | - | 6 | 6 | 8 |
|   factorisations | - | - | 6 | 6 | 8 |
|   max dev / max x_ref | - | - | 9.618e-14 | 4.424e-14 | 6.714e-14 |
|   d log_ev final | - | - | 0 | 0 | 0 |
|   Jaccard(Z_final, A* 1e-6) | - | - | - | - | - |
|   d log_ev at 1 pass(es) [raw] | - | - | 0.8373 | -17.97 | -17.73 |
|   d log_ev at 1 pass(es) [clipped] | - | - | -33.82 | -118.9 | -51.36 |
|   d log_ev at 2 pass(es) [raw] | - | - | 0.9646 | 0.8022 | 4.355 |
|   d log_ev at 2 pass(es) [clipped] | - | - | -4.895 | -2.76 | -28.38 |
|   d log_ev at 3 pass(es) [raw] | - | - | -1.558e-06 | -0.03952 | 0.5291 |
|   d log_ev at 3 pass(es) [clipped] | - | - | -1.558e-06 | -0.04224 | -0.6166 |
|   d log_ev at 4 pass(es) [raw] | - | - | -4.539e-08 | -3.779e-08 | 0.006472 |
|   d log_ev at 4 pass(es) [clipped] | - | - | -4.539e-08 | -3.779e-08 | -0.008019 |
|   d log_ev at 6 pass(es) [raw] | - | - | 0 | 0 | -3.965e-10 |
|   d log_ev at 6 pass(es) [clipped] | - | - | 0 | 0 | -3.965e-10 |
|   d log_ev at 8 pass(es) [raw] | - | - | - | - | 0 |
|   d log_ev at 8 pass(es) [clipped] | - | - | - | - | 0 |
|   d log_ev at 12 pass(es) [raw] | - | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | - | - | - | - | - |
| **edge_zeroed__free_one** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 30 | 25 | 29 | 30 |
|   factorisations | 41 | 31 | 26 | 30 | 31 |
|   max dev / max x_ref | 0.8031 | 6.432e-14 | 9.618e-14 | 4.424e-14 | 6.714e-14 |
|   d log_ev final | -4.563 | 0 | 0 | 0 | 0 |
|   Jaccard(Z_final, A* 1e-6) | 0.2473 | 0.434 | 0.4962 | 0.4788 | 0.4328 |
|   d log_ev at 1 pass(es) [raw] | 17.55 | 16.08 | 14.06 | 13.76 | 10.83 |
|   d log_ev at 1 pass(es) [clipped] | -4.07e+05 | -12.73 | -7.882 | -10.75 | -7.229 |
|   d log_ev at 2 pass(es) [raw] | -25.19 | 2.498 | 2.774 | 3.003 | 0.8762 |
|   d log_ev at 2 pass(es) [clipped] | -3.538e+04 | -1.243 | -0.546 | -0.9344 | -0.9461 |
|   d log_ev at 3 pass(es) [raw] | -31.75 | -0.2938 | -0.04869 | -0.2009 | -0.3271 |
|   d log_ev at 3 pass(es) [clipped] | -32 | -0.3485 | -0.04869 | -0.2048 | -0.4878 |
|   d log_ev at 4 pass(es) [raw] | -10.21 | -0.2578 | -0.0116 | -0.137 | -0.2701 |
|   d log_ev at 4 pass(es) [clipped] | -730.6 | -0.2603 | -0.0116 | -0.137 | -0.2701 |
|   d log_ev at 6 pass(es) [raw] | -10.34 | -0.05716 | -1.269e-05 | -0.03988 | -0.003998 |
|   d log_ev at 6 pass(es) [clipped] | -10.34 | -0.05716 | -1.269e-05 | -0.03988 | -0.003998 |
|   d log_ev at 8 pass(es) [raw] | -10.22 | -0.005648 | -2.374e-06 | -0.01269 | -3.521e-05 |
|   d log_ev at 8 pass(es) [clipped] | -10.22 | -0.005648 | -2.374e-06 | -0.01269 | -3.521e-05 |
|   d log_ev at 12 pass(es) [raw] | -6.021 | -8.885e-06 | -2.047e-07 | -2.747e-06 | -1.865e-06 |
|   d log_ev at 12 pass(es) [clipped] | -6.021 | -8.885e-06 | -2.047e-07 | -2.747e-06 | -1.865e-06 |
| **pure__free_all** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 12 | 12 | 13 | 13 |
|   factorisations | 41 | 13 | 13 | 14 | 14 |
|   max dev / max x_ref | 1.7 | 2.354e-06 | 2.517e-06 | 2.11e-06 | 2.45e-06 |
|   d log_ev final | 12.3 | 2.692e-10 | 2.401e-10 | 2.183e-10 | 4.839e-10 |
|   Jaccard(Z_final, A* 1e-6) | 0.6538 | 1 | 1 | 1 | 0.981 |
|   d log_ev at 1 pass(es) [raw] | -2.387 | 16.67 | 14.86 | 12.69 | 11.1 |
|   d log_ev at 1 pass(es) [clipped] | -8.645e+05 | -13.51 | -11.27 | -13.15 | -8.284 |
|   d log_ev at 2 pass(es) [raw] | -5.151 | 5.609 | 3.245 | 4.387 | 4.616 |
|   d log_ev at 2 pass(es) [clipped] | -3.598e+07 | -2.033 | -1.128 | -2.365 | -2.25 |
|   d log_ev at 3 pass(es) [raw] | -0.1693 | 0.3436 | 0.002435 | 0.03169 | 0.1009 |
|   d log_ev at 3 pass(es) [clipped] | -3.153e+05 | -0.5954 | -0.008915 | -0.02676 | -0.03464 |
|   d log_ev at 4 pass(es) [raw] | -24.47 | -2.903e-06 | 0.0003446 | -6.709e-07 | -5.374e-06 |
|   d log_ev at 4 pass(es) [clipped] | -2.476e+04 | -2.903e-06 | -0.000322 | -6.709e-07 | -1.018e-05 |
|   d log_ev at 6 pass(es) [raw] | 1.684 | -1.466e-07 | -1.047e-07 | -6.366e-08 | -1.41e-07 |
|   d log_ev at 6 pass(es) [clipped] | -5.387e+09 | -1.466e-07 | -1.047e-07 | -6.366e-08 | -1.41e-07 |
|   d log_ev at 8 pass(es) [raw] | 5.713 | 2.437e-10 | 1.855e-10 | 1.346e-10 | 2.365e-10 |
|   d log_ev at 8 pass(es) [clipped] | -1.901e+07 | 2.437e-10 | 1.855e-10 | 1.346e-10 | 2.365e-10 |
|   d log_ev at 12 pass(es) [raw] | 7.294 | 2.692e-10 | 2.401e-10 | 2.183e-10 | 4.839e-10 |
|   d log_ev at 12 pass(es) [clipped] | -1.093e+06 | 2.692e-10 | 2.401e-10 | 2.183e-10 | 4.839e-10 |
| **pure__free_one** | | | | | |
|   certified | no | no | no | no | no |
|   passes to certification | - | - | - | - | - |
|   factorisations | 41 | 41 | 41 | 41 | 41 |
|   max dev / max x_ref | 1.041 | 0.0003423 | 0.0004664 | 0.0003631 | 0.0002385 |
|   d log_ev final | -3.675 | -7.729e-06 | -1.791e-05 | -6.359e-06 | -2.757e-06 |
|   Jaccard(Z_final, A* 1e-6) | 0.4201 | 0.7103 | 0.7182 | 0.7515 | 0.7007 |
|   d log_ev at 1 pass(es) [raw] | -2.387 | 16.67 | 14.86 | 12.69 | 11.1 |
|   d log_ev at 1 pass(es) [clipped] | -8.645e+05 | -13.51 | -11.27 | -13.15 | -8.284 |
|   d log_ev at 2 pass(es) [raw] | -27.6 | 2.503 | 2.596 | 2.723 | 0.8375 |
|   d log_ev at 2 pass(es) [clipped] | -7340 | -1.285 | -0.7249 | -1.219 | -0.9659 |
|   d log_ev at 3 pass(es) [raw] | -37.22 | -0.3047 | -0.2681 | -0.4835 | -0.3638 |
|   d log_ev at 3 pass(es) [clipped] | -1934 | -0.3594 | -0.2788 | -0.4874 | -0.5246 |
|   d log_ev at 4 pass(es) [raw] | -41.08 | -0.2688 | -0.1414 | -0.4025 | -0.3071 |
|   d log_ev at 4 pass(es) [clipped] | -41.08 | -0.2713 | -0.1414 | -0.4026 | -0.3071 |
|   d log_ev at 6 pass(es) [raw] | -16.51 | -0.04613 | -0.02534 | -0.219 | -0.01889 |
|   d log_ev at 6 pass(es) [clipped] | -16.51 | -0.04613 | -0.02534 | -0.219 | -0.01889 |
|   d log_ev at 8 pass(es) [raw] | -10.13 | -0.0007635 | -0.003773 | -0.04932 | -0.0005984 |
|   d log_ev at 8 pass(es) [clipped] | -10.13 | -0.0007635 | -0.003773 | -0.04932 | -0.0005984 |
|   d log_ev at 12 pass(es) [raw] | -8.211 | -0.0004512 | -0.0004636 | -0.006649 | -0.0003768 |
|   d log_ev at 12 pass(es) [clipped] | -8.211 | -0.0004512 | -0.0004636 | -0.006649 | -0.0003768 |
| **masked JAX (edge-zeroed, free_all)** | | | | | |
|   pass budget | 6 | 7 | 7 | 7 | 8 |
|   certified at pass | - | 7 | 7 | 7 | 8 |
|   max dev vs library / max x | 1.234 | 6.121e-14 | 9.548e-14 | 4.346e-14 | 6.546e-14 |
|   d log_ev | 3.328 | 0 | 0 | 0 | 0 |

Per-pass fixed-set history, `edge_zeroed__free_all` (pass: |Z|, primal viol., dual viol.):

- `S0_current_mge60` — 1:(249,27,21), 2:(255,11,7), 3:(259,6,13), 4:(252,7,6), 5:(253,6,20), 6:(239,14,10), 7:(243,9,3), 8:(249,5,10), 9:(244,8,7), 10:(245,5,6), 11:(244,6,5), 12:(245,4,6), 13:(243,6,2), 14:(247,2,6), 15:(243,4,2), 16:(245,2,5), 17:(242,5,0), 18:(247,2,4), 19:(245,4,2), 20:(247,1,7) ...
- `S3_mge_converted_to_regular` — 1:(231,26,20), 2:(237,11,7), 3:(241,3,4), 4:(240,0,2), 5:(238,0,2), 6:(236,0,2), 7:(234,0,0)
- `S3_mge_converted_to_regular__er_plus_1pct` — 1:(242,36,9), 2:(269,11,8), 3:(272,1,5), 4:(268,0,3), 5:(265,0,2), 6:(263,0,1), 7:(262,0,0)
- `S3_mge_converted_to_regular__er_minus_1pct` — 1:(241,31,13), 2:(259,17,12), 3:(264,4,3), 4:(265,0,2), 5:(263,0,2), 6:(261,0,2), 7:(259,0,0)
- `S3_mge_converted_to_regular__ell0_plus_0.01` — 1:(226,27,19), 2:(234,20,6), 3:(248,3,5), 4:(246,1,3), 5:(244,0,2), 6:(242,0,2), 7:(240,0,2), 8:(238,0,0)

Cross-draw Jaccard of A* (tol 1e-6):

| pair | Jaccard |
|---|---|
| er_plus_1pct vs fiducial | 0.513 |
| er_minus_1pct vs fiducial | 0.474 |
| er_minus_1pct vs er_plus_1pct | 0.373 |
| ell0_plus_0.01 vs fiducial | 0.441 |
| ell0_plus_0.01 vs er_plus_1pct | 0.421 |
| ell0_plus_0.01 vs er_minus_1pct | 0.367 |
