# Fixed-lens-light source-only probe — mesh rectangular, 400 source pixels

`rectangular_400.json` — autolens 2026.8.17.1, instrument hst, reg constant, tau_rel = 1e-09, pass cap 40. CPU, fp64, dense.

`d log_ev` is nats against the reference for that convention — the library's own edge-zeroed reconstruction for `edge_zeroed`, the PDIP solution for `pure`. A **positive** value means an infeasible early iterate scoring above the constrained optimum, which is not a usable evidence.

| quantity | S0_current_mge60 | S3_mge_converted_to_regular | S3_mge_converted_to_regular__er_plus_1pct | S3_mge_converted_to_regular__er_minus_1pct | S3_mge_converted_to_regular__ell0_plus_0.01 |
|---|---|---|---|---|---|
| n_params | 460 | 400 | 400 | 400 | 400 |
| n_funcs (MGE columns) | 60 | 0 | 0 | 0 | 0 |
| edge-zeroed pixels | 76 | 76 | 76 | 76 | 76 |
| cond(F+lH) | 4.16e+10 | 1.306e+07 | - | - | - |
| PDIP iterations | 19 | 15 | 16 | 16 | 16 |
| n(A*) tol 1e-08 | 29 | 36 | 42 | 42 | 40 |
| n(A*) tol 1e-06 | 29 | 36 | 42 | 42 | 41 |
| n(A*) tol 0.0001 | 49 | 51 | 60 | 55 | 55 |
| n(N) = #(x_unc < 0) | 78 | 59 | 66 | 59 | 68 |
| Jaccard(N, A* 1e-6) | 0.2892 | 0.4179 | 0.4026 | 0.3836 | 0.4342 |
| log_ev(PDIP) | 2.916e+04 | 2.907e+04 | 2.764e+04 | 2.735e+04 | 2.857e+04 |
| log_ev(library, edge-zeroed) | 2.885e+04 | 2.885e+04 | 2.742e+04 | 2.713e+04 | 2.834e+04 |
| d log_ev unconstrained - PDIP | 21.66 | 104.4 | 112.3 | 99.18 | 95.91 |
| **edge_zeroed__free_all** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 4 | 5 | 5 | 5 |
|   factorisations | 41 | 5 | 6 | 6 | 6 |
|   max dev / max x_ref | 0.7001 | 2.423e-14 | 2.928e-14 | 1.68e-14 | 2.459e-14 |
|   d log_ev final | 3.092 | 0 | 0 | 3.638e-12 | 0 |
|   Jaccard(Z_final, A* 1e-6) | 0.2083 | 0.3462 | 0.3784 | 0.3818 | 0.3761 |
|   d log_ev at 1 pass(es) [raw] | 11.62 | 9.863 | 11.73 | 6.537 | 7.87 |
|   d log_ev at 1 pass(es) [clipped] | -4.177e+06 | -10.1 | -8.215 | -4.747 | -2.993 |
|   d log_ev at 2 pass(es) [raw] | -6.829 | 0.5774 | 0.9908 | 0.6067 | 0.006064 |
|   d log_ev at 2 pass(es) [clipped] | -7.91 | -0.5905 | -0.9978 | -0.4707 | -0.001592 |
|   d log_ev at 3 pass(es) [raw] | 3.695 | 0.1244 | -1.106e-09 | -1.349e-07 | -3.152e-08 |
|   d log_ev at 3 pass(es) [clipped] | -5.302e+07 | -0.1077 | -1.106e-09 | -1.349e-07 | -3.152e-08 |
|   d log_ev at 4 pass(es) [raw] | 2.71 | 0 | -7.276e-11 | -1.157e-08 | -2.983e-10 |
|   d log_ev at 4 pass(es) [clipped] | -9.479e+05 | 0 | -7.276e-11 | -1.157e-08 | -2.983e-10 |
|   d log_ev at 6 pass(es) [raw] | 2.719 | - | - | - | - |
|   d log_ev at 6 pass(es) [clipped] | -2.091e+05 | - | - | - | - |
|   d log_ev at 8 pass(es) [raw] | 2.718 | - | - | - | - |
|   d log_ev at 8 pass(es) [clipped] | -2.686e+08 | - | - | - | - |
|   d log_ev at 12 pass(es) [raw] | -4.908 | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | -5.463e+06 | - | - | - | - |
| **edge_zeroed__free_all_warm_start** | | | | | |
|   certified | - | - | yes | yes | yes |
|   passes to certification | - | - | 6 | 4 | 6 |
|   factorisations | - | - | 6 | 4 | 6 |
|   max dev / max x_ref | - | - | 2.928e-14 | 1.68e-14 | 2.459e-14 |
|   d log_ev final | - | - | 0 | 3.638e-12 | 0 |
|   Jaccard(Z_final, A* 1e-6) | - | - | - | - | - |
|   d log_ev at 1 pass(es) [raw] | - | - | 7.681 | 1.679 | -5.213 |
|   d log_ev at 1 pass(es) [clipped] | - | - | -10.93 | -4.777 | -12.41 |
|   d log_ev at 2 pass(es) [raw] | - | - | -0.03255 | -4.777e-07 | 3.851 |
|   d log_ev at 2 pass(es) [clipped] | - | - | -0.04448 | -4.777e-07 | -16.1 |
|   d log_ev at 3 pass(es) [raw] | - | - | -1.548e-08 | -3.634e-08 | 0.05185 |
|   d log_ev at 3 pass(es) [clipped] | - | - | -1.548e-08 | -3.634e-08 | -0.009903 |
|   d log_ev at 4 pass(es) [raw] | - | - | -1.106e-09 | 3.638e-12 | -3.321e-07 |
|   d log_ev at 4 pass(es) [clipped] | - | - | -1.106e-09 | 3.638e-12 | -3.321e-07 |
|   d log_ev at 6 pass(es) [raw] | - | - | 0 | - | 0 |
|   d log_ev at 6 pass(es) [clipped] | - | - | 0 | - | 0 |
|   d log_ev at 8 pass(es) [raw] | - | - | - | - | - |
|   d log_ev at 8 pass(es) [clipped] | - | - | - | - | - |
|   d log_ev at 12 pass(es) [raw] | - | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | - | - | - | - | - |
| **edge_zeroed__free_one** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 13 | 13 | 13 | 12 |
|   factorisations | 41 | 14 | 14 | 14 | 13 |
|   max dev / max x_ref | 1.274 | 2.423e-14 | 2.928e-14 | 1.68e-14 | 2.459e-14 |
|   d log_ev final | -4.942 | 0 | 0 | 3.638e-12 | 0 |
|   Jaccard(Z_final, A* 1e-6) | 0.1724 | 0.3462 | 0.3784 | 0.3818 | 0.3761 |
|   d log_ev at 1 pass(es) [raw] | 11.62 | 9.863 | 11.73 | 6.537 | 7.87 |
|   d log_ev at 1 pass(es) [clipped] | -4.177e+06 | -10.1 | -8.215 | -4.747 | -2.993 |
|   d log_ev at 2 pass(es) [raw] | -13.45 | 0.5773 | 0.2658 | 0.4674 | 0.006063 |
|   d log_ev at 2 pass(es) [clipped] | -7630 | -0.5905 | -0.5852 | -0.3867 | -0.001593 |
|   d log_ev at 3 pass(es) [raw] | -14.29 | 0.1207 | -0.1431 | -0.03791 | -1.388e-06 |
|   d log_ev at 3 pass(es) [clipped] | -14.29 | -0.1114 | -0.1431 | -0.03791 | -1.388e-06 |
|   d log_ev at 4 pass(es) [raw] | -8.788 | -1.589e-05 | -4.493e-06 | -2.23e-06 | -6.051e-07 |
|   d log_ev at 4 pass(es) [clipped] | -90.3 | -1.589e-05 | -4.493e-06 | -2.23e-06 | -6.051e-07 |
|   d log_ev at 6 pass(es) [raw] | -3.286 | -4.273e-07 | -4.318e-07 | -6.028e-07 | -1.032e-07 |
|   d log_ev at 6 pass(es) [clipped] | -3.971 | -4.273e-07 | -4.318e-07 | -6.028e-07 | -1.032e-07 |
|   d log_ev at 8 pass(es) [raw] | -2.812 | -5.193e-08 | -6.14e-08 | -7.681e-08 | -2.447e-08 |
|   d log_ev at 8 pass(es) [clipped] | -2.812 | -5.193e-08 | -6.14e-08 | -7.681e-08 | -2.447e-08 |
|   d log_ev at 12 pass(es) [raw] | -6.792 | -8.367e-11 | -7.276e-11 | -2.274e-09 | 0 |
|   d log_ev at 12 pass(es) [clipped] | -6.792 | -8.367e-11 | -7.276e-11 | -2.274e-09 | 0 |
| **pure__free_all** | | | | | |
|   certified | no | yes | yes | yes | yes |
|   passes to certification | - | 10 | 10 | 9 | 10 |
|   factorisations | 41 | 11 | 11 | 10 | 11 |
|   max dev / max x_ref | 1.257 | 7.055e-07 | 3.835e-08 | 2.749e-07 | 3.53e-07 |
|   d log_ev final | 6.907 | 1.455e-11 | 0 | 3.638e-12 | 7.276e-12 |
|   Jaccard(Z_final, A* 1e-6) | 0.5938 | 1 | 1 | 1 | 0.9756 |
|   d log_ev at 1 pass(es) [raw] | 1.336 | 9.826 | 11.87 | 6.963 | 7.874 |
|   d log_ev at 1 pass(es) [clipped] | -1.897e+07 | -10.05 | -10.01 | -5.255 | -3.005 |
|   d log_ev at 2 pass(es) [raw] | -8.566 | 0.5773 | 0.982 | 0.6063 | 0.005965 |
|   d log_ev at 2 pass(es) [clipped] | -57.94 | -0.5902 | -0.9909 | -0.4705 | -0.001578 |
|   d log_ev at 3 pass(es) [raw] | 4.869 | 0.1245 | -1.465e-05 | -4.781e-06 | -2.194e-06 |
|   d log_ev at 3 pass(es) [clipped] | -1.288e+08 | -0.1078 | -1.465e-05 | -4.781e-06 | -2.194e-06 |
|   d log_ev at 4 pass(es) [raw] | 8.421 | -4.201e-07 | -2.221e-06 | -7.95e-07 | -1.873e-07 |
|   d log_ev at 4 pass(es) [clipped] | -7.876e+08 | -4.201e-07 | -2.221e-06 | -7.95e-07 | -1.873e-07 |
|   d log_ev at 6 pass(es) [raw] | 7.497 | -8.127e-09 | -4.17e-08 | -7.24e-10 | -1.051e-09 |
|   d log_ev at 6 pass(es) [clipped] | -9.772e+06 | -8.127e-09 | -4.17e-08 | -7.24e-10 | -1.051e-09 |
|   d log_ev at 8 pass(es) [raw] | 8.847 | -1.673e-10 | -9.386e-10 | -1.819e-11 | -2.183e-11 |
|   d log_ev at 8 pass(es) [clipped] | -2.723e+08 | -1.673e-10 | -9.386e-10 | -1.819e-11 | -2.183e-11 |
|   d log_ev at 12 pass(es) [raw] | 8.853 | - | - | - | - |
|   d log_ev at 12 pass(es) [clipped] | -7.398e+08 | - | - | - | - |
| **pure__free_one** | | | | | |
|   certified | no | yes | no | yes | yes |
|   passes to certification | - | 39 | - | 38 | 40 |
|   factorisations | 41 | 40 | 41 | 39 | 41 |
|   max dev / max x_ref | 0.637 | 7.055e-07 | 2.809e-06 | 2.749e-07 | 3.53e-07 |
|   d log_ev final | -0.1697 | 1.455e-11 | -1.783e-10 | 3.638e-12 | 7.276e-12 |
|   Jaccard(Z_final, A* 1e-6) | 0.4828 | 1 | 1 | 1 | 0.9756 |
|   d log_ev at 1 pass(es) [raw] | 1.336 | 9.826 | 11.87 | 6.963 | 7.874 |
|   d log_ev at 1 pass(es) [clipped] | -1.897e+07 | -10.05 | -10.01 | -5.255 | -3.005 |
|   d log_ev at 2 pass(es) [raw] | -14.29 | 0.5772 | 0.7282 | 0.5378 | 0.005938 |
|   d log_ev at 2 pass(es) [clipped] | -2085 | -0.5903 | -1.1 | -0.4777 | -0.001605 |
|   d log_ev at 3 pass(es) [raw] | -16.56 | 0.1207 | -0.1429 | -0.0001731 | -3.484e-05 |
|   d log_ev at 3 pass(es) [clipped] | -766.7 | -0.1116 | -0.1429 | -0.0001731 | -3.484e-05 |
|   d log_ev at 4 pass(es) [raw] | -15.15 | -0.0001313 | -7.98e-05 | -7.53e-05 | -2.943e-05 |
|   d log_ev at 4 pass(es) [clipped] | -15.15 | -0.0001313 | -7.98e-05 | -7.53e-05 | -2.943e-05 |
|   d log_ev at 6 pass(es) [raw] | -13.94 | -4.12e-05 | -5.321e-05 | -3.733e-05 | -2.036e-05 |
|   d log_ev at 6 pass(es) [clipped] | -13.95 | -4.12e-05 | -5.321e-05 | -3.733e-05 | -2.036e-05 |
|   d log_ev at 8 pass(es) [raw] | -12.56 | -1.825e-05 | -2.629e-05 | -2.189e-05 | -1.549e-05 |
|   d log_ev at 8 pass(es) [clipped] | -12.56 | -1.825e-05 | -2.629e-05 | -2.189e-05 | -1.549e-05 |
|   d log_ev at 12 pass(es) [raw] | -11.92 | -6.194e-06 | -7.912e-06 | -7.729e-06 | -7.757e-06 |
|   d log_ev at 12 pass(es) [clipped] | -11.92 | -6.194e-06 | -7.912e-06 | -7.729e-06 | -7.757e-06 |
| **masked JAX (edge-zeroed, free_all)** | | | | | |
|   pass budget | 6 | 4 | 5 | 5 | 5 |
|   certified at pass | - | 4 | 5 | 5 | 5 |
|   max dev vs library / max x | 0.5703 | 2.848e-14 | 2.928e-14 | 1.701e-14 | 2.386e-14 |
|   d log_ev | 2.719 | 0 | 0 | 0 | 0 |

Per-pass fixed-set history, `edge_zeroed__free_all` (pass: |Z|, primal viol., dual viol.):

- `S0_current_mge60` — 1:(117,14,8), 2:(123,1,14), 3:(110,9,4), 4:(115,3,6), 5:(112,7,3), 6:(116,5,4), 7:(117,3,3), 8:(117,4,3), 9:(118,3,7), 10:(114,6,4), 11:(116,4,2), 12:(118,6,6), 13:(118,4,9), 14:(113,7,6), 15:(114,5,2), 16:(117,2,4), 17:(115,5,1), 18:(119,2,12), 19:(109,8,4), 20:(113,4,4) ...
- `S3_mge_converted_to_regular` — 1:(105,7,5), 2:(107,3,4), 3:(106,1,3), 4:(104,0,0)
- `S3_mge_converted_to_regular__er_plus_1pct` — 1:(111,7,7), 2:(111,5,3), 3:(113,0,1), 4:(112,0,1), 5:(111,0,0)
- `S3_mge_converted_to_regular__er_minus_1pct` — 1:(108,12,5), 2:(115,2,2), 3:(115,0,3), 4:(112,0,2), 5:(110,0,0)
- `S3_mge_converted_to_regular__ell0_plus_0.01` — 1:(112,7,3), 2:(116,1,4), 3:(113,0,3), 4:(110,0,1), 5:(109,0,0)

Cross-draw Jaccard of A* (tol 1e-6):

| pair | Jaccard |
|---|---|
| er_plus_1pct vs fiducial | 0.733 |
| er_minus_1pct vs fiducial | 0.660 |
| er_minus_1pct vs er_plus_1pct | 0.585 |
| ell0_plus_0.01 vs fiducial | 0.481 |
| ell0_plus_0.01 vs er_plus_1pct | 0.537 |
| ell0_plus_0.01 vs er_minus_1pct | 0.482 |
