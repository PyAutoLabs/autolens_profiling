# Fixed lens light — source-pixel scaling — `local_rtx2060_fp64`

RTX 2060 Max-Q 6 GB, fp64, single call. Config name `local_rtx2060_fp64_fixed_light`.

Every cell is measured. `—` means the leg produced no value; `OOM` /
`timeout` / `failed` / `not run` in the *fits* table say why. Nothing in
this file is extrapolated.

## rectangular

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ 0.50 GB | ✓ 1.02 GB | ✓ 1.54 GB | ✓ 2.53 GB | ✓ 3.13 GB |
| certifying budget | 5 | 8 | 7 | 10 | 6 |
| PDIP iterations | 15 | 15 | 15 | 15 | 15 |

### Per call, ms

| Row | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 11.2 | 44.4 | 89.5 | 467.6 | 1,091 | 2.24 | 0.985 |
| certified active set (phase 3 safe budget) | 21.7 | 58.9 | 134.6 | 504.7 | 1,875 | 2.13 | 0.971 |
| Cholesky solve (unconstrained) | 2.704 | 5.912 | 11.6 | 41.7 | 147.6 | 1.91 | 0.953 |
| NNLS PDIP (reference) | 33.9 | 87.0 | 185.0 | 683.6 | 2,401 | 2.04 | 0.968 |
| F+λH build (dense) | 54.6 | 175.9 | 370.8 | 1,062 | 2,819 | 1.89 | 0.994 |
| log det Cholesky (F+λH) | 2.774 | 4.975 | 10.9 | 38.0 | 161.9 | 1.94 | 0.930 |
| log det Cholesky (H) | 2.803 | 5.058 | 10.9 | 38.0 | 146.4 | 1.90 | 0.936 |
| library likelihood call (S3) | 123.1 | 346.2 | 673.9 | 1,855 | 5,460 | 1.80 | 0.986 |
| library likelihood call (S0) | 153.0 | 386.6 | 770.3 | 1,969 | 6,136 | 1.74 | 0.980 |

### Where the whole call crosses a threshold

- **100 ms** — not measured: the threshold lies outside the measured range, and this table never extrapolates.
- **1 s** — N ≈ 1846 (library likelihood call, S3)

## delaunay

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 | 1000 | 1500 | 2500 | 4000 |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ 0.50 GB | ✓ 1.00 GB | ✓ 1.52 GB | ✓ 2.53 GB | ✓ 3.13 GB |
| certifying budget | 1 | 1 | 2 | 1 | 2 |
| PDIP iterations | 14 | 16 | 17 | 20 | 19 |

### Per call, ms

| Row | 500 | 1000 | 1500 | 2500 | 4000 | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 4.493 | 10.7 | 44.0 | 80.7 | 449.1 | 2.18 | 0.963 |
| certified active set (phase 3 safe budget) | 15.2 | 54.3 | 90.0 | 343.2 | 1,228 | 2.08 | 0.982 |
| Cholesky solve (unconstrained) | 2.895 | 6.370 | 12.5 | 39.1 | 157.7 | 1.91 | 0.960 |
| NNLS PDIP (reference) | 33.9 | 93.9 | 210.9 | 867.7 | 3,032 | 2.19 | 0.979 |
| F+λH build (dense) | 56.1 | 185.7 | 373.7 | 1,059 | 2,868 | 1.89 | 0.997 |
| log det Cholesky (F+λH) | 2.903 | 5.245 | 11.7 | 38.8 | 149.0 | 1.92 | 0.950 |
| log det Cholesky (H) | 2.933 | 5.136 | 11.6 | 39.4 | 140.7 | 1.90 | 0.950 |
| library likelihood call (S3) | 141.1 | 377.8 | 727.7 | 2,225 | 6,604 | 1.85 | 0.986 |
| library likelihood call (S0) | 176.0 | 437.1 | 847.1 | 2,291 | 7,385 | 1.78 | 0.982 |

### Where the whole call crosses a threshold

- **100 ms** — not measured: the threshold lies outside the measured range, and this table never extrapolates.
- **1 s** — N ≈ 1735 (library likelihood call, S3)

