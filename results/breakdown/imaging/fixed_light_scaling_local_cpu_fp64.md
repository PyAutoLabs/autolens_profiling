# Fixed lens light — source-pixel scaling — `local_cpu_fp64`

JAX-CPU i9-10885H, NPROC 8 / BLAS 1, fp64, single call. Config name `local_cpu_fp64_fixed_light`.

Every cell is measured. `—` means the leg produced no value; `OOM` /
`timeout` / `failed` / `not run` in the *fits* table say why. Nothing in
this file is extrapolated.

## rectangular

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ | ✓ | ✓ | ✓ | ✓ |
| certifying budget | 5 | 8 | 7 | 10 | 6 |
| PDIP iterations | 15 | 15 | 15 | 15 | 15 |

### Per call, ms

| Row | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 16.1 | 140.1 | 311.3 | 1,601 | 3,615 | 2.61 | 0.994 |
| certified active set (phase 3 safe budget) | 29.8 | 182.8 | 468.4 | 1,731 | 6,305 | 2.53 | 0.999 |
| Cholesky solve (unconstrained) | 11.2 | 18.5 | 37.2 | 140.8 | 506.8 | 1.84 | 0.923 |
| NNLS PDIP (reference) | 162.3 | 240.1 | 584.2 | 2,291 | 8,333 | 1.93 | 0.917 |
| F+λH build (dense) | 330.7 | 422.9 | 727.9 | 1,835 | 4,638 | 1.28 | 0.906 |
| log det Cholesky (F+λH) | 8.159 | 12.0 | 32.9 | 132.9 | 476.5 | 2.01 | 0.921 |
| log det Cholesky (H) | 8.076 | 11.7 | 30.0 | 123.9 | 449.2 | 1.98 | 0.916 |
| library likelihood call (S3) | 383.5 | 1,192 | 2,132 | 6,307 | 15,396 | 1.76 | 0.992 |
| library likelihood call (S0) | 454.7 | 1,248 | 2,533 | 6,751 | 19,019 | 1.78 | 0.988 |

### Where the whole call crosses a threshold

- **100 ms** — not measured: the threshold lies outside the measured range, and this table never extrapolates.
- **1 s** — N ≈ 912 (library likelihood call, S3)

## delaunay

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 | 1000 | 1500 | 2500 | 4000 |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ | ✓ | ✓ | ✓ | ✓ |
| certifying budget | 1 | 1 | 2 | 1 | 2 |
| PDIP iterations | 14 | 16 | 17 | 20 | 19 |

### Per call, ms

| Row | 500 | 1000 | 1500 | 2500 | 4000 | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 5.681 | 27.1 | 109.5 | 301.3 | 1,801 | 2.73 | 0.990 |
| certified active set (phase 3 safe budget) | 23.7 | 101.5 | 286.6 | 1,170 | 4,270 | 2.52 | 0.996 |
| Cholesky solve (unconstrained) | 2.964 | 15.3 | 35.6 | 140.3 | 556.3 | 2.50 | 0.996 |
| NNLS PDIP (reference) | 37.8 | 227.4 | 638.6 | 3,138 | 10,836 | 2.74 | 0.999 |
| F+λH build (dense) | 88.2 | 358.4 | 657.1 | 1,842 | 4,674 | 1.89 | 0.998 |
| log det Cholesky (F+λH) | 2.111 | 35.7 | 27.9 | 126.1 | 489.6 | 2.44 | 0.941 |
| log det Cholesky (H) | 2.068 | 22.5 | 28.0 | 123.0 | 505.8 | 2.52 | 0.977 |
| library likelihood call (S3) | 391.2 | 1,133 | 2,357 | 6,999 | 18,976 | 1.88 | 0.994 |
| library likelihood call (S0) | 452.8 | 1,294 | 2,745 | 8,192 | 23,758 | 1.91 | 0.992 |

### Where the whole call crosses a threshold

- **100 ms** — not measured: the threshold lies outside the measured range, and this table never extrapolates.
- **1 s** — N ≈ 922 (library likelihood call, S3)

