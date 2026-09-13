# Fixed lens light — source-pixel scaling — `hpc_a100_fp64`

A100 80GB PCIe (RAL gpu-2), fp64, @vmap 16. Config name `hpc_a100_fp64_fixed_light`.

Every cell is measured. `—` means the leg produced no value; `OOM` /
`timeout` / `failed` / `not run` in the *fits* table say why. Nothing in
this file is extrapolated.

## rectangular

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ 0.43 GB | ✓ 1.02 GB | ✓ 1.72 GB | ✓ 3.42 GB | ✓ 7.76 GB |
| certifying budget | 5 | 8 | 7 | 10 | 6 |
| PDIP iterations | 15 | 15 | 15 | 15 | 15 |

### Per call, ms

| Row | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 2.872 | 8.155 | 11.2 | 25.0 | 30.0 | 1.15 | 0.978 |
| certified active set (phase 3 safe budget) | 5.766 | 10.6 | 16.6 | 27.4 | 51.0 | 1.03 | 0.991 |
| Cholesky solve (unconstrained) | 0.626 | 1.035 | 1.540 | 2.473 | 4.363 | 0.92 | 0.986 |
| NNLS PDIP (reference) | 9.325 | 16.8 | 26.1 | 42.0 | 73.3 | 0.98 | 0.993 |
| F+λH build (dense) | 1.012 | 2.200 | 5.001 | 11.9 | 34.5 | 1.69 | 0.977 |
| log det Cholesky (F+λH) | 0.520 | 0.804 | 1.208 | 1.881 | 3.189 | 0.87 | 0.983 |
| log det Cholesky (H) | 0.540 | 0.802 | 1.351 | 1.844 | 3.242 | 0.85 | 0.976 |
| library likelihood call (S3) | 13.4 | 25.4 | 38.6 | 67.1 | 134.3 | 1.08 | 0.986 |
| library likelihood call (S0) | 20.3 | 34.2 | 51.1 | 87.1 | 162.5 | 0.99 | 0.981 |

### Where the whole call crosses a threshold

- **100 ms** — N ≈ 3261 (library likelihood call, S3)
- **1 s** — not measured: the threshold lies outside the measured range, and this table never extrapolates.

## delaunay

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 | 1000 | 1500 | 2500 | 4000 |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ 0.45 GB | ✓ 1.00 GB | ✓ 1.69 GB | ✓ 3.42 GB | ✓ 7.85 GB |
| certifying budget | 1 | 1 | 2 | 1 | 2 |
| PDIP iterations | 14 | 16 | 17 | 20 | 19 |

### Per call, ms

| Row | 500 | 1000 | 1500 | 2500 | 4000 | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 1.056 | 2.221 | 4.267 | 4.802 | 12.1 | 1.11 | 0.963 |
| certified active set (phase 3 safe budget) | 3.829 | 7.229 | 10.8 | 18.4 | 32.1 | 1.02 | 0.997 |
| Cholesky solve (unconstrained) | 0.606 | 1.059 | 1.457 | 2.404 | 3.958 | 0.90 | 0.995 |
| NNLS PDIP (reference) | 8.461 | 18.4 | 28.1 | 55.2 | 87.3 | 1.14 | 0.999 |
| F+λH build (dense) | 0.999 | 2.423 | 4.816 | 12.1 | 32.4 | 1.68 | 0.990 |
| log det Cholesky (F+λH) | 0.486 | 0.814 | 1.159 | 1.802 | 2.961 | 0.86 | 0.995 |
| log det Cholesky (H) | 0.497 | 0.812 | 1.190 | 1.827 | 2.957 | 0.86 | 0.995 |
| library likelihood call (S3) | 15.7 | 31.9 | 49.4 | 93.4 | 165.2 | 1.14 | 0.998 |
| library likelihood call (S0) | 26.6 | 47.2 | 65.3 | 111.5 | 200.4 | 0.96 | 0.990 |

### Where the whole call crosses a threshold

- **100 ms** — N ≈ 2645 (library likelihood call, S3)
- **1 s** — not measured: the threshold lies outside the measured range, and this table never extrapolates.

