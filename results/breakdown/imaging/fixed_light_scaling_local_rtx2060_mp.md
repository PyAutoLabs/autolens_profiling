# Fixed lens light — source-pixel scaling — `local_rtx2060_mp`

RTX 2060 Max-Q 6 GB, mixed precision, single call. Config name `local_rtx2060_mp_fixed_light`.

Every cell is measured. `—` means the leg produced no value; `OOM` /
`timeout` / `failed` / `not run` in the *fits* table say why. Nothing in
this file is extrapolated.

## rectangular

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ 0.63 GB | ✓ 1.13 GB | ✓ 1.78 GB | ✓ 2.83 GB | ✓ 3.63 GB |
| certifying budget | 4 | 4 | 4 | 5 | 5 |
| PDIP iterations | 15 | 15 | 15 | 15 | 15 |

### Per call, ms

| Row | 500 (484) | 1000 (1024) | 1500 (1521) | 2500 | 4000 (3969) | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 12.4 | 25.9 | 65.7 | 263.0 | 904.4 | 2.09 | 0.956 |
| certified active set (phase 3 safe budget) | 21.7 | 65.4 | 136.5 | 523.0 | 1,907 | 2.13 | 0.975 |
| Cholesky solve (unconstrained) | 3.298 | 6.318 | 12.8 | 39.4 | 154.7 | 1.83 | 0.941 |
| NNLS PDIP (reference) | 35.8 | 91.2 | 197.0 | 664.6 | 2,485 | 2.02 | 0.968 |
| F+λH build (dense) | 55.8 | 184.8 | 373.3 | 1,075 | 2,874 | 1.88 | 0.994 |
| log det Cholesky (F+λH) | 3.003 | 5.229 | 12.2 | 41.2 | 153.9 | 1.90 | 0.937 |
| log det Cholesky (H) | 2.971 | 5.140 | 12.5 | 38.6 | 143.1 | 1.87 | 0.940 |
| library likelihood call (S3) | 111.8 | 321.0 | 630.1 | 1,758 | 5,363 | 1.83 | 0.985 |
| library likelihood call (S0) | 139.5 | 359.2 | 724.0 | 1,898 | 6,076 | 1.78 | 0.979 |

### Where the whole call crosses a threshold

- **100 ms** — not measured: the threshold lies outside the measured range, and this table never extrapolates.
- **1 s** — N ≈ 1902 (library likelihood call, S3)

## delaunay

Requested N (built N in brackets where they differ).

### Fits, and at what peak

| N | 500 | 1000 | 1500 | 2500 | 4000 |
|---|---:|---:|---:|---:|---:|
| outcome | ✓ 0.50 GB | ✓ 1.00 GB | ✓ 1.77 GB | ✓ 2.83 GB | ✓ 3.63 GB |
| certifying budget | 1 | 1 | 2 | 1 | 2 |
| PDIP iterations | 14 | 16 | 17 | 20 | 19 |

### Per call, ms

| Row | 500 | 1000 | 1500 | 2500 | 4000 | exponent α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified active set (at its certifying budget) | 4.291 | 12.3 | 46.3 | 82.4 | 472.6 | 2.21 | 0.970 |
| certified active set (phase 3 safe budget) | 14.9 | 57.0 | 89.5 | 364.8 | 1,232 | 2.10 | 0.983 |
| Cholesky solve (unconstrained) | 3.440 | 6.303 | 12.1 | 41.9 | 151.9 | 1.84 | 0.946 |
| NNLS PDIP (reference) | 32.5 | 95.7 | 209.3 | 969.9 | 3,129 | 2.24 | 0.981 |
| F+λH build (dense) | 56.7 | 185.6 | 371.6 | 1,099 | 2,911 | 1.90 | 0.997 |
| log det Cholesky (F+λH) | 2.631 | 5.025 | 11.8 | 40.7 | 153.8 | 1.98 | 0.957 |
| log det Cholesky (H) | 2.717 | 5.235 | 12.0 | 38.7 | 145.4 | 1.93 | 0.958 |
| library likelihood call (S3) | 126.1 | 349.9 | 696.6 | 2,149 | 6,395 | 1.89 | 0.988 |
| library likelihood call (S0) | 157.7 | 406.6 | 803.7 | 2,219 | 7,258 | 1.83 | 0.983 |

### Where the whole call crosses a threshold

- **100 ms** — not measured: the threshold lies outside the measured range, and this table never extrapolates.
- **1 s** — N ≈ 1767 (library likelihood call, S3)

