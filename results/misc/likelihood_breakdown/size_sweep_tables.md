<!-- `python scripts/misc/likelihood_breakdown/size_sweep_table.py --out` output, rendered from the harvested result JSONs and `results/sweep/size_sweep_manifest.jsonl` at commit d2a3848. Regenerate rather than hand-edit. -->

# Reduced N_src size sweep — autolens_profiling#247

- results: `/home/jammy/Code/PyAutoLabs-wt/matrix-free-pixelized-likelihood/autolens_profiling/results/breakdown/imaging`
- manifest: `/home/jammy/Code/PyAutoLabs-wt/matrix-free-pixelized-likelihood/autolens_profiling/results/sweep/size_sweep_manifest.jsonl`
- logs: `/home/jammy/Code/PyAutoLabs-wt/matrix-free-pixelized-likelihood/autolens_profiling/hpc/batch_gpu`
- legs known: 35 (26 submitted per the manifest)

## 1. Fits — does this path still run at this mesh size?

`✓` with a peak-device-memory figure where the leg records one (the matrix-free cell
reports `matrix_free.peak_bytes`; the dense and sparse cells record none today, so their
`✓` is bare). `OOM` is a **datum**, not a failure: it is the answer to the question the
sweep was reduced to ask. `—` is a leg the sweep never submitted.

| mesh | N (requested) | built | dense | sparse | matrix-free |
|---|---|---|---|---|---|
| rectangular | 1500 | 1521 | ✓ | ✓ | ✓ 0.9 GB |
| delaunay | 1500 | 1500 | ✓ | ✓ | ✓ 0.9 GB |
| delaunay_nn | 1500 | 1500 | ✓ | ✓ | ✓ 0.9 GB |
| rectangular | 3000 | 3025 | ✓ | ✓ | ✓ 0.6 GB |
| delaunay | 3000 | 3000 | ✓ | ✓ | ✓ 0.6 GB |
| delaunay_nn | 3000 | 3000 | ✓ | ✓ | — |
| rectangular | 5000 | 5041 | ✓ | ✓ | ✓ 1.6 GB |
| delaunay | 5000 | 5000 | ✓ | ✓ | ✓ 1.6 GB |
| delaunay_nn | 5000 | 5000 | ✓ | ✓ | — |
| rectangular | 8000 | 7921 | ✓ | ✓ | — |
| delaunay | 8000 | 8000 | ✓ | ✓ | — |
| rectangular | 12000 | 12100 | ✓ | ✓ | ✓ 7.7 GB |
| delaunay | 12000 | 12000 | ✓ | ✓ | ✓ 7.6 GB |

## 2. Per call — what one likelihood evaluation costs

`per-call ms` is the **exact paths'** whole-likelihood step-sum total
(`total_step_by_step`) and the **matrix-free** path's three crossover rows summed
(`PCG solve (unconstrained, Jacobi est.)` + `SLQ log det (F+λH reduced) p=16 m=320` + `SLQ log det (λH reduced) p=16 m=320`). Those two are different
quantities and must not be compared column-to-column — the like-for-like comparison is
`components`, which on the exact paths is the solve plus the two log-det Choleskys that
the matrix-free rows replace. `solve @vmap16` is the batched per-call of each path's own
solve row (`NNLS PDIP @vmap N` / `PCG solve @vmap N`). `n/a` is a value this leg does
not record; no cell is ever interpolated.

| mesh | N | path | per-call ms | components (ms) | solve @vmap16 ms | cg iters | cond(F+λH) | log-ev err (nats) |
|---|---|---|---|---|---|---|---|---|
| rectangular | 1500 | dense | 59.889 | chol solve 1.556 · logdet(F+λH) 1.271 · logdet(H) 1.191 | 21.874 | n/a | n/a | n/a |
| rectangular | 1500 | sparse | 70.808 | chol solve n/a · logdet(F+λH) n/a · logdet(H) n/a | n/a | n/a | n/a | n/a |
| rectangular | 1500 | matrix_free | 1462.284 | pcg 1179.947 · slq(F+λH) 253.708 · slq(λH) 28.630 | 285.191 | 5578 | 4.12e+10 | -324.549 |
| delaunay | 1500 | dense | 67.157 | chol solve 1.576 · logdet(F+λH) 1.156 · logdet(H) 1.147 | 21.926 | n/a | n/a | n/a |
| delaunay | 1500 | sparse | 73.737 | chol solve n/a · logdet(F+λH) n/a · logdet(H) n/a | n/a | n/a | n/a | n/a |
| delaunay | 1500 | matrix_free | 2644.256 | pcg 2385.656 · slq(F+λH) 229.664 · slq(λH) 28.936 | 560.622 | 11512 | 4.11e+10 | -95.487 |
| delaunay_nn | 1500 | dense | 80.436 | chol solve 1.522 · logdet(F+λH) 1.164 · logdet(H) 1.184 | 21.911 | n/a | n/a | n/a |
| delaunay_nn | 1500 | sparse | 84.733 | chol solve n/a · logdet(F+λH) n/a · logdet(H) n/a | n/a | n/a | n/a | n/a |
| delaunay_nn | 1500 | matrix_free | 2258.815 | pcg 1913.830 · slq(F+λH) 311.267 · slq(λH) 33.719 | 510.639 | 7716 | 4.11e+10 | -103.986 |
| rectangular | 3000 | dense | 132.343 | chol solve 3.017 · logdet(F+λH) 2.238 · logdet(H) 2.251 | 90.867 | n/a | n/a | n/a |
| rectangular | 3000 | sparse | 137.414 | chol solve 3.038 · logdet(F+λH) 2.226 · logdet(H) 2.240 | 91.012 | n/a | n/a | n/a |
| rectangular | 3000 | matrix_free | 1680.532 | pcg 1389.980 · slq(F+λH) 256.198 · slq(λH) 34.354 | 316.991 | 6429 | 4.1e+10 | -4.651 |
| delaunay | 3000 | dense | 137.240 | chol solve 2.920 · logdet(F+λH) 2.162 · logdet(H) 2.142 | 83.544 | n/a | n/a | n/a |
| delaunay | 3000 | sparse | 139.663 | chol solve 2.935 · logdet(F+λH) 2.236 · logdet(H) 2.286 | 83.937 | n/a | n/a | n/a |
| delaunay | 3000 | matrix_free | 3042.661 | pcg 2774.956 · slq(F+λH) 232.465 · slq(λH) 35.240 | 615.669 | 13543 | 4.08e+10 | +3.499 |
| delaunay_nn | 3000 | dense | 161.281 | chol solve 2.922 · logdet(F+λH) 2.123 · logdet(H) 2.149 | 91.136 | n/a | n/a | n/a |
| delaunay_nn | 3000 | sparse | 159.223 | chol solve 2.908 · logdet(F+λH) 2.110 · logdet(H) 2.130 | 91.196 | n/a | n/a | n/a |
| rectangular | 5000 | dense | 256.486 | chol solve 6.900 · logdet(F+λH) 5.477 · logdet(H) 5.410 | 265.849 | n/a | n/a | n/a |
| rectangular | 5000 | sparse | 244.210 | chol solve 6.871 · logdet(F+λH) 5.464 · logdet(H) 5.489 | 269.047 | n/a | n/a | n/a |
| rectangular | 5000 | matrix_free | 1874.490 | pcg 1582.548 · slq(F+λH) 262.883 · slq(λH) 29.059 | 356.782 | 7221 | 4.08e+10 | -13.865 |
| delaunay | 5000 | dense | 273.683 | chol solve 6.629 · logdet(F+λH) 5.156 · logdet(H) 5.147 | 280.938 | n/a | n/a | n/a |
| delaunay | 5000 | sparse | 295.331 | chol solve 6.769 · logdet(F+λH) 5.146 · logdet(H) 5.181 | 284.781 | n/a | n/a | n/a |
| delaunay | 5000 | matrix_free | 2886.973 | pcg 2619.359 · slq(F+λH) 235.295 · slq(λH) 32.318 | 578.915 | 12366 | 4.1e+10 | +5.542 |
| delaunay_nn | 5000 | dense | 308.057 | chol solve 6.661 · logdet(F+λH) 5.195 · logdet(H) 5.198 | 286.279 | n/a | n/a | n/a |
| delaunay_nn | 5000 | sparse | 275.969 | chol solve 6.611 · logdet(F+λH) 5.169 · logdet(H) 5.193 | 287.452 | n/a | n/a | n/a |
| rectangular | 8000 | dense | 679.077 | chol solve 18.163 · logdet(F+λH) 16.182 · logdet(H) 15.251 | 975.675 | n/a | n/a | n/a |
| rectangular | 8000 | sparse | 631.817 | chol solve 18.247 · logdet(F+λH) 15.505 · logdet(H) 15.295 | 989.608 | n/a | n/a | n/a |
| delaunay | 8000 | dense | 839.140 | chol solve 17.512 · logdet(F+λH) 14.956 · logdet(H) 14.955 | 1262.937 | n/a | n/a | n/a |
| delaunay | 8000 | sparse | 791.309 | chol solve 17.715 · logdet(F+λH) 15.211 · logdet(H) 14.970 | 1284.714 | n/a | n/a | n/a |
| rectangular | 12000 | dense | 1824.739 | chol solve 46.853 · logdet(F+λH) 42.795 · logdet(H) 42.728 | n/a | n/a | n/a | n/a |
| rectangular | 12000 | sparse | 1684.022 | chol solve 47.217 · logdet(F+λH) 43.266 · logdet(H) 42.674 | n/a | n/a | n/a | n/a |
| rectangular | 12000 | matrix_free | 2218.441 | pcg 1934.678 · slq(F+λH) 248.838 · slq(λH) 34.925 | 453.652 | 8984 | 4.06e+10 | -67.096 |
| delaunay | 12000 | dense | 2011.462 | chol solve 47.505 · logdet(F+λH) 41.714 · logdet(H) 41.406 | n/a | n/a | n/a | n/a |
| delaunay | 12000 | sparse | 1844.219 | chol solve 48.054 · logdet(F+λH) 42.491 · logdet(H) 41.291 | n/a | n/a | n/a | n/a |
| delaunay | 12000 | matrix_free | 3106.604 | pcg 2831.271 · slq(F+λH) 236.114 · slq(λH) 39.219 | 669.077 | 13320 | 4.06e+10 | +4.571 |

## 3. Crossover — computed from the rows above, never interpolated

**rectangular.**
- Dense fits at every measured N up to 12000; where it stops is n/a — measured, not extrapolated.
- Sparse fits at every measured N up to 12000; where it stops is n/a — measured, not extrapolated.
- Matrix-free (unconstrained PCG + SLQ m=320) vs the exact solve + two log-det Choleskys — N = 1500: matrix-free 1462.284 ms loses to the dense exact 4.019 ms; N = 3000: matrix-free 1680.532 ms loses to the dense exact 7.507 ms; N = 5000: matrix-free 1874.490 ms loses to the dense exact 17.786 ms; N = 12000: matrix-free 2218.441 ms loses to the dense exact 132.376 ms.
- Matrix-free does not win at any measured N; a crossover above the largest measured N is n/a — measured, not extrapolated.
- SLQ log-evidence error — N = 1500: -324.549 nats (outside the 0.5-nat bar); N = 3000: -4.651 nats (outside the 0.5-nat bar); N = 5000: -13.865 nats (outside the 0.5-nat bar); N = 12000: -67.096 nats (outside the 0.5-nat bar).

**delaunay.**
- Dense fits at every measured N up to 12000; where it stops is n/a — measured, not extrapolated.
- Sparse fits at every measured N up to 12000; where it stops is n/a — measured, not extrapolated.
- Matrix-free (unconstrained PCG + SLQ m=320) vs the exact solve + two log-det Choleskys — N = 1500: matrix-free 2644.256 ms loses to the dense exact 3.880 ms; N = 3000: matrix-free 3042.661 ms loses to the dense exact 7.224 ms; N = 5000: matrix-free 2886.973 ms loses to the dense exact 16.932 ms; N = 12000: matrix-free 3106.604 ms loses to the dense exact 130.625 ms.
- Matrix-free does not win at any measured N; a crossover above the largest measured N is n/a — measured, not extrapolated.
- SLQ log-evidence error — N = 1500: -95.487 nats (outside the 0.5-nat bar); N = 3000: +3.499 nats (outside the 0.5-nat bar); N = 5000: +5.542 nats (outside the 0.5-nat bar); N = 12000: +4.571 nats (outside the 0.5-nat bar).

**delaunay_nn.**
- Dense fits at every measured N up to 5000; where it stops is n/a — measured, not extrapolated.
- Sparse fits at every measured N up to 5000; where it stops is n/a — measured, not extrapolated.
- Matrix-free (unconstrained PCG + SLQ m=320) vs the exact solve + two log-det Choleskys — N = 1500: matrix-free 2258.815 ms loses to the dense exact 3.870 ms.
- Matrix-free does not win at any measured N; a crossover above the largest measured N is n/a — measured, not extrapolated.
- SLQ log-evidence error — N = 1500: -103.986 nats (outside the 0.5-nat bar).
