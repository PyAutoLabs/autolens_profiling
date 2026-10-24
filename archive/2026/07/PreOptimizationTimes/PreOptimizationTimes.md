# PreOptimizationTimes

Named baseline snapshot (12 cells; convention: [`design_lock_in.md`](../../notes/design_lock_in.md)). Full-pipeline per-call cost per cell × config; `(vmap …)` is the vmap per-call where measured.

| Cell | local_cpu_fp64 | local_cpu_mp | local_cpu_fp64_sparse | local_cpu_mp_sparse |
|---|---|---|---|---|
| `datacube/delaunay/sma` | — | — | — | — |
| `imaging/delaunay/hst` | 16.73 s (vmap 12.66 s) | 17.84 s (vmap 16.03 s) | 4.14 s (vmap 5.22 s) | 4.49 s (vmap 4.95 s) |
| `imaging/delaunay/jwst` | 48.81 s (vmap 26.23 s) | 22.43 s (vmap 19.51 s) | 10.42 s (vmap 12.30 s) | 14.05 s (vmap 12.85 s) |
| `imaging/mge/ao` | 3.11 s (vmap 4.92 s) | 5.71 s (vmap 4.02 s) | — | — |
| `imaging/mge/hst` | 117.7 ms (vmap 165.1 ms) | 164.3 ms (vmap 178.9 ms) | 256.1 ms (vmap 187.1 ms) | 140.5 ms (vmap 222.8 ms) |
| `imaging/mge/jwst` | 716.2 ms (vmap 574.8 ms) | 678.5 ms (vmap 527.1 ms) | 387.3 ms (vmap 381.7 ms) | 488.7 ms (vmap 380.6 ms) |
| `imaging/pixelization/hst` | 13.72 s (vmap 12.50 s) | 14.78 s (vmap 13.05 s) | 5.79 s (vmap 5.16 s) | 5.25 s (vmap 6.42 s) |
| `imaging/pixelization/jwst` | 21.78 s (vmap 28.32 s) | 43.58 s (vmap 29.39 s) | 9.57 s (vmap 9.69 s) | 9.42 s (vmap 9.47 s) |
| `interferometer/delaunay/alma` | — | 6.51 s (vmap 7.77 s) | — | — |
| `interferometer/delaunay/sma` | 2.58 s (vmap 2.36 s) | 3.34 s (vmap 2.60 s) | — | — |
| `interferometer/mge/sma` | 230.7 ms (vmap 237.1 ms) | 231.5 ms (vmap 254.6 ms) | — | — |
| `interferometer/pixelization/sma` | 2.04 s (vmap 2.03 s) | 2.39 s (vmap 2.48 s) | — | — |
