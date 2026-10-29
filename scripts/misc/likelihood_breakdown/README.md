# likelihood_breakdown

Per-step JIT decomposition of the PyAutoLens likelihood function. The headline question is:

> *"Where, inside this likelihood, is the time actually going?"*

Run one of these scripts when you want to find the **next source-code-level optimization target** for a given dataset class / model. Each script JIT-compiles every step of the pipeline as an isolated JAX program and reports its lower / compile / first-call / steady-state time, then writes a per-step JSON + horizontal bar chart so the dominant step is immediately visible.

For *how long the likelihood actually takes* on production hardware — i.e. a single end-to-end number per (hardware, precision) config — use the sibling package [`likelihood_runtime/`](../likelihood_runtime/) instead. The two packages are deliberately disjoint so neither has to pay the other's cost.

## Latest results

<!-- BEGIN auto-table:breakdown -->
| Cell | Instrument | Setup | Host | Inversion path | Step-sum total | PyAutoLens version |
|------|------------|-------|------|----------------|----------------|--------------------|
| `cluster/image_plane` | — | _not recorded_ | local_cpu_fp64 | dense (mapping) | 4.81 s | v2026.8.17.1 |
| `cluster/image_plane` | — | _not recorded_ | hpc_ral_cpu_fp64 | dense (mapping) | 4.53 s | v2026.8.17.1 |
| `cluster/source_plane` | — | _not recorded_ | local_cpu_fp64 | dense (mapping) | 4.5 ms | v2026.7.23.1 (archived) |
| `datacube/delaunay` | alma_high | _not recorded_ | hpc_a100_fp64 | dense (mapping) | — | v2026.7.6.649 |
| `datacube/delaunay` | alma_high | _not recorded_ | hpc_a100_mp | dense (mapping) | — | v2026.7.6.649 |
| `datacube/inversion` | alma_high | _not recorded_ | hpc_a100_fp64 | dense (mapping) | — | v2026.7.6.649 |
| `datacube/inversion` | alma_high | _not recorded_ | hpc_a100_mp | dense (mapping) | — | v2026.7.6.649 |
| `imaging/delaunay` | ao | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 54.92 s | v2026.8.17.1 |
| `imaging/delaunay` | ao | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 10.52 s | v2026.8.17.1 |
| `imaging/delaunay` | ao | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 57.52 s | v2026.8.17.1 |
| `imaging/delaunay` | ao | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 360.7 ms | v2026.8.17.1 |
| `imaging/delaunay` | ao | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 900.4 ms | v2026.8.17.1 |
| `imaging/delaunay` | euclid | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 1.37 s | v2026.8.17.1 |
| `imaging/delaunay` | euclid | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 736.7 ms | v2026.8.17.1 |
| `imaging/delaunay` | euclid | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 1.19 s | v2026.8.17.1 |
| `imaging/delaunay` | euclid | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 47.3 ms | v2026.8.17.1 |
| `imaging/delaunay` | euclid | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 51.0 ms | v2026.8.17.1 |
| `imaging/delaunay` | hst | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 2.34 s | v2026.8.17.1 |
| `imaging/delaunay` | hst | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 1.22 s | v2026.8.17.1 |
| `imaging/delaunay` | hst | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 2.65 s | v2026.8.17.1 |
| `imaging/delaunay` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 64.5 ms | v2026.8.17.1 |
| `imaging/delaunay` | hst | `jax_gpu_dense_mixed` | hpc_a100 | dense (mapping) | 96.8 ms | v2026.7.6.649 |
| `imaging/delaunay` | hst | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 80.4 ms | v2026.8.17.1 |
| `imaging/delaunay` | hst | `jax_gpu_sparse_mixed` | hpc_a100 | sparse (w-tilde) | 95.5 ms | v2026.7.6.649 |
| `imaging/delaunay` | hst | _not recorded_ | local_cpu_fp64 | dense (mapping) | 3.34 s | v2026.8.17.1 (archived) |
| `imaging/delaunay` | hst | _not recorded_ | local_cpu_fp64 | sparse (w-tilde) | 8.81 s | v2026.7.6.649 (archived) |
| `imaging/delaunay` | jwst | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 4.74 s | v2026.8.17.1 |
| `imaging/delaunay` | jwst | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 2.05 s | v2026.8.17.1 |
| `imaging/delaunay` | jwst | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 5.05 s | v2026.8.17.1 |
| `imaging/delaunay` | jwst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 89.9 ms | v2026.8.17.1 |
| `imaging/delaunay` | jwst | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 127.4 ms | v2026.8.17.1 |
| `imaging/delaunay/light_fixed` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | — | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 6.98 s | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 1.46 s | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 6.23 s | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 90.2 ms | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 1.20 s | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | _not recorded_ | local_cpu_fp64 | dense (mapping) | 4.49 s | v2026.8.17.1 (archived) |
| `imaging/delaunay_nn/light_fixed` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | — | v2026.8.17.1 |
| `imaging/delaunay_numba` | euclid | _not recorded_ | local_cpu_fp64 | sparse (numba) | 255.0 ms | v2026.8.17.1 (archived) |
| `imaging/delaunay_numba` | hst | _not recorded_ | local_cpu_fp64 | sparse (numba) | 2.04 s | v2026.8.17.1 (archived) |
| `imaging/mge` | ao | `jax_cpu_dense_fp64` | hpc_ral_cpu | dense (mapping) | 776.0 ms | v2026.8.17.1 |
| `imaging/mge` | ao | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 23.2 ms | v2026.8.17.1 |
| `imaging/mge` | euclid | `jax_cpu_dense_fp64` | hpc_ral_cpu | dense (mapping) | 16.4 ms | v2026.8.17.1 |
| `imaging/mge` | euclid | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 8.6 ms | v2026.8.17.1 |
| `imaging/mge` | hst | `jax_cpu_dense_fp64` | hpc_ral_cpu | dense (mapping) | 33.3 ms | v2026.8.17.1 |
| `imaging/mge` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 9.0 ms | v2026.8.17.1 |
| `imaging/mge` | hst | _not recorded_ | local_cpu_fp64 | dense (mapping) | 40.3 ms | v2026.8.17.1 (archived) |
| `imaging/mge` | jwst | `jax_cpu_dense_fp64` | hpc_ral_cpu | dense (mapping) | 66.5 ms | v2026.8.17.1 |
| `imaging/mge` | jwst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 10.5 ms | v2026.8.17.1 |
| `imaging/pixelization` | hst | _not recorded_ | local_cpu_fp64 | dense (mapping) | 3.40 s | v2026.8.17.1 (archived) |
| `imaging/pixelization` | hst | _not recorded_ | local_cpu_fp64 | sparse (w-tilde) | 10.17 s | v2026.7.6.649 (archived) |
| `imaging/pixelization_numba` | euclid | _not recorded_ | local_cpu_fp64 | sparse (numba) | 611.9 ms | v2026.8.17.1 (archived) |
| `imaging/pixelization_numba` | hst | _not recorded_ | local_cpu_fp64 | sparse (numba) | 1.30 s | v2026.8.17.1 (archived) |
| `imaging/rectangular` | ao | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 52.36 s | v2026.8.17.1 |
| `imaging/rectangular` | ao | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 10.98 s | v2026.8.17.1 |
| `imaging/rectangular` | ao | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 51.62 s | v2026.8.17.1 |
| `imaging/rectangular` | ao | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 316.5 ms | v2026.8.17.1 |
| `imaging/rectangular` | ao | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 988.4 ms | v2026.8.17.1 |
| `imaging/rectangular` | euclid | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 935.7 ms | v2026.8.17.1 |
| `imaging/rectangular` | euclid | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 408.0 ms | v2026.8.17.1 |
| `imaging/rectangular` | euclid | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 961.7 ms | v2026.8.17.1 |
| `imaging/rectangular` | euclid | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 33.8 ms | v2026.8.17.1 |
| `imaging/rectangular` | euclid | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 43.2 ms | v2026.8.17.1 |
| `imaging/rectangular` | hst | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 2.31 s | v2026.8.17.1 |
| `imaging/rectangular` | hst | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 739.5 ms | v2026.8.17.1 |
| `imaging/rectangular` | hst | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 2.20 s | v2026.8.17.1 |
| `imaging/rectangular` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 49.7 ms | v2026.8.17.1 |
| `imaging/rectangular` | hst | `jax_gpu_dense_mixed` | hpc_a100 | dense (mapping) | 56.4 ms | v2026.7.6.649 |
| `imaging/rectangular` | hst | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 70.4 ms | v2026.8.17.1 |
| `imaging/rectangular` | hst | `jax_gpu_sparse_mixed` | hpc_a100 | sparse (w-tilde) | 55.5 ms | v2026.7.6.649 |
| `imaging/rectangular` | jwst | `jax_cpu_sparse_fp64` | hpc_ral_cpu | sparse (w-tilde) | 4.17 s | v2026.8.17.1 |
| `imaging/rectangular` | jwst | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 1.76 s | v2026.8.17.1 |
| `imaging/rectangular` | jwst | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 4.35 s | v2026.8.17.1 |
| `imaging/rectangular` | jwst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 75.4 ms | v2026.8.17.1 |
| `imaging/rectangular` | jwst | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 125.6 ms | v2026.8.17.1 |
| `imaging/rectangular/light_fixed` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | — | v2026.8.17.1 |
| `imaging/rectangular/source_pixels_784` | hst | `numba_cpu_sparse_fp64` | hpc_ral_cpu | sparse (numba) | 628.9 ms | v2026.8.17.1 |
| `imaging/rectangular/source_pixels_784` | hst | `jax_cpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 1.34 s | v2026.8.17.1 |
| `imaging/rectangular/source_pixels_784` | hst | `jax_gpu_dense_fp64` | hpc_a100 | dense (mapping) | 32.3 ms | v2026.8.17.1 |
| `imaging/rectangular/source_pixels_784` | hst | `jax_gpu_sparse_fp64` | hpc_a100 | sparse (w-tilde) | 49.9 ms | v2026.8.17.1 |
| `interferometer/delaunay` | alma | `jax_cpu_sparse_fp64_wtilde` | local_laptop | sparse (w-tilde) | 4.18 s | v2026.8.17.1 |
| `interferometer/delaunay` | alma | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 2.43 s | v2026.8.17.1 |
| `interferometer/delaunay` | alma | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (numba) | 827.4 ms | v2026.8.17.1 |
| `interferometer/delaunay` | alma | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 41.7 ms | v2026.8.17.1 |
| `interferometer/delaunay` | alma | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 51.3 ms | v2026.8.17.1 |
| `interferometer/delaunay` | alma_high | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 6.33 s | v2026.8.17.1 |
| `interferometer/delaunay` | alma_high | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 12.76 s | v2026.8.17.1 |
| `interferometer/delaunay` | alma_high | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 84.7 ms | v2026.8.17.1 |
| `interferometer/delaunay` | alma_high | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 99.6 ms | v2026.8.17.1 |
| `interferometer/delaunay` | jvla | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 56.20 s | v2026.8.17.1 |
| `interferometer/delaunay` | jvla | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 47.63 s | v2026.8.17.1 |
| `interferometer/delaunay` | jvla | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 445.1 ms | v2026.8.17.1 |
| `interferometer/delaunay` | jvla | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 548.8 ms | v2026.8.17.1 |
| `interferometer/delaunay` | sdp81 | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (numba) | 1.49 s | v2026.8.17.1 |
| `interferometer/delaunay` | sdp81 | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 55.4 ms | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `jax_cpu_sparse_fp64_wtilde` | local_laptop | sparse (w-tilde) | 1.52 s | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `numba_cpu_sparse_fp64_wtilde` | local_laptop | sparse (numba) | 472.0 ms | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `jax_cpu_dense_fp64_nufft` | hpc_ral_cpu | dense (mapping) | 7.27 s | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 1.05 s | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (numba) | 529.7 ms | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `jax_gpu_dense_fp64_nufft` | hpc_a100 | dense (mapping) | 57.2 ms | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 29.5 ms | v2026.8.17.1 |
| `interferometer/delaunay` | sma | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 33.6 ms | v2026.8.17.1 |
| `interferometer/delaunay_numba_direct_conv` | alma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 1.80 s | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_direct_conv` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 378.2 ms | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_jax` | alma | _not recorded_ | local_cpu_fp64 | sparse (w-tilde) | 3.28 s | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_jax` | sma | _not recorded_ | local_cpu_fp64 | sparse (w-tilde) | 727.7 ms | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_reference` | alma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 4.54 s | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_reference` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 535.9 ms | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_source_loop` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 461.7 ms | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_symmetric` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 443.1 ms | v2026.8.17.1 (archived) |
| `interferometer/delaunay_numba_two_stage` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 388.7 ms | v2026.8.17.1 (archived) |
| `interferometer/mge` | alma | `jax_cpu_dense_fp64_nufft` | local_laptop | dense (mapping) | 41.19 s | v2026.8.17.1 |
| `interferometer/mge` | alma | `jax_cpu_sparse_fp64_wtilde` | local_laptop | sparse (w-tilde) | 32.7 ms | v2026.8.17.1 |
| `interferometer/mge` | alma | `jax_gpu_dense_fp64_nufft` | hpc_a100 | dense (mapping) | 943.2 ms | v2026.8.17.1 |
| `interferometer/mge` | alma | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 2.7 ms | v2026.8.17.1 |
| `interferometer/mge` | alma_high | `jax_cpu_dense_fp64_nufft` | local_laptop | dense (mapping) | 212.19 s | v2026.8.17.1 |
| `interferometer/mge` | alma_high | `jax_gpu_dense_fp64_nufft` | hpc_a100 | dense (mapping) | 3.88 s | v2026.8.17.1 |
| `interferometer/mge` | alma_high | `jax_gpu_dense_mixed_nufft` | hpc_a100 | dense (mapping) | 3.93 s | v2026.8.17.1 |
| `interferometer/mge` | alma_high | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 19.6 ms | v2026.8.17.1 |
| `interferometer/mge` | jvla | `jax_gpu_dense_fp64_nufft` | hpc_a100 | dense (mapping) | 23.81 s | v2026.8.17.1 |
| `interferometer/mge` | jvla | `jax_gpu_dense_mixed_nufft` | hpc_a100 | dense (mapping) | 23.72 s | v2026.8.17.1 |
| `interferometer/mge` | jvla | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 16.7 ms | v2026.8.17.1 |
| `interferometer/mge` | sdp81 | `jax_gpu_dense_fp64_nufft` | hpc_a100 | dense (mapping) | 975.0 ms | v2026.8.17.1 |
| `interferometer/mge` | sdp81 | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 4.2 ms | v2026.8.17.1 |
| `interferometer/mge` | sma | `jax_cpu_dense_fp64_nufft` | local_laptop | dense (mapping) | 84.0 ms | v2026.8.17.1 |
| `interferometer/mge` | sma | `jax_cpu_sparse_fp64_wtilde` | local_laptop | sparse (w-tilde) | 11.8 ms | v2026.8.17.1 |
| `interferometer/mge` | sma | `jax_cpu_dense_fp64_nufft` | hpc_ral_cpu | dense (mapping) | 96.4 ms | v2026.8.17.1 |
| `interferometer/mge` | sma | `jax_gpu_dense_fp64_nufft` | hpc_a100 | dense (mapping) | 4.3 ms | v2026.8.17.1 |
| `interferometer/mge_dft` | sma | _not recorded_ | local_cpu_fp64 | dense (mapping) | 32.9 ms | v2026.8.17.1 (archived) |
| `interferometer/pixelization_numba_direct_conv` | alma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 1.76 s | v2026.8.17.1 (archived) |
| `interferometer/pixelization_numba_direct_conv` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 184.5 ms | v2026.8.17.1 (archived) |
| `interferometer/pixelization_numba_jax` | alma | _not recorded_ | local_cpu_fp64 | sparse (w-tilde) | 2.28 s | v2026.8.17.1 (archived) |
| `interferometer/pixelization_numba_jax` | sma | _not recorded_ | local_cpu_fp64 | sparse (w-tilde) | 584.6 ms | v2026.8.17.1 (archived) |
| `interferometer/pixelization_numba_reference` | alma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 7.31 s | v2026.8.17.1 (archived) |
| `interferometer/pixelization_numba_reference` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | 480.5 ms | v2026.8.17.1 (archived) |
| `interferometer/preload` | alma | _not recorded_ | local_cpu_fp64 | sparse (numba) | — | v2026.8.17.1 (archived) |
| `interferometer/preload` | alma_high | _not recorded_ | local_cpu_fp64 | sparse (numba) | — | v2026.8.17.1 (archived) |
| `interferometer/preload` | sma | _not recorded_ | local_cpu_fp64 | sparse (numba) | — | v2026.8.17.1 (archived) |
| `interferometer/rectangular` | alma | `jax_cpu_sparse_fp64_wtilde` | local_laptop | sparse (w-tilde) | 3.69 s | v2026.8.17.1 |
| `interferometer/rectangular` | alma | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 2.80 s | v2026.8.17.1 |
| `interferometer/rectangular` | alma | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (numba) | 998.0 ms | v2026.8.17.1 |
| `interferometer/rectangular` | alma | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 37.1 ms | v2026.8.17.1 |
| `interferometer/rectangular` | alma | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 45.5 ms | v2026.8.17.1 |
| `interferometer/rectangular` | alma_high | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 8.33 s | v2026.8.17.1 |
| `interferometer/rectangular` | alma_high | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 7.25 s | v2026.8.17.1 |
| `interferometer/rectangular` | alma_high | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 76.2 ms | v2026.8.17.1 |
| `interferometer/rectangular` | alma_high | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 95.2 ms | v2026.8.17.1 |
| `interferometer/rectangular` | jvla | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 59.71 s | v2026.8.17.1 |
| `interferometer/rectangular` | jvla | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 63.41 s | v2026.8.17.1 |
| `interferometer/rectangular` | jvla | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 446.6 ms | v2026.8.17.1 |
| `interferometer/rectangular` | jvla | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 552.9 ms | v2026.8.17.1 |
| `interferometer/rectangular` | sdp81 | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (numba) | 1.85 s | v2026.8.17.1 |
| `interferometer/rectangular` | sdp81 | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 46.2 ms | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `jax_cpu_sparse_fp64_wtilde` | local_laptop | sparse (w-tilde) | 1.24 s | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `numba_cpu_sparse_fp64_wtilde` | local_laptop | sparse (numba) | 1.15 s | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `jax_cpu_dense_fp64_nufft` | hpc_ral_cpu | dense (mapping) | 4.05 s | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `jax_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (w-tilde) | 655.3 ms | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `numba_cpu_sparse_fp64_wtilde` | hpc_ral_cpu | sparse (numba) | 309.2 ms | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `jax_gpu_dense_fp64_nufft` | hpc_a100 | dense (mapping) | 50.8 ms | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `jax_gpu_sparse_fp64_wtilde` | hpc_a100 | sparse (w-tilde) | 24.5 ms | v2026.8.17.1 |
| `interferometer/rectangular` | sma | `jax_gpu_sparse_mixed_wtilde` | hpc_a100 | sparse (w-tilde) | 29.4 ms | v2026.8.17.1 |
| `point_source_image/image_plane` | simple | `jax_cpu_na_fp64` | local_laptop | dense (mapping) | 62.7 ms | v2026.8.17.1 |
| `point_source_image/image_plane` | simple | `jax_cpu_na_fp64` | hpc_ral_cpu | dense (mapping) | 24.7 ms | v2026.8.17.1 |
| `point_source_image/image_plane` | simple | `jax_gpu_na_fp64` | hpc_a100 | dense (mapping) | 1.1 ms | v2026.8.17.1 |
| `point_source_source/source_plane` | simple | `jax_cpu_na_fp64` | local_laptop | dense (mapping) | 438 μs | v2026.8.17.1 |
| `point_source_source/source_plane` | simple | `jax_cpu_na_fp64` | hpc_ral_cpu | dense (mapping) | 146 μs | v2026.8.17.1 |
| `point_source_source/source_plane` | simple | `jax_gpu_na_fp64` | hpc_a100 | dense (mapping) | 222 μs | v2026.8.17.1 |
<!-- END auto-table:breakdown -->

Auto-generated by `scripts/misc/tooling/build_readme.py` from the versioned artifacts under `results/breakdown/`.

## The shared helper package

Since 2026-09-10 this directory is also an importable package (`likelihood_breakdown`),
alongside its README. `scripts/misc` is already on `sys.path` in every cell, so the
imaging cells do:

```python
from likelihood_breakdown import reconstruction_steps, sparse_steps, timing
```

| Module | What it holds |
|--------|---------------|
| `timing.py` | `Timer`, `block`, `jit_profile`, `vmap_profile`, `parse_vmap_batch`, `split_by_successive_differences` — the JIT/vmap harness the cells shared by copy-paste until the copies drifted. `jit_profile` / `vmap_profile` record `{lower_s, compile_s, first_call_s, steady_per_call_s}` per label into the cell's `jit_records` dict, which lands in the result JSON as `jit_phases`; before this, compile time existed only in SLURM stdout. |
| `sparse_steps.py` | Standalone JAX functions reproducing `InversionImagingSparse` for the **func-list + mapper** case (an MGE lens-light basis alongside one pixelized `Mapper`) — the configuration all three production-fiducial imaging cells build. |
| `reconstruction_steps.py` | Standalone JAX pieces of the *inside* of steps 12 and 13: `jacobi_scaled`, `nnls_pdip` / `nnls_pdip_one_iteration` (the PDIP driver, which keeps the iteration count the library's `custom_vjp` primal throws away), `cholesky_curvature_reg`, `cholesky_solve`, `log_det_cholesky`, `log_evidence_terms`. They feed the **overlapping** `steps_reconstruction_sub_rows` table on both legs. |
| `overhead_verdict.py` | `abba_overhead_verdict` — the ONE PASS / FAIL / FAIL_GROSS / INCONCLUSIVE verdict for the ABBA instrumentation overhead: one-sided Student-t bounds of the excess in ms against a ms budget. Shared by `fixed_light_numba.py` and its CI test (identity asserted); #362 fix phase 1, rule in `wiki/campaigns/misc/timing_noise_audit_2026_10.md`. |

`block()` synchronises every leaf of a returned pytree. That matters: the rectangular
cell's older local copy tested `hasattr(x, "block_until_ready")`, so a tuple-returning
prefix was timed asynchronously.

### Why `sparse_steps.py` exists

Until 2026-09-10 the `_sparse` breakdown rows were **dense tables with a sparse dataset
attached**. The cells called `dataset.apply_sparse_operator()` and then timed
`inversion.operated_mapping_matrix`, `curvature_matrix_via_mapping_matrix_from` and
`mapped_reconstructed_data_via_mapping_matrix_from` — none of which the w-tilde path
ever calls. That is why dense and `_sparse` rows agreed to <1 %: they were the same
computation twice.

### Dense ↔ sparse row correspondence

| dense row | sparse row | library call |
|-----------|------------|--------------|
| Ray-trace grids / mesh grid | *same* | path-independent |
| Lens light images (pre-PSF) | *same* | path-independent |
| Blurred image (PSF convolution) | *same* | path-independent |
| Profile-subtracted image | *same* | path-independent |
| Overlay grid / triangulation | *same* | path-independent |
| Inversion setup (steps 4-8 / 5-8 combined) | Inversion setup (sparse, … combined) | `params -> triplets + MGE operated basis + psf_weighted_data` |
| Mapping matrix | Sparse triplets (data + curvature) | `Mapper.sparse_triplets_data` / `sparse_triplets_curvature` |
| Blurred mapping matrix (PSF) | MGE operated basis (params prefix) + PSF-weighted data | `LightProfileLinearObjFuncList.operated_mapping_matrix_override` (**not** `psf.convolved_mapping_matrix_from` — a linear light profile overrides the inversion's convolution so that flux outside the mask blurring into it is included, and the mapping matrix has no columns for that region); `inversion_imaging_util.psf_weighted_data_from` |
| Data vector (D) | Data vector (D, w-tilde) | `data_vector_via_psf_weighted_data_from` (mapper block) + `data_vector_via_blurred_mapping_matrix_from` (MGE block) |
| Curvature matrix (F) | Curvature matrix (F, w-tilde) | the three blocks + `curvature_matrix_mirrored_from` + `curvature_matrix_with_added_to_diag_from` |
| — | *sub-row* F diag (FFT blocks) | `ImagingSparseOperator.curvature_matrix_diag_from` — `ceil(S / batch_size)` rFFT2 column blocks |
| — | *sub-row* F off-diag (mapper × MGE) | `ImagingSparseOperator.curvature_matrix_off_diag_func_list_from` |
| — | *sub-row* F MGE × MGE GEMM | `(N, 60)ᵀ (N, 60)` dense matmul on `B / noise` |
| Regularization matrix (H) | *same* | `regularization.regularization_matrix_from` |
| Regularized reconstruction | *same* | `reconstruction_positive_only_from` (NNLS) |
| Mapped recon + log evidence | Mapped recon + log evidence (sparse) | `mapped_reconstructed_operated_data_via_sparse_operator_from` + `psf.convolved_image_from`; every other term identical |

H, `F + λH`, the NNLS solve and both Choleskys are **the same code in both legs**. That
is the point of the comparison: the w-tilde path replaces the mapping matrix, not the
log-det. It is also why the two legs share one pinned log evidence — a sparse row that
does not reproduce the dense evidence is a bug in the w-tilde path, not a different
measurement.

The MGE operated basis stays dense (60 PSF-convolved columns) and is reported as its own
row rather than folded away: a matrix-free line has to beat it too. Its row is timed as a
`params -> basis` prefix because the override makes it a function of the model parameters;
convolving the unoperated mapping matrix at image resolution instead is off by 2.1e-2
relative on F.

### Extra JSON keys on the sparse leg

| key | meaning |
|-----|---------|
| `configuration.inversion_path` | `"sparse"` |
| `configuration.sparse_batch_size` | `--sparse-batch-size` (default 128), the operator's FFT column-block width |
| `configuration.sparse_nnz` | non-zeros in the mapper's sparse triplets |
| `configuration.total_params` | source pixels + MGE columns |
| `configuration.mixed_precision_note` | present with `--use-mixed-precision`: the operator hard-casts to float64, so the flag does not reach the w-tilde blocks |
| `steps_sparse_setup_rows` | the three setup pieces timed standalone (they overlap the combined row, so they are not in `total_step_by_step`) |
| `steps_sparse_sub_rows` | the three F blocks (they sum to the single F row) |
| `sparse_equivalence` | max relative difference of the standalone F and D against `fit.inversion`, and the 1e-8 tolerance the cell asserts |

`jit_phases` (both legs) carries lower / compile / first-call / steady time per timed
JIT, so compile cost is recoverable from the artifact rather than only from job stdout.

### Why `reconstruction_steps.py` exists

The 2026-09-10 A100 baseline found `Regularized reconstruction` to be 36.7–37.9 ms on
every pixelized cell — 61–71 % of the per-call cost, and identical dense vs sparse. That
row is one fused JIT unit, and it is **not separable into stages**: the library never
factorises `F + λH` once. `reconstruction_positive_only_from` Jacobi-rescales the system
and hands it to a PDIP `lax.while_loop` (`autoarray/util/jax_nnls.py`, cap 50 iterations)
in which *every iteration is a fresh dense Cholesky of the (n, n) KKT system*, inside the
external `jaxnnls` package. Worse, `solve_nnls_primal`'s `custom_vjp` discards
`converged` and `pdip_iter`, so the iteration count is invisible through the library call.

So the sub-rows are **comparators measured on the same matrices**, not a partition:

| sub-row | what it measures |
|---------|------------------|
| `Cholesky (F+λH)` | one factorisation of the full matrix the NNLS receives |
| `Cholesky solve (unconstrained)` | that factorisation plus two triangular solves — the reconstruction with the non-negativity constraint dropped. **This is the number a matrix-free CG line has to beat**, not the ~37 ms row |
| `NNLS PDIP (cell-driven, max_iter 50)` | the same solve, driven from the cell through `jax_nnls.solve_nnls` so the iteration count survives; agrees with the library reconstruction to 1e-8 |
| `NNLS PDIP one iteration` | `max_iter=1` — initialisation plus one PDIP step, the cross-check on `row / iterations` |
| `NNLS PDIP @vmap N (identical lanes)` | only with `--vmap-batch N`, and only a best case: identical lanes converge on the same iteration, so the `while_loop` never waits for a straggler |
| `Log det Cholesky (F+λH reduced)` | the first of the two log-dets folded into the step-13 row, on the rank-stripped block |
| `Log det Cholesky (H reduced)` | the second |

### Extra JSON keys on both legs (reconstruction split)

| key | meaning |
|-----|---------|
| `steps_reconstruction_sub_rows` | the table above, in seconds per call. **Overlapping** — never in `steps` or `total_step_by_step` |
| `nnls` | `iterations`, `converged`, `ms_per_iteration` (the cell-driven row / iterations), `one_iteration_ms`, `max_iter`, `solver_tol`, `jacobi_preconditioning`, `reconstruction_max_abs_diff_vs_library` and its 1e-8 tolerance, plus a `drift` record when that tolerance is breached (recorded, never asserted; the rectangular cell also copies it into `pinned_drift`) |
| `log_evidence_terms` | `chi_squared`, `regularization_term`, `log_det_curvature_reg`, `log_det_regularization`, `noise_normalization`, `log_evidence` — evaluated eagerly once, not inside a timed row, and from the **inversion's own** reconstruction and reduced blocks (the quantities the cell's `log_evidence_check` compares to `FitImaging`). The step-by-step chain accumulates ~1 % of drift through the ill-conditioned solve, so its evidence is not the one an SLQ estimate should be checked against |
| `steps_reconstruction_vmap_note` | present only when the `@vmap` row ran: says in words that identical lanes make it a best case |

Because H, `F + λH`, the NNLS solve and both Choleskys are the same code on both legs,
these rows are measured on the dense and `--sparse` legs alike and are expected to agree.

**The sparse rows are a comparator, not a production path.** Production GPU runs fit the
plain dataset (memory `jax-path-never-applies-sparse-operator`); these rows exist to say
what a half matrix-free line already costs.

## Methodology

For each pipeline step (e.g. *ray-trace grids* → *blurred mapping matrix* → *curvature matrix F* → *NNLS reconstruction* → *log-evidence*), the script:

1. Wraps the step in a small Python function whose only argument is the upstream JAX array(s) it consumes.
2. Calls a per-step `jit_profile(func, label, *args, n_repeats=10)` helper that records:
   - **lower** — `jax.jit(func).lower(...)` (JAX → MLIR tracing time)
   - **compile** — XLA → device-binary compile time
   - **first call** — initial execution including any deferred kernel setup
   - **steady_state × 10** — average over ten subsequent calls; this is the number that goes into the per-step bar.
3. Asserts the JIT output matches the eager FitImaging / FitInterferometer reference at `rtol=1e-4`, so the per-step decomposition is provably equivalent to the production path.
4. Emits a JSON with `{steps: {name: per_call_s}, total_step_by_step: ...}` and a single horizontal bar chart sorted by step cost.

## Platform policy: aim for GPU, fall back to CPU

**A breakdown should always aim to be a full GPU breakdown.** Production
likelihoods run on GPU, and a step's bottleneck shape genuinely changes across
backends (XLA fusion behaviour, sparse-precision-matrix layout, callback
boundaries), so a CPU-only decomposition can point optimization at the wrong
step. The canonical decomposition for a cell is therefore **A100 fp64** (plus
mixed precision where the source supports it), dispatched via the
`hpc/batch_gpu/submit_breakdown_*` scripts.

CPU fp64 remains the fallback when no GPU is available (laptop-only sessions,
HPC downtime) — it is cheap to run locally and still catches the
step-dominance picture for compile-bound and callback-bound steps, but treat it
as provisional until the GPU decomposition confirms it.

The per-step JIT compile is expensive — for a 1000-vertex Delaunay cell it's
tens of minutes of compile time per config — so the grid stays deliberately
small: one representative instrument per dataset class, GPU (+mp) as the
target, CPU as the fallback. Cross-hardware *runtime* comparisons stay in the
runtime package; this package answers where the time goes, on the hardware
that matters.

## XLA fusion caveat

XLA can — and frequently does — fuse adjacent steps into a single kernel at full-pipeline JIT time, so the **sum of per-step bars is an *upper bound* on the production cost**, not the production cost itself. If

    sum(steps) ≫ full_pipeline_single_jit  (from likelihood_runtime/)

then fusion is doing most of the optimization for you and the per-step decomposition is a misleading guide; focus optimisation on whichever step is large *and* doesn't fuse cleanly (typically the ones that involve a Python-level callback, a serial NNLS solve, or a non-uniform memory pattern).

If, on the other hand, the two numbers agree closely, the per-step bars are a faithful map of the per-call cost and the biggest bar is your target.

## Scripts

| Script | Dataset class | Source model | Notes |
|--------|--------------|--------------|-------|
| `scripts/imaging/mge/likelihood_breakdown.py` | Imaging | MGE linear bulge | Linear MGE source; 8-step pipeline. |
| `scripts/imaging/rectangular/likelihood_breakdown.py` | Imaging | RectangularBilinearAdaptImage (`--rect-mesh rtu` for the RTU variant) | 13-step pipeline incl. mesh + regularisation. |
| `scripts/imaging/rectangular/likelihood_breakdown_numba.py` | Imaging | Adaptive rectangular + free `Adapt` (numba CPU, `use_jax=False`) | Rectangular sibling of `scripts/imaging/delaunay/likelihood_breakdown_numba.py`, production-configured on the subhalo `rect_adapt` stage since 2026-09-08 (autolens_profiling#235); same `--variant` / `--memo` / `--n-instances` / `--cold-evals` protocol. |
| `scripts/imaging/delaunay/likelihood_breakdown.py` | Imaging | DelaunayBrightnessImage | 13-step pipeline; Hilbert-curve mesh. |
| `scripts/imaging/delaunay_nn/likelihood_breakdown.py` | Imaging | DelaunayBrightnessImage with `al.mesh.DelaunayNN` | Like-for-like sibling of `scripts/imaging/delaunay/likelihood_breakdown.py` — same 13 steps, same configuration, Sibson natural-neighbour interpolation (cap 32) instead of barycentric. |
| `scripts/interferometer/delaunay/likelihood_breakdown.py` | Interferometer | DelaunayBrightnessImage + sparse-DFT | 11-step pipeline. The transform-mapping-matrix step is the interferometer-specific replacement for imaging's PSF convolution. |
| `scripts/datacube/delaunay/likelihood_breakdown.py` | Datacube | DelaunayBrightnessImage × N channels | 8-step pipeline. Channel-invariant steps profiled once; channel-variant steps profiled on channel 0 and multiplied by `N_channels` for the cube cost. |
| `scripts/imaging/delaunay/likelihood_breakdown_numba.py` | Imaging | Delaunay + free `AdaptSplit` (numba CPU, `use_jax=False`) | 18-step pipeline; sparse-operator CPU path. **Production-configured since 2026-09-08** (autolens_profiling#235): `--instrument euclid` builds the Euclid `vis_pix` stage, `--instrument hst` the subhalo `source_pix[2]` stage — mesh, S/N-driven pixelization over-sampling, MGE basis, positions penalty and thread pinning all matched; the instance stream is seeded iid and the NNLS memo is off and recorded. `--variant legacy` rebuilds the old fiducial. See [`wiki/campaigns/imaging/production_representative_cells.md`](../../../wiki/campaigns/imaging/production_representative_cells.md). |

Four cells are intentionally absent from this package:
- `interferometer/mge` — full-pipeline-by-design, no per-step decomposition (see runtime).
- `interferometer/pixelization` — same reason; the sparse precision-operator path doesn't decompose meaningfully.
- `point_source_image/image_plane`, `point_source_source/source_plane` — single short JIT shots.

These four live only in `likelihood_runtime/`.

`imaging/delaunay_numba_nnls_iterations.py` used to sit in this package. It is a
diagnostic A/B of the NNLS cross-evaluation warm-start memo, not a per-step
breakdown, and it moved to [`scripts/misc/nnls_warm_start/`](../nnls_warm_start/README.md)
(results under `results/misc/nnls_warm_start/`) on 2026-09-08 so no default cell or README
table cites a memo-on number.

## How to read the output

For each cell, the script writes two files into `results/breakdown/<class>/`:

- `<model>_breakdown_<instrument>_v<al_version>.json` — schema:
  ```jsonc
  {
    "autolens_version": "2026.5.14.2",
    "device": {"backend": "cpu", "device": "TFRT_CPU_0"},
    "instrument": "hst",
    "configuration": { ... mask + mesh + pixel-scale snapshot ... },
    "steps": {
      "Ray-trace grids":           0.0020,
      "Blurred image (PSF)":       0.0016,
      "Mapping matrix":            0.0586,
      "Blurred mapping matrix":    0.1493,
      "Curvature matrix (F)":      0.0039,
      "NNLS reconstruction":       0.0007,
      "Mapped reconstructed image":0.0004,
      "Chi-squared / log_evidence":0.0008
    },
    "total_step_by_step": 0.2173,
    "log_likelihood_eager": ...
  }
  ```
- `<model>_breakdown_<instrument>_v<al_version>.png` — horizontal bar chart, one bar per step, sorted by cost.

### Reading the bar chart

The top bar is the next thing to optimise. Compare against the runtime package's `full_pipeline_single_jit` for the same cell — if `total_step_by_step` is within a factor of ~2 of the full-pipeline number, the per-step decomposition is faithful. If it's much larger, XLA fusion is hiding the real cost; treat per-step as a coarse guide.

## Running

From the autolens_profiling root:

```bash
# Default (CPU fp64, sma/hst as appropriate to the cell)
python scripts/imaging/mge/likelihood_breakdown.py

# GPU backend (local GPU; the A100 tier goes via hpc/batch_gpu/submit_breakdown_*)
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu python scripts/imaging/mge/likelihood_breakdown.py

# Mixed precision (paired with the fp64 run on GPU tiers; supported by all cells except imaging/mge)
python scripts/imaging/mge/likelihood_breakdown.py --use-mixed-precision
```

The dataset is auto-simulated via `simulators/<dataset_type>.py --instrument <name>` if `dataset/<class>/<instrument>/` is missing. Subsequent runs reuse the cached dataset.

## When to choose breakdown vs runtime

| Question | Package |
|----------|---------|
| "Where should I focus PyAutoLens optimisation work for this cell?" | **breakdown** |
| "How long will my A100 sampler run take per likelihood call?" | runtime |
| "Does mixed precision actually save time on this cell?" | runtime |
| "Which step fuses cleanly under XLA and which doesn't?" | breakdown (compare total_step_by_step vs runtime's full_pipeline_single_jit) |
| "How does the bottleneck shape change between consumer GPU and A100?" | runtime (and re-run breakdown on the new hardware if the shape changed) |
