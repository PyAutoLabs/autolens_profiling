# `searches/` — first-class search profiling

This section profiles **first-class PyAutoFit search objects** end-to-end:
`af.Nautilus` today, with the registry shape ready for `af.DynestyStatic`,
`af.BlackJAXNUTS`, `af.Emcee`, etc. Unlike `likelihood_runtime/` (which
profiles `analysis.log_likelihood_function` in isolation), every cell here
runs `search.fit(model=model, analysis=analysis)` — so visualization,
samples I/O, `samples_info.json`, latent variables, and every other piece
of PyAutoFit machinery is exercised and measured.

## Latest results

<!-- BEGIN auto-table:searches -->
| Sampler | Cell | Config | max logL | logZ | Wall | Evals | Time / eval | Basis | Target | ESS | Version |
|---------|------|--------|---------:|-----:|-----:|------:|------------:|-------|--------|----:|---------|
| `multi_start_adam` | `group/mge/hst` | `local_gpu_fp64` | -231,891.0 | — | 2368.26 s | — | — | stored | — | — | v2026.7.9.1 |
| `multi_start_prodigy` | `cluster/image_plane_solved/simple` | `hpc_a100_fp64` | -1,708.9 | — | 832.23 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `cluster/source_plane_solved/simple` | `hpc_a100_fp64` | 21.4 | — | 250.86 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `cluster/source_plane_tensor/simple` | `hpc_a100_fp64` | -11,000.5 | — | 261.36 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane/simple` | `default` | -79.9 | — | 852.82 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane/simple` | `hpc_a100_fp64` | -68.5 | — | 90.31 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane/simple` | `starts256` | -47.7 | — | 3515.50 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane_solved/simple` | `default` | 2.4 | — | 981.87 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane_solved/simple` | `hpc_a100_fp64` | 9.7 | — | 117.74 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane/simple` | `default` | -109.7 | — | 19.41 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane/simple` | `hpc_a100_fp64` | -109.7 | — | 35.85 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane_solved/simple` | `hpc_a100_fp64` | 4.5 | — | 44.77 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane_tensor/simple` | `hpc_a100_fp64` | 13.4 | — | 31.68 s | — | — | stored | — | — | v2026.7.23.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n128_seed100` | 31,787.9 | — | 198.29 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n128_seed101` | 31,787.9 | — | 144.51 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n128_seed102` | 31,787.9 | — | 137.26 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n128_seed103` | 31,787.9 | — | 153.15 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n128_seed104` | 31,787.9 | — | 159.00 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed0` | 31,787.9 | — | 245.80 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed0_diag_theta_e` | 31,787.9 | — | 271.44 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed1` | -95,708.6 | — | 111.70 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed1_diag_theta_e` | 31,787.9 | — | 283.98 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed2` | -134,315.3 | — | 127.69 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed2_diag_theta_e` | 26,076.8 | — | 133.22 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed3` | -125,319.1 | — | 114.45 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed3_diag_theta_e` | 31,787.9 | — | 105.88 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed4` | 26,076.8 | — | 123.04 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n16_seed4_diag_theta_e` | -128,632.1 | — | 137.83 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed0` | 31,787.9 | — | 224.77 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed0_pos_t0.3_f1e8` | 31,764.3 | — | 553.39 s | 247,808 | 2.2 ms | evals | `bf3d096f` | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed1` | 31,787.9 | — | 216.51 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed105` | 31,787.9 | — | 174.55 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed106` | 31,787.9 | — | 179.25 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed107` | 31,787.9 | — | 168.77 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed108` | 31,787.9 | — | 182.25 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed109` | 31,787.9 | — | 184.13 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed110` | 31,787.9 | — | 182.97 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed111` | 31,787.9 | — | 164.87 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed112` | 31,787.9 | — | 178.90 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed113` | 31,787.9 | — | 169.37 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed114` | 31,787.9 | — | 179.30 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed1_pos_t0.3_f1e8` | 31,785.6 | — | 232.52 s | 70,656 | 3.3 ms | evals | `bf3d096f` | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed2` | 31,787.9 | — | 176.52 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed2_pos_t0.3_f1e8` | 31,787.4 | — | 209.26 s | 61,184 | 3.4 ms | evals | `bf3d096f` | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed3` | 31,787.9 | — | 172.18 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed3_pos_t0.3_f1e8` | 31,702.5 | — | 487.85 s | 173,312 | 2.8 ms | evals | `bf3d096f` | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed4` | 31,787.9 | — | 191.40 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n256_seed4_pos_t0.3_f1e8` | 16,727.5 | — | 172.66 s | 32,000 | 5.4 ms | evals | `bf3d096f` | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n64_seed0` | 31,787.9 | — | 172.13 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n64_seed1` | 31,787.9 | — | 117.63 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n64_seed2` | 31,787.9 | — | 116.10 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n64_seed3` | 31,787.9 | — | 279.98 s | — | — | stored | — | — | v2026.8.17.1 |
| `multi_start_prodigy_autoconv` | `imaging/mge/hst` | `hpc_a100_fp64_n64_seed4` | 31,787.9 | — | 121.12 s | — | — | stored | — | — | v2026.8.17.1 |
| `nautilus` | `cluster/image_plane_solved/simple` | `hpc_a100_fp64` | 31.5 | -1.8 | 742.08 s | 8,400 | 44.1 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `cluster/source_plane/simple` | `hpc_a100_fp64` | 42.3 | -36.5 | 551.28 s | 34,600 | 8.5 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `cluster/source_plane_solved/simple` | `hpc_a100_fp64` | 43.4 | 12.7 | 442.81 s | 7,200 | 14.4 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `cluster/source_plane_tensor/simple` | `hpc_a100_fp64` | 69.8 | 8.5 | 482.73 s | 20,500 | 8.8 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `imaging/delaunay/hst` | `hpc_a100_fp64` | 30,623.2 | 30,562.2 | 1938.52 s | 30,240 | 62.5 ms | evals | — | — | v2026.8.17.1 |
| `nautilus` | `imaging/delaunay/hst` | `hpc_a100_fp64_nbatch16` | 30,623.8 | 30,562.1 | 2088.15 s | 30,432 | 66.7 ms | evals | `a45841ff` | 2,464.3 | v2026.8.17.1 |
| `nautilus` | `imaging/delaunay/hst` | `hpc_a100_fp64_nbatch256` | 30,622.8 | 30,562.2 | 1758.41 s | 33,536 | 50.8 ms | evals | `a45841ff` | 2,387.1 | v2026.8.17.1 |
| `nautilus` | `imaging/delaunay/hst` | `hpc_a100_fp64_nbatch64` | 30,623.0 | 30,562.2 | 1750.83 s | 32,064 | 53.0 ms | evals | `a45841ff` | 2,335.1 | v2026.8.17.1 |
| `nautilus` | `imaging/delaunay/hst` | `hpc_a100_fp64_ref_pos_tauto0.2_f1e8` | 31,339.0 | 31,264.8 | 3265.47 s | 50,736 | 63.2 ms | evals | `6d52f9dc` | 4,128.9 | v2026.8.17.1 |
| `nautilus` | `imaging/delaunay_nn/hst` | `hpc_a100_fp64_ref` | 30,650.8 | 30,591.1 | 3230.23 s | 46,400 | 68.3 ms | evals | `6a13b9a4` | 4,400.7 | v2026.8.17.1 |
| `nautilus` | `imaging/delaunay_nn/hst` | `hpc_a100_fp64_ref_pos_t0.3_f1e8` | 31,347.9 | 31,275.2 | 3111.05 s | 49,800 | 61.1 ms | evals | `1e007f22` | 4,364.7 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64` | 31,786.6 | 31,690.5 | 775.11 s | 62,208 | 11.4 ms | evals | — | — | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_nbatch1000` | 31,787.0 | 31,690.4 | 525.55 s | 77,000 | 5.9 ms | evals | `770ccd47` | 4,693.6 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_nbatch128` | 31,786.7 | 31,690.5 | 680.84 s | 64,768 | 9.5 ms | evals | `770ccd47` | 4,247.8 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_nbatch256` | 31,786.9 | 31,690.5 | 612.28 s | 66,304 | 8.1 ms | evals | `770ccd47` | 4,156.3 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_nbatch512` | 31,786.9 | 31,690.5 | 633.18 s | 67,584 | 8.4 ms | evals | `770ccd47` | 4,574.5 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_nbatch64` | 31,786.9 | 31,690.5 | 737.83 s | 63,424 | 10.6 ms | evals | `770ccd47` | 4,304.1 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_ref_pos_tauto0.2_f1e8` | 31,786.8 | 31,690.4 | 1015.28 s | 96,704 | 9.7 ms | evals | `bf3d096f` | 7,586.1 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed0` | 31,786.7 | 31,690.5 | 773.51 s | 63,552 | 11.1 ms | evals | `770ccd47` | 4,314.8 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed0_pos_t0.3_f1e8` | 31,786.5 | 31,690.5 | 764.23 s | 63,104 | 11.0 ms | evals | `bf3d096f` | 4,393.4 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed1` | 31,786.8 | 31,690.5 | 734.27 s | 63,936 | 10.4 ms | evals | `770ccd47` | 4,302.5 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed1_pos_t0.3_f1e8` | 31,786.8 | 31,690.5 | 735.44 s | 64,000 | 10.5 ms | evals | `bf3d096f` | 4,261.7 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed2` | 31,786.6 | 31,690.5 | 742.06 s | 63,360 | 10.6 ms | evals | `770ccd47` | 4,274.3 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed2_pos_t0.3_f1e8` | 31,786.5 | 31,690.5 | 720.05 s | 63,104 | 10.3 ms | evals | `bf3d096f` | 4,487.6 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed3` | 31,786.6 | 31,690.5 | 716.20 s | 62,464 | 10.4 ms | evals | `770ccd47` | 4,220.9 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed3_pos_t0.3_f1e8` | 31,786.5 | 31,690.5 | 768.62 s | 63,104 | 11.0 ms | evals | `bf3d096f` | 4,357.0 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed4` | 31,786.9 | 31,690.5 | 758.43 s | 62,976 | 10.8 ms | evals | `770ccd47` | 4,209.0 | v2026.8.17.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64_seed4_pos_t0.3_f1e8` | 31,786.7 | 31,690.5 | 810.36 s | 63,616 | 11.6 ms | evals | `bf3d096f` | 4,622.7 | v2026.8.17.1 |
| `nautilus` | `imaging/pixelization/hst` | `hpc_a100_fp64` | 29,670.4 | 29,590.1 | 3351.29 s | 55,984 | 58.8 ms | evals | `801ba27b` | 2,215.8 | v2026.8.17.1 |
| `nautilus` | `imaging/pixelization/hst` | `hpc_a100_fp64_ref` | 29,670.3 | 29,590.7 | 6535.37 s | 101,296 | 64.0 ms | evals | `801ba27b` | 3,688.0 | v2026.8.17.1 |
| `nautilus` | `imaging/slam_source_pix/hst` | `hpc_a100_fp64_ref` | 30,809.5 | 30,718.0 | 5663.27 s | 127,300 | 44.0 ms | evals | `37150b62` | 4,386.8 | v2026.8.17.1 |
| `nautilus` | `imaging/slam_source_pix/hst` | `hpc_a100_fp64_ref_pos_tauto0.2_f1e8` | 31,411.4 | 31,305.1 | 5387.13 s | 123,700 | 43.1 ms | evals | `b29616db` | 4,737.8 | v2026.8.17.1 |
| `nautilus` | `point_source/image_plane/simple` | `default` | 9.6 | -16.9 | 739.70 s | 13,760 | 53.8 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane/simple` | `hpc_a100_fp64` | 9.6 | -16.8 | 217.11 s | 14,464 | 9.8 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_repeat_solved/simple` | `hpc_a100_fp64` | 7.9 | -8.0 | 162.72 s | 7,100 | 15.5 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_repeat_solved/simple_missing` | `hpc_a100_fp64` | 13.1 | -8.2 | 186.72 s | 9,800 | 13.5 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_solved/near_caustic` | `hpc_a100_fp64` | 29.1 | 5.0 | 176.35 s | 10,500 | 11.7 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_solved/simple` | `hpc_a100_fp64` | 10.6 | -5.3 | 147.21 s | 7,500 | 12.2 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_solved/simple_missing` | `hpc_a100_fp64` | 22.1 | 4.8 | 193.29 s | 11,700 | 12.0 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane/simple` | `default` | -313.2 | -343.9 | 168.00 s | 15,552 | 10.8 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane/simple` | `hpc_a100_fp64` | -313.9 | -344.7 | 217.83 s | 14,400 | 11.2 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_solved/near_caustic` | `hpc_a100_fp64` | 16.5 | -7.8 | 112.85 s | 10,200 | 9.1 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_solved/simple` | `hpc_a100_fp64` | 6.4 | -10.6 | 86.58 s | 7,700 | 8.3 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_tensor/near_caustic` | `hpc_a100_fp64` | 34.5 | -8.9 | 270.32 s | 23,400 | 9.3 ms | evals | — | — | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_tensor/simple` | `hpc_a100_fp64` | 15.3 | -11.8 | 184.36 s | 14,900 | 9.0 ms | evals | — | — | v2026.7.23.1 |
| `nss` | `imaging/delaunay/hst` | `hpc_a100_fp64` | 30,622.2 | 30,567.8 | 29770.41 s | 206,448 | 144.0 ms | evals | — | — | v2026.5.21.1 |
| `nss` | `imaging/delaunay/hst` | `hpc_a100_fp64_mainline` | 30,624.1 | 30,565.2 | 34777.23 s | 150,991 | 230.0 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64` | 31,786.6 | 31,698.9 | 882.78 s | 234,498 | 3.6 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30` | 31,785.8 | 31,691.2 | 4260.68 s | 1,492,747 | 2.8 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_n1000` | 31,786.6 | 31,690.2 | 20319.56 s | 7,519,030 | 2.7 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_n500` | 31,786.6 | 31,691.4 | 10002.55 s | 3,718,760 | 2.7 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_nd100` | 31,786.3 | 31,691.4 | 3572.52 s | 1,236,644 | 2.9 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_nd100_dlogz10` | 31,787.3 | 31,691.8 | 3724.98 s | 1,344,578 | 2.7 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_nd100_seed43` | 31,786.3 | 31,691.0 | 3522.12 s | 1,246,385 | 2.8 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_nd100_seed44` | 31,786.3 | 31,691.7 | 3490.55 s | 1,236,569 | 2.8 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_nd100_seed45` | 31,786.2 | 31,691.5 | 3467.15 s | 1,235,718 | 2.8 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_nd100_seed46` | 31,786.2 | 31,692.0 | 3498.88 s | 1,237,066 | 2.8 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner30_nd20` | 31,786.0 | 31,690.7 | 6175.09 s | 1,638,384 | 3.7 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64_inner45` | 31,786.3 | 31,690.0 | 6382.99 s | 2,264,857 | 2.8 ms | evals | — | — | v2026.8.17.1 |
| `nss` | `imaging/pixelization/hst` | `hpc_a100_fp64` | 29,142.5 | 29,078.9 | 19190.01 s | 266,043 | 72.0 ms | evals | — | — | v2026.5.21.1 |

_`Basis` — what `likelihood_evals` counts in that row. `evals` = reject-inclusive evaluations, comparable across rows. `stored` = a pre-schema-v2 MultiStart run that recorded stored samples, not evaluations; its step count was never written, so `Evals` and `Time / eval` are withheld rather than guessed. Never compare a per-eval figure against a `stored` row (issue #177)._
<!-- END auto-table:searches -->

Auto-generated by `scripts/misc/tooling/build_readme.py` from the search-run artifacts under `results/searches/<sampler>/<class>/<model>/<instrument>/` (latest version per sampler × cell × config).

## Design

| Dimension      | Values                                                                    |
|----------------|---------------------------------------------------------------------------|
| Sampler        | `nautilus`, `multi_start_{adam,prodigy,prodigy_autoconv,lion,adabelief}` (via `_samplers.SAMPLER_BUILDERS`) |
| Dataset class  | `imaging`, `interferometer`, `point_source`, `datacube`, `group`         |
| Model type     | `mge`, `pixelization`, `delaunay`, `image_plane`, `source_plane`          |
| Instrument     | per-dataset-class (HST/Euclid/JWST/AO; SMA/ALMA/ALMA-high/JVLA; simple)   |
| Hardware       | `local_cpu`, `local_gpu`, `hpc_a100` (external dispatch)                  |
| Precision      | `fp64`, `mp` (mixed precision via `al.Settings(use_mixed_precision=...)`) |

Layout:

```
searches/
  README.md                 # this file
  _setup.py                 # dataset/model/analysis dispatchers
  _samplers.py              # sampler registry + per-(ds, model) n_live
  _metrics.py               # viz wall-time interception + result reader
  _runner.py                # shared driver (every leaf calls run_search)
  sweep.py                  # matrix driver, resume-by-default
  aggregate.py              # comparison.json + comparison.png per cell
  nautilus/
    imaging/{mge, pixelization, delaunay}.py
    interferometer/{mge, pixelization, delaunay}.py
    point_source/{image_plane, source_plane}.py
    datacube/delaunay.py
```

## Key design choices

**MAP optimizers alongside samplers.** `multi_start_adam` (`af.MultiStartAdam`,
a JAX/optax multi-start gradient MAP optimizer) is registered as a first-class
search too, but only for the `imaging/mge` cell — the benchmark-proven cell where
a gradient MAP optimizer is meaningful (pixelization/Delaunay/interferometer/
point-source are outside its use case). It is JAX-only (a pure-NumPy config
raises) and has no `n_live` (it records `n_starts`/`n_steps`; the JSON stores
`n_live: null`).

**First-class only.** No more wrapping `nautilus.Sampler` directly. The
old `simple.py` / `jax.py` scripts are deleted. Every cell goes through
`af.Nautilus.fit(model, analysis)`, so visualization, output writes,
sample I/O, and latent-variable computation are part of the profile.

**SLaM-matched `n_live`.** Per `autolens_workspace/scripts/guides/modeling/
slam_start_here.py`: MGE / point-source / parametric phases use
`n_live=200` (matches `source_lp[1]`); pixelization / Delaunay phases
use `n_live=150` (matches `source_pix[1]`).

**`number_of_cores=1` always.** This profile measures per-evaluation
end-to-end cost. Production scaling via `number_of_cores > 1` is a
separate axis a future sweep can introduce.

**JAX rows force `force_x1_cpu=True` and `use_jax_vmap=True`.** This is
mandatory: `nautilus.Sampler` forking under multiprocessing corrupts
JAX state. The trade-off is one batched evaluation per Nautilus step.

**Positions penalty is opt-in (`SEARCHES_POSITIONS`, Phase 4 Stage 1).**
Off by default; see the dedicated "Position likelihood" section below.

**Visualization wall-time is split out.** `_metrics.attach_viz_timer`
wraps every visualize-family hook on the analysis (`visualize`,
`visualize_combined`, `visualize_before_fit`,
`visualize_before_fit_combined`) plus the search's `plot_results`. The
JSON reports `total_wall_s`, `viz_wall_s` and the derived
`sampler_wall_s = total_wall_s - viz_wall_s` so you can ask both "how
long did the full first-class fit take?" and "how much was viz?".

**`sweep.py` wipes search state by default.** PyAutoFit's resume gate is
the `.completed` sentinel file under `<output_path>/searches/...` — once
a `search.fit()` finishes sampling, that file is written and the next
attempt at the same `path_prefix` short-circuits to a cached-result load.
For *production* (SLaM-style chained phases) this is correct behaviour.
For *profiling* it produces 2-3× phantom speedups when a re-run after
a post-fit crash hits the cached `samples.csv`. `sweep.py` therefore
removes `<output_path>/searches/<sampler>/<ds>/<model>/<instrument>/<config>/`
before each cell run by default. Pass `--keep-completed` to opt out
(e.g. when iterating on the post-fit visualization path).

`force_pickle_overwrite=True` is also set on every search, but it only
controls whether output pickles in the `files/` directory get re-written
when an existing search is *resumed* — it does **not** bypass the
`.completed` gate. The sweep-level wipe is what makes re-runs honest.

## Group-scale truth-recovery benchmark (`group/mge`)

The `group` dataset class is the high-dimensional stress test for the JAX
gradient MAP optimizers (autolens_profiling#82): **4 deflector galaxies**
(MGE light + `Isothermal` mass, `ExternalShear` on the primary) lensing **4
background MGE sources** — ~54 free parameters, versus ~14 for the single-lens
`imaging/mge` cell. It answers "do the multi-start gradient optimizers scale to
a harder, higher-dimensional model, and do they recover the input truth?"

- **Simulator.** `simulators/group4_mge.py` builds the tracer from a single
  truth structure (`GROUP4_TRUTH`) and writes the dataset **plus a
  `truth.json`** to `dataset/imaging/group4_mge/<instrument>/`. Auto-simulated
  on first run via the standard `auto_simulate_if_missing` hook
  (`dataset_type="group4_mge"`).
- **Centres are seeded, geometry is not.** Every galaxy's light + mass centre
  gets a modest-sigma Gaussian prior at its known position — the honest prior
  for individually-visible group members, and what breaks the permutation
  symmetry among the 4 lenses / 4 sources. Einstein radii, ellipticities and
  shear keep broad default priors, so the search still has to *find* the mass
  model.
- **Truth recovery is scored.** `searches/_recovery.py` compares the fit's
  `max_log_likelihood_instance` to `truth.json` (per-lens Einstein-radius
  fractional error + mass-centre distance, primary shear, per-source centre)
  and writes a `"recovery"` block — including `overall_pass` — into the summary
  JSON. Nautilus (`nautilus/group/mge`) is the **reference anchor**: if it can't
  recover here, the simulation/model is wrong before any optimizer is judged.
- **Two optimizer modes.** The MultiStart family runs both **fixed-step**
  (`n_steps=300`) and, for `multi_start_prodigy_autoconv`,
  **auto-convergence** (each start early-stops via
  `af.MultiStartGradientConvergence` when its figure-of-merit plateaus). The
  `sampler_config` block records the convergence criterion so the
  early-stop-vs-fixed-300 comparison is self-describing.

## Datacube multi-channel fitting

`datacube/delaunay.py` fits `_DATACUBE_N_CHANNELS` (default 4) identical
interferometer channels via `af.FactorGraphModel`. Each channel becomes
its own `al.AnalysisInterferometer`, wrapped in an `af.AnalysisFactor`
paired with `model.copy()`, then combined under a single global model —
the same pattern documented in
`autolens_workspace/scripts/multi_dataset/modeling.py`. The N channels are
identical copies of the per-instrument dataset; the profile measures
cube-cost scaling, not band-wavelength variation.

To change the channel count, edit `_DATACUBE_N_CHANNELS` in `_setup.py`
(34 matches the existing ALMA cube fiducial; 4 keeps profiling
turnaround sane).

## Standalone instruments

- **`multi_start_nan_accounting_overhead.py`** — does `MultiStartGradient`'s
  per-step value-NaN / gradient-NaN accounting (PyAutoFit#1472) cost any run
  time? Times four ways of obtaining per-lane gradient finiteness against a
  real likelihood, with a duplicate-baseline **control** so the measurement
  noise floor is reported alongside the overhead — the shipped `fused` variant
  is meant to sit below it. Counter-intuitively the `eager` variant (reduce on
  device but outside the jit) is the *worst*, not the best: an un-jitted
  reduction buys a kernel dispatch plus a host round-trip to save a few KB that
  were never the cost. Run it on **both** CPU and GPU — on CPU a device→host
  copy is a same-address-space memcpy, so the `host` variant's true cost is
  invisible there.

  Local CPU row (`mge`, 16 starts, ndim 15, 1.03 s/step): **fused 4.1 us
  (0.0004% of a step)**, host 7.0 us, eager 29.4 us. Absolute values move ~50%
  between CPU runs; the *ordering* (fused < host < eager) has been stable across
  three independent measurements, so read the ranking as the result and the
  magnitudes as order-of-magnitude.

  The `fused` figure comes from a proxy objective at the real array shapes, not
  from the real likelihood — differencing two ~1 s jitted calls cannot resolve a
  ~4 us effect (it returns negative "costs"). That makes it an upper bound on
  the reduction in isolation; XLA's fusion into the real backward pass is not
  captured. The end-to-end loop is ~1200x too coarse to see any of this and
  reports itself as bounding, not measuring.

  Related but distinct: `misc/hazards/checks/nonfinite_gradient.py` *detects*
  non-finite gradients on a likelihood surface; this one measures what it costs
  to *count* them during a fit.

## Position likelihood (`SEARCHES_POSITIONS`)

Phase 4 Stage 1 (issue #159). Attaches a real `al.PositionsLH` penalty to a
cell's analysis — the SLaM `result.positions_likelihood_from` idiom
(`autolens_workspace` SLaM scripts: `factor=3.0, minimum_threshold=0.2`) —
sourced from the simulator's own **truth** positions rather than positions
re-solved from a completed prior search's max-likelihood model (there is no
prior search here to chain from; this is an idealisation, not the production
SLaM chained-fit workflow).

Off by default. Three env vars, all validated (an unrecognised value raises
rather than falling back silently):

| Env var | Values | Default |
|---------|--------|---------|
| `SEARCHES_POSITIONS` | `off` \| `on` | `off` |
| `SEARCHES_POSITIONS_THRESHOLD` | a float, or `auto` | `0.3` |
| `SEARCHES_POSITIONS_FACTOR` | a float (e.g. `1e5`, `1e8`) | `1e8` |

- `SEARCHES_NAUTILUS_SEED=<int>` — af.Nautilus RNG seed (identifier field) for repeated-seed campaigns; carry it in `--config-name` (W2, #160).
- `SEARCHES_LOG_DET_METHOD=cholesky|slogdet` — overrides the evidence log-det. Default (W8, #165): `slogdet` for gradient-work samplers (MultiStart*/NUTS) on pixelized cells when JAX runs on a GPU; `cholesky` (packaged default) for nested samplers and on CPU. Recorded in the results JSON as `log_det_method`. Setting it **explicitly** also appends `ld_<method>` to a MultiStart arm's `unique_tag`, and therefore to its output directory and identifier (#175) — without that suffix a `cholesky` arm and a `slogdet` arm share one directory and the second returns the first's `.completed` fit. Leaving it unset (the W8-resolved default path) adds no suffix, so every recorded cell keeps its exact output path.
- `SEARCHES_BIJECTOR=none|auto_log|log_reg|logit` — the MultiStart* per-parameter bijector arm (W5 Phase 8B, #162). See "Bijector A/B" below.
- `SEARCHES_LANE_HISTORY=1` — turns on `record_lane_nan_history` (PyAutoFit PR#1525): per-step, per-lane value/grad-NaN booleans recorded into the results JSON's `diagnostics` block. Off by default (extra memory/time cost per step).
- `SEARCHES_TRACE_PARAMS=<comma-separated dotted paths>` — turns on `trace_param_indices` (PyAutoFit PR#1525): per-step physical values for the named parameters, recorded as `diagnostics.trace_history`. Paths use `model.path_for_prior` format (the same format `SEARCHES_POSITIONS`/scaler/bijector info blocks use elsewhere in this repo).

`auto` replicates SLaM's `positions_threshold_from(factor=3.0,
minimum_threshold=0.2)`: `max(3.0 * max_sep(truth positions, truth tracer),
0.2)`. Because truth positions trace back to ~zero separation through the
truth tracer by construction, `auto` **collapses to the 0.2 floor** for every
dataset this Stage-1 plumbing supports — a caveat recorded verbatim in every
run's `positions_settings()` block, not left implicit.

**Supported cells**: single-plane `imaging` / `interferometer` only. `group`,
`cluster`, `datacube`(`_img`/`_img_hetero`) and `point_source` all raise
`NotImplementedError` when `SEARCHES_POSITIONS=on` — a per-system or
multi-plane structure the single `PositionsLH` list here does not model, and
a silently-ignored positions request is exactly the failure this plumbing
must not permit.

**Truth positions**: loaded from `dataset/<class>/<instrument>/positions.json`
when present (every shipped `interferometer` dataset already has one). For
`dataset/imaging/hst/` — which does not — they are derived **once** from the
committed `tracer.json` via the simulator's own `PointSolver.for_grid(
pixel_scale_precision=0.001, magnification_threshold=0.1)` recipe
(`scripts/misc/simulators/imaging.py`) and written to
`dataset/imaging/hst/positions.json` (committed to the repo, loudly logged
when it happens). This never touches `data.fits` / `psf.fits` /
`noise_map.fits` — it does not re-simulate.

**Correctness guard — output-path/identifier collisions.** PyAutoFit's
identifier hashes only `[search, model, unique_tag]`; the `Analysis` object
— and therefore whether a `positions_likelihood_list` is attached — is never
part of that hash. A positions-on and positions-off run of the "same"
search/model/config_name would otherwise resolve to the *same* output
directory and identifier, and the `.completed` resume gate would make the
second run's `fit()` silently return the first run's cached result.
`_setup.positions_arm_tag()` (e.g. `pos_t0.3_f1e8`, or `pos_tauto0.2_f1e5`
for the auto/`1e5` arm) is composed into every sampler's `unique_tag` via
`_samplers.arm_unique_tag()`, and appended to `run_search`'s `config_name`,
so this can never happen silently. `_samplers.assert_disjoint_output_paths`
is the reusable guard (see `scripts/misc/test/test_searches_positions.py`).

Every cell's results JSON always carries a `"positions"` block
(`{"enabled": false}` when off) in both `sampler_config` and the top-level
summary, so a positions-on and a positions-off artifact are never ambiguous
after the fact.

Run the Stage-1 hazard/reliability transects with:

```bash
JAX_ENABLE_X64=True python scripts/misc/searches/positions_transects.py --quick
JAX_ENABLE_X64=True python scripts/misc/searches/positions_transects.py
```

See `results/notes/inference/phase_04_positions/transects/RESULTS.md` for the
write-up (classification rule, arms run, and the measured C0 hinge /
interior-plateau / argmax-switch numbers).

## Bijector A/B (`SEARCHES_BIJECTOR`)

W5 Phase 8B (issue #162), following on from Phase 8A's `slogdet_ab.py`. The
PyAutoFit half (`autofit/non_linear/bijector.py`, `MultiStartGradient
(bijector=...)`, opt-in `record_lane_nan_history` / `trace_param_indices`
diagnostics) is PyAutoFit PR#1525. `_samplers.py` resolves an arm label to a
live `af.Bijector*` object and composes it into `multi_start_unique_tag` (a
bijector arm never resumes another arm's `.completed` fit, same discipline as
`SEARCHES_POSITIONS`/`SEARCHES_CLIPPER`/`SEARCHES_SCALER` above).

| Label | Object | Notes |
|---|---|---|
| `none` (default) | `af.BijectorNone` | bit-identical to no bijector at all |
| `auto_log` | `af.BijectorAuto` | log on every eligible `LogUniformPrior`/`LogGaussianPrior`, unrestricted |
| `log_reg` | `af.BijectorPerPath` | log ONLY on paths containing `"regularization."` backed by a `LogUniformPrior` — resolved via a throwaway probe model (`_samplers._probe_model`), never the real dataset |
| `logit` | `af.BijectorLogit` | secondary/opt-in arm — see `autofit.non_linear.bijector`'s module docstring on why it is not a default |

`SEARCHES_LANE_HISTORY=1` / `SEARCHES_TRACE_PARAMS=<paths>` (see the env-var
list above) are the diagnostics the A/B reads: `_per_lane.per_lane_block`
forwards `lane_value_nan_history` / `lane_grad_nan_history` / `trace_history`
/ `trace_param_indices` from `search_internal` straight into every
MultiStart* results JSON's `diagnostics` block when the search was built with
them on.

Driver: `scripts/misc/searches/bijector_ab.py` — pre-registration, the 39-arm
table (`delaunay_adapt_split` x {cholesky,slogdet} x {none,log_reg} x 5 seeds;
`knn` x {none,log_reg,logit} x 5 seeds; `mge` control x {none,log_reg} x 2
seeds), readouts and the F1-F5 falsification scorer are all in its module
docstring. Run:

```bash
python3 scripts/misc/searches/bijector_ab.py --stage run     # 39 real searches (GPU)
python3 scripts/misc/searches/bijector_ab.py --stage score   # F1-F5 verdict from the JSONs
```

Results land under
`results/searches/multi_start_prodigy/imaging/<cell>/hst/phase8b/` (per-arm
JSON+PNG) and `results/notes/inference/phase_08_regularization/bijector_ab/`
(the verdict artifact). See `results/notes/inference/phase_08_regularization/RESULTS.md`
"8B" for the write-up once run, and `hpc/batch_gpu/submit_phase8b_bijector_a100`
for the A100 submit.

## What this *doesn't* profile (yet)

- **Pool scaling.** `number_of_cores > 1` sweeps are future work.
- **Adapt-image regeneration across phases.** Pixelization / Delaunay
  cells use a truth-derived `lensed_source.fits` cached next to the
  dataset. Production SLaM regenerates this between phases.
- **A100 dispatch.** The local sweep generates only CPU and laptop-GPU
  rows. The `hpc_a100_fp64` / `hpc_a100_mp` config names exist in
  `sweep.py` for parity with `likelihood_runtime/`; the actual dispatch
  to RAL HPC happens externally (same mechanism as the likelihood
  sweep).
- **Samplers other than Nautilus.** The registry is in place; adding
  `dynesty`, `blackjax_nuts`, `emcee`, etc. is one function per sampler
  in `_samplers.py`.

## Running

Single cell (CPU NumPy, fastest path):

```bash
python searches/nautilus/imaging/mge.py \
    --instrument hst --config-name local_cpu_fp64
```

Single cell (laptop GPU, JAX-vmap):

```bash
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python searches/nautilus/imaging/mge.py \
    --instrument hst --config-name local_gpu_fp64
```

Full sweep (every cell × instrument × config) — warning, this is long:

```bash
python searches/sweep.py
```

Iteration sweep (one cell, one instrument, CPU only):

```bash
python searches/sweep.py \
    --only nautilus/imaging/mge \
    --instrument hst \
    --skip-gpu --skip-mp
```

Aggregate post-sweep:

```bash
python searches/aggregate.py
```

## Output layout

```
results/searches/
  <sampler>/<dataset_class>/<model>/<instrument>/
    <config_name>.json         # per-config headline metrics
    <config_name>.png          # per-config bar chart
    <config_name>.log          # subprocess stdout/stderr (sweep only)
    comparison.json            # cross-config aggregation (aggregate.py)
    comparison.png             # cross-config bar chart (aggregate.py)
```

The PyAutoFit search itself writes its own output (`samples.csv`,
`samples_info.json`, `search.summary`, visualization, ...) to the
autoconf `output_path` under `path_prefix=searches/<sampler>/
<dataset_class>/<model>/<instrument>`. The metric JSON+PNG above live
separately under `results/searches/`.

## Pixelized-mesh multi-start cells (#117 campaign knowledge)

The `multi_start_prodigy/{pixelization,knn,delaunay}.py` imaging cells were
promoted from the autolens_workspace_developer#117 broad-start campaign
(2026-07; full record: `autolens_workspace_developer/searches_minimal/
pix_prodigy_findings.md`). The durable lessons their configs encode:

- **Gradient multi-start works on pixelized sources** — the #100/#101
  "Nautilus wins pix decisively" verdict inverted once the library search
  gained per-start vmapped state, lr-free Prodigy, and resurrection. knn:
  +29724 @ r_E 1.599 vs a matched-settings Nautilus's +5704 @ r_E 1.011.
- **The regularization axis decides searchability**, not the mesh landscape:
  AdaptSplit's double-squared coefficients make its high-coefficient region
  an escape-taxed floor (knn) or an outright NaN wall (delaunay). Fixed or
  inherited reg (the SLaM `source_pix[1]` pattern) is the fast path
  (~150-250 steps to truth); free Matérn is the safe free parametrization
  (same fit ceiling, no wall) — hence the `delaunay_matern` model type.
- **Budgets**: 16 starts recover the basin; `batch_size=4` is mandatory
  (unbatched 16-start pixelized jvp ≈ 58 GB); 3000 steps because reg-mode
  crossings arrive late (~1300-2000 steps) — a long plateau is a reg mode,
  not convergence.
- **Mesh smoothness class predicts gradient efficiency** (kernel-CDF C∞ >
  knn Wendland > delaunay C0-at-flip-seams) and, by extension, which
  posterior kernels each mesh can host (Hamiltonian on the smooth meshes;
  tempered SMC or warm refits for delaunay).
- **Rectangular caveats (campaign-close state)**: the kernel-CDF
  `value_and_grad` step cost is anomalously high on CPU (~17x knn vs ~4.5x
  forward-eval ratio — profile it on the A100 before drawing landscape
  conclusions), and its *fixed-reg* arm stalled where every other mesh's
  converged — implicating the sharp `bandwidth=0.1` (narrow gradient
  support), with `bandwidth=1.0` costing only ~1.4k nats of fit ceiling.
  Prefer "search smooth, refine sharp" as an annealing schedule; do NOT free
  bandwidth as a model parameter without first checking the joint
  (bandwidth, reg) evidence scan for an interior optimum — a MAP objective
  may rail it at the staircase limit.
- **Ops**: multi-start resume chains do not survive library upgrades that
  touch FoM bookkeeping (the resume sanity check refuses, by design) — pin
  the HPC mirrors for a campaign or plan to restart in-flight chains.
