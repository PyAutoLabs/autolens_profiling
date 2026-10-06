<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · alma

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/alma/f39751ede135253ef378`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | {"cache_existed": true, "mask_radius_arcsec": 2.0, "md5_after": "6bf5afdf64a9396dcf743e05c2693317", "md5_before": "6bf5afdf64a9396dcf743e05c2693317", "regenerated": false} | recorded |  |
| arms | ["numba", "numpy_fft"] | recorded |  |
| cpu_count | 124 | recorded |  |
| curvature_blocks_fft | 12 | recorded |  |
| headline_arm | "numba" | recorded |  |
| host_load_avg_end | [12.48, 5.62, 2.14] | recorded |  |
| host_load_avg_start | [4.33, 0.99, 0.33] | recorded |  |
| hostname | "euclid-ral-gpu-1" | recorded |  |
| image_pixels_masked | 5024 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance_stream | {"arm_order": "ABBA (alternates per instance)", "kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| inversion_class | "InversionInterferometerSparseNumba" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 2.0 | recorded |  |
| mask_radius_preset_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | "RectangularBilinearAdaptImage" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| n_instances | 20 | recorded |  |
| n_sub_repeats | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| n_warm | 19 | recorded |  |
| nnz_per_source_column | 13.212360289283366 | recorded |  |
| numba_gate | 1000000000.0 | recorded |  |
| numba_gate_library_default | 60.0 | recorded |  |
| numba_num_threads_env | "1" | recorded |  |
| numba_parallel_kernel | false | recorded |  |
| numba_thread_pool | 1 | recorded |  |
| numba_threads_headline | 1 | recorded |  |
| numba_version | "0.65.1" | recorded |  |
| operator_M | 6400 | recorded |  |
| operator_extent_shape | [80, 80] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| pixels_solved | 1369 | recorded |  |
| positive_only_solver | "fnnls" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "scheme": "constant"} | name |  |
| solve_subset_edge_zeroed | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1521 | count |  |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| thread_env | {"AUTOLENS_PROFILING_LEVER_NUMBA_THREADS": "1", "NUMBA_NUM_THREADS_before": "1", "n_threads": 1, "note": "Set by _production_config.pin_thread_env before numpy import; mirrors the production SLURM submit scripts.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": "1", "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 100000 | recorded |  |
| visibilities | 1000000 | recorded |  |
| would_route_numba_at_default_gate | true | recorded |  |
| xp | "numpy" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>23 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0ec573aa302958e96923` | steps.Regularization term sᵀHs | {"steps.Regularization term s\u1d40Hs": 0.0010524033124583137} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Regularization term sᵀHs` |
| `measurement/25368437d00eca78838e` | steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.020883249992039055} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/2a9609a8804082fa278b` | steps.Log-det (F + H) (fnnls Cholesky reuse) | {"steps.Log-det (F + H) (fnnls Cholesky reuse)": 0.053040737843778184} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/2e41ccece4ee0e819714` | steps_median.Regularization matrix H | {"steps_median.Regularization matrix H": 0.007988484983798116} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Regularization matrix H` |
| `measurement/3a06953085b4212b481c` | steps.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.009015774158270736} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/42f10993c497f5bfd0aa` | steps_median.Mapping matrix L | {"steps_median.Mapping matrix L": 0.01056725301896222} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Mapping matrix L` |
| `measurement/59f31c7ee8aa0f1b7d31` | steps_median.Noise normalisation + evidence assembly | {"steps_median.Noise normalisation + evidence assembly": 0.015117909002583474} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Noise normalisation + evidence assembly` |
| `measurement/5faf9e63801bee17677a` | steps.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.005160658267988383} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/6942493bfc481cd72d90` | steps_median.Log-det H | {"steps_median.Log-det H": 0.0058363599819131196} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Log-det H` |
| `measurement/743968844b6fb18c00fb` | steps_median.Curvature F + H: F by numba direct_conv | {"steps_median.Curvature F + H: F by numba direct_conv": 0.18593546500778757} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Curvature F + H: F by numba direct_conv` |
| `measurement/762d8c3a4691e6d2fe13` | steps_median.Regularization term sᵀHs | {"steps_median.Regularization term s\u1d40Hs": 0.0010260269918944687} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Regularization term sᵀHs` |
| `measurement/7bd2c1a83c67cae20568` | steps.Log-det H | {"steps.Log-det H": 0.00582750873505383} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Log-det H` |
| `measurement/8a0614fff49ea21ea386` | component_total | {"component_total": 0.7980945491106373} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/total_step_by_step` |
| `measurement/9456e4814530e6f41bdf` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.008222959947919375} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Regularization matrix H` |
| `measurement/96fbe233e4a0aa1bcd1c` | steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.0050977200153283775} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/9ca2d17b5c65576b1aaf` | steps.Noise normalisation + evidence assembly | {"steps.Noise normalisation + evidence assembly": 0.015080858156771251} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Noise normalisation + evidence assembly` |
| `measurement/9d9de6d3b7252da94929` | steps_median.Reconstruction: D = Lᵀ d~ + fnnls | {"steps_median.Reconstruction: D = L\u1d40 d~ + fnnls": 0.48184509298880585} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/b3422f6eb51e07f02229` | steps.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.02102225263431472} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/bb8828fb2344ee5ca61f` | steps.Curvature F + H: F by numba direct_conv | {"steps.Curvature F + H: F by numba direct_conv": 0.18776300983920105} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Curvature F + H: F by numba direct_conv` |
| `measurement/c36044587a341e7358d6` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.010687391739338636} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Mapping matrix L` |
| `measurement/e10cb4e82c49dc0d6604` | steps_median.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps_median.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.00854626702493988} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/e94f62b55f5ebf34eddb` | steps.Reconstruction: D = Lᵀ d~ + fnnls | {"steps.Reconstruction: D = L\u1d40 d~ + fnnls": 0.48122099447554273} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/eedebc59e66ea0bc3b6c` | steps_median.Log-det (F + H) (fnnls Cholesky reuse) | {"steps_median.Log-det (F + H) (fnnls Cholesky reuse)": 0.05278469799668528} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Log-det (F + H) (fnnls Cholesky reuse)` |

</details>

### compile

<details><summary>1 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/d41f3dbf7ca40d944868` | operator_setup | {"operator_setup": 14.958129753998946} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/operator_build_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/d4f62280e58f69048932` | host_peak_rss | {"host_peak_rss": 2298.421875} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/13585ad24b057e4d6558` | full_call.max_s | {"full_call.max_s": 0.8827419249864761} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/max_s` |
| `measurement/17ebc1ff1f272d7ba1c1` | full_call.median_s | {"full_call.median_s": 0.7820138109964319} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/median_s` |
| `measurement/b2cf1ac228da79346f6d` | full_call.mean_s | {"full_call.mean_s": 0.7831767298431307} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/mean_s` |
| `measurement/e636461901472f1f7de9` | full_call.min_s | {"full_call.min_s": 0.7236182310152799} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/min_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "e281abf3257e9efd4915f1daa9acc0498be250a5",
      "PyAutoFit": "c156a9d8e8eb9f82b6da169b1d50b7e64580fe98",
      "PyAutoGalaxy": "ba8a08fabcb13d9472f2bd43ef4fea6039284d50",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "autolens_profiling": "92a20614c6c23faefae429e7b9ec4cfae7c9611c"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 19,
    "statistic": "max",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": false,
    "host": "euclid-ral-gpu-1",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "e281abf3257e9efd4915f1daa9acc0498be250a5",
      "PyAutoFit": "c156a9d8e8eb9f82b6da169b1d50b7e64580fe98",
      "PyAutoGalaxy": "ba8a08fabcb13d9472f2bd43ef4fea6039284d50",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "autolens_profiling": "92a20614c6c23faefae429e7b9ec4cfae7c9611c"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 19,
    "statistic": "mean",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": false,
    "host": "euclid-ral-gpu-1",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "e281abf3257e9efd4915f1daa9acc0498be250a5",
      "PyAutoFit": "c156a9d8e8eb9f82b6da169b1d50b7e64580fe98",
      "PyAutoGalaxy": "ba8a08fabcb13d9472f2bd43ef4fea6039284d50",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "autolens_profiling": "92a20614c6c23faefae429e7b9ec4cfae7c9611c"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 19,
    "statistic": "median",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": false,
    "host": "euclid-ral-gpu-1",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "e281abf3257e9efd4915f1daa9acc0498be250a5",
      "PyAutoFit": "c156a9d8e8eb9f82b6da169b1d50b7e64580fe98",
      "PyAutoGalaxy": "ba8a08fabcb13d9472f2bd43ef4fea6039284d50",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "autolens_profiling": "92a20614c6c23faefae429e7b9ec4cfae7c9611c"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 19,
    "statistic": "min",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": false,
    "host": "euclid-ral-gpu-1",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "e281abf3257e9efd4915f1daa9acc0498be250a5",
      "PyAutoFit": "c156a9d8e8eb9f82b6da169b1d50b7e64580fe98",
      "PyAutoGalaxy": "ba8a08fabcb13d9472f2bd43ef4fea6039284d50",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "autolens_profiling": "92a20614c6c23faefae429e7b9ec4cfae7c9611c"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": null,
    "statistic": "median",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "repetitions": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": false,
    "host": "euclid-ral-gpu-1",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "e281abf3257e9efd4915f1daa9acc0498be250a5",
      "PyAutoFit": "c156a9d8e8eb9f82b6da169b1d50b7e64580fe98",
      "PyAutoGalaxy": "ba8a08fabcb13d9472f2bd43ef4fea6039284d50",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "autolens_profiling": "92a20614c6c23faefae429e7b9ec4cfae7c9611c"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": null,
    "statistic": "peak",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "repetitions": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": false,
    "host": "euclid-ral-gpu-1",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "e281abf3257e9efd4915f1daa9acc0498be250a5",
      "PyAutoFit": "c156a9d8e8eb9f82b6da169b1d50b7e64580fe98",
      "PyAutoGalaxy": "ba8a08fabcb13d9472f2bd43ef4fea6039284d50",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "autolens_profiling": "92a20614c6c23faefae429e7b9ec4cfae7c9611c"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": null,
    "statistic": null,
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "repetitions": "Not recorded in this legacy evidence; no current default substituted.",
      "statistic": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": false,
    "host": "euclid-ral-gpu-1",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

## Hazards

No explicitly bound applicable evidence; this does not establish absence of hazards or validate a setting.

## Recommendations

No explicitly bound applicable evidence; this does not establish absence of hazards or validate a setting.

## Scripts

Navigation links identify the model family; they do not reconstruct historical settings or promise an executable replay.

- [likelihood_breakdown](../../../../scripts/interferometer/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/rectangular/likelihood_runtime.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
