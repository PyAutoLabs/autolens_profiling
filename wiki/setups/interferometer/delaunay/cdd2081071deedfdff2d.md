<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · alma

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/alma/e003216d750509981b5e`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | {"cache_existed": true, "mask_radius_arcsec": 5.0, "md5_after": "6bf5afdf64a9396dcf743e05c2693317", "md5_before": "6bf5afdf64a9396dcf743e05c2693317", "regenerated": false} | recorded |  |
| arms | ["numba", "numpy_fft"] | recorded |  |
| cpu_count | 124 | recorded |  |
| curvature_blocks_fft | 12 | recorded |  |
| delaunay_vertices | 1500 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| headline_arm | "numba" | recorded |  |
| hilbert_pixels | 1500 | recorded |  |
| host_load_avg_end | [6.98, 7.06, 3.72] | recorded |  |
| host_load_avg_start | [4.33, 0.99, 0.33] | recorded |  |
| hostname | "euclid-ral-gpu-1" | recorded |  |
| image_mesh | "Hilbert(weight_power=1.0, weight_floor=0.0)" | recorded |  |
| image_pixels_masked | 31428 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance_stream | {"arm_order": "ABBA (alternates per instance)", "kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| inversion_class | "InversionInterferometerSparseNumba" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 5.0 | recorded |  |
| mask_radius_preset_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | "Delaunay" | recorded |  |
| n_instances | 20 | recorded |  |
| n_sub_repeats | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| n_warm | 19 | recorded |  |
| nnz_per_source_column | 55.58133333333333 | recorded |  |
| numba_gate | 1000000000.0 | recorded |  |
| numba_gate_library_default | 60.0 | recorded |  |
| numba_num_threads_env | "1" | recorded |  |
| numba_parallel_kernel | false | recorded |  |
| numba_thread_pool | 1 | recorded |  |
| numba_threads_headline | 1 | recorded |  |
| numba_version | "0.65.1" | recorded |  |
| operator_M | 40000 | recorded |  |
| operator_extent_shape | [200, 200] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| pixels_solved | 1500 | recorded |  |
| positive_only_solver | "fnnls" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| solve_subset_edge_zeroed | false | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
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

[Original setup artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>23 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0951e09639e615570705` | steps.Regularization term sᵀHs | {"steps.Regularization term s\u1d40Hs": 0.0009369081572482461} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Regularization term sᵀHs` |
| `measurement/0b8e3c645714e7733249` | steps.Curvature F + H: F by numba direct_conv | {"steps.Curvature F + H: F by numba direct_conv": 3.399189680626334} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Curvature F + H: F by numba direct_conv` |
| `measurement/2178a65268c1504516f6` | steps.Log-det (F + H) (fnnls Cholesky reuse) | {"steps.Log-det (F + H) (fnnls Cholesky reuse)": 0.017327738366714727} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/33ddb2565ab598c5d4be` | steps.Log-det H | {"steps.Log-det H": 0.006987828842523557} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Log-det H` |
| `measurement/39e626e1b9813be0af6b` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.01000570236723625} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/4f846bb427ef83edd14a` | steps_median.Log-det H | {"steps_median.Log-det H": 0.006938456004718319} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Log-det H` |
| `measurement/512d6c7728c6d3232988` | steps_median.Log-det (F + H) (fnnls Cholesky reuse) | {"steps_median.Log-det (F + H) (fnnls Cholesky reuse)": 0.017242002999410033} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/5448e12f7dad9ce464ee` | steps_median.Reconstruction: D = Lᵀ d~ + fnnls | {"steps_median.Reconstruction: D = L\u1d40 d~ + fnnls": 0.24525592400459573} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/562b937a9595084bc40e` | steps.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.04588552831525081} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/788a52a27e5c38d6849c` | steps_median.Noise normalisation + evidence assembly | {"steps_median.Noise normalisation + evidence assembly": 0.013863234024029225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Noise normalisation + evidence assembly` |
| `measurement/7b54c1c28a13ea040330` | steps.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.008192087735024918} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/9276e66ac8d5294acbc0` | steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.008180862001609057} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/9975babaf46709765efb` | steps_median.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps_median.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.007652958011021838} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/ac9134249d782fd9e408` | steps_median.Regularization term sᵀHs | {"steps_median.Regularization term s\u1d40Hs": 0.0009198990010190755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Regularization term sᵀHs` |
| `measurement/b3a7d2afc6c44f73cdb8` | steps_median.Regularization matrix H | {"steps_median.Regularization matrix H": 0.010007389995735139} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Regularization matrix H` |
| `measurement/c54501b386f773cdf731` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.03909684652860865} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/d67ea07be8c8f97e6d3d` | steps.Reconstruction: D = Lᵀ d~ + fnnls | {"steps.Reconstruction: D = L\u1d40 d~ + fnnls": 0.24839036200060755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/da62776950b6d9b667ea` | steps.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.007606996788557521} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/dca0a223c84b4c8c9d02` | steps_median.Mapping matrix L | {"steps_median.Mapping matrix L": 0.03575251100119203} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Mapping matrix L` |
| `measurement/e06b50aedae699e7382f` | steps_median.Curvature F + H: F by numba direct_conv | {"steps_median.Curvature F + H: F by numba direct_conv": 3.3917168279876933} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Curvature F + H: F by numba direct_conv` |
| `measurement/e0bc9c580545c681873a` | steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.04600604000734165} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps_median/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/f877f65fccd1ca2d26f0` | steps.Noise normalisation + evidence assembly | {"steps.Noise normalisation + evidence assembly": 0.013973253370127886} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/steps/Noise normalisation + evidence assembly` |
| `measurement/fe7f75fca7bb90086c79` | component_total | {"component_total": 3.797592933098234} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/total_step_by_step` |

</details>

### compile

<details><summary>1 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/d04f3111492aa3430213` | operator_setup | {"operator_setup": 13.042849515972193} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/operator_build_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/5a6205916b7b945c890d` | host_peak_rss | {"host_peak_rss": 2397.0625} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/28f0acd73962193ba02b` | full_call.mean_s | {"full_call.mean_s": 3.7887282652143193} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/full_call/mean_s` |
| `measurement/3c8d675ebec9468e2fe2` | full_call.min_s | {"full_call.min_s": 3.7353024360199925} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/full_call/min_s` |
| `measurement/a8d7bbc5c96151699011` | full_call.max_s | {"full_call.max_s": 3.8420657179958653} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/full_call/max_s` |
| `measurement/d13f37eb44338a8e114b` | full_call.median_s | {"full_call.median_s": 3.7776144609961193} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/r5.0/breakdown.json) `/full_call/median_s` |

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

- [likelihood_breakdown](../../../../scripts/interferometer/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/interferometer/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/interferometer/delaunay/quick_update.py)
- [_streaming](../../../../scripts/interferometer/delaunay/_streaming.py)
- [streaming_accumulate](../../../../scripts/interferometer/delaunay/streaming_accumulate.py)
- [streaming_in_memory](../../../../scripts/interferometer/delaunay/streaming_in_memory.py)
- [streaming_parity](../../../../scripts/interferometer/delaunay/streaming_parity.py)
- [streaming_plot_scaling](../../../../scripts/interferometer/delaunay/streaming_plot_scaling.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
