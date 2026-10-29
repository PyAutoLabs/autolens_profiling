<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/alma_high/a86349ca39db79f5fe04`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | {"cache_existed": true, "mask_radius_arcsec": 3.5, "md5_after": "c3d20edcef4343c530ce40a57c45237d", "md5_before": "c3d20edcef4343c530ce40a57c45237d", "regenerated": false} | recorded |  |
| arms | ["numba", "numpy_fft"] | recorded |  |
| cpu_count | 124 | recorded |  |
| curvature_blocks_fft | 12 | recorded |  |
| headline_arm | "numba" | recorded |  |
| host_load_avg_end | [1.02, 1.89, 2.75] | recorded |  |
| host_load_avg_start | [4.33, 0.99, 0.33] | recorded |  |
| hostname | "euclid-ral-gpu-1" | recorded |  |
| image_pixels_masked | 61572 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance_stream | {"arm_order": "ABBA (alternates per instance)", "kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| inversion_class | "InversionInterferometerSparseNumba" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mask_radius_preset_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | "RectangularBilinearAdaptImage" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| n_instances | 20 | recorded |  |
| n_sub_repeats | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| n_warm | 19 | recorded |  |
| nnz_per_source_column | 161.9250493096647 | recorded |  |
| numba_gate | 1000000000.0 | recorded |  |
| numba_gate_library_default | 60.0 | recorded |  |
| numba_num_threads_env | "1" | recorded |  |
| numba_parallel_kernel | false | recorded |  |
| numba_thread_pool | 1 | recorded |  |
| numba_threads_headline | 1 | recorded |  |
| numba_version | "0.65.1" | recorded |  |
| operator_M | 78400 | recorded |  |
| operator_extent_shape | [280, 280] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.025 | recorded |  |
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
| transformer_chunk_size | 1000000 | recorded |  |
| visibilities | 5000000 | recorded |  |
| would_route_numba_at_default_gate | false | recorded |  |
| xp | "numpy" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>23 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/083a8691f7f30c179ae0` | steps.Log-det (F + H) (fnnls Cholesky reuse) | {"steps.Log-det (F + H) (fnnls Cholesky reuse)": 0.04759899595305078} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/0a42516001c474b54b67` | steps_median.Regularization term sᵀHs | {"steps_median.Regularization term s\u1d40Hs": 0.0008640800078865141} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Regularization term sᵀHs` |
| `measurement/1353401db7a4fa38ff2c` | steps.Reconstruction: D = Lᵀ d~ + fnnls | {"steps.Reconstruction: D = L\u1d40 d~ + fnnls": 0.44415727326199167} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/1ac805c5e4ccdc1d0190` | steps_median.Log-det H | {"steps_median.Log-det H": 0.005487904010806233} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Log-det H` |
| `measurement/2b7fff337c029a12bb97` | steps_median.Reconstruction: D = Lᵀ d~ + fnnls | {"steps_median.Reconstruction: D = L\u1d40 d~ + fnnls": 0.44659629598027095} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/3f726ca2aae6b8e9f4fa` | component_total | {"component_total": 19.026710038747346} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/total_step_by_step` |
| `measurement/41956dcfda7cb02fe07e` | steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.08615937901777215} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/41a69a286436a511c390` | steps.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.08676941936688595} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/4301ee3f172f61caa84c` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.0069704488391595845} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/4906e4acc35c77afea6c` | steps_median.Curvature F + H: F by numba direct_conv | {"steps_median.Curvature F + H: F by numba direct_conv": 18.169811366999056} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Curvature F + H: F by numba direct_conv` |
| `measurement/61d617c9fcb73fb09fa0` | steps.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.01692922094585619} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/63a8184d03c577a029c7` | steps_median.Noise normalisation + evidence assembly | {"steps_median.Noise normalisation + evidence assembly": 0.07922194601269439} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Noise normalisation + evidence assembly` |
| `measurement/66faa4f844a104209655` | steps_median.Log-det (F + H) (fnnls Cholesky reuse) | {"steps_median.Log-det (F + H) (fnnls Cholesky reuse)": 0.04733900399878621} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/6724d8860b863638ceb2` | steps.Noise normalisation + evidence assembly | {"steps.Noise normalisation + evidence assembly": 0.08109336073730926} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Noise normalisation + evidence assembly` |
| `measurement/6e816758d02c60e330ee` | steps_median.Mapping matrix L | {"steps_median.Mapping matrix L": 0.103046436997829} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Mapping matrix L` |
| `measurement/78e61d0ec00f786c143d` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.10357815911227494} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/7eef5f06ab4946b219a0` | steps_median.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps_median.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.041536664008162916} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/7f339f66787e0045129e` | steps.Log-det H | {"steps.Log-det H": 0.0055558778449745945} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Log-det H` |
| `measurement/c057ce590f99575429cf` | steps.Curvature F + H: F by numba direct_conv | {"steps.Curvature F + H: F by numba direct_conv": 18.19065180294657} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Curvature F + H: F by numba direct_conv` |
| `measurement/ce5fceb24e34a1658094` | steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.01680927799316123} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/e1405483f864f41293f8` | steps_median.Regularization matrix H | {"steps_median.Regularization matrix H": 0.00686549799866043} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps_median/Regularization matrix H` |
| `measurement/ec110c3a4e13cb62716a` | steps.Regularization term sᵀHs | {"steps.Regularization term s\u1d40Hs": 0.00089157779094469} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Regularization term sᵀHs` |
| `measurement/ecf9b846ae62d856de7c` | steps.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.042513901948328374} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |

</details>

### compile

<details><summary>1 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/72e5d1f109dd3f8e30b2` | operator_setup | {"operator_setup": 17.81747134399484} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/operator_build_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/6eec5dc94a2273f19386` | host_peak_rss | {"host_peak_rss": 8812.2421875} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/5e4381fdbead1e0a249a` | full_call.median_s | {"full_call.median_s": 18.969985739997355} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/full_call/median_s` |
| `measurement/8e0baf80d1222e1ab50d` | full_call.mean_s | {"full_call.mean_s": 19.043804775264825} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/full_call/mean_s` |
| `measurement/a312b53344f14017d66e` | full_call.min_s | {"full_call.min_s": 18.72270956399734} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/full_call/min_s` |
| `measurement/e448961b442ddf9fedc0` | full_call.max_s | {"full_call.max_s": 19.361711283010663} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/full_call/max_s` |

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
- [likelihood_runtime_matrixfree](../../../../scripts/interferometer/rectangular/likelihood_runtime_matrixfree.py)
- [likelihood_runtime_numba](../../../../scripts/interferometer/rectangular/likelihood_runtime_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
