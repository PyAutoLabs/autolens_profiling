<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · alma

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/alma/a6559c490d9e13f44f27`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| arms | ["numba", "numpy_fft"] | recorded |  |
| curvature_blocks_fft | 32 | recorded |  |
| delaunay_vertices | 4000 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| headline_arm | "numba" | recorded |  |
| hilbert_pixels | 4000 | recorded |  |
| host_load_avg_end | [4.0, 4.2, 3.66] | recorded |  |
| host_load_avg_start | [4.68, 5.54, 3.28] | recorded |  |
| image_mesh | "Hilbert(weight_power=1.0, weight_floor=0.0)" | recorded |  |
| image_pixels_masked | 15380 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance_stream | {"arm_order": "ABBA (alternates per instance)", "kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| inversion_class | "InversionInterferometerSparseNumba" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mask_radius_preset_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | "Delaunay" | recorded |  |
| n_instances | 12 | recorded |  |
| n_sub_repeats | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| n_warm | 11 | recorded |  |
| nnz_per_source_column | 11.1645 | recorded |  |
| numba_gate | 1000000000.0 | recorded |  |
| numba_gate_library_default | 60.0 | recorded |  |
| numba_num_threads_env | "1" | recorded |  |
| numba_parallel_kernel | false | recorded |  |
| numba_version | "0.65.1" | recorded |  |
| operator_M | 19600 | recorded |  |
| operator_extent_shape | [140, 140] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| pixels_solved | 4000 | recorded |  |
| positive_only_solver | "fnnls" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| solve_subset_edge_zeroed | false | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 4000 | count |  |
| source_pixels_requested | 4000 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| thread_env | {"NUMBA_NUM_THREADS_before": "1", "n_threads": 1, "note": "Set by _production_config.pin_thread_env before numpy import; mirrors the production SLURM submit scripts.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": "1", "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 100000 | recorded |  |
| visibilities | 1000000 | recorded |  |
| would_route_numba_at_default_gate | true | recorded |  |
| xp | "numpy" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>23 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0dbf9c31d1e2d0d16496` | steps_median.Log-det H | {"steps_median.Log-det H": 0.028116439993027598} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Log-det H` |
| `measurement/0dfe575b937bcdbd1171` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.03809360400191508} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/11fba417c6f9c3438ab7` | steps_median.Log-det (F + H) (fnnls Cholesky reuse) | {"steps_median.Log-det (F + H) (fnnls Cholesky reuse)": 0.2108510340040084} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/268d4a22cf49f71abe0c` | steps.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d; F, D re-evaluated) | {"steps.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d; F, D re-evaluated)": 1.2986709215461318} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d; F, D re-evaluated)` |
| `measurement/28add30215609a83afd6` | steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.006289388024015352} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/2e14711580cb57bd2556` | steps.Curvature F + H: F by numba direct_conv | {"steps.Curvature F + H: F by numba direct_conv": 1.2339782803610433} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Curvature F + H: F by numba direct_conv` |
| `measurement/34863ba987ce8ffba3a8` | steps.Reconstruction: D = Lᵀ d~ + fnnls | {"steps.Reconstruction: D = L\u1d40 d~ + fnnls": 3.538124156631635} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/353997af13586eb07f60` | steps_median.Noise normalisation + evidence assembly | {"steps_median.Noise normalisation + evidence assembly": 0.013488411990692839} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Noise normalisation + evidence assembly` |
| `measurement/44495539a2be5f02efd6` | steps.Noise normalisation + evidence assembly | {"steps.Noise normalisation + evidence assembly": 0.01351847772391259} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Noise normalisation + evidence assembly` |
| `measurement/575242d46001cfbd6fa7` | steps_median.Regularization term sᵀHs | {"steps_median.Regularization term s\u1d40Hs": 0.0048422489780932665} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Regularization term sᵀHs` |
| `measurement/77080754842df7ed9885` | steps_median.Mapping matrix L | {"steps_median.Mapping matrix L": 0.03805513301631436} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Mapping matrix L` |
| `measurement/9c9ea88438b4ed0e6bbb` | steps_median.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d; F, D re-evaluated) | {"steps_median.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d; F, D re-evaluated)": 1.2959445369779132} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d; F, D re-evaluated)` |
| `measurement/9ebe8b1c658bb29ae5f9` | steps.Log-det H | {"steps.Log-det H": 0.028192539813145148} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Log-det H` |
| `measurement/ad0e91ae1adb57325ffe` | steps.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.060485491002592345} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/b339e0572dc984289e6c` | steps.Log-det (F + H) (fnnls Cholesky reuse) | {"steps.Log-det (F + H) (fnnls Cholesky reuse)": 0.2113874276427933} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/bd054066ebc55fa01a03` | steps.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.006279967542716556} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/c2798b6b67ae49cbdea4` | steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.06084967899369076} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/e63483ede36bc6bcd1c3` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.03648191536194645} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/f1d496b6a5f16369d3ea` | component_total | {"component_total": 6.470073108265007} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/total_step_by_step` |
| `measurement/f698735366419edc7b30` | steps_median.Regularization matrix H | {"steps_median.Regularization matrix H": 0.03658469699439593} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Regularization matrix H` |
| `measurement/f8d8b93cc95c35798133` | steps_median.Reconstruction: D = Lᵀ d~ + fnnls | {"steps_median.Reconstruction: D = L\u1d40 d~ + fnnls": 3.5581870059832} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/fdf82b65fc735361053a` | steps.Regularization term sᵀHs | {"steps.Regularization term s\u1d40Hs": 0.0048603266371752725} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps/Regularization term sᵀHs` |
| `measurement/fec2deb592b73bbccf78` | steps_median.Curvature F + H: F by numba direct_conv | {"steps_median.Curvature F + H: F by numba direct_conv": 1.236243170016678} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/steps_median/Curvature F + H: F by numba direct_conv` |

</details>

### compile

<details><summary>1 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/908d707fe47534afd293` | operator_setup | {"operator_setup": 4.716435328999069} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/operator_build_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/fba5479d89b7472e46db` | host_peak_rss | {"host_peak_rss": 2374.21875} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/3395fca1c6ef2d448065` | full_call.max_s | {"full_call.max_s": 6.867436844011536} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/full_call/max_s` |
| `measurement/c09af90659425ce6c8bc` | full_call.min_s | {"full_call.min_s": 6.225477468018653} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/full_call/min_s` |
| `measurement/d42501d4f9f40b56c954` | full_call.median_s | {"full_call.median_s": 6.473636320995865} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/full_call/median_s` |
| `measurement/f59ce8b0050d36e714f6` | full_call.mean_s | {"full_call.mean_s": 6.461742007734508} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/scans/n_sweep_n4000/breakdown.json) `/full_call/mean_s` |

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
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "5468c6cefa0ae8bb08a449497cbd3c620f3720e2",
      "PyAutoGalaxy": "879a9308b709b3705eb95904de6b349a5c034d61",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "eb27da2464e2f07e0ad67e7fb18ee84be76c479a",
      "autolens_profiling": "9721d3912a565ec05bbf53e539eadaaba538cb25"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 11,
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
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "5468c6cefa0ae8bb08a449497cbd3c620f3720e2",
      "PyAutoGalaxy": "879a9308b709b3705eb95904de6b349a5c034d61",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "eb27da2464e2f07e0ad67e7fb18ee84be76c479a",
      "autolens_profiling": "9721d3912a565ec05bbf53e539eadaaba538cb25"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 11,
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
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "5468c6cefa0ae8bb08a449497cbd3c620f3720e2",
      "PyAutoGalaxy": "879a9308b709b3705eb95904de6b349a5c034d61",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "eb27da2464e2f07e0ad67e7fb18ee84be76c479a",
      "autolens_profiling": "9721d3912a565ec05bbf53e539eadaaba538cb25"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 11,
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
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "5468c6cefa0ae8bb08a449497cbd3c620f3720e2",
      "PyAutoGalaxy": "879a9308b709b3705eb95904de6b349a5c034d61",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "eb27da2464e2f07e0ad67e7fb18ee84be76c479a",
      "autolens_profiling": "9721d3912a565ec05bbf53e539eadaaba538cb25"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": 11,
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
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "5468c6cefa0ae8bb08a449497cbd3c620f3720e2",
      "PyAutoGalaxy": "879a9308b709b3705eb95904de6b349a5c034d61",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "eb27da2464e2f07e0ad67e7fb18ee84be76c479a",
      "autolens_profiling": "9721d3912a565ec05bbf53e539eadaaba538cb25"
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
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "5468c6cefa0ae8bb08a449497cbd3c620f3720e2",
      "PyAutoGalaxy": "879a9308b709b3705eb95904de6b349a5c034d61",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "eb27da2464e2f07e0ad67e7fb18ee84be76c479a",
      "autolens_profiling": "9721d3912a565ec05bbf53e539eadaaba538cb25"
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
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "5468c6cefa0ae8bb08a449497cbd3c620f3720e2",
      "PyAutoGalaxy": "879a9308b709b3705eb95904de6b349a5c034d61",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "eb27da2464e2f07e0ad67e7fb18ee84be76c479a",
      "autolens_profiling": "9721d3912a565ec05bbf53e539eadaaba538cb25"
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
- [quick_update](../../../../scripts/interferometer/delaunay/quick_update.py)
- [_streaming](../../../../scripts/interferometer/delaunay/_streaming.py)
- [streaming_accumulate](../../../../scripts/interferometer/delaunay/streaming_accumulate.py)
- [streaming_in_memory](../../../../scripts/interferometer/delaunay/streaming_in_memory.py)
- [streaming_parity](../../../../scripts/interferometer/delaunay/streaming_parity.py)
- [streaming_plot_scaling](../../../../scripts/interferometer/delaunay/streaming_plot_scaling.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
