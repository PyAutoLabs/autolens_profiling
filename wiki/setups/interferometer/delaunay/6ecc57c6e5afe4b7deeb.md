<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · sma

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/sma/fa389784ee8efbf199b6`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | {"cache_existed": true, "mask_radius_arcsec": 3.5, "md5_after": "a978d8b16bad31379a7810a35bfd71b1", "md5_before": "a978d8b16bad31379a7810a35bfd71b1", "regenerated": false} | recorded |  |
| arms | ["numba", "numpy_fft"] | recorded |  |
| cpu_count | 124 | recorded |  |
| curvature_blocks_fft | 12 | recorded |  |
| delaunay_vertices | 1500 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| headline_arm | "numba" | recorded |  |
| hilbert_pixels | 1500 | recorded |  |
| host_load_avg_end | [12.39, 4.19, 1.49] | recorded |  |
| host_load_avg_start | [4.33, 0.99, 0.33] | recorded |  |
| hostname | "euclid-ral-gpu-1" | recorded |  |
| image_mesh | "Hilbert(weight_power=1.0, weight_floor=0.0)" | recorded |  |
| image_pixels_masked | 3852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance_stream | {"arm_order": "ABBA (alternates per instance)", "kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| inversion_class | "InversionInterferometerSparseNumba" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mask_radius_preset_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | "Delaunay" | recorded |  |
| n_instances | 20 | recorded |  |
| n_sub_repeats | 5 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| n_warm | 19 | recorded |  |
| nnz_per_source_column | 6.236 | recorded |  |
| numba_gate | 1000000000.0 | recorded |  |
| numba_gate_library_default | 60.0 | recorded |  |
| numba_num_threads_env | "1" | recorded |  |
| numba_parallel_kernel | false | recorded |  |
| numba_thread_pool | 1 | recorded |  |
| numba_threads_headline | 1 | recorded |  |
| numba_version | "0.65.1" | recorded |  |
| operator_M | 4900 | recorded |  |
| operator_extent_shape | [70, 70] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.1 | recorded |  |
| pixels_solved | 1500 | recorded |  |
| positive_only_solver | "fnnls" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [256, 256] | recorded |  |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| solve_subset_edge_zeroed | false | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| thread_env | {"AUTOLENS_PROFILING_LEVER_NUMBA_THREADS": "1", "NUMBA_NUM_THREADS_before": "1", "n_threads": 1, "note": "Set by _production_config.pin_thread_env before numpy import; mirrors the production SLURM submit scripts.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": "1", "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| visibilities | 190 | recorded |  |
| would_route_numba_at_default_gate | true | recorded |  |
| xp | "numpy" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>23 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0b33aaac107a9a974326` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.007613192735190846} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/0be2218ae37ea2a89cde` | steps.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.02036335136998739} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/0e5fd8ee17dcf8ada51c` | steps.Curvature F + H: F by numba direct_conv | {"steps.Curvature F + H: F by numba direct_conv": 0.0848198537822004} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Curvature F + H: F by numba direct_conv` |
| `measurement/1297e0c3f58c851d4439` | steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.001371313992422074} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/17ec6099e722aa5e6fa2` | steps_median.Curvature F + H: F by numba direct_conv | {"steps_median.Curvature F + H: F by numba direct_conv": 0.08235961399623193} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Curvature F + H: F by numba direct_conv` |
| `measurement/20cadb484b3cfbab91d6` | steps_median.Regularization matrix H | {"steps_median.Regularization matrix H": 0.007123294984921813} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Regularization matrix H` |
| `measurement/343e034af2b726b92660` | steps_median.Log-det (F + H) (fnnls Cholesky reuse) | {"steps_median.Log-det (F + H) (fnnls Cholesky reuse)": 0.0004587740113493055} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/3949b476cadc5909d8bf` | steps.Log-det (F + H) (fnnls Cholesky reuse) | {"steps.Log-det (F + H) (fnnls Cholesky reuse)": 0.0007314211581711118} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/442628440d80d6df3367` | steps_median.Mapping matrix L | {"steps_median.Mapping matrix L": 0.004552556027192622} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Mapping matrix L` |
| `measurement/4f2b016dc34ab7a29790` | steps.Regularization term sᵀHs | {"steps.Regularization term s\u1d40Hs": 0.0009812598980620112} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Regularization term sᵀHs` |
| `measurement/5b87083488477437d5b2` | steps.Reconstruction: D = Lᵀ d~ + fnnls | {"steps.Reconstruction: D = L\u1d40 d~ + fnnls": 0.1494350759971176} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/62c39c68db59451bb360` | steps.Noise normalisation + evidence assembly | {"steps.Noise normalisation + evidence assembly": 4.477779025604066e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Noise normalisation + evidence assembly` |
| `measurement/6a3ffb041e91520ea401` | steps_median.Regularization term sᵀHs | {"steps_median.Regularization term s\u1d40Hs": 0.0008860790112521499} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Regularization term sᵀHs` |
| `measurement/6b9680b44b093e006d1d` | steps_median.Reconstruction: D = Lᵀ d~ + fnnls | {"steps_median.Reconstruction: D = L\u1d40 d~ + fnnls": 0.14153448698925786} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/9108bce475c8d3efc68f` | steps_median.Log-det H | {"steps_median.Log-det H": 0.00715686398325488} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Log-det H` |
| `measurement/92b9f3ea23bbb6354ff3` | steps_median.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps_median.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.001000488002318889} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/96a6dbb6de7020259070` | steps.Log-det H | {"steps.Log-det H": 0.007411900785212454} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Log-det H` |
| `measurement/b10f09f9766ff11bf09e` | component_total | {"component_total": 0.27887675409376816} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/ba91c777201e3d907099` | steps_median.Noise normalisation + evidence assembly | {"steps_median.Noise normalisation + evidence assembly": 4.248999175615609e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Noise normalisation + evidence assembly` |
| `measurement/c9ee2375040e10bc2a0d` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.004986170420459912} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/d089eb8631ebb5929d78` | steps.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.0010813559469840441} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/e65fc328f6c78b70154f` | steps.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.001408394210126349} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/ec2e8a2b7af5e6aa8468` | steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.02018677900196053} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps_median/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |

</details>

### compile

<details><summary>1 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/c7a02496e8be4ce352fb` | operator_setup | {"operator_setup": 1.9308421520108823} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/operator_build_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/827d3f62f31af266a1e7` | host_peak_rss | {"host_peak_rss": 844.5546875} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/24180587e99a98b00e51` | full_call.mean_s | {"full_call.mean_s": 0.2777473999502897} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/full_call/mean_s` |
| `measurement/d71d6e71662ffed364a9` | full_call.max_s | {"full_call.max_s": 0.3446794850169681} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/full_call/max_s` |
| `measurement/f768410cbe6e1dcfa5c8` | full_call.median_s | {"full_call.median_s": 0.2661237760039512} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/full_call/median_s` |
| `measurement/fa53842c1fd6035e682b` | full_call.min_s | {"full_call.min_s": 0.25717922201147303} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/numba_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/full_call/min_s` |

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
