<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · sma

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/sma/86ea908cc7f490df7041`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | {"cache_existed": true, "mask_radius_arcsec": 2.0, "md5_after": "a978d8b16bad31379a7810a35bfd71b1", "md5_before": "a978d8b16bad31379a7810a35bfd71b1", "regenerated": false} | recorded |  |
| arms | ["numba", "numpy_fft"] | recorded |  |
| cpu_count | 124 | recorded |  |
| curvature_blocks_fft | 12 | recorded |  |
| delaunay_vertices | 1500 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| headline_arm | "numba" | recorded |  |
| hilbert_pixels | 1500 | recorded |  |
| host_load_avg_end | [74.09, 70.07, 68.5] | recorded |  |
| host_load_avg_start | [72.9, 69.14, 68.13] | recorded |  |
| hostname | "euclid-ral-gpu-2" | recorded |  |
| image_mesh | "Hilbert(weight_power=1.0, weight_floor=0.0)" | recorded |  |
| image_pixels_masked | 1264 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance_stream | {"arm_order": "ABBA (alternates per instance)", "kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| inversion_class | "InversionInterferometerSparseNumba" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 2.0 | recorded |  |
| mask_radius_preset_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | "Delaunay" | recorded |  |
| n_instances | 20 | recorded |  |
| n_sub_repeats | 5 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| n_warm | 19 | recorded |  |
| nnz_per_source_column | 1.8346666666666667 | recorded |  |
| numba_gate | 1000000000.0 | recorded |  |
| numba_gate_library_default | 60.0 | recorded |  |
| numba_num_threads_env | "1" | recorded |  |
| numba_parallel_kernel | false | recorded |  |
| numba_thread_pool | 1 | recorded |  |
| numba_threads_headline | 1 | recorded |  |
| numba_version | "0.65.1" | recorded |  |
| operator_M | 1600 | recorded |  |
| operator_extent_shape | [40, 40] | recorded |  |
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

[Original setup artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>23 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/06c2be9c62250721dba7` | steps_median.Log-det (F + H) (fnnls Cholesky reuse) | {"steps_median.Log-det (F + H) (fnnls Cholesky reuse)": 0.0004679789999499917} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/13709a94f1b30d5b6228` | steps.Log-det (F + H) (fnnls Cholesky reuse) | {"steps.Log-det (F + H) (fnnls Cholesky reuse)": 0.00047905590250401904} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/139f892abc7b13c5d92f` | steps.Curvature F + H: F by numba direct_conv | {"steps.Curvature F + H: F by numba direct_conv": 0.026033332525433873} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Curvature F + H: F by numba direct_conv` |
| `measurement/1e0984e3edda4582884a` | steps_median.Noise normalisation + evidence assembly | {"steps_median.Noise normalisation + evidence assembly": 1.1499971151351929e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Noise normalisation + evidence assembly` |
| `measurement/24070484222104fa4473` | steps_median.Regularization matrix H | {"steps_median.Regularization matrix H": 0.008022836991585791} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Regularization matrix H` |
| `measurement/26ae0ff66ae6f3074fc4` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.00810436110048996} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Regularization matrix H` |
| `measurement/3542704410566b82584c` | steps.Log-det H | {"steps.Log-det H": 0.008012652008083501} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Log-det H` |
| `measurement/49776f9bcc24e7ad3570` | steps.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.022097607948939856} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/50df107110ab5a650cf2` | steps.Reconstruction: D = Lᵀ d~ + fnnls | {"steps.Reconstruction: D = L\u1d40 d~ + fnnls": 0.16451681979110858} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/63cd6f679b7b3cd168bb` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.0019968810956925154} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Mapping matrix L` |
| `measurement/646e39226227c0360889` | steps_median.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps_median.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.0011307660024613142} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/6880ea89b1ef5d668041` | steps_median.Log-det H | {"steps_median.Log-det H": 0.007982637034729123} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Log-det H` |
| `measurement/783f58ab5b5894063316` | steps.Noise normalisation + evidence assembly | {"steps.Noise normalisation + evidence assembly": 1.1559370537533572e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Noise normalisation + evidence assembly` |
| `measurement/887bcc2723a6b47c692b` | steps.Regularization term sᵀHs | {"steps.Regularization term s\u1d40Hs": 0.001132254749168887} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Regularization term sᵀHs` |
| `measurement/94ab845d58ccee230335` | steps_median.Reconstruction: D = Lᵀ d~ + fnnls | {"steps_median.Reconstruction: D = L\u1d40 d~ + fnnls": 0.16519176797010005} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/a6aabfccd582c4b5da3c` | steps_median.Mapping matrix L | {"steps_median.Mapping matrix L": 0.0018724449910223484} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Mapping matrix L` |
| `measurement/acc4e83b1887774c9a45` | steps.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.0010671295244001637} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/c3bfa8956ebbd8acebfc` | steps_median.Regularization term sᵀHs | {"steps_median.Regularization term s\u1d40Hs": 0.0010459780460223556} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Regularization term sᵀHs` |
| `measurement/d284e2c3c3c6f1e69cbf` | steps_median.Curvature F + H: F by numba direct_conv | {"steps_median.Curvature F + H: F by numba direct_conv": 0.026193894969765097} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Curvature F + H: F by numba direct_conv` |
| `measurement/d37c89aa8591c212286d` | steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.0010782770114019513} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/d403e6371e01adc117db` | steps.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.0011748177409907313} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/f39e10513ed682df75f2` | steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.02176139800576493} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/steps_median/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/f7e78fa016c94da9371b` | component_total | {"component_total": 0.2346264717573496} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/total_step_by_step` |

</details>

### compile

<details><summary>1 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/a4e377de43dceee07d35` | operator_setup | {"operator_setup": 3.413633315998595} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/operator_build_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/111c4439b2d7d3e021d8` | host_peak_rss | {"host_peak_rss": 841.34375} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/578ce23bf993f46898fb` | full_call.min_s | {"full_call.min_s": 0.22736833000089973} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/min_s` |
| `measurement/5fe10f222cbb9775d983` | full_call.median_s | {"full_call.median_s": 0.24693199497414753} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/median_s` |
| `measurement/7f7b83e5fa5e6dca6f5e` | full_call.mean_s | {"full_call.mean_s": 0.2449406423226097} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/mean_s` |
| `measurement/fee701d78f99c5ed12ce` | full_call.max_s | {"full_call.max_s": 0.2632517690071836} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/delaunay_numba_hpc_ral_cpu_fp64_r2.0.json) `/full_call/max_s` |

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
      "hostname": "euclid-ral-gpu-2",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_revisions": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_revisions": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_revisions": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "efd13c4cdfb7ab0d4a37549468998e06e31f27d7",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_revisions": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "autolens_profiling.library_revisions": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-2",
    "measured_at": "2026-09-30T18:29:53Z",
    "qualified": false,
    "unknowns": {}
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
      "hostname": "euclid-ral-gpu-2",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_revisions": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_revisions": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_revisions": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "efd13c4cdfb7ab0d4a37549468998e06e31f27d7",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_revisions": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "autolens_profiling.library_revisions": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-2",
    "measured_at": "2026-09-30T18:29:53Z",
    "qualified": false,
    "unknowns": {}
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
      "hostname": "euclid-ral-gpu-2",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_revisions": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_revisions": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_revisions": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "efd13c4cdfb7ab0d4a37549468998e06e31f27d7",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_revisions": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "autolens_profiling.library_revisions": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-2",
    "measured_at": "2026-09-30T18:29:53Z",
    "qualified": false,
    "unknowns": {}
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
      "hostname": "euclid-ral-gpu-2",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_revisions": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_revisions": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_revisions": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "efd13c4cdfb7ab0d4a37549468998e06e31f27d7",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_revisions": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "autolens_profiling.library_revisions": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-2",
    "measured_at": "2026-09-30T18:29:53Z",
    "qualified": false,
    "unknowns": {}
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
      "hostname": "euclid-ral-gpu-2",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_revisions": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_revisions": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_revisions": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "efd13c4cdfb7ab0d4a37549468998e06e31f27d7",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_revisions": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "autolens_profiling.library_revisions": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-2",
    "measured_at": "2026-09-30T18:29:53Z",
    "qualified": false,
    "unknowns": {}
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
      "hostname": "euclid-ral-gpu-2",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_revisions": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_revisions": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_revisions": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "efd13c4cdfb7ab0d4a37549468998e06e31f27d7",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_revisions": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "autolens_profiling.library_revisions": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-2",
    "measured_at": "2026-09-30T18:29:53Z",
    "qualified": false,
    "unknowns": {}
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
      "hostname": "euclid-ral-gpu-2",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_revisions": "7a89e19a09760a0daf22f40e8b111b5388767f73",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_revisions": "b13169e2ced2c44ed2b4a43f3a6ef1c92df989d4",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_revisions": "4c834cedee64a6fe530c7ac66d77de21cca346e8",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "efd13c4cdfb7ab0d4a37549468998e06e31f27d7",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_revisions": "1ec1c829acec0ce85a7eddf791eb66574d507884",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "autolens_profiling.library_revisions": "e9a7ade6f7ff07b4a8e9e2009d1e0ef645404307",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-2",
    "measured_at": "2026-09-30T18:29:53Z",
    "qualified": false,
    "unknowns": {}
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
