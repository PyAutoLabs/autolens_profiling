<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/alma_high/9a8a8ebdd77498ef7153`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | {"cache_existed": true, "mask_radius_arcsec": 5.0, "md5_after": "c3d20edcef4343c530ce40a57c45237d", "md5_before": "c3d20edcef4343c530ce40a57c45237d", "regenerated": false} | recorded |  |
| arms | ["numba", "numpy_fft"] | recorded |  |
| cpu_count | 124 | recorded |  |
| curvature_blocks_fft | 12 | recorded |  |
| headline_arm | "numba" | recorded |  |
| host_load_avg_end | [2.71, 5.49, 10.96] | recorded |  |
| host_load_avg_start | [65.19, 70.49, 74.58] | recorded |  |
| hostname | "euclid-ral-gpu-1" | recorded |  |
| image_pixels_masked | 125676 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance_stream | {"arm_order": "ABBA (alternates per instance)", "kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| inversion_class | "InversionInterferometerSparseNumba" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 5.0 | recorded |  |
| mask_radius_preset_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | "RectangularBilinearAdaptImage" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| n_instances | 20 | recorded |  |
| n_sub_repeats | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| n_warm | 19 | recorded |  |
| nnz_per_source_column | 330.50887573964496 | recorded |  |
| numba_gate | 1000000000.0 | recorded |  |
| numba_gate_library_default | 60.0 | recorded |  |
| numba_num_threads_env | "1" | recorded |  |
| numba_parallel_kernel | false | recorded |  |
| numba_thread_pool | 1 | recorded |  |
| numba_threads_headline | 1 | recorded |  |
| numba_version | "0.65.1" | recorded |  |
| operator_M | 160000 | recorded |  |
| operator_extent_shape | [400, 400] | recorded |  |
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

[Original setup artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>23 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/263f01e05f5c87f1ceda` | steps_median.Log-det (F + H) (fnnls Cholesky reuse) | {"steps_median.Log-det (F + H) (fnnls Cholesky reuse)": 0.049538511026185006} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/2aca538112812a33aef3` | steps.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.10440332452325445} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/304fb1e92943270af94b` | steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps_median.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.026762346969917417} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/30a200a0bee20180ef52` | steps_median.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps_median.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.0009364010184071958} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |
| `measurement/324e10f474fe026d2a79` | component_total | {"component_total": 77.66100187811351} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/total_step_by_step` |
| `measurement/3d3a92adb0463df181a1` | steps.Log-det H | {"steps.Log-det H": 0.006303907050383522} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Log-det H` |
| `measurement/4c6bd309bf6824c9e5ea` | steps_median.Curvature F + H: F by numba direct_conv | {"steps_median.Curvature F + H: F by numba direct_conv": 74.66901423997479} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Curvature F + H: F by numba direct_conv` |
| `measurement/57fb74f63bebb950d946` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.22775086457841098} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Mapping matrix L` |
| `measurement/5e88ca02dd57643e0447` | steps.Reconstruction: D = Lᵀ d~ + fnnls | {"steps.Reconstruction: D = L\u1d40 d~ + fnnls": 0.7228173606318274} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/5f12975715a0d2f11dfe` | steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate) | {"steps_median.Inversion build: ray-trace + mesh + mapper (+ routing gate)": 0.10539262101519853} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Inversion build: ray-trace + mesh + mapper (+ routing gate)` |
| `measurement/60c688df25a4823364c6` | steps.Noise normalisation + evidence assembly | {"steps.Noise normalisation + evidence assembly": 2.3537480860556426e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Noise normalisation + evidence assembly` |
| `measurement/67b80f9ecbfdb40334ec` | steps_median.Mapping matrix L | {"steps_median.Mapping matrix L": 0.22232217999408022} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Mapping matrix L` |
| `measurement/6bc993f7255c68a22575` | steps.Regularization term sᵀHs | {"steps.Regularization term s\u1d40Hs": 0.0009352292124132969} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Regularization term sᵀHs` |
| `measurement/7016990922a9d5b0a6ad` | steps.F marshalling: kernel_index_arrays (CSR / CSC / extent) | {"steps.F marshalling: kernel_index_arrays (CSR / CSC / extent)": 0.02760907478851119} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/F marshalling: kernel_index_arrays (CSR ~1 CSC ~1 extent)` |
| `measurement/7c8f5294ee2188ddd51c` | steps_median.Regularization matrix H | {"steps_median.Regularization matrix H": 0.007093433989211917} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Regularization matrix H` |
| `measurement/87ca56494aeeb8892904` | steps.Curvature F + H: F by numba direct_conv | {"steps.Curvature F + H: F by numba direct_conv": 76.51282737941763} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Curvature F + H: F by numba direct_conv` |
| `measurement/9b890fdaa4388464f8dc` | steps_median.Log-det H | {"steps_median.Log-det H": 0.0062078419723547995} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Log-det H` |
| `measurement/a4457d1fb53b56e31f09` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.007232678055420126} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Regularization matrix H` |
| `measurement/c749bb210cd12018e8a3` | steps_median.Reconstruction: D = Lᵀ d~ + fnnls | {"steps_median.Reconstruction: D = L\u1d40 d~ + fnnls": 0.7052882579737343} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Reconstruction: D = Lᵀ d~0 + fnnls` |
| `measurement/e1ea4fe8b60e9cf55b76` | steps_median.Noise normalisation + evidence assembly | {"steps_median.Noise normalisation + evidence assembly": 2.6229012291878462e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Noise normalisation + evidence assembly` |
| `measurement/e85b278b806b107d7811` | steps.Log-det (F + H) (fnnls Cholesky reuse) | {"steps.Log-det (F + H) (fnnls Cholesky reuse)": 0.05015138500235289} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Log-det (F + H) (fnnls Cholesky reuse)` |
| `measurement/ea6ab5e7781a6d9a53c2` | steps_median.Regularization term sᵀHs | {"steps_median.Regularization term s\u1d40Hs": 0.0009142509661614895} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps_median/Regularization term sᵀHs` |
| `measurement/fe31da5359837b3a7065` | steps.Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d) | {"steps.Fast chi-squared (s\u1d40Fs - 2s\u1d40D + d\u1d40N\u207b\u00b9d)": 0.0009471373724457072} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/steps/Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d)` |

</details>

### compile

<details><summary>1 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/b6d000851b424ab5246f` | operator_setup | {"operator_setup": 16.720286474970635} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/operator_build_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/5978caa3419652cca567` | host_peak_rss | {"host_peak_rss": 8819.01953125} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/34be77b4b3fa795ea2dd` | full_call.min_s | {"full_call.min_s": 74.16008806799073} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/full_call/min_s` |
| `measurement/c831439d47e0e8697161` | full_call.mean_s | {"full_call.mean_s": 77.51281580257867} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/full_call/mean_s` |
| `measurement/d0b43bdf2024c48bdd82` | full_call.median_s | {"full_call.median_s": 75.32136200997047} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/full_call/median_s` |
| `measurement/fb1948194eaac12e2794` | full_call.max_s | {"full_call.max_s": 85.70186311099678} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json) `/full_call/max_s` |

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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-30T20:44:53Z",
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
      "hostname": "euclid-ral-gpu-1",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-30T20:44:53Z",
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
      "hostname": "euclid-ral-gpu-1",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-30T20:44:53Z",
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
      "hostname": "euclid-ral-gpu-1",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-30T20:44:53Z",
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
      "hostname": "euclid-ral-gpu-1",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-30T20:44:53Z",
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
      "hostname": "euclid-ral-gpu-1",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-30T20:44:53Z",
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
      "hostname": "euclid-ral-gpu-1",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-30T20:44:53Z",
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

- [likelihood_breakdown](../../../../scripts/interferometer/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/rectangular/likelihood_runtime.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
