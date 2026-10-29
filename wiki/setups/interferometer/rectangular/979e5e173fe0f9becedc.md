<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/alma_high/277111109edac15e0d3e`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 12 | recorded |  |
| host_load_avg_end | [8.99, 4.84, 2.01] | recorded |  |
| host_load_avg_start | [0.64, 0.13, 0.12] | recorded |  |
| image_pixels_masked | 61572 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh | "RectangularBilinearAdaptImage" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 3 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 78400 | recorded |  |
| operator_extent_shape | [280, 280] | recorded |  |
| operator_fft_shape | [560, 560] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.025 | recorded |  |
| pixels_solved | 1369 | recorded |  |
| positive_only_solver_requested | "pdip" | recorded |  |
| positive_only_solver_used | "pdip" | recorded |  |
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
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 1000000 | recorded |  |
| transformer_chunk_size_preset | 1000000 | recorded |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 5000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>27 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/004622b00b539270ccc0` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.17870214866707101} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/0dc8fd1ed568e209e0a7` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.1815576116593244} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/223f7e980638d8f2057c` | steps.Curvature matrix F = Aᵀ W~ A (12 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (12 blocks of 128)": 6.971792447002372} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (12 blocks of 128)` |
| `measurement/22be461a38a2628ed748` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.006122890665816764} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/29eb3ccb27ecb281256e` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.03826270832602555} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/40fd5d5f6e09bb4402c6` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.0700793266587425} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/4cd7460e0bbfafb6d8fd` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0019889263300380358} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/4f1fab8f761e58e98604` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.47141347199794836} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/5bbee0880d702e216690` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.05127034399386806} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/5ec0aef679a9092f279c` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.47003064533540356} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/796b0e2e849caef2c1c0` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.040201142333292715} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/7c5a0eee76929e506b3b` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.47003064533540356} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/7d7c92bea91f82cdec9c` | component_total | {"component_total": 7.907522288985395} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/7edb4c744b657e2005d4` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 7.600348007671225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/99f252cd699bbd8431bc` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 6.971792447002372} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/9d640ae80eb6f97498f2` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.0005369233258534223} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/a9c23582e80613430924` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.04928141766383003} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/b12a419acadfc8265fef` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.12743180467320295} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/b5aec13480db966a1a81` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.0005369233258534223} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/bfd5fcc10f2a25a859a2` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.17332419267040677} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/c19b7409c4115da05449` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.21776619433270147} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/c3d2aa4cf713c7436e08` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.0700793266587425} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/ca8855072cd1d65b0c1f` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.040201142333292715} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/e2e77da97e2c37d8901b` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.1815576116593244} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/e4812772dc02c70e9ba7` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 7.538792986335466} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/f8dd7803fa3032caef4b` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.624859984673094} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/fe7c1ea4419a28e2ebdd` | steps.Ray-trace grids (data) | {"steps.Ray-trace grids (data)": 0.0019889263300380358} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace grids (data)` |

</details>

### compile

<details><summary>52 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/05f78aeb63644a663939` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.038887418020749465} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/068d6f57bcc1ff528c08` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.054994295991491526} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/0d7796ed4b0ffd115985` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.05301760000293143} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/1481d01b38c2600d5149` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.005282827012706548} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/1d037c4568de0c1df672` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.6459115809993818} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/1f92ea77b57523405ec6` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.012059127009706572} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/2112426758a1b97b9721` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.00600567800574936} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/215cf3e9b02421c9a75a` | log_dets.first_call_s | {"log_dets.first_call_s": 0.07013829599600285} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/21e03bbcbdeaf13ee49d` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0008212709799408913} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/2ab35ae50d241540f80c` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.04515471198828891} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/2c66ab07d98e73cf4484` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.18965393499820493} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/31c6472bcc839f912345` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 2.3383103809901513} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/31db389c76bc91c707be` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.013622568018035963} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/3390ec9f605519806b76` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.12380195801961236} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/36ad228417797121443e` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.43346969698905014} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/38102ce4898908b0384f` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.06501284800469875} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/4264c8e7c2a40aab1a99` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 0.9220636590034701} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/4dbb1cbf9b523c298088` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0023671720118727535} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/4dd2dda3e98a00f55617` | full_pipeline.compile_s | {"full_pipeline.compile_s": 2.3020417719963007} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/587c832b877d103ca3c4` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.14885515999048948} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/6867072e75f786a53710` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.2159912520146463} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/6a2b11577005bdf3390c` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.0481562580098398} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/6be7ca538d97cefef33b` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.1029578759917058} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/6c8259fb2f01edcf7bed` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 7.8274229539965745} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/6f32cc059ee081ebcb9f` | operator_setup | {"operator_setup": 25.337876725010574} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/operator_build_s` |
| `measurement/741db5a2be310589f0b9` | log_dets.lower_s | {"log_dets.lower_s": 0.02344417999847792} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/7939629448b40c015aff` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 7.822485743003199} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/7ad17e1621ee886d2560` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.2822897739824839} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/7b5f73d654d50a3d95ee` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.01454430801095441} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/7e578d8a2641f37f68f7` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.6381392840121407} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/7f6c9deec93ccfa38ead` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.06092696500127204} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/83595ba6c6ed8c313a74` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.6920853019983042} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/85992834f44872cf4a30` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.0592852359986864} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/8a49fa49f4fb557e6890` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.012040596979204565} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/9192f13865f0342e4f28` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04487782600335777} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/93cfa9664e69bbb65439` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.012131396011682227} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/9cb5353c803c7bf7053b` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.0398202090000268} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/9da93e078b254cfa24e3` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.0870659250067547} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/9dc176e0ee12b936415e` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.43799297299119644} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/a6ac90b11385f49e45d3` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 0.96101255499525} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/aea7a131b7cbdf78aa43` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.6531579049769789} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/bd48b3e9ed8ed7f06927` | log_dets.compile_s | {"log_dets.compile_s": 0.0866985390021} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/c899562c3a9baaa9aae1` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 0.9903420160117093} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/d4c0184b9908579b9de1` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 6.911511333979433} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/d7b9816a96b9dd132ffd` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.4678557690058369} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/dceca8e5d834e79bd6db` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.4721259069920052} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/de985dcb388e8a7fe9a2` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.8466409149987157} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/e302e74246c627d36b5f` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.1492944760248065} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/f1f6bf02cb55d9f9ceab` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.03161412398912944} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/f2878c9dffbec1d0feaa` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.18965745501918718} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/f5c3387652e10e5a7765` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.10935635000350885} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/ff1d90ec4daefd3f8df6` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.5557439230033197} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/d86ef60042d05f99637d` | host_peak_rss | {"host_peak_rss": 8920.890625} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/2d11b3ed1e2c3f4c3bae` | single_jit_block | {"single_jit_block": 7.600348007671225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/full_pipeline_single_jit` |
| `measurement/b396480e7c791191aa37` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 7.600348007671225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/breakdown.json) `/full_pipeline/per_call_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 204,
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
      "autotune_cache_entries_at_start": 204,
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

- [likelihood_breakdown](../../../../scripts/interferometer/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/rectangular/likelihood_runtime.py)
- [likelihood_runtime_matrixfree](../../../../scripts/interferometer/rectangular/likelihood_runtime_matrixfree.py)
- [likelihood_runtime_numba](../../../../scripts/interferometer/rectangular/likelihood_runtime_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
