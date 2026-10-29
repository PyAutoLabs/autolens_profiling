<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/alma_high/7a2c06e5c176ed0aff20`.

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

[Original setup artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>27 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1fabf3c2ef7bd54f3433` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.04928141766383003} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/222050004289641dc750` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 7.538792986335466} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/234e7bfff92293ee73a3` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0019889263300380358} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/2bd8c7c289ceaac42dfd` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.47003064533540356} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/458ca0aff6a6c6f9fe89` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.1815576116593244} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/523795f3b4d29e614574` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.0005369233258534223} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/56a3903ba90055ff1309` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.47141347199794836} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/5830f052216b4c4feb2b` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.03826270832602555} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/597f9564331c1b5105eb` | steps.Ray-trace grids (data) | {"steps.Ray-trace grids (data)": 0.0019889263300380358} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Ray-trace grids (data)` |
| `measurement/67b8f3975cbc2d020673` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.0700793266587425} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/6ed723a15cc45cb1ac98` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 6.971792447002372} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/76e19d99c41afcbb0530` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.0700793266587425} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/7b66758b5db214a37905` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.17332419267040677} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/8594d31de5188c0b5df5` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.05127034399386806} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/85bf6f71ecf9b7c9e2cb` | component_total | {"component_total": 7.907522288985395} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/total_step_by_step` |
| `measurement/8a2ee599270bfe77287a` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.47003064533540356} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/95f5d37a9bfa837fbd02` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.040201142333292715} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/9a5f3f3de01ccb5f169e` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.006122890665816764} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/9c92f4ce300256b99007` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.0005369233258534223} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/a4b797b23afcb56f9d66` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.12743180467320295} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/a4c42a665db5d114a78c` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.17870214866707101} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/b9355e57a2d76cae98b5` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 7.600348007671225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/be23cfe27e23190e801f` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.21776619433270147} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/c42f4d613d7fd3b7b507` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.040201142333292715} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/d4136b4b88c7c9d033aa` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.1815576116593244} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/d948e26bda41180b775b` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.624859984673094} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/e3450c2fb7901bc8885b` | steps.Curvature matrix F = Aᵀ W~ A (12 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (12 blocks of 128)": 6.971792447002372} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (12 blocks of 128)` |

</details>

### compile

<details><summary>52 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/04246a5f9981dffe7eb9` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.01454430801095441} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/0972f1b5cbec5caca0ed` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.1029578759917058} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/0d3eb9cc47fb74652a69` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 7.822485743003199} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/15ae6217b6007cf62c64` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.2159912520146463} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/1e91ba2563bc487cf083` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.038887418020749465} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/1fb5db5d2940b849b03b` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.0592852359986864} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/2bb82a2b364a49aa4a4d` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 0.9903420160117093} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/2cd7cc1c79f46c9f1db9` | log_dets.compile_s | {"log_dets.compile_s": 0.0866985390021} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/2d1519d8f083aa40e814` | full_pipeline.compile_s | {"full_pipeline.compile_s": 2.3020417719963007} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/326727931abf7cdef950` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.6531579049769789} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/33510a77ffb699e95ebb` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.05301760000293143} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/3a9f1151a170d69432b9` | operator_setup | {"operator_setup": 25.337876725010574} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/operator_build_s` |
| `measurement/40a43c82de2ef3d3512f` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.6920853019983042} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/436d09e46df90518c974` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.10935635000350885} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/47b68243cb60488defa1` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.4678557690058369} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/56532d5c332e94958330` | log_dets.first_call_s | {"log_dets.first_call_s": 0.07013829599600285} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/6ee1d02452fe0bae2250` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.1492944760248065} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/728101e03cab7cc6db52` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.0481562580098398} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/79ed8f325ff7b6ce27d1` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.18965745501918718} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/8612e9c4be97de31d842` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.5557439230033197} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/875d5637b7a04e6b97f5` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 6.911511333979433} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/886e4fe554ca8e75cc14` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04487782600335777} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/893ed408b5b0b20b0cdb` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.6381392840121407} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/894d2b2b7c38c1f27f0e` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 2.3383103809901513} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/897b994dbb759c7c866f` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.005282827012706548} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/8f34d6133966402b8883` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.012040596979204565} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/942655de084f014a50b3` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.18965393499820493} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/9bf78d597320acebdacb` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0023671720118727535} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/9d87dbce326d02560a69` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.013622568018035963} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/9daf00fc80a3a31e7467` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 7.8274229539965745} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/a74c26f4a562365ad29f` | log_dets.lower_s | {"log_dets.lower_s": 0.02344417999847792} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/aea86d53b43105a4b89d` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.8466409149987157} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/b60bbb2ae379358002ab` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 0.96101255499525} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/b7dc681f15936d0c21b2` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0008212709799408913} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/bd72ae739698310c14fa` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.054994295991491526} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/c2b397fa8801c7fd5afc` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.12380195801961236} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/c50ef253c67b81c83d9f` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.03161412398912944} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/c68e46f0949b41b3b7bf` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.06092696500127204} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/c7daf29fc2750ee122b6` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.6459115809993818} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/d21b78042e00c068bfb7` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.2822897739824839} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/d358a4f96d370f44ea39` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.14885515999048948} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/deb58740c0cb65fa060f` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.43346969698905014} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/df601432ca5023de4ea1` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.06501284800469875} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/dfb13d4703045e524c17` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.0870659250067547} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/e546f4dabc37676c0ac1` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.00600567800574936} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/edaf818312ad41966389` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.012131396011682227} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/ef64546804c7abba351c` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.43799297299119644} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/f15b5934a83cf454f794` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.04515471198828891} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/f5203a76e30ce421f916` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.0398202090000268} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/f7d6ac43c4608d21c1b0` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 0.9220636590034701} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/fe65662bc657abba5a9b` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.012059127009706572} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/feaa3123abdf31c5ed8a` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.4721259069920052} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/61baa17f193ca344ff1d` | host_peak_rss | {"host_peak_rss": 8920.890625} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0702954eec1870ed0c35` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 7.600348007671225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/full_pipeline/per_call_s` |
| `measurement/d85f7af86971cc1ccbb8` | single_jit_block | {"single_jit_block": 7.600348007671225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.8.17.1_2026-09-27/breakdown.json) `/full_pipeline_single_jit` |

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
