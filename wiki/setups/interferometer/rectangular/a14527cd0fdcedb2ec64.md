<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/alma_high/b0640efbd3c289d79f86`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 12 | recorded |  |
| host_load_avg_end | [4.59, 2.35, 0.92] | recorded |  |
| host_load_avg_start | [1.95, 0.5, 0.17] | recorded |  |
| image_pixels_masked | 125676 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 5.0 | recorded |  |
| mesh | "RectangularBilinearAdaptImage" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 160000 | recorded |  |
| operator_extent_shape | [400, 400] | recorded |  |
| operator_fft_shape | [800, 800] | recorded |  |
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
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 1000000 | recorded |  |
| transformer_chunk_size_preset | 1000000 | recorded |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 5000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>28 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0184454843a9ea448091` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.0001680549990851432} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/136dd42f1b22f67b866e` | steps.Ray-trace grids (data) | {"steps.Ray-trace grids (data)": 0.00033315400360152126} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Ray-trace grids (data)` |
| `measurement/19728f574e6e598cc9ed` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.0004123314982280135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/1b901d8ea8539b44b8ee` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.0001680549990851432} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/2b5f7eb71b5b96eb0799` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.0011688859027344733} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/2e34660b927853ce1bac` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.0011688859027344733} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/38c077ad9c25902fd8cb` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.1985539660032373} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/45947a38bf9ee7d4d393` | full_pipeline_vmap16.steady_per_call_s | {"full_pipeline_vmap16.steady_per_call_s": 0.18731248345648055} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline_vmap16/steady_per_call_s` |
| `measurement/4a8920f524b25479a03b` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.002202098403358832} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/4c7a61b0eaa38c981dfb` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 0.17114060019957833} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/4cfafb3f52f126e65167` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.0035969548975117505} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/67efbfe4a3b590804d9d` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.0004123314982280135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/69d28b28060cce0c9a3c` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.0016405388945713638} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/79adadbd3232d61bf5b7` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.001649745902977884} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/7e32ed3f080ebb8f385b` | steps.Curvature matrix F = Aᵀ W~ A (12 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (12 blocks of 128)": 0.17114060019957833} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (12 blocks of 128)` |
| `measurement/826c388f7b726a9e3315` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.00033932700171135366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/8941c90660c7c96e4338` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.001973692898172885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/ae838a0781f85a66b446` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 0.19819412909564563} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/bfc85ca06adb4256d760` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.02345715289702639} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/c2a6b1060b4efbb78109` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.0035500741039868446} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/d25ebfc31af5ad9ceb9f` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.003623438801150769} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/d2f2065e9584c0dc17f7` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.002202098403358832} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/d90f9b93c8f5b89b26db` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.023446284100646152} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/d9e60bc2efeac3687369` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.011479554604738951} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/de200d5992b6912d0b83` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.010098786000162363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/e58ccfb335c40d7c3a8a` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.02345715289702639} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/eb16a8222e44cd0eb8e7` | component_total | {"component_total": 0.20209919800399803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/total_step_by_step` |
| `measurement/f1a146200c8832135834` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.00033315400360152126} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |

</details>

### compile

<details><summary>53 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00a39556ff750843d56a` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 0.175617731991224} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/066235167ad86161bafa` | full_pipeline.compile_s | {"full_pipeline.compile_s": 7.97230673301965} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/0a8728d389585b100741` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.8121723040239885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/1f39d5d4f5060e1b61f6` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.012719651975203305} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/2f2df408ae84bf9f9211` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 7.936866480973549} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/3047d97d91ab0f773761` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.2564558850135654} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/313d4c255957225cda0e` | log_dets.first_call_s | {"log_dets.first_call_s": 0.003027161001227796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/317f9f0c6fc4fc814445` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0009785040165297687} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/368ed7f8aa23219bd70d` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.7720758889918216} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/37f661723432a18cd737` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 3.0071541279903613} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/3bbe0919cbbe592cdd5d` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.47523722099140286} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/3dba94d5643c6369cdf8` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04939110699342564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/3ddc71e4dbfd29fc6556` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 0.23605070100165904} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/3fa81dc21d807ae47dff` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.06294694397365674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/41d29b34af485e3876fb` | log_dets.lower_s | {"log_dets.lower_s": 0.02732070197816938} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/48aaaf837239a7a5ad8a` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.0009018850396387279} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/49e427c999f2fcba2bb0` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.03441952803405002} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/4fe06c9cff75bda400d0` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.016367728996556252} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/5ca8afecc16440ab8fd6` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.25379755202447996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/5f423d18ec3f022d8304` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.7733094930299558} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/6203612e15c28c094963` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.0013714919914491475} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/62c3967610c22555cbb2` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.09074238297762349} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/630b50123acf7da78e44` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.7210675120004453} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/67dcf1781d7e235e23e6` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.07382032700115815} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/6befc7ee91993be70dd1` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.01444865099620074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/6d92c88e97d1837ca4dd` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.015717804024461657} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/6e4e27bfc33be857673f` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.049708804988768} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/6e5c16bcb97cd0911a5e` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.14635918097337708} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/6fae2aa487d71b08c9fb` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.009200682980008423} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/72ad2950d88f3c3e835a` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.01118283101823181} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/78fae039b65b422dba3a` | log_dets.compile_s | {"log_dets.compile_s": 0.1574120440054685} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/7a5d92e852349e75a454` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.025229315971955657} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/7f2f6be48d2b08a63896` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0025757740368135273} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/967314a5ff7796158431` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.015372735972050577} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/9e166f6e7a8d28bb68e7` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 2.831153856997844} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/9e4475d7863d702f941a` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.02545346401166171} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/aa1311bfed98e52ba558` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.011205721995793283} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/abad0ef538fb9b523d60` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.1226216179784387} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/ace9f92449bed0333cf0` | operator_setup | {"operator_setup": 3.111069659993518} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/operator_build_s` |
| `measurement/b23c92c01117312a0680` | full_pipeline_vmap16.first_call_s | {"full_pipeline_vmap16.first_call_s": 12.72563906799769} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline_vmap16/first_call_s` |
| `measurement/b77699d3648ba92cdeea` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.11325849499553442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/b7d81a4d84f6c9e9c13f` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.0016972990124486387} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/c2cf3288b277f76f0781` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.0040908349910750985} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/c31878215b393d758ca1` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.08327352802734822} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/c68351d25214f5b8bbf6` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.05669566203141585} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/cb18df3fb702e3fe3611` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.5183810580056161} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/e740c2d68189bd963e42` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.006875599035993218} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/ec900cbb1eca1b9a4ee3` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 2.9419860069756396} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/f269097ecac39357dae7` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.2482290559564717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/f3c807bbeca6c8875993` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.49104262504260987} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/f656ffde467fe4d17357` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.04456163599388674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/fcf6ea0e4b88343aacf0` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.06402977701509371} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/ff4ff06c068a6d509c3e` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.011205851973500103} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/c2643ca54d77746904bd` | host_peak_rss | {"host_peak_rss": 4087.03125} | MiB | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/13868fb526bf48283c2e` | single_jit_block | {"single_jit_block": 0.1985539660032373} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/full_pipeline_single_jit` |
| `measurement/224243dcd88d28042cf4` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 0.1985539660032373} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/full_pipeline/per_call_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 204,
      "backend": "gpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-2",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 61373 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "9428eca24a02b528edde68550dbcf82e609a652a",
      "PyAutoArray.library_revisions": "9428eca24a02b528edde68550dbcf82e609a652a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "404b3e5f7760d26f4bb41c7b1535ae5be3d113a0",
      "PyAutoFit.library_revisions": "404b3e5f7760d26f4bb41c7b1535ae5be3d113a0",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "c9609825660f5ee8d6b3254b5ee90971216dd272",
      "PyAutoGalaxy.library_revisions": "c9609825660f5ee8d6b3254b5ee90971216dd272",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "21b520be4d0dd103aaaaba800e42c5092fedeb29",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "PyAutoNerves.library_revisions": "bf104102312852537159b7fad2f998ab6677b4e4",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "fe0d4b5184124c5d076c80deff52e8beef94091b",
      "autolens_profiling.library_revisions": "fe0d4b5184124c5d076c80deff52e8beef94091b",
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
    "measured_at": "2026-09-28T19:31:31Z",
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
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 204,
      "backend": "gpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-2",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 61373 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "9428eca24a02b528edde68550dbcf82e609a652a",
      "PyAutoArray.library_revisions": "9428eca24a02b528edde68550dbcf82e609a652a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "404b3e5f7760d26f4bb41c7b1535ae5be3d113a0",
      "PyAutoFit.library_revisions": "404b3e5f7760d26f4bb41c7b1535ae5be3d113a0",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "c9609825660f5ee8d6b3254b5ee90971216dd272",
      "PyAutoGalaxy.library_revisions": "c9609825660f5ee8d6b3254b5ee90971216dd272",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "21b520be4d0dd103aaaaba800e42c5092fedeb29",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "bf104102312852537159b7fad2f998ab6677b4e4",
      "PyAutoNerves.library_revisions": "bf104102312852537159b7fad2f998ab6677b4e4",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "fe0d4b5184124c5d076c80deff52e8beef94091b",
      "autolens_profiling.library_revisions": "fe0d4b5184124c5d076c80deff52e8beef94091b",
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
    "measured_at": "2026-09-28T19:31:31Z",
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
