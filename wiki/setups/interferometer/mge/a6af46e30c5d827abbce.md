<!-- generated: build_setup_wiki.py; do not edit -->
# mge · sma

[Model index](index.md)

Exact setup ID: `interferometer/mge/sma/8675753d737c33d91353`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| chunk_size | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| chunk_size_note | "TransformerNUFFT.transform_mapping_matrix ignores chunk_size: step 3 is one nufft2d2 over every column and every visibility." | recorded |  |
| host_load_avg_end | [1.76, 0.71, 0.47] | recorded |  |
| host_load_avg_start | [0.08, 0.17, 0.28] | recorded |  |
| image_pixels_masked | 3852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerMapping" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| linear_gaussians | 20 | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "raw" | recorded |  |
| nnls_solver | "pdip" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.1 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [256, 256] | recorded |  |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transform | "library" | recorded |  |
| transform_column_batch | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| transform_vis_chunk | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| transformer | "TransformerNUFFT" | name |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 190 | recorded |  |
| vmap_batch | 4 | recorded |  |
| w_tilde_arm | true | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>30 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/03dce28d70ff6ed922fe` | w_tilde_chain_from_mapping_matrix.steady_per_call_s | {"w_tilde_chain_from_mapping_matrix.steady_per_call_s": 0.0024031713997828773} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/steady_per_call_s` |
| `measurement/048347d6513437e41b5a` | w_tilde_reconstruction.steady_per_call_s | {"w_tilde_reconstruction.steady_per_call_s": 0.0023077550002199134} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_reconstruction/steady_per_call_s` |
| `measurement/0a080b5f8052ffad7ae5` | log_likelihood_residual_visibilities.steady_per_call_s | {"log_likelihood_residual_visibilities.steady_per_call_s": 0.00016570199950365349} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/log_likelihood_residual_visibilities/steady_per_call_s` |
| `measurement/0e9bdffc89db5c458b7e` | w_tilde_figure_of_merit.steady_per_call_s | {"w_tilde_figure_of_merit.steady_per_call_s": 0.00013649519969476386} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_figure_of_merit/steady_per_call_s` |
| `measurement/17e7cb7839679428eda4` | steps.Reconstruction (PDIP NNLS) | {"steps.Reconstruction (PDIP NNLS)": 0.002290947100118501} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps/Reconstruction (PDIP NNLS)` |
| `measurement/1832324402c8eb9ff192` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0005365548000554554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/1b71ac73726955af3228` | component_total | {"component_total": 0.8561593482001626} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/total_step_by_step` |
| `measurement/1c6d022cd5757c4f6343` | ray_trace_standalone.steady_per_call_s | {"ray_trace_standalone.steady_per_call_s": 0.0001548799999000039} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/ray_trace_standalone/steady_per_call_s` |
| `measurement/2bb3ae619ed59278ebbf` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.0001348961995972786} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/4169e0ac53b49b49cc18` | dense_chain_from_mapping_matrix.steady_per_call_s | {"dense_chain_from_mapping_matrix.steady_per_call_s": 0.8549736594999559} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/dense_chain_from_mapping_matrix/steady_per_call_s` |
| `measurement/53299bf2a9c675c19061` | full_pipeline_vmap4.steady_per_call_s | {"full_pipeline_vmap4.steady_per_call_s": 0.2313859617250273} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/full_pipeline_vmap4/steady_per_call_s` |
| `measurement/5f40028f9f6334eac282` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0001522149999800604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps/Data vector (D)` |
| `measurement/7ee8ec891bae53005521` | transform_mapping_matrix_standalone.steady_per_call_s | {"transform_mapping_matrix_standalone.steady_per_call_s": 0.8612041606997082} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/transform_mapping_matrix_standalone/steady_per_call_s` |
| `measurement/8033a002a24b76be2574` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.0005365548000554554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps/Ray-trace grids` |
| `measurement/85c8f10f83d56b416134` | w_tilde_curvature.steady_per_call_s | {"w_tilde_curvature.steady_per_call_s": 0.0003484788998321164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_curvature/steady_per_call_s` |
| `measurement/8c8634494730aa88e24f` | reconstruction.steady_per_call_s | {"reconstruction.steady_per_call_s": 0.002290947100118501} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/reconstruction/steady_per_call_s` |
| `measurement/94740190ad3086ed69a1` | setup_prefix_1_vmap4.steady_per_call_s | {"setup_prefix_1_vmap4.steady_per_call_s": 0.00011032932488888036} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_1_vmap4/steady_per_call_s` |
| `measurement/9ef10042af92f80595e8` | steps_vmap_per_call.Full pipeline | {"steps_vmap_per_call.Full pipeline": 0.2313859617250273} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps_vmap_per_call/Full pipeline` |
| `measurement/a1227c97f75414313e72` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.0001348961995972786} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/be9b0a95b76cf3fff85e` | setup_prefix_2_vmap4.steady_per_call_s | {"setup_prefix_2_vmap4.steady_per_call_s": 0.00022998285003268394} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_2_vmap4/steady_per_call_s` |
| `measurement/cc85fc097936a6b2ba5d` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.000722938599938061} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/cf73f4a3b6d05e9022a3` | setup_prefix_3_vmap4.steady_per_call_s | {"setup_prefix_3_vmap4.steady_per_call_s": 0.22493705582492113} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_3_vmap4/steady_per_call_s` |
| `measurement/ddcbae4e27d528cf0596` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.00020726069997181184} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps/Curvature matrix (F)` |
| `measurement/e1487cd43251dbaa23b8` | steps.Mapping matrix (20 Gaussians) | {"steps.Mapping matrix (20 Gaussians)": 0.0001863837998826056} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps/Mapping matrix (20 Gaussians)` |
| `measurement/e263ae0c41fe75e708ea` | curvature_matrix.steady_per_call_s | {"curvature_matrix.steady_per_call_s": 0.00020726069997181184} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/curvature_matrix/steady_per_call_s` |
| `measurement/e2ed81c768c91dd8c360` | data_vector.steady_per_call_s | {"data_vector.steady_per_call_s": 0.0001522149999800604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/data_vector/steady_per_call_s` |
| `measurement/e42895f32ee9827689a9` | w_tilde_data_vector.steady_per_call_s | {"w_tilde_data_vector.steady_per_call_s": 0.0001259121003386099} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_data_vector/steady_per_call_s` |
| `measurement/f5ad27507bbdd36915da` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.853374029200495} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/f79881dcdb318fed31ac` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.8545848549001676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/fd4ec7e22e0603288317` | steps.Transformed mapping matrix (NUFFT) | {"steps.Transformed mapping matrix (NUFFT)": 0.8526510906005569} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/steps/Transformed mapping matrix (NUFFT)` |

</details>

### compile

<details><summary>55 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0211199517a44e6a5394` | reconstruction.compile_s | {"reconstruction.compile_s": 0.363252218994603} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/reconstruction/compile_s` |
| `measurement/08cf6b258bda6b122d17` | w_tilde_curvature.compile_s | {"w_tilde_curvature.compile_s": 0.1486746100054006} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_curvature/compile_s` |
| `measurement/0e80231155f2142ad024` | w_tilde_curvature.first_call_s | {"w_tilde_curvature.first_call_s": 0.02238815400050953} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_curvature/first_call_s` |
| `measurement/1d338836bb6371306921` | w_tilde_reconstruction.first_call_s | {"w_tilde_reconstruction.first_call_s": 0.002656133998243604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_reconstruction/first_call_s` |
| `measurement/1e46a7fe6dc2351aafdc` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.004973889001121279} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/1ecbde967796b3941a2b` | full_pipeline.compile_s | {"full_pipeline.compile_s": 3.3955634080048185} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/251a75f03a60bd5d5b05` | w_tilde_curvature.lower_s | {"w_tilde_curvature.lower_s": 0.09239961999992374} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_curvature/lower_s` |
| `measurement/26239f22399329bac48e` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.012376455000776332} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/2a8d8e9918f16c33795c` | w_tilde_chain_from_mapping_matrix.first_call_s | {"w_tilde_chain_from_mapping_matrix.first_call_s": 0.02464633999625221} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/first_call_s` |
| `measurement/2b3b0daa6a2c5543b7bf` | w_tilde_chain_from_mapping_matrix.lower_s | {"w_tilde_chain_from_mapping_matrix.lower_s": 0.04046356499748072} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/lower_s` |
| `measurement/2d26fafcb3a8ece8955e` | curvature_matrix.first_call_s | {"curvature_matrix.first_call_s": 0.04141892000188818} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/curvature_matrix/first_call_s` |
| `measurement/314c32cda58825d11b2d` | w_tilde_data_vector.first_call_s | {"w_tilde_data_vector.first_call_s": 0.0007709960045758635} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_data_vector/first_call_s` |
| `measurement/3a3fb7e0c77d7eed428d` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.2579321869998239} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/3b0a3a9095c2d986cd68` | log_likelihood_residual_visibilities.lower_s | {"log_likelihood_residual_visibilities.lower_s": 0.010328756994567811} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/log_likelihood_residual_visibilities/lower_s` |
| `measurement/3fc142efa1d7c909b18b` | log_likelihood_residual_visibilities.compile_s | {"log_likelihood_residual_visibilities.compile_s": 0.1036974509988795} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/log_likelihood_residual_visibilities/compile_s` |
| `measurement/4e5d1a1b72fd2ce4cd6e` | curvature_matrix.compile_s | {"curvature_matrix.compile_s": 0.061640077001356985} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/curvature_matrix/compile_s` |
| `measurement/58276411a5bf1818c2b2` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.47625927500484977} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/5b04319081c288fc0d1a` | w_tilde_data_vector.lower_s | {"w_tilde_data_vector.lower_s": 0.00624181299644988} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_data_vector/lower_s` |
| `measurement/5b951ab31ae7d1844b3d` | transform_mapping_matrix_standalone.compile_s | {"transform_mapping_matrix_standalone.compile_s": 0.9135201850003796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/transform_mapping_matrix_standalone/compile_s` |
| `measurement/5f8b771cdfbfcacc5f60` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.08507949500199175} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/62570486468e07d6ef8e` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.0009591740017640404} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/65c4d7900fc619fd4bbf` | curvature_matrix.lower_s | {"curvature_matrix.lower_s": 0.017541193999932148} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/curvature_matrix/lower_s` |
| `measurement/6ad261df57dfc12c69e5` | dense_chain_from_mapping_matrix.lower_s | {"dense_chain_from_mapping_matrix.lower_s": 0.07720434200018644} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/dense_chain_from_mapping_matrix/lower_s` |
| `measurement/76fa5defe2f96c464de3` | log_likelihood_residual_visibilities.first_call_s | {"log_likelihood_residual_visibilities.first_call_s": 0.0009165649971691892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/log_likelihood_residual_visibilities/first_call_s` |
| `measurement/7b96aa9b54b095972646` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 2.999664777998987} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/7d920947de2f624f467f` | full_pipeline_vmap4.first_call_s | {"full_pipeline_vmap4.first_call_s": 5.585983219003538} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/full_pipeline_vmap4/first_call_s` |
| `measurement/7e48e12feba58de5d456` | data_vector.compile_s | {"data_vector.compile_s": 0.05481472800602205} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/data_vector/compile_s` |
| `measurement/7fdf43024668c7a336bf` | setup_prefix_2_vmap4.first_call_s | {"setup_prefix_2_vmap4.first_call_s": 2.5265546729933703} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_2_vmap4/first_call_s` |
| `measurement/812017d0b0b561d00a7c` | data_vector.lower_s | {"data_vector.lower_s": 0.01386440599890193} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/data_vector/lower_s` |
| `measurement/8bc3bc2eda5d28b89847` | dense_chain_from_mapping_matrix.compile_s | {"dense_chain_from_mapping_matrix.compile_s": 1.3340673290003906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/dense_chain_from_mapping_matrix/compile_s` |
| `measurement/9257a0987427e281f467` | w_tilde_reconstruction.lower_s | {"w_tilde_reconstruction.lower_s": 0.0003220490034436807} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_reconstruction/lower_s` |
| `measurement/96cce416a73419d311ce` | w_tilde_figure_of_merit.lower_s | {"w_tilde_figure_of_merit.lower_s": 0.00631832200451754} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_figure_of_merit/lower_s` |
| `measurement/9cab261113d80b0d21a8` | w_tilde_chain_from_mapping_matrix.compile_s | {"w_tilde_chain_from_mapping_matrix.compile_s": 0.5075659650028683} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/compile_s` |
| `measurement/a0e314c3a7c362d8d017` | ray_trace_standalone.first_call_s | {"ray_trace_standalone.first_call_s": 0.0012607619937625714} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/ray_trace_standalone/first_call_s` |
| `measurement/a22d180219a2dd1c20f5` | w_tilde_reconstruction.compile_s | {"w_tilde_reconstruction.compile_s": 9.039998985826969e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_reconstruction/compile_s` |
| `measurement/aae3f78dd793e938b16c` | data_vector.first_call_s | {"data_vector.first_call_s": 0.0008754450027481653} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/data_vector/first_call_s` |
| `measurement/ab6c07a97f415686ff37` | dense_chain_from_mapping_matrix.first_call_s | {"dense_chain_from_mapping_matrix.first_call_s": 0.8935040959986509} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/dense_chain_from_mapping_matrix/first_call_s` |
| `measurement/aee64beb35275cb76048` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 1.90093526399869} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/b7cf0f66c9dce0170989` | setup_prefix_1_vmap4.first_call_s | {"setup_prefix_1_vmap4.first_call_s": 0.573421467001026} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_1_vmap4/first_call_s` |
| `measurement/c861e95dd6c6cc74cb2d` | full_pipeline.lower_s | {"full_pipeline.lower_s": 0.950594990994432} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/c8c76f34f73a560d25b2` | setup_prefix_3_vmap4.first_call_s | {"setup_prefix_3_vmap4.first_call_s": 4.245932127996639} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_3_vmap4/first_call_s` |
| `measurement/cc78aee785510cd0ede6` | w_tilde_data_vector.compile_s | {"w_tilde_data_vector.compile_s": 0.10463022599287797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_data_vector/compile_s` |
| `measurement/ce21840519a6ca701bac` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.878659037000034} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/ce2f14dda01240502356` | ray_trace_standalone.lower_s | {"ray_trace_standalone.lower_s": 0.07203542299976107} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/ray_trace_standalone/lower_s` |
| `measurement/d0cee30396a8dda66a06` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0021109880035510287} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/d472992dfad82e3e869e` | reconstruction.first_call_s | {"reconstruction.first_call_s": 0.013721577000978868} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/reconstruction/first_call_s` |
| `measurement/d828ec9f95ed3c0078e1` | ray_trace_standalone.compile_s | {"ray_trace_standalone.compile_s": 0.42625011799827917} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/ray_trace_standalone/compile_s` |
| `measurement/e2074a2096fa1d267040` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.861017303999688} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/e3d8b47cbe8e75459262` | transform_mapping_matrix_standalone.first_call_s | {"transform_mapping_matrix_standalone.first_call_s": 0.8590792350005358} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/transform_mapping_matrix_standalone/first_call_s` |
| `measurement/e402475fa64b5b306a48` | reconstruction.lower_s | {"reconstruction.lower_s": 0.06654838800022844} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/reconstruction/lower_s` |
| `measurement/f447902a1bfde13c1a4e` | transform_mapping_matrix_standalone.lower_s | {"transform_mapping_matrix_standalone.lower_s": 0.044414531002985314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/transform_mapping_matrix_standalone/lower_s` |
| `measurement/f79386422d2d433414df` | w_tilde_figure_of_merit.first_call_s | {"w_tilde_figure_of_merit.first_call_s": 0.0006994260038482025} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_figure_of_merit/first_call_s` |
| `measurement/f9aaf2a77807f6b5b9d7` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.4432927239977289} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/f9e6c2899c5996fbd39a` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.10127062600076897} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/ffff8413e544bc9f05c9` | w_tilde_figure_of_merit.compile_s | {"w_tilde_figure_of_merit.compile_s": 0.04738554399955319} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/jit_phases/w_tilde_figure_of_merit/compile_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/9aef33b8b993112160dd` | host_peak_rss | {"host_peak_rss": 1328.48046875} | MiB | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>1 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/71f3c58ee81dff4dceef` | single_jit_block | {"single_jit_block": 0.8545848549001676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/sma/mge_hpc_a100_fp64.json) `/full_pipeline_single_jit` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 2087 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoLens": "2026.8.17.1"
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
    "host": "euclid-ral-gpu-2",
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
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 204,
      "backend": "gpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-2",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 2087 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoLens": "2026.8.17.1"
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
    "host": "euclid-ral-gpu-2",
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

- [likelihood_breakdown](../../../../scripts/interferometer/mge/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/interferometer/mge/likelihood_runtime.py)
- [quick_update](../../../../scripts/interferometer/mge/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
