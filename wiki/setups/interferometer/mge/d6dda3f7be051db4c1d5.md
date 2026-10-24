<!-- generated: build_setup_wiki.py; do not edit -->
# mge · jvla

[Model index](index.md)

Exact setup ID: `interferometer/mge/jvla/78db21001c35cb4f1f35`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| chunk_size | 1000000 | recorded |  |
| chunk_size_note | "TransformerNUFFT.transform_mapping_matrix ignores chunk_size: step 3 is one nufft2d2 over every column and every visibility." | recorded |  |
| dense_inversion_class | "InversionInterferometerMapping" | recorded |  |
| host_load_avg_end | [0.91, 0.99, 0.86] | recorded |  |
| host_load_avg_start | [0.38, 0.08, 0.03] | recorded |  |
| image_pixels_masked | 384852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| library_sparse_arm | true | recorded |  |
| linear_gaussians | 20 | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "raw" | recorded |  |
| nnls_solver | "pdip" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.01 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transform | "script_chunked" | recorded |  |
| transform_column_batch | 4 | recorded |  |
| transform_vis_chunk | 1000000 | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 25000000 | recorded |  |
| w_tilde_arm | true | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>37 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0222860cabc9ed6cb588` | dense_steps.Transformed mapping matrix (NUFFT) | {"dense_steps.Transformed mapping matrix (NUFFT)": 23.55589655590011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/dense_steps/Transformed mapping matrix (NUFFT)` |
| `measurement/03099886cafd029f656e` | component_total | {"component_total": 0.01671951220050687} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/036f1e4182af4dbdd8e4` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 23.55903510650096} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/05a694f5234f9112306b` | dense_steps.Mapping matrix (20 Gaussians) | {"dense_steps.Mapping matrix (20 Gaussians)": 0.002543545899970923} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/dense_steps/Mapping matrix (20 Gaussians)` |
| `measurement/0a5b7d4751f6752adc92` | steps.Fast chi-squared + figure of merit (FitInterferometer) | {"steps.Fast chi-squared + figure of merit (FitInterferometer)": 0.00032169619953492656} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/steps/Fast chi-squared + figure of merit (FitInterferometer)` |
| `measurement/12d095a9a1eb7c185362` | curvature_matrix.steady_per_call_s | {"curvature_matrix.steady_per_call_s": 0.6211710942996433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix/steady_per_call_s` |
| `measurement/24f728b9d373cda8db78` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.0008513687993399799} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/264222fd2ea706015e37` | library_sparse_prefix_1.steady_per_call_s | {"library_sparse_prefix_1.steady_per_call_s": 0.0005919308998272755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_1/steady_per_call_s` |
| `measurement/2bcb30e45be5275649e3` | steps.Reconstruction (PDIP NNLS) | {"steps.Reconstruction (PDIP NNLS)": 0.0019322836000355906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/steps/Reconstruction (PDIP NNLS)` |
| `measurement/36f8cf89733a07b9d275` | dense_steps.Fast chi-squared + figure of merit | {"dense_steps.Fast chi-squared + figure of merit": 0.0008513687993399799} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/dense_steps/Fast chi-squared + figure of merit` |
| `measurement/38100411ae9438d98eac` | log_likelihood_residual_visibilities.steady_per_call_s | {"log_likelihood_residual_visibilities.steady_per_call_s": 0.011065398198843468} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/steady_per_call_s` |
| `measurement/3caf38418011d115eede` | w_tilde_figure_of_merit.steady_per_call_s | {"w_tilde_figure_of_merit.steady_per_call_s": 0.00014158929989207535} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_figure_of_merit/steady_per_call_s` |
| `measurement/3eb5c909096f423475be` | w_tilde_data_vector.steady_per_call_s | {"w_tilde_data_vector.steady_per_call_s": 0.000254006999603007} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_data_vector/steady_per_call_s` |
| `measurement/428b8448433e32915df8` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.003138550600851886} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/45b0a36821fa8dad9141` | w_tilde_reconstruction.steady_per_call_s | {"w_tilde_reconstruction.steady_per_call_s": 0.002550121299282182} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_reconstruction/steady_per_call_s` |
| `measurement/4c0f16434ec6dc534616` | dense_chain_from_mapping_matrix.steady_per_call_s | {"dense_chain_from_mapping_matrix.steady_per_call_s": 24.06797760019981} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/steady_per_call_s` |
| `measurement/59ee8d1c1957fb14822d` | w_tilde_curvature.steady_per_call_s | {"w_tilde_curvature.steady_per_call_s": 0.011166564999439289} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_curvature/steady_per_call_s` |
| `measurement/67282c2b181a350bf932` | dense_steps.Data vector (D) | {"dense_steps.Data vector (D)": 0.01569888440135401} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/dense_steps/Data vector (D)` |
| `measurement/6e927f57e9e15de1dbe1` | steps.Mapping matrix (20 Gaussians) | {"steps.Mapping matrix (20 Gaussians)": 0.0026181016000919044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/steps/Mapping matrix (20 Gaussians)` |
| `measurement/746ffb47a32e605bb593` | library_sparse_fit_log_likelihood.steady_per_call_s | {"library_sparse_fit_log_likelihood.steady_per_call_s": 0.09312331569963135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_fit_log_likelihood/steady_per_call_s` |
| `measurement/76daab23646501d893d6` | ray_trace_standalone.steady_per_call_s | {"ray_trace_standalone.steady_per_call_s": 0.00028469459939515216} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/ray_trace_standalone/steady_per_call_s` |
| `measurement/7aeca5ab568951cf9a55` | library_sparse_prefix_3.steady_per_call_s | {"library_sparse_prefix_3.steady_per_call_s": 0.014465532400936353} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_3/steady_per_call_s` |
| `measurement/8715b4d75c49e200e264` | library_sparse_prefix_4.steady_per_call_s | {"library_sparse_prefix_4.steady_per_call_s": 0.016397816000971944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_4/steady_per_call_s` |
| `measurement/8cda256d95cea4b66b7d` | w_tilde_chain_from_mapping_matrix.steady_per_call_s | {"w_tilde_chain_from_mapping_matrix.steady_per_call_s": 0.013550146200577729} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/steady_per_call_s` |
| `measurement/9b5da35dc436746172ae` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.0005919308998272755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/steps/Ray-trace grids` |
| `measurement/a3498aaafe4f036d7bb7` | dense_steps.Ray-trace grids | {"dense_steps.Ray-trace grids": 0.0005950047008809634} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/dense_steps/Ray-trace grids` |
| `measurement/a851467cc8fe866ebbeb` | library_sparse_full_pipeline_vmap16.steady_per_call_s | {"library_sparse_full_pipeline_vmap16.steady_per_call_s": 0.01076319722506014} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_full_pipeline_vmap16/steady_per_call_s` |
| `measurement/ac8f9db5bed67e4c1a64` | library_sparse_full_pipeline.steady_per_call_s | {"library_sparse_full_pipeline.steady_per_call_s": 0.016708837200712878} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_full_pipeline/steady_per_call_s` |
| `measurement/ae924cf4e9b7387877e2` | library_sparse_prefix_2.steady_per_call_s | {"library_sparse_prefix_2.steady_per_call_s": 0.0032100324999191797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_2/steady_per_call_s` |
| `measurement/b329612b0b71ddc757bf` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0005950047008809634} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/c0a597a61b9a53253ac4` | reconstruction.steady_per_call_s | {"reconstruction.steady_per_call_s": 0.0026788069008034654} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction/steady_per_call_s` |
| `measurement/cb1c254e0b42048f8b6f` | transform_mapping_matrix_standalone.steady_per_call_s | {"transform_mapping_matrix_standalone.steady_per_call_s": 23.516417920000094} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/steady_per_call_s` |
| `measurement/cd6fce7f3fec7af807fa` | dense_steps.Curvature matrix (F) | {"dense_steps.Curvature matrix (F)": 0.6211710942996433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/dense_steps/Curvature matrix (F)` |
| `measurement/db5528915a8179ea9417` | steps.Data vector + curvature matrix (library W~: Bᵀ d~, Bᵀ W~ B) | {"steps.Data vector + curvature matrix (library W~: B\u1d40 d~, B\u1d40 W~ B)": 0.011255499901017173} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/steps/Data vector + curvature matrix (library W~0: Bᵀ d~0, Bᵀ W~0 B)` |
| `measurement/efadbd1482e972409ae3` | dense_steps.Reconstruction (PDIP NNLS) | {"dense_steps.Reconstruction (PDIP NNLS)": 0.0026788069008034654} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/dense_steps/Reconstruction (PDIP NNLS)` |
| `measurement/f1e77d07fc41917da7c9` | library_sparse_prefix_5.steady_per_call_s | {"library_sparse_prefix_5.steady_per_call_s": 0.01671951220050687} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_5/steady_per_call_s` |
| `measurement/fae087f2ef07ef3957f8` | data_vector.steady_per_call_s | {"data_vector.steady_per_call_s": 0.01569888440135401} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector/steady_per_call_s` |

</details>

### compile

<details><summary>70 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0132ec358a7905525354` | w_tilde_data_vector.lower_s | {"w_tilde_data_vector.lower_s": 0.007142193004256114} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_data_vector/lower_s` |
| `measurement/04d8b452cdfff9f01506` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.2671676800091518} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/0cf5a800b0124264bc91` | w_tilde_curvature.first_call_s | {"w_tilde_curvature.first_call_s": 0.06605864599987399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_curvature/first_call_s` |
| `measurement/0f8fa9c78b00cfcf463c` | log_likelihood_residual_visibilities.compile_s | {"log_likelihood_residual_visibilities.compile_s": 0.23537980399851222} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/compile_s` |
| `measurement/142f7179f0f55ce514f6` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.680078795994632} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/150cff2b8ba831312238` | library_sparse_prefix_3.compile_s | {"library_sparse_prefix_3.compile_s": 3.519532648002496} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_3/compile_s` |
| `measurement/18e8d84f0b7117e7331a` | w_tilde_reconstruction.lower_s | {"w_tilde_reconstruction.lower_s": 0.0003591249987948686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_reconstruction/lower_s` |
| `measurement/2468ffed404ba41561fe` | w_tilde_chain_from_mapping_matrix.compile_s | {"w_tilde_chain_from_mapping_matrix.compile_s": 0.7173584340052912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/compile_s` |
| `measurement/25ea75312975cacc0cc4` | library_sparse_prefix_4.compile_s | {"library_sparse_prefix_4.compile_s": 3.5989321539964294} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_4/compile_s` |
| `measurement/2ae60fa61b9a2550fb7e` | library_sparse_prefix_2.lower_s | {"library_sparse_prefix_2.lower_s": 0.39147681900067255} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_2/lower_s` |
| `measurement/2d57c34952f8f0c9864a` | dense_chain_from_mapping_matrix.lower_s | {"dense_chain_from_mapping_matrix.lower_s": 0.5335578810045263} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/lower_s` |
| `measurement/2dfad5181552048dc0ae` | reconstruction.lower_s | {"reconstruction.lower_s": 0.02692010600003414} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction/lower_s` |
| `measurement/33897fbdd5b08d743178` | w_tilde_curvature.lower_s | {"w_tilde_curvature.lower_s": 0.1393884459976107} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_curvature/lower_s` |
| `measurement/391e19d1ab9772090524` | w_tilde_reconstruction.compile_s | {"w_tilde_reconstruction.compile_s": 1.3319993740879e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_reconstruction/compile_s` |
| `measurement/3c7e8fa6ddd1e367763c` | transform_mapping_matrix_standalone.first_call_s | {"transform_mapping_matrix_standalone.first_call_s": 23.650242440999136} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/first_call_s` |
| `measurement/3cb4fa177494e49e53ce` | library_sparse_prefix_5.first_call_s | {"library_sparse_prefix_5.first_call_s": 0.1053698049945524} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_5/first_call_s` |
| `measurement/3d920df0ac9ccc180e3a` | curvature_matrix.compile_s | {"curvature_matrix.compile_s": 0.06787159500527196} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix/compile_s` |
| `measurement/3eb2ed35346a4bd655a3` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.23807483099517412} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/410a9a17b7c4784dd7c9` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 23.097305574003258} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/45da9d91e52580cb96db` | dense_chain_from_mapping_matrix.first_call_s | {"dense_chain_from_mapping_matrix.first_call_s": 24.12508748000255} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/first_call_s` |
| `measurement/475badd0221ddd384e95` | library_sparse_fit_log_likelihood.first_call_s | {"library_sparse_fit_log_likelihood.first_call_s": 0.2220654130069306} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_fit_log_likelihood/first_call_s` |
| `measurement/4d04991a3bfd16ab3c3d` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.004737220995593816} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/5365b98cab55ed64f5a8` | library_sparse_prefix_3.first_call_s | {"library_sparse_prefix_3.first_call_s": 0.03543324299971573} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_3/first_call_s` |
| `measurement/5bc8d66a613c55764abd` | library_sparse_prefix_4.lower_s | {"library_sparse_prefix_4.lower_s": 0.84149273399089} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_4/lower_s` |
| `measurement/6137bd42a5fe4d3e4e56` | reconstruction.first_call_s | {"reconstruction.first_call_s": 0.24643961001129355} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction/first_call_s` |
| `measurement/643b6ed4943b409c68d4` | w_tilde_figure_of_merit.lower_s | {"w_tilde_figure_of_merit.lower_s": 0.005909549989155494} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_figure_of_merit/lower_s` |
| `measurement/6a0d135e227cd3a719fa` | data_vector.lower_s | {"data_vector.lower_s": 0.00873370400222484} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector/lower_s` |
| `measurement/6c0876dee57091ec38c6` | w_tilde_curvature.compile_s | {"w_tilde_curvature.compile_s": 0.34213374099635985} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_curvature/compile_s` |
| `measurement/718d7fdbda0bef933342` | curvature_matrix.lower_s | {"curvature_matrix.lower_s": 0.011254123994149268} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix/lower_s` |
| `measurement/7375f2706065dbe426d5` | library_sparse_prefix_2.first_call_s | {"library_sparse_prefix_2.first_call_s": 0.010256198002025485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_2/first_call_s` |
| `measurement/761d88eeed701bc597c2` | library_sparse_full_pipeline.compile_s | {"library_sparse_full_pipeline.compile_s": 21.12670554300712} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_full_pipeline/compile_s` |
| `measurement/783addd3a53e0f71ca8d` | w_tilde_chain_from_mapping_matrix.first_call_s | {"w_tilde_chain_from_mapping_matrix.first_call_s": 0.03069199199671857} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/first_call_s` |
| `measurement/78c4d8ff0982025f147b` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.5960747549979715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/79f0a5983e7c77e8a7d8` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.010190324988798238} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/80e5287221475b9b7dbd` | library_sparse_prefix_5.lower_s | {"library_sparse_prefix_5.lower_s": 1.87506643900997} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_5/lower_s` |
| `measurement/81cd88cdee45c4294ccc` | w_tilde_data_vector.first_call_s | {"w_tilde_data_vector.first_call_s": 0.0013032639981247485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_data_vector/first_call_s` |
| `measurement/8c73872423a4fd58a3ed` | library_sparse_prefix_4.first_call_s | {"library_sparse_prefix_4.first_call_s": 0.03997067001182586} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_4/first_call_s` |
| `measurement/9a6ff5cbe8914b9c5999` | library_sparse_fit_log_likelihood.compile_s | {"library_sparse_fit_log_likelihood.compile_s": 65.58099107501039} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_fit_log_likelihood/compile_s` |
| `measurement/a096186905c55f16817f` | w_tilde_figure_of_merit.first_call_s | {"w_tilde_figure_of_merit.first_call_s": 0.0007751009979983792} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_figure_of_merit/first_call_s` |
| `measurement/ab1e2931619a97919e87` | log_likelihood_residual_visibilities.lower_s | {"log_likelihood_residual_visibilities.lower_s": 0.007948544996907003} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/lower_s` |
| `measurement/ab937ff08deedaca4405` | data_vector.compile_s | {"data_vector.compile_s": 0.15218997199553996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector/compile_s` |
| `measurement/ac5b37e1fa221d8c122c` | transform_mapping_matrix_standalone.compile_s | {"transform_mapping_matrix_standalone.compile_s": 7.56156497199845} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/compile_s` |
| `measurement/acfc8110af94444ae59d` | library_sparse_prefix_3.lower_s | {"library_sparse_prefix_3.lower_s": 1.9959377769991988} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_3/lower_s` |
| `measurement/ae837c2fde65a1668dfd` | library_sparse_full_pipeline.lower_s | {"library_sparse_full_pipeline.lower_s": 1.7356350640038727} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_full_pipeline/lower_s` |
| `measurement/af5fc4e792a0265178c3` | curvature_matrix.first_call_s | {"curvature_matrix.first_call_s": 0.6422817059938097} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix/first_call_s` |
| `measurement/b009128a56b7f76f5b51` | log_likelihood_residual_visibilities.first_call_s | {"log_likelihood_residual_visibilities.first_call_s": 0.011772628000471741} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/first_call_s` |
| `measurement/b1dece6ebd65ce3568fe` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.07666105199314188} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/b59f50fec1a87af4e6f5` | reconstruction.compile_s | {"reconstruction.compile_s": 0.35769834301027004} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction/compile_s` |
| `measurement/b92680b9fef92286547c` | library_sparse_prefix_2.compile_s | {"library_sparse_prefix_2.compile_s": 0.15179741599422414} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_2/compile_s` |
| `measurement/be24256117bbd3913449` | w_tilde_reconstruction.first_call_s | {"w_tilde_reconstruction.first_call_s": 0.0029399350023595616} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_reconstruction/first_call_s` |
| `measurement/c3970b32f9fff90fe8d7` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.14513231300225016} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/ca6e43d15ef62476181f` | library_sparse_full_pipeline.first_call_s | {"library_sparse_full_pipeline.first_call_s": 0.10977632200228982} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_full_pipeline/first_call_s` |
| `measurement/cee35afaac37b7731f88` | w_tilde_data_vector.compile_s | {"w_tilde_data_vector.compile_s": 0.10082191899709869} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_data_vector/compile_s` |
| `measurement/d80e5f42501da9c67463` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.010615313003654592} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/d9a4fdc5b9256339c58b` | library_sparse_prefix_5.compile_s | {"library_sparse_prefix_5.compile_s": 21.652438559001894} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_5/compile_s` |
| `measurement/db68fa180787db42577a` | dense_chain_from_mapping_matrix.compile_s | {"dense_chain_from_mapping_matrix.compile_s": 12.182932203999371} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/compile_s` |
| `measurement/e520b4010769306c85df` | library_sparse_fit_log_likelihood.lower_s | {"library_sparse_fit_log_likelihood.lower_s": 2.306960212998092} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_fit_log_likelihood/lower_s` |
| `measurement/e61367ac5252c8db7138` | transform_mapping_matrix_standalone.lower_s | {"transform_mapping_matrix_standalone.lower_s": 0.2108222209935775} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/lower_s` |
| `measurement/eada34927b550dd20b79` | library_sparse_prefix_1.compile_s | {"library_sparse_prefix_1.compile_s": 0.6875088230008259} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_1/compile_s` |
| `measurement/eaefdec0c3dcd8b79fb2` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.001636000000871718} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/eb911734abad39eeadba` | library_sparse_prefix_1.lower_s | {"library_sparse_prefix_1.lower_s": 0.07689901599951554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_1/lower_s` |
| `measurement/f026bfa0aa945f013bdb` | ray_trace_standalone.first_call_s | {"ray_trace_standalone.first_call_s": 0.0015563019987894222} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/ray_trace_standalone/first_call_s` |
| `measurement/f0a9ea8272ab827edd59` | ray_trace_standalone.lower_s | {"ray_trace_standalone.lower_s": 0.07108584599336609} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/ray_trace_standalone/lower_s` |
| `measurement/f15df83641bb687a4a02` | w_tilde_figure_of_merit.compile_s | {"w_tilde_figure_of_merit.compile_s": 0.04848963700351305} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_figure_of_merit/compile_s` |
| `measurement/f2ce87904d2da1ace1f3` | w_tilde_chain_from_mapping_matrix.lower_s | {"w_tilde_chain_from_mapping_matrix.lower_s": 0.04414006999286357} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/lower_s` |
| `measurement/fbb70ec4723bc24a31c9` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 8.801544950998505} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/fc50f1981180ecb3dca4` | library_sparse_full_pipeline_vmap16.first_call_s | {"library_sparse_full_pipeline_vmap16.first_call_s": 24.982725670997752} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_full_pipeline_vmap16/first_call_s` |
| `measurement/fcd12aaf2f0f9d4998e5` | ray_trace_standalone.compile_s | {"ray_trace_standalone.compile_s": 0.392052285009413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/ray_trace_standalone/compile_s` |
| `measurement/fd6def08ed93b313b596` | data_vector.first_call_s | {"data_vector.first_call_s": 0.016334234998794273} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector/first_call_s` |
| `measurement/feaa20c28719af21c0a9` | library_sparse_prefix_1.first_call_s | {"library_sparse_prefix_1.first_call_s": 0.004462796001462266} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_sparse_prefix_1/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/66c10a51f7d86924d851` | host_peak_rss | {"host_peak_rss": 28881.8203125} | MiB | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/jvla/jax_gpu_sparse_fp64_wtilde/hpc_a100/breakdown.json) `/peak_rss_mb` |

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
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 47903 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "b5ef2e89eaeb7ab4b240adb0e3f035f8104323ec",
      "PyAutoFit": "dd9fbe0aab60c4a57702b1444ca031b317c91331",
      "PyAutoGalaxy": "36d1b43605444ea2a213204d71360e53936cf76b",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "1fa613aa8d89a79613009fb66f5b4e2d21deb57a",
      "autolens_profiling": "bb6ec21f28a83177edadb725f12eb07af8c5dad5"
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
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 204,
      "backend": "gpu",
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 47903 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "b5ef2e89eaeb7ab4b240adb0e3f035f8104323ec",
      "PyAutoFit": "dd9fbe0aab60c4a57702b1444ca031b317c91331",
      "PyAutoGalaxy": "36d1b43605444ea2a213204d71360e53936cf76b",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "1fa613aa8d89a79613009fb66f5b4e2d21deb57a",
      "autolens_profiling": "bb6ec21f28a83177edadb725f12eb07af8c5dad5"
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

- [likelihood_breakdown](../../../../scripts/interferometer/mge/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/interferometer/mge/likelihood_runtime.py)
- [quick_update](../../../../scripts/interferometer/mge/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
