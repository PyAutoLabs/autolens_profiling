<!-- generated: build_setup_wiki.py; do not edit -->
# mge · alma

[Model index](index.md)

Exact setup ID: `interferometer/mge/alma/f1f501c8d539e67cced4`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| chunk_size | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| chunk_size_note | "TransformerNUFFT.transform_mapping_matrix ignores chunk_size: step 3 is one nufft2d2 over every column and every visibility." | recorded |  |
| dense_inversion_class | "InversionInterferometerMapping" | recorded |  |
| host_load_avg_end | [6.478515625, 6.06591796875, 4.3447265625] | recorded |  |
| host_load_avg_start | [3.7744140625, 3.9072265625, 3.18994140625] | recorded |  |
| image_pixels_masked | 15380 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| library_sparse_arm | true | recorded |  |
| linear_gaussians | 20 | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| n_repeats | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "raw" | recorded |  |
| nnls_solver | "pdip" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transform | "library" | recorded |  |
| transform_column_batch | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| transform_vis_chunk | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| transformer | "TransformerNUFFT" | name |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 1000000 | recorded |  |
| w_tilde_arm | true | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>35 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/03ae6fc8bba521dc3da3` | w_tilde_reconstruction.steady_per_call_s | {"w_tilde_reconstruction.steady_per_call_s": 0.0003945500011468539} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_reconstruction/steady_per_call_s` |
| `measurement/0522a3a975c7e115e83b` | curvature_matrix.steady_per_call_s | {"curvature_matrix.steady_per_call_s": 0.12631909999981872} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/curvature_matrix/steady_per_call_s` |
| `measurement/0c5f1b40b59f2a63e3a4` | library_sparse_prefix_5.steady_per_call_s | {"library_sparse_prefix_5.steady_per_call_s": 0.03267810000033933} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_5/steady_per_call_s` |
| `measurement/0c8f16f2121aac7314b0` | w_tilde_data_vector.steady_per_call_s | {"w_tilde_data_vector.steady_per_call_s": 0.0005810000002384186} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_data_vector/steady_per_call_s` |
| `measurement/0e310fd9c4a316c54f69` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0011793000012403354} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/204e81c6ce1a82f94ed1` | dense_steps.Fast chi-squared + figure of merit | {"dense_steps.Fast chi-squared + figure of merit": 0.00796115000048303} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/dense_steps/Fast chi-squared + figure of merit` |
| `measurement/2db3d576259009b4898e` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.0016214999996009283} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/steps/Ray-trace grids` |
| `measurement/31ce37ff5c8d4057774a` | dense_steps.Ray-trace grids | {"dense_steps.Ray-trace grids": 0.0011793000012403354} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/dense_steps/Ray-trace grids` |
| `measurement/37c5f5e62bc2cab476d2` | reconstruction.steady_per_call_s | {"reconstruction.steady_per_call_s": 0.00031024999952933285} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/reconstruction/steady_per_call_s` |
| `measurement/3d7991940f4fe1b14138` | steps.Data vector + curvature matrix (library W~: Bᵀ d~, Bᵀ W~ B) | {"steps.Data vector + curvature matrix (library W~: B\u1d40 d~, B\u1d40 W~ B)": 0.04759050000211573} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/steps/Data vector + curvature matrix (library W~0: Bᵀ d~0, Bᵀ W~0 B)` |
| `measurement/42c3517b97d85a90a340` | w_tilde_curvature.steady_per_call_s | {"w_tilde_curvature.steady_per_call_s": 0.030999149999843212} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_curvature/steady_per_call_s` |
| `measurement/43db8bd9bd796b0d4cda` | dense_steps.Curvature matrix (F) | {"dense_steps.Curvature matrix (F)": 0.12631909999981872} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/dense_steps/Curvature matrix (F)` |
| `measurement/56dc6f4e060fba01a92c` | steps.Mapping matrix (20 Gaussians) | {"steps.Mapping matrix (20 Gaussians)": 0.0021684999992430676} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/steps/Mapping matrix (20 Gaussians)` |
| `measurement/5ce496317feb19536e91` | dense_steps.Reconstruction (PDIP NNLS) | {"dense_steps.Reconstruction (PDIP NNLS)": 0.00031024999952933285} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/dense_steps/Reconstruction (PDIP NNLS)` |
| `measurement/5dd2ad6a788ecd79df32` | w_tilde_figure_of_merit.steady_per_call_s | {"w_tilde_figure_of_merit.steady_per_call_s": 6.749999920430128e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_figure_of_merit/steady_per_call_s` |
| `measurement/62878e0aaa895a77577a` | library_sparse_fit_log_likelihood.steady_per_call_s | {"library_sparse_fit_log_likelihood.steady_per_call_s": 2.86328605000017} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_fit_log_likelihood/steady_per_call_s` |
| `measurement/63f86577d43c0f13f4d1` | ray_trace_standalone.steady_per_call_s | {"ray_trace_standalone.steady_per_call_s": 0.0009015499999804888} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/ray_trace_standalone/steady_per_call_s` |
| `measurement/64101e479971a61712eb` | transform_mapping_matrix_standalone.steady_per_call_s | {"transform_mapping_matrix_standalone.steady_per_call_s": 11.950081900000441} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/transform_mapping_matrix_standalone/steady_per_call_s` |
| `measurement/66167b80929ab49721d1` | dense_steps.Data vector (D) | {"dense_steps.Data vector (D)": 0.06816414999957487} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/dense_steps/Data vector (D)` |
| `measurement/69394d2679299354c1ec` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.0032964000001811655} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/7cab7fe3fe7a2989b1ba` | dense_steps.Transformed mapping matrix (NUFFT) | {"dense_steps.Transformed mapping matrix (NUFFT)": 10.844914899998912} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/dense_steps/Transformed mapping matrix (NUFFT)` |
| `measurement/85e8bc4d70111e72a53d` | component_total | {"component_total": 0.03267810000033933} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/total_step_by_step` |
| `measurement/8a18a3b9440b48c4387c` | dense_steps.Mapping matrix (20 Gaussians) | {"dense_steps.Mapping matrix (20 Gaussians)": 0.00211709999894083} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/dense_steps/Mapping matrix (20 Gaussians)` |
| `measurement/911f7188157bbc82cdab` | w_tilde_chain_from_mapping_matrix.steady_per_call_s | {"w_tilde_chain_from_mapping_matrix.steady_per_call_s": 0.03472499999952561} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/steady_per_call_s` |
| `measurement/9231aa4a04222c444c12` | library_sparse_prefix_3.steady_per_call_s | {"library_sparse_prefix_3.steady_per_call_s": 0.05138050000095973} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_3/steady_per_call_s` |
| `measurement/974eb16faebc0c7f479a` | log_likelihood_residual_visibilities.steady_per_call_s | {"log_likelihood_residual_visibilities.steady_per_call_s": 0.025181499999234802} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/log_likelihood_residual_visibilities/steady_per_call_s` |
| `measurement/98bcf4450c5d80757b6a` | library_sparse_prefix_4.steady_per_call_s | {"library_sparse_prefix_4.steady_per_call_s": 0.04428514999926847} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_4/steady_per_call_s` |
| `measurement/a187c19db99ff6e5e884` | library_sparse_prefix_2.steady_per_call_s | {"library_sparse_prefix_2.steady_per_call_s": 0.003789999998843996} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_2/steady_per_call_s` |
| `measurement/ab373187861f26eaca86` | library_sparse_prefix_1.steady_per_call_s | {"library_sparse_prefix_1.steady_per_call_s": 0.0016214999996009283} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_1/steady_per_call_s` |
| `measurement/afed8f3d5fb38b6a9815` | library_sparse_full_pipeline.steady_per_call_s | {"library_sparse_full_pipeline.steady_per_call_s": 0.0346518999995169} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_full_pipeline/steady_per_call_s` |
| `measurement/c9e8b3e4e1451378020d` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.00796115000048303} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/cbf0b63dca5626d256e4` | dense_chain_from_mapping_matrix.steady_per_call_s | {"dense_chain_from_mapping_matrix.steady_per_call_s": 11.869372299999668} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/dense_chain_from_mapping_matrix/steady_per_call_s` |
| `measurement/d203324f03f45b524a04` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 10.848211299999093} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/dbf896af2498cb2070d0` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 13.10301575000085} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/ea7730b3ac1fdbec3604` | data_vector.steady_per_call_s | {"data_vector.steady_per_call_s": 0.06816414999957487} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/data_vector/steady_per_call_s` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/016c4ad4b59bbfdc6f10` | curvature_matrix.first_call_s | {"curvature_matrix.first_call_s": 0.1378813999981503} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/curvature_matrix/first_call_s` |
| `measurement/04331aa90bc87922deb7` | library_sparse_prefix_2.first_call_s | {"library_sparse_prefix_2.first_call_s": 0.006607399998756591} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_2/first_call_s` |
| `measurement/04ae8a5854c6d37ec380` | library_sparse_prefix_4.first_call_s | {"library_sparse_prefix_4.first_call_s": 0.0443249999989348} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_4/first_call_s` |
| `measurement/0848999c78777e6e941c` | reconstruction.lower_s | {"reconstruction.lower_s": 0.06941110000116169} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/reconstruction/lower_s` |
| `measurement/0ad9e74b8dcd1b49c1f6` | curvature_matrix.compile_s | {"curvature_matrix.compile_s": 0.11654300000009243} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/curvature_matrix/compile_s` |
| `measurement/13f9dc0adcb39820e220` | library_sparse_prefix_1.lower_s | {"library_sparse_prefix_1.lower_s": 0.12813630000164267} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_1/lower_s` |
| `measurement/15042e795df927c54c43` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.485694599999988} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/1596d7291061c322911f` | w_tilde_data_vector.first_call_s | {"w_tilde_data_vector.first_call_s": 0.0010947999980999157} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_data_vector/first_call_s` |
| `measurement/1a3eedd028ddc7bf62e2` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0023723000012978446} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/1c0bfd970d74cdcd7c1f` | library_sparse_prefix_4.compile_s | {"library_sparse_prefix_4.compile_s": 0.16725159999987227} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_4/compile_s` |
| `measurement/1c4e5e2a5e7a87fe9348` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 0.4181484000000637} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/1e00169b84a3f53fe0c2` | log_likelihood_residual_visibilities.compile_s | {"log_likelihood_residual_visibilities.compile_s": 0.0910223999999289} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/log_likelihood_residual_visibilities/compile_s` |
| `measurement/1e3f2d1d0ad9a99df186` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 11.82999109999946} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/20630b889c9eb42c2272` | library_sparse_full_pipeline.first_call_s | {"library_sparse_full_pipeline.first_call_s": 0.03738629999861587} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_full_pipeline/first_call_s` |
| `measurement/235defd21b150e3350cb` | ray_trace_standalone.first_call_s | {"ray_trace_standalone.first_call_s": 0.0015269999967131298} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/ray_trace_standalone/first_call_s` |
| `measurement/23983de4a1ec146af922` | data_vector.first_call_s | {"data_vector.first_call_s": 0.06958460000168998} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/data_vector/first_call_s` |
| `measurement/2544fdcd11ad42ffd11c` | full_pipeline.compile_s | {"full_pipeline.compile_s": 6.669821499999671} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/28a7ef68ad31d1e03347` | w_tilde_curvature.first_call_s | {"w_tilde_curvature.first_call_s": 0.0297379000003275} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_curvature/first_call_s` |
| `measurement/2a4565beaeae4b0674eb` | full_pipeline.lower_s | {"full_pipeline.lower_s": 0.7641280999996525} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/2aa100086edcea1b5e1d` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.41619489999720827} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/2d0e791afe982855fbae` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.005364500000723638} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/377c8e1e48240466265e` | library_sparse_prefix_3.first_call_s | {"library_sparse_prefix_3.first_call_s": 0.04983210000136751} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_3/first_call_s` |
| `measurement/520d5bc64c6a126f29ea` | dense_chain_from_mapping_matrix.lower_s | {"dense_chain_from_mapping_matrix.lower_s": 0.125999399999273} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/dense_chain_from_mapping_matrix/lower_s` |
| `measurement/559142e4fcb0ec2bfa1c` | w_tilde_curvature.lower_s | {"w_tilde_curvature.lower_s": 0.1042323999972723} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_curvature/lower_s` |
| `measurement/5cf39dce356e9b2e4836` | curvature_matrix.lower_s | {"curvature_matrix.lower_s": 0.01684180000302149} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/curvature_matrix/lower_s` |
| `measurement/5f59d3f9c7afefff30b7` | library_sparse_fit_log_likelihood.lower_s | {"library_sparse_fit_log_likelihood.lower_s": 1.4906588000012562} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_fit_log_likelihood/lower_s` |
| `measurement/62735f178d510c3479f7` | transform_mapping_matrix_standalone.lower_s | {"transform_mapping_matrix_standalone.lower_s": 0.07632839999860153} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/transform_mapping_matrix_standalone/lower_s` |
| `measurement/630da85466c8cd5a72c3` | w_tilde_reconstruction.compile_s | {"w_tilde_reconstruction.compile_s": 1.720000000204891e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_reconstruction/compile_s` |
| `measurement/692a9e1737aa67876518` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.012466600001062034} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/6ab6ff16f7cefabb44d1` | w_tilde_reconstruction.first_call_s | {"w_tilde_reconstruction.first_call_s": 0.000667500000417931} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_reconstruction/first_call_s` |
| `measurement/6ce86bdee0d6bb1db5fe` | library_sparse_prefix_5.compile_s | {"library_sparse_prefix_5.compile_s": 3.3019069000001764} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_5/compile_s` |
| `measurement/6fcf9e5cee095f915fdd` | library_sparse_full_pipeline.lower_s | {"library_sparse_full_pipeline.lower_s": 1.8468243000024813} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_full_pipeline/lower_s` |
| `measurement/70d36168d4731e307e13` | w_tilde_reconstruction.lower_s | {"w_tilde_reconstruction.lower_s": 0.0004760999981954228} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_reconstruction/lower_s` |
| `measurement/7110deda870cb826cac5` | library_sparse_prefix_1.first_call_s | {"library_sparse_prefix_1.first_call_s": 0.002727700000832556} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_1/first_call_s` |
| `measurement/7177689dd1aba59ac046` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 12.217262300000584} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/794eb578c8b1f94252d6` | reconstruction.compile_s | {"reconstruction.compile_s": 0.043527800000447314} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/reconstruction/compile_s` |
| `measurement/7a1c1fcb2665a99bdf10` | dense_chain_from_mapping_matrix.first_call_s | {"dense_chain_from_mapping_matrix.first_call_s": 13.282875500000955} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/dense_chain_from_mapping_matrix/first_call_s` |
| `measurement/7baf6d3e7eb77e907a40` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.0686688999994658} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/7d5ff6afcf554f99565d` | library_sparse_prefix_3.compile_s | {"library_sparse_prefix_3.compile_s": 1.5400558000001183} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_3/compile_s` |
| `measurement/8777c47a5efda1234d1c` | transform_mapping_matrix_standalone.first_call_s | {"transform_mapping_matrix_standalone.first_call_s": 11.510467500000232} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/transform_mapping_matrix_standalone/first_call_s` |
| `measurement/8bdf3ed919e8b7c94b45` | ray_trace_standalone.compile_s | {"ray_trace_standalone.compile_s": 0.20179339999958756} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/ray_trace_standalone/compile_s` |
| `measurement/916f63eebfe1233a2618` | library_sparse_prefix_4.lower_s | {"library_sparse_prefix_4.lower_s": 1.31041660000119} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_4/lower_s` |
| `measurement/9411a12357abca38393b` | w_tilde_chain_from_mapping_matrix.first_call_s | {"w_tilde_chain_from_mapping_matrix.first_call_s": 0.03512500000215368} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/first_call_s` |
| `measurement/94946841fa3147cff957` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.08763700000054087} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/95b251a529cdabf3c17f` | w_tilde_curvature.compile_s | {"w_tilde_curvature.compile_s": 0.12649749999764026} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_curvature/compile_s` |
| `measurement/9b0cbb48ca915e084c04` | w_tilde_chain_from_mapping_matrix.compile_s | {"w_tilde_chain_from_mapping_matrix.compile_s": 0.03440509999927599} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/compile_s` |
| `measurement/9d4806547f45d441778b` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.021703500002331566} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/a0276ee93b9dfe7b8770` | w_tilde_figure_of_merit.lower_s | {"w_tilde_figure_of_merit.lower_s": 0.009449100001802435} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_figure_of_merit/lower_s` |
| `measurement/a5bd75dbf4f36a92da3b` | w_tilde_chain_from_mapping_matrix.lower_s | {"w_tilde_chain_from_mapping_matrix.lower_s": 0.07971760000145878} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_chain_from_mapping_matrix/lower_s` |
| `measurement/a71af2b8210220c04eb5` | library_sparse_prefix_5.lower_s | {"library_sparse_prefix_5.lower_s": 2.4104630000001634} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_5/lower_s` |
| `measurement/b364f8d0b83c5ce6c3b8` | library_sparse_prefix_2.lower_s | {"library_sparse_prefix_2.lower_s": 0.48702830000183894} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_2/lower_s` |
| `measurement/b4bbd4a6a67bffee0539` | w_tilde_data_vector.lower_s | {"w_tilde_data_vector.lower_s": 0.007709800000156974} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_data_vector/lower_s` |
| `measurement/b933203f76b7ee55af67` | library_sparse_fit_log_likelihood.first_call_s | {"library_sparse_fit_log_likelihood.first_call_s": 2.869493899997906} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_fit_log_likelihood/first_call_s` |
| `measurement/bd64db9ac0965d8ea2e5` | reconstruction.first_call_s | {"reconstruction.first_call_s": 0.0035170000010111835} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/reconstruction/first_call_s` |
| `measurement/bdbb34681b07e6bc1908` | library_sparse_fit_log_likelihood.compile_s | {"library_sparse_fit_log_likelihood.compile_s": 5.248003900000185} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_fit_log_likelihood/compile_s` |
| `measurement/c6a368c2059bfdec0b6a` | dense_chain_from_mapping_matrix.compile_s | {"dense_chain_from_mapping_matrix.compile_s": 0.6531188000008115} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/dense_chain_from_mapping_matrix/compile_s` |
| `measurement/cab692773208ef70061c` | ray_trace_standalone.lower_s | {"ray_trace_standalone.lower_s": 0.09671589999925345} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/ray_trace_standalone/lower_s` |
| `measurement/cd4775d890f9ad476fd8` | library_sparse_prefix_5.first_call_s | {"library_sparse_prefix_5.first_call_s": 0.03973039999982575} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_5/first_call_s` |
| `measurement/ce1199562146dd24eb72` | data_vector.compile_s | {"data_vector.compile_s": 0.08387759999823174} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/data_vector/compile_s` |
| `measurement/d26a67df61b8fbd00c67` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.01690489999964484} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/d4153a5a19c01be0a619` | library_sparse_prefix_3.lower_s | {"library_sparse_prefix_3.lower_s": 1.3485734000023513} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_3/lower_s` |
| `measurement/d9721270eb285fd2d6b6` | w_tilde_figure_of_merit.first_call_s | {"w_tilde_figure_of_merit.first_call_s": 0.00041989999954239465} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_figure_of_merit/first_call_s` |
| `measurement/dc7107d5612fb37d86d8` | log_likelihood_residual_visibilities.lower_s | {"log_likelihood_residual_visibilities.lower_s": 0.014427699999941979} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/log_likelihood_residual_visibilities/lower_s` |
| `measurement/e40d2a3f0cdabd02a797` | library_sparse_prefix_1.compile_s | {"library_sparse_prefix_1.compile_s": 0.021930799997790018} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_1/compile_s` |
| `measurement/eba43cf29ddb9dcabd73` | library_sparse_full_pipeline.compile_s | {"library_sparse_full_pipeline.compile_s": 2.879134200000408} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_full_pipeline/compile_s` |
| `measurement/f349638b45917e702d6e` | transform_mapping_matrix_standalone.compile_s | {"transform_mapping_matrix_standalone.compile_s": 0.858147300001292} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/transform_mapping_matrix_standalone/compile_s` |
| `measurement/f562c70b409627fce5e6` | log_likelihood_residual_visibilities.first_call_s | {"log_likelihood_residual_visibilities.first_call_s": 0.032608100002107676} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/log_likelihood_residual_visibilities/first_call_s` |
| `measurement/f756ffee4a41ec0673cd` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.09278619999895454} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/f89df237b4790debfa02` | w_tilde_figure_of_merit.compile_s | {"w_tilde_figure_of_merit.compile_s": 0.053215200001432095} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_figure_of_merit/compile_s` |
| `measurement/fb09e9d319f9dd187be0` | library_sparse_prefix_2.compile_s | {"library_sparse_prefix_2.compile_s": 0.09292270000150893} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/library_sparse_prefix_2/compile_s` |
| `measurement/fe5d03a0a4e854b9baf7` | data_vector.lower_s | {"data_vector.lower_s": 0.0116740999983449} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/data_vector/lower_s` |
| `measurement/ff107e3048ab85e541f2` | w_tilde_data_vector.compile_s | {"w_tilde_data_vector.compile_s": 0.09280779999971855} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/jit_phases/w_tilde_data_vector/compile_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/fdeee33db3babd5149fe` | host_peak_rss | {"host_peak_rss": 5226.08984375} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>1 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/535f66d0f5d99fca232b` | single_jit_block | {"single_jit_block": 13.10301575000085} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/mge_breakdown_alma_v2026.8.17.1_sparse.json) `/full_pipeline_single_jit` |

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
      "cache_fresh": true,
      "cpu_count": 8,
      "device": "cpu:0",
      "hostname": "DESKTOP-H143S82",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "b5ef2e89eaeb7ab4b240adb0e3f035f8104323ec",
      "PyAutoFit": "326f611b1b40afd89bb73956441faa7379328407",
      "PyAutoGalaxy": "36d1b43605444ea2a213204d71360e53936cf76b",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "8021c41dd110e7f3a70e3c650234e138a05cfb8a",
      "autolens_profiling": "ee671dcf345cc56b0439fec167468d78cd921ddf"
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
    "host": "DESKTOP-H143S82",
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
      "cache_fresh": true,
      "cpu_count": 8,
      "device": "cpu:0",
      "hostname": "DESKTOP-H143S82",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "b5ef2e89eaeb7ab4b240adb0e3f035f8104323ec",
      "PyAutoFit": "326f611b1b40afd89bb73956441faa7379328407",
      "PyAutoGalaxy": "36d1b43605444ea2a213204d71360e53936cf76b",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "8021c41dd110e7f3a70e3c650234e138a05cfb8a",
      "autolens_profiling": "ee671dcf345cc56b0439fec167468d78cd921ddf"
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
    "host": "DESKTOP-H143S82",
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
