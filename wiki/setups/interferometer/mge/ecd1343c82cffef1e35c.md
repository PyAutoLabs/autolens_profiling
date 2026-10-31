<!-- generated: build_setup_wiki.py; do not edit -->
# mge · sma

[Model index](index.md)

Exact setup ID: `interferometer/mge/sma/5afd2b9b2018fa18a0ee`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| chunk_size | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| chunk_size_note | "TransformerNUFFT.transform_mapping_matrix ignores chunk_size: step 3 is one nufft2d2 over every column and every visibility." | recorded |  |
| dense_inversion_class | "InversionInterferometerMapping" | recorded |  |
| host_load_avg_end | [0.99, 1.32, 3.58] | recorded |  |
| host_load_avg_start | [0.99, 1.35, 3.67] | recorded |  |
| image_pixels_masked | 3852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerMapping" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| library_sparse_arm | false | recorded |  |
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
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transform | "library" | recorded |  |
| transform_column_batch | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| transform_vis_chunk | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| transformer | "TransformerNUFFT" | name |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 190 | recorded |  |
| w_tilde_arm | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>20 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/19eacf3a79a4b397e70e` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 5.523549625650048e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/1f2f9d599e9d8ddf9e45` | steps.Reconstruction (PDIP NNLS) | {"steps.Reconstruction (PDIP NNLS)": 0.00015641170321032405} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Reconstruction (PDIP NNLS)` |
| `measurement/23afbbe9e382b6134abc` | steps.Mapping matrix (20 Gaussians) | {"steps.Mapping matrix (20 Gaussians)": 0.000672765204217285} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Mapping matrix (20 Gaussians)` |
| `measurement/2468a96d5a640115b850` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.07504674040246755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/34accbe31d19b2bb70a4` | dense_chain_from_mapping_matrix.steady_per_call_s | {"dense_chain_from_mapping_matrix.steady_per_call_s": 0.07383336109342054} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/steady_per_call_s` |
| `measurement/425232db28699a638f4f` | component_total | {"component_total": 0.07594982479931786} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/529d217a594fd8be6e75` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.0004506518947891891} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace grids` |
| `measurement/5383514184d70482c00a` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0004506518947891891} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/6f77c6a2e262777b66a7` | log_likelihood_residual_visibilities.steady_per_call_s | {"log_likelihood_residual_visibilities.steady_per_call_s": 4.610869800671935e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/steady_per_call_s` |
| `measurement/8d73f2dc30b7d9eb5a1c` | ray_trace_standalone.steady_per_call_s | {"ray_trace_standalone.steady_per_call_s": 0.0002253795973956585} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_standalone/steady_per_call_s` |
| `measurement/8da5d46b8e9af6ea9a44` | curvature_matrix.steady_per_call_s | {"curvature_matrix.steady_per_call_s": 5.523549625650048e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix/steady_per_call_s` |
| `measurement/99a162882d851e6cccf1` | data_vector.steady_per_call_s | {"data_vector.steady_per_call_s": 1.695690443739295e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector/steady_per_call_s` |
| `measurement/afc3cc788dc5a832874b` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 4.306629998609424e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/b177c677b1126cf7dc50` | steps.Data vector (D) | {"steps.Data vector (D)": 1.695690443739295e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Data vector (D)` |
| `measurement/b312d2770f354d282b09` | steps.Transformed mapping matrix (NUFFT) | {"steps.Transformed mapping matrix (NUFFT)": 0.07455473729642108} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Transformed mapping matrix (NUFFT)` |
| `measurement/d81dda1169fb60a5e587` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.001123417099006474} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/e12d9fbe8cdb92e040d9` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.07567815439542755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/eb371f8fcd973e28f119` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 4.306629998609424e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/eeed2d32e6db5f8bc98e` | reconstruction.steady_per_call_s | {"reconstruction.steady_per_call_s": 0.00015641170321032405} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction/steady_per_call_s` |
| `measurement/f61c1cb7d8abf3d96bf3` | transform_mapping_matrix_standalone.steady_per_call_s | {"transform_mapping_matrix_standalone.steady_per_call_s": 0.07312805179972201} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/steady_per_call_s` |

</details>

### compile

<details><summary>36 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1ae31b2c54c39d8ac815` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 0.6073866810183972} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/1fb3ba59f0f7796827cb` | reconstruction.compile_s | {"reconstruction.compile_s": 0.22009285900276154} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction/compile_s` |
| `measurement/23122d2e0d61e458ab0b` | dense_chain_from_mapping_matrix.first_call_s | {"dense_chain_from_mapping_matrix.first_call_s": 0.0846095229499042} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/first_call_s` |
| `measurement/26c45b4ab75eed57e8d5` | transform_mapping_matrix_standalone.first_call_s | {"transform_mapping_matrix_standalone.first_call_s": 0.0822550089797005} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/first_call_s` |
| `measurement/29875f17283214f440ea` | ray_trace_standalone.compile_s | {"ray_trace_standalone.compile_s": 0.15436042204964906} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_standalone/compile_s` |
| `measurement/410e6d7e4208e9af2473` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.17261345707811415} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/43073ecf1b13388a25ae` | transform_mapping_matrix_standalone.compile_s | {"transform_mapping_matrix_standalone.compile_s": 0.20670495193917304} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/compile_s` |
| `measurement/45a758d4f5117fecf6ff` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.00477685802616179} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/490ae14948330ccae283` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.002748070051893592} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/4da2f0b87a827c86ed1c` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.45249320892617106} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/4e4d5e6c97954c21033b` | reconstruction.first_call_s | {"reconstruction.first_call_s": 0.0007806419162079692} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction/first_call_s` |
| `measurement/5a1268d385a729bd167c` | dense_chain_from_mapping_matrix.compile_s | {"dense_chain_from_mapping_matrix.compile_s": 0.4184494810178876} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/compile_s` |
| `measurement/5c4fee1e1f2b6f9353b5` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.09756557503715158} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/5d0610d0f4a976df3c53` | ray_trace_standalone.lower_s | {"ray_trace_standalone.lower_s": 0.05539782508276403} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_standalone/lower_s` |
| `measurement/612460ac39677cb4ba4d` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.0810760889435187} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/6e64c0da51e16a433474` | log_likelihood_residual_visibilities.compile_s | {"log_likelihood_residual_visibilities.compile_s": 0.06389612099155784} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/compile_s` |
| `measurement/75c022f2448df5dd30aa` | curvature_matrix.lower_s | {"curvature_matrix.lower_s": 0.014834366040304303} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix/lower_s` |
| `measurement/7fd5356c551e2af75a14` | full_pipeline.compile_s | {"full_pipeline.compile_s": 0.8703660160535946} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/89b759f018786f978bae` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.00025308597832918167} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/8a9d727dd1494be403b5` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.007540480932220817} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/8c0c2394ec0d33f3a27d` | curvature_matrix.first_call_s | {"curvature_matrix.first_call_s": 0.0003234769683331251} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix/first_call_s` |
| `measurement/922771af64f92731f8f1` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.06123595801182091} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/964756dec4777acd4a0b` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.08167963300365955} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/a3967f92eb54688180b7` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.43530389689840376} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/a96daf8305ab8dc9e7ae` | transform_mapping_matrix_standalone.lower_s | {"transform_mapping_matrix_standalone.lower_s": 0.03025086096022278} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transform_mapping_matrix_standalone/lower_s` |
| `measurement/b6d64a5601aee5b227a5` | data_vector.compile_s | {"data_vector.compile_s": 0.05364381498657167} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector/compile_s` |
| `measurement/b8b466b57edf56118492` | dense_chain_from_mapping_matrix.lower_s | {"dense_chain_from_mapping_matrix.lower_s": 0.06149712507613003} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/dense_chain_from_mapping_matrix/lower_s` |
| `measurement/baf636db4a90c769745d` | log_likelihood_residual_visibilities.first_call_s | {"log_likelihood_residual_visibilities.first_call_s": 0.0002911769552156329} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/first_call_s` |
| `measurement/c95bfe65200375acf2fa` | full_pipeline.lower_s | {"full_pipeline.lower_s": 0.6471973709994927} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/ceab8a107b4e5f41b3b7` | data_vector.first_call_s | {"data_vector.first_call_s": 0.00027461303398013115} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector/first_call_s` |
| `measurement/d048482f0991aca79699` | ray_trace_standalone.first_call_s | {"ray_trace_standalone.first_call_s": 0.0014820980140939355} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_standalone/first_call_s` |
| `measurement/d9655dab24f24232e4cd` | log_likelihood_residual_visibilities.lower_s | {"log_likelihood_residual_visibilities.lower_s": 0.00512229697778821} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_likelihood_residual_visibilities/lower_s` |
| `measurement/dbfc75c2570b4c115bbb` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.228112526005134} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/ddc92e73b32777258994` | curvature_matrix.compile_s | {"curvature_matrix.compile_s": 0.05646979296579957} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix/compile_s` |
| `measurement/e522ccb4799320490ac6` | reconstruction.lower_s | {"reconstruction.lower_s": 0.06165908498223871} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction/lower_s` |
| `measurement/fc42552b303cac5f8e02` | data_vector.lower_s | {"data_vector.lower_s": 0.006966794026084244} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector/lower_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/3c267b5faf4410f5bef7` | host_peak_rss | {"host_peak_rss": 927.24609375} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0a767d294b1ce518d26f` | single_jit_median | {"single_jit_median": 0.07446866849204525} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/full_pipeline_single_jit_median` |
| `measurement/30d8ae571088e79c03d4` | single_jit_block | {"single_jit_block": 0.07504674040246755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/mge/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/full_pipeline_single_jit` |

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
      "cpu_count": 236,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-10-3",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "7c682ccbca9eb2bcad63b659e5d6328644a99617",
      "autolens_profiling.library_revisions": "7c682ccbca9eb2bcad63b659e5d6328644a99617",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.11.2",
      "jaxlib.dependency_versions": "0.11.2",
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
    "host": "euclid-ral-compute-10-3",
    "measured_at": "2026-10-30T08:32:44Z",
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
      "cache_fresh": true,
      "cpu_count": 236,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-10-3",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "7c682ccbca9eb2bcad63b659e5d6328644a99617",
      "autolens_profiling.library_revisions": "7c682ccbca9eb2bcad63b659e5d6328644a99617",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.11.2",
      "jaxlib.dependency_versions": "0.11.2",
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
    "host": "euclid-ral-compute-10-3",
    "measured_at": "2026-10-30T08:32:44Z",
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
      "cache_fresh": true,
      "cpu_count": 236,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-10-3",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "7c682ccbca9eb2bcad63b659e5d6328644a99617",
      "autolens_profiling.library_revisions": "7c682ccbca9eb2bcad63b659e5d6328644a99617",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.11.2",
      "jaxlib.dependency_versions": "0.11.2",
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
    "host": "euclid-ral-compute-10-3",
    "measured_at": "2026-10-30T08:32:44Z",
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

- [likelihood_breakdown](../../../../scripts/interferometer/mge/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/interferometer/mge/likelihood_runtime.py)
- [quick_update](../../../../scripts/interferometer/mge/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
