<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · jvla

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/jvla/695e99f941d3638242c7`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 12 | recorded |  |
| host_load_avg_end | [1.74, 2.89, 2.15] | recorded |  |
| host_load_avg_start | [5.85, 3.47, 1.43] | recorded |  |
| image_pixels_masked | 384852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh | "RectangularBilinearAdaptImage" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| mixed_precision | {"abs_diff_nats": 0.004424631595611572, "bar_nats": 0.5, "figure_of_merit_fp64": -301318490.2833924, "figure_of_merit_mp": -301318490.2789678, "holds_bar": true, "note": "use_mixed_precision reaches the mapper / mapping matrix only; InterferometerSparseOperator hard-casts the W~ path to float64."} | recorded |  |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 490000 | recorded |  |
| operator_extent_shape | [700, 700] | recorded |  |
| operator_fft_shape | [1400, 1400] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.01 | recorded |  |
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
| use_mixed_precision | true | recorded |  |
| visibilities | 25000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>30 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/03795af55085d8ebdd90` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.023367844699532726} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Reconstruction (pdip)` |
| `measurement/044321d6bd04bead84ef` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.5500911986993742} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/0b555b1ea35b7461711e` | steps.Ray-trace grids (data) | {"steps.Ray-trace grids (data)": 0.0004153718997258693} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Ray-trace grids (data)` |
| `measurement/0d89d284772f48e169e6` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.00022236429940676318} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/0eaaed2f72a90b893b41` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.0056215096003143115} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/13dee64c2057726b9d4d` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.00637890499929199} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/188053fe7d20bff44034` | curvature_sparse_fp32.steady_per_call_s | {"curvature_sparse_fp32.steady_per_call_s": 7.179529850400286} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse_fp32/steady_per_call_s` |
| `measurement/248c263fb849cc55bc3b` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.0022069922008085994} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/36f4ff7cb8c6b0d12166` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 0.549823053900036} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/3ad5a6b89beb567407ab` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.0005660831011482514} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/3d8398cf182558b18910` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.0041497534009977246} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/4c00735a569067def014` | full_pipeline_vmap4.steady_per_call_s | {"full_pipeline_vmap4.steady_per_call_s": 0.5611165388749214} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/full_pipeline_vmap4/steady_per_call_s` |
| `measurement/5d7bc7feb27ebfd296e0` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 0.518806538300123} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/60caf8797e0650283fd6` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.03715971199999331} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/75c8a9df4e17ce635886` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.023318014900723938} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/7944bc8e1e054c39e25e` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0004153718997258693} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/7e27eb0924db8d08c2bc` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.001471756199316587} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Mapping matrix L` |
| `measurement/82eaaaf4e5eb81571387` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.0009236746991518885} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/86477e18428436362a9c` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.0037343815012718553} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/895c789cb3a8e794feed` | component_total | {"component_total": 0.5528624588987441} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/total_step_by_step` |
| `measurement/b17042666dd4bba83a86` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.005627972500224132} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/bddad2c48dbaf341d7f2` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.00022236429940676318} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/c351c561cfcfd81e41bb` | steps.Curvature matrix F = Aᵀ W~ A (12 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (12 blocks of 128)": 0.518806538300123} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Curvature matrix F = Aᵀ W~0 A (12 blocks of 128)` |
| `measurement/d4a8971b3471e80e388e` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.006630943100026343} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/e3c12a9fcfed9b626aa6` | steps.Regularization matrix H | {"steps.Regularization matrix H": 6.4628999098207546e-06} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Regularization matrix H` |
| `measurement/e9705ab35265126919ab` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.0017070721994969062} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/e9f594cc31b7b52cbb5d` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.0017070721994969062} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/ed98e864f8cee3056d5c` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.023367844699532726} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/f847c118419bbf5b66ed` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.0022069922008085994} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/fac35c0730c0a7691f41` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.0009236746991518885} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/steps/Fast chi-squared + figure of merit` |

</details>

### compile

<details><summary>56 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00aac99ad6719cdbbd12` | curvature_sparse_fp32.compile_s | {"curvature_sparse_fp32.compile_s": 0.2950089320074767} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse_fp32/compile_s` |
| `measurement/0d9b405b2e4ab34ba203` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.012303600000450388} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/0e3e65bd38fcc0c8650c` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.015049945999635383} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/0f4b020b0159775a0c75` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.08533600199734792} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/0fd8ddc672ab2cba3a26` | operator_setup | {"operator_setup": 7.436979058998986} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/operator_build_s` |
| `measurement/142b2a3fc8fb2045ee9b` | curvature_sparse_fp32.first_call_s | {"curvature_sparse_fp32.first_call_s": 7.369378806994064} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse_fp32/first_call_s` |
| `measurement/1e97e441693b0d0577b8` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.741196923001553} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/266db7f5dfb2ea9d118c` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.060437491003540345} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/291674d8e8cda52d01db` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.014747131994226947} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/3024490d6b09a9dc98fc` | full_pipeline.compile_s | {"full_pipeline.compile_s": 23.020094268009416} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/394ffaa057905e7a101e` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.7864693080045981} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/40166edcc1729a38efdf` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.6868094439996639} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/41345d417615849fe1f3` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.02502731500135269} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/41c0357663fdb37f76c4` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.014448774003540166} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/44025e2fece3c8a6859a` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.09782141800678801} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/45ab2cb8c4384d9df4e5` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.061443288999726065} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/47b13a16c207bd8e17c9` | full_pipeline_vmap4.first_call_s | {"full_pipeline_vmap4.first_call_s": 27.435358292001183} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/full_pipeline_vmap4/first_call_s` |
| `measurement/48f7ef3e2f935f089065` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.663810445999843} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/4a490cc58af376dbee33` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.002135444010491483} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/4b60a917862a3ce2d7e9` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.02524513500975445} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/4b8d5a368b6c960edc5f` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.01397861199802719} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/4fc50a61dd66f15503ef` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04132408200530335} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/4fd05e78a50236bc42e5` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 3.4153996630047914} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/52324edd0e54507bae9c` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.04983753900160082} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/5937df46b4e219698e46` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.00876102400070522} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/5cd0e5e0f1b50bc1da1d` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.8253507269982947} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/616f64f6a22ac0998906` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.006118974997662008} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/63202081cb72873e0199` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.6774158579937648} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/6e4addf573901458c955` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.0011042359983548522} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/6efd0d1bf62224080ed0` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0039615120040252805} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/6f9961c27e23e3bad59c` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.2620906609954545} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/7202de9aee12339800fd` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.01715048198821023} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/733ab63486846814d367` | log_dets.first_call_s | {"log_dets.first_call_s": 0.002866234994144179} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/764de522eb00f2ff3bd2` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.28566350499750115} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/79bb8d4cdc9be08ae986` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.4886625350045506} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/812737f755a21a26daed` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.04444622900336981} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/81bf09e430e51bcaff78` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.8580578190012602} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/87454d4bd78039aa1c12` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 0.5643331040046178} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/8832fd4f17874cf3251d` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.007374231005087495} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/8efb4efbe6aa922b31e8` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.2125887280126335} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/90c86c64692406ad8ea8` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.06015515800390858} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/9134a880e355110facd3` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.014742291008587927} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/9334f647b49550192dc9` | log_dets.lower_s | {"log_dets.lower_s": 0.024720859990338795} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/log_dets/lower_s` |
| `measurement/a1dfab3fb57d37d87528` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 0.6197787889977917} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/a72b642da2b695af60ef` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 23.10107031100779} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/a908fd814c6053aa5b22` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.12052371299068909} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/b3903d920093b0058baf` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.4753949790028855} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/c30d9d1f493db37b53b3` | curvature_sparse_fp32.lower_s | {"curvature_sparse_fp32.lower_s": 0.048649114003637806} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/curvature_sparse_fp32/lower_s` |
| `measurement/c3b28ecc474f4b7c8b02` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.8476694650016725} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/cb3d8d8da519d00a4990` | log_dets.compile_s | {"log_dets.compile_s": 0.1614203350036405} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/log_dets/compile_s` |
| `measurement/d92b0185e9bff43b601c` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 3.4889353070029756} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/e7e3e3efe56c9446d122` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.13176103700243402} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/ee1d6aba2785269b46c8` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.010234705987386405} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/f6febd38b6c46e38599d` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0009584890067344531} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/ff02d85b8bdc963b1824` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.09179822300211526} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/ffe1564e9bbdf1773171` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.0020981940033379942} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/jit_phases/data_vector_sparse/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/f58362f3077735484830` | host_peak_rss | {"host_peak_rss": 10225.2421875} | MiB | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/e53f46930bd614cab438` | single_jit_block | {"single_jit_block": 0.5500911986993742} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/full_pipeline_single_jit` |
| `measurement/f2b0e9b28745e5b1f972` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 0.5500911986993742} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/jvla/pixelization_hpc_a100_mp.json) `/full_pipeline/per_call_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 61381 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "mixed",
    "software": {
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "cf83504e8c7f3cc196ffef4d8a4ff86dd456bb6b",
      "PyAutoGalaxy": "0e4b89cff42882a6cf5bbed6c9686f7212d7e24c",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "2b3bc53388a15e1d51fb70b583061bc2ee3294f0",
      "autolens_profiling": "66e45e906ce77dc9a58fe2cf7870adc343c98929"
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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 61381 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "mixed",
    "software": {
      "PyAutoArray": "14d63360f341912b209f2045d7e388e264692f1d",
      "PyAutoFit": "cf83504e8c7f3cc196ffef4d8a4ff86dd456bb6b",
      "PyAutoGalaxy": "0e4b89cff42882a6cf5bbed6c9686f7212d7e24c",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "2b3bc53388a15e1d51fb70b583061bc2ee3294f0",
      "autolens_profiling": "66e45e906ce77dc9a58fe2cf7870adc343c98929"
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

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
