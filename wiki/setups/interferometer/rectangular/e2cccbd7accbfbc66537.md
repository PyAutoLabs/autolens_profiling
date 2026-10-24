<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · jvla

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/jvla/b812652e5e6dd54b2583`.

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

[Original setup artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>30 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/01a1a8b8af8ceacee2e6` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.0017070721994969062} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/061fd9bb9d2f8c521d8e` | steps.Curvature matrix F = Aᵀ W~ A (12 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (12 blocks of 128)": 0.518806538300123} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (12 blocks of 128)` |
| `measurement/0657bc2eeba481bd6ca7` | curvature_sparse_fp32.steady_per_call_s | {"curvature_sparse_fp32.steady_per_call_s": 7.179529850400286} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse_fp32/steady_per_call_s` |
| `measurement/150cac4ac6c10458fd9c` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.005627972500224132} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/16b40761cd8dbe3b61ea` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.0005660831011482514} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/2605c065a839790e5b0b` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.5500911986993742} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/38237c495565467c7e2d` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.0017070721994969062} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/3bbbade01dc18bbd8ca7` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.00637890499929199} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/4943c3fe571c58986ce2` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.0009236746991518885} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/4de78fc52cdc8c5f33d6` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.006630943100026343} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/506d1b8f259f0c010c8f` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.0041497534009977246} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/576a8a18084b166da742` | full_pipeline_vmap4.steady_per_call_s | {"full_pipeline_vmap4.steady_per_call_s": 0.5611165388749214} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/full_pipeline_vmap4/steady_per_call_s` |
| `measurement/605f5271429912baf277` | steps.Ray-trace grids (data) | {"steps.Ray-trace grids (data)": 0.0004153718997258693} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Ray-trace grids (data)` |
| `measurement/685b56e04ee237b2deb1` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.023318014900723938} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/79b0d557a40a467b523b` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.00022236429940676318} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/89d3c9f1a9176e9aafb0` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.023367844699532726} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/915004e7803dd7ef887f` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.001471756199316587} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/98d1428d3af3168bf559` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 0.549823053900036} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/9c378c186476a5bf9e05` | steps.Regularization matrix H | {"steps.Regularization matrix H": 6.4628999098207546e-06} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/b078c6ad629bc8a8fa8e` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.00022236429940676318} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/b2f0a32e7312cdf1519f` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0004153718997258693} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/bbd51f09048ce013dc41` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.0056215096003143115} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/bc4fd264f3d28d471e6a` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.0022069922008085994} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/cf4cba4d128e0ee08483` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.023367844699532726} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/d92a2f6e9ebf52dc5b95` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.03715971199999331} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/e8aea1c0ba1d6285c8d2` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 0.518806538300123} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/e8d7a000f26956fbaddd` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.0037343815012718553} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/f31b8b78efd92af9ea4e` | component_total | {"component_total": 0.5528624588987441} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/fa7a2c6e1834904f885c` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.0009236746991518885} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/fca1d009101aaa453342` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.0022069922008085994} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |

</details>

### compile

<details><summary>56 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0b5ad4758c43cfc757b6` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.6774158579937648} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/12469ed91e116fc5b830` | log_dets.first_call_s | {"log_dets.first_call_s": 0.002866234994144179} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/15a204fe76c294938e25` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.02502731500135269} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/194bc85c6fe559d9fe1f` | full_pipeline_vmap4.first_call_s | {"full_pipeline_vmap4.first_call_s": 27.435358292001183} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/full_pipeline_vmap4/first_call_s` |
| `measurement/1a73dda626e85ca3f59d` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.2620906609954545} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/2157160ee774369fd4ad` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.12052371299068909} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/2299e7d0b7b0ba39a06c` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.00876102400070522} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/22fddfaa2f03f3078b96` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.09782141800678801} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/24873cdcb21797c1d531` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 0.6197787889977917} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/2af11d033c554ef83271` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.4886625350045506} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/34a17404b72ee22b621f` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.7864693080045981} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/3549e0d8bb1701f0874b` | curvature_sparse_fp32.compile_s | {"curvature_sparse_fp32.compile_s": 0.2950089320074767} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse_fp32/compile_s` |
| `measurement/392490b3e40519d8a588` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.0020981940033379942} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/3de9a2529a6b54c98f57` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.28566350499750115} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/447d71dcb807df625826` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.2125887280126335} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/4aaa7498297537ab4c8a` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.014448774003540166} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/53b5542d4f1e0a729fe4` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 23.10107031100779} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/58749af9ca8d48997a58` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.012303600000450388} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/5ba63e253379be526ccd` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0039615120040252805} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/5f45e90c4a753ec43abd` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.8253507269982947} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/5f80a936b4d84aec422b` | log_dets.compile_s | {"log_dets.compile_s": 0.1614203350036405} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/675256bcead65b0420ce` | operator_setup | {"operator_setup": 7.436979058998986} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/operator_build_s` |
| `measurement/6dd97a8b5398595eef78` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.01397861199802719} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/76a14c350ce206f995eb` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.08533600199734792} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/7dc098f5169215c433da` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.007374231005087495} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/7e88496686a0d012dc11` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.02524513500975445} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/84edc90035d5cd8c8a76` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.061443288999726065} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/8a23bd25a592da552a19` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04132408200530335} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/8aa14a1b5da43ff4d151` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.010234705987386405} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/9268837859cc140e202b` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.4753949790028855} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/968d96eba17a119662b0` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.8476694650016725} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/97d6e43fdb72626380ba` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.04444622900336981} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/9c45dbb8312a795c4787` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.06015515800390858} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/9c711fa52b6ea81757e2` | curvature_sparse_fp32.lower_s | {"curvature_sparse_fp32.lower_s": 0.048649114003637806} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse_fp32/lower_s` |
| `measurement/a7489dab0e39bc59c1fc` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.014742291008587927} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/ac39c62d9ea5ce05e505` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.014747131994226947} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/b45f156b672abda00a8d` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 0.5643331040046178} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/b641301aaf6172de21e5` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.0011042359983548522} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/b7b81ec426b118eb85a8` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.01715048198821023} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/b9edc8750d1b3f93e61b` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 3.4889353070029756} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/bb106f3da589ef9c50f1` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.060437491003540345} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/bb3c0067c4e9110a6994` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.6868094439996639} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/bb56921eb58f58aa5463` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.015049945999635383} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/c0ce3535f6b2afb2d282` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.663810445999843} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/c2bfdcf60c29a6bb5e55` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.09179822300211526} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/c890dbe276caed9ad62c` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.006118974997662008} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/ca6306f9a207537fab8c` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.8580578190012602} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/cd1176b5d1accbbef46a` | full_pipeline.compile_s | {"full_pipeline.compile_s": 23.020094268009416} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/ce6e7dc5e2d4401832ee` | log_dets.lower_s | {"log_dets.lower_s": 0.024720859990338795} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/d5c5fd0410a70b0a1c3b` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.04983753900160082} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/e261933b33c64e588916` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.741196923001553} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/e3629c6dd13bb0089792` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.13176103700243402} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/e69e334f0629c9ce6494` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 3.4153996630047914} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/e9d6a69233b5a0c80f0d` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0009584890067344531} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/ee926e408806059d0dc1` | curvature_sparse_fp32.first_call_s | {"curvature_sparse_fp32.first_call_s": 7.369378806994064} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/curvature_sparse_fp32/first_call_s` |
| `measurement/fc9b0bbe6cd163340cf7` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.002135444010491483} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/812d3c0c355348a33b61` | host_peak_rss | {"host_peak_rss": 10225.2421875} | MiB | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/189a48cf2357c9bf18b9` | single_jit_block | {"single_jit_block": 0.5500911986993742} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/full_pipeline_single_jit` |
| `measurement/1de43607ea4ae221861f` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 0.5500911986993742} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_gpu_sparse_mixed_wtilde/hpc_a100/breakdown.json) `/full_pipeline/per_call_s` |

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
