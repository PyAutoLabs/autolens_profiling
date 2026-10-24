<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · alma

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/alma/6e62f3ddb48b6167ffbf`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 12 | recorded |  |
| delaunay_vertices | 1500 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| hilbert_pixels | 1500 | recorded |  |
| host_load_avg_end | [5.7, 2.25, 0.83] | recorded |  |
| host_load_avg_start | [1.58, 0.38, 0.13] | recorded |  |
| image_mesh | "Hilbert(weight_power=1.0, weight_floor=0.0)" | recorded |  |
| image_pixels_masked | 31428 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 5.0 | recorded |  |
| mesh | "Delaunay" | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 40000 | recorded |  |
| operator_extent_shape | [200, 200] | recorded |  |
| operator_fft_shape | [400, 400] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| pixels_solved | 1500 | recorded |  |
| positive_only_solver_requested | "pdip" | recorded |  |
| positive_only_solver_used | "pdip" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| solve_subset_edge_zeroed | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 100000 | recorded |  |
| transformer_chunk_size_preset | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| visibilities | 1000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>29 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0cb2b505d963607a57a7` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.00043864569743163885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/1c44b0d2dede2fee5c23` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.000325360998976976} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/2d15eb1b3ed1cb9e248b` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.023331188602605836} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/385548960e9fff738f50` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.0022435278980992733} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/487957e1641e70a979ed` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.00032388410181738436} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/5044527377b818a14ce0` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.00043864569743163885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/6c2169377dc26423d1ba` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 0.07005416590254754} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/6e084fb788407a62ab3b` | full_pipeline_vmap16.steady_per_call_s | {"full_pipeline_vmap16.steady_per_call_s": 0.057037864931044166} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline_vmap16/steady_per_call_s` |
| `measurement/780415761a5297efc90d` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.0029917670006398113} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/83e27a50e9b278765c55` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.02351227630279027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/92cddada78a02decb125` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.006192007398931309} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/a058a8c6c33099bb7942` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.00032388410181738436} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/a6a33e816db8cd979bd5` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.0702364546013996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/a77faa5a7d50124ff6e7` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.006518827402032912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/a83b66209ae9748cb0fc` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 0.03767945049912669} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/a973a91b20d65d50a927` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.00017931080074049533} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/af95af309c51d42dadd9` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.0001356013002805412} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/bc29131baadcf926c889` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.006958662997931242} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/bd4da4158088efc1cce7` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.00011447459692135419} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/be40ceb9efd77a6a971e` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.0022435278980992733} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/c111d20180f2b4633162` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.0001356013002805412} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/c4bff035c6833db43816` | component_total | {"component_total": 0.07129204879747704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/total_step_by_step` |
| `measurement/c722a068b97da792e008` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.006844188401009888} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/cb45c2b322d67296a958` | steps.Ray-trace grids (data + mesh vertices) | {"steps.Ray-trace grids (data + mesh vertices)": 0.0003268200031016022} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Ray-trace grids (data + mesh vertices)` |
| `measurement/d5cf5a3edbe876de9053` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.006817017705179751} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/d651731c840aa6995a4c` | steps.Curvature matrix F = Aᵀ W~ A (12 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (12 blocks of 128)": 0.03767945049912669} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (12 blocks of 128)` |
| `measurement/d7fd62049c1e1551f306` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.02351227630279027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/eca90380f8bf07bd7b4c` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.0006279494031332433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/ff0bf01655bff76f5ae7` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0003268200031016022} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |

</details>

### compile

<details><summary>53 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/003c574e67984e675ccd` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.13551849499344826} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/0168622cd7871002624e` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.011222944012843072} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/0f063b8b03407670aedf` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.009449106000829488} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/113d8d941398b7e09660` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.013640684017445892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/15141eb587bec27d0458` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.053522755042649806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/21467de1b95cee876adf` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 2.075242585968226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/29692ab6171b4093fc95` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.025326304021291435} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/2b1a93434475afa9db3d` | log_dets.first_call_s | {"log_dets.first_call_s": 0.0029051349847577512} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/32d37b2e0ee1ce873e1f` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.5014442400424741} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/3de06fdad474169edea7` | log_dets.compile_s | {"log_dets.compile_s": 0.14369456702843308} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/3f25804253f683bdc5de` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.014688082970678806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/3f84e548674dd6fa7a0a` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 3.35643402999267} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/45e1eca1e85a786f9dff` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.0928113599657081} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/48b03838d95cecfac792` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.015187867975328118} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/5088b8f49bd0003c3fd0` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.114925469039008} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/5615cd0f2357329ae118` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.18974840105511248} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/5eb5f60b47256e77f458` | full_pipeline_vmap16.first_call_s | {"full_pipeline_vmap16.first_call_s": 7.674576886987779} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline_vmap16/first_call_s` |
| `measurement/6367701c59ecb1efb649` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.77697620698018} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/6c3824e72fbff25f3e80` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0008518890244886279} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/6dc3076d7f0fa244cbfe` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.0011679449817165732} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/733b491e77f5572aaa1a` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0021736729540862143} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/746395e4d6f319bb0da8` | log_dets.lower_s | {"log_dets.lower_s": 0.02592904696939513} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/75b93300902eca4225d1` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.000968398991972208} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/7645ab7772a014bc8867` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.0007927700062282383} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/7e3b17c34cc94f4a95b6` | operator_setup | {"operator_setup": 5.067651065997779} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/operator_build_s` |
| `measurement/84b707662b1afb5b9a2c` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.06321807799395174} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/89fb0fa12ef26efd4c42` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.22729896695818752} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/93c6606c86d8d9a95e27` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.0064955810084939} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/950123e54b00ec8bf91a` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.02540586399845779} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/98f4bebee7f769544570` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 0.055718317977152765} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/9a3dd30055b9e5d9e71f` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.8725902540027164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/9cecd7f7e3601b31111f` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 0.09433327300939709} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/9d09e9e545dc7be833b1` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.011566021014004946} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/9e69cdedabc3ab566e43` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 1.9499133590143174} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/a1bc2c8caad53d1ef8e9` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.77859239699319} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/a2c02e6e003012da4086` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.04739697900367901} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/a5066ae4aa2c798758d1` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.770246927975677} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/accc6535fc990cb61f80` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.11446603003423661} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/b680a847d3567356ad92` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.06209585099713877} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/b6c9d987a75e0aa978bd` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.2760806590085849} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/b8d74e439af2526ef99c` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.0016154399490915239} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/ba2cfeb15585947b1bf9` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.0107205170206726} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/be7644c80f08138db429` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.011811496980953962} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/c47426a74cc70fdc85db` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.041220233018975705} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/c9d094faf84477b3b235` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.043565883999690413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/cf16bd499a9ef866840f` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.06719827000051737} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/d2f11d56a60c337b3b81` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.5278098430135287} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/dbfb7b1f9c4a728d36f1` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.05723443900933489} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/de1a8c65af74e966995b` | full_pipeline.compile_s | {"full_pipeline.compile_s": 4.077846255968325} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/e383fa83e25188ab6aeb` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.8199415290146135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/e8648d9a21b09dadec23` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.01510640699416399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/f402b15f480e34b98d97` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 4.110081637976691} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/f701bc0010f4bdaf00af` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.012770065979566425} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/67cb78f6b504444b6aad` | host_peak_rss | {"host_peak_rss": 2614.62890625} | MiB | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/683a2fe120b26843860d` | single_jit_block | {"single_jit_block": 0.0702364546013996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/full_pipeline_single_jit` |
| `measurement/ef5a27665d6a576f9882` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 0.0702364546013996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma/jax_gpu_sparse_fp64_wtilde/hpc_a100/scans/r5.0/breakdown.json) `/full_pipeline/per_call_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 37351 MiB, 81920 MiB",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-28T19:30:50Z",
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
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 37351 MiB, 81920 MiB",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-09-28T19:30:50Z",
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

- [likelihood_breakdown](../../../../scripts/interferometer/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/delaunay/likelihood_runtime.py)
- [quick_update](../../../../scripts/interferometer/delaunay/quick_update.py)
- [_streaming](../../../../scripts/interferometer/delaunay/_streaming.py)
- [streaming_accumulate](../../../../scripts/interferometer/delaunay/streaming_accumulate.py)
- [streaming_in_memory](../../../../scripts/interferometer/delaunay/streaming_in_memory.py)
- [streaming_parity](../../../../scripts/interferometer/delaunay/streaming_parity.py)
- [streaming_plot_scaling](../../../../scripts/interferometer/delaunay/streaming_plot_scaling.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
