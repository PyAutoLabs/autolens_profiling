<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · alma

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/alma/c66214f34a970f2b4c66`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 32 | recorded |  |
| host_load_avg_end | [1.83, 2.91, 1.68] | recorded |  |
| host_load_avg_start | [3.63, 3.43, 1.68] | recorded |  |
| image_pixels_masked | 15380 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh | "RectangularBilinearAdaptImage" | recorded |  |
| mesh_shape | [64, 64] | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 19600 | recorded |  |
| operator_extent_shape | [140, 140] | recorded |  |
| operator_fft_shape | [280, 280] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| pixels_solved | 3844 | recorded |  |
| positive_only_solver_requested | "pdip" | recorded |  |
| positive_only_solver_used | "pdip" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "scheme": "constant"} | name |  |
| solve_subset_edge_zeroed | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 4096 | count |  |
| source_pixels_requested | 4096 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 100000 | recorded |  |
| transformer_chunk_size_preset | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| visibilities | 1000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>33 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/02e58de3728f4a2d6bc1` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.05338332829996943} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/13d2a8ef5f3c4723c170` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.11920346550032264} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/1db07163c0c86eb7d2e8` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.0005199705992708914} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/269e82520b7e5f1b9a70` | steps.Curvature matrix F = Aᵀ W~ A (32 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (32 blocks of 128)": 0.04884882699989248} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Curvature matrix F = Aᵀ W~0 A (32 blocks of 128)` |
| `measurement/2a2a12fbf85aff577e75` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 0.04884882699989248} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/2e27d214b5e4d614e995` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.06273236679990077} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Reconstruction (pdip)` |
| `measurement/55ca2d02df6df43714d1` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.00017177090048789979} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/57d1e1fccd70c3a7d133` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.0003392917991732246} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Mapping matrix L` |
| `measurement/5a500220f1f283af6c79` | curvature_sparse_b64.steady_per_call_s | {"curvature_sparse_b64.steady_per_call_s": 0.05262181030120701} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b64/steady_per_call_s` |
| `measurement/5c6292fe55a1bf55dd8e` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.00013756510015809907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/60a93cab821c4a2c0139` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.005797357199480757} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/767ff9b8311b0ebeeeec` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.0017138480994617566} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/7cfea8b51e3d8580b486` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.00027750519948313015} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/820fba6d873b85e7de5c` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.005797357199480757} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/9406321353e4f56a3e32` | full_pipeline_certified.steady_per_call_s | {"full_pipeline_certified.steady_per_call_s": 0.10996443119947799} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline_certified/steady_per_call_s` |
| `measurement/a7097e0a2c4460a0f69d` | curvature_sparse_b256.steady_per_call_s | {"curvature_sparse_b256.steady_per_call_s": 0.0506221549003385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b256/steady_per_call_s` |
| `measurement/a781ac6e7d99e8b0819d` | steps.Ray-trace grids (data) | {"steps.Ray-trace grids (data)": 0.00027750519948313015} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Ray-trace grids (data)` |
| `measurement/b10ba605118ebc718316` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.0015165344011620618} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/b813272eb36062b1da09` | curvature_sparse_fp32.steady_per_call_s | {"curvature_sparse_fp32.steady_per_call_s": 0.1147835434996523} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_fp32/steady_per_call_s` |
| `measurement/c25bfd34ee2434cd8082` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.00012408320035319788} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Regularization matrix H` |
| `measurement/c374ff00b0104b2bc445` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.0018379312998149544} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/c86f168d518700ba1fce` | component_total | {"component_total": 0.12041486059897578} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/total_step_by_step` |
| `measurement/cb4f6813e7c6a2c9d600` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 0.11922416320012416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/cdcb5c2b7767cea4368c` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.001374556300288532} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/d4ced95404098e7a7a8c` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.06270994290098315} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/d565f9c6fa359dc98cb6` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.00013756510015809907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/d825b9cff1de32e9bd9b` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.0010970511008054017} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/d9d4ce31cbeff26276f1` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.0005199705992708914} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/ddd34883a0ce33c9a548` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.0005408426004578359} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/e8a3b399a09539ee7287` | full_pipeline_vmap16.steady_per_call_s | {"full_pipeline_vmap16.steady_per_call_s": 0.14527321181876687} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline_vmap16/steady_per_call_s` |
| `measurement/f00839477b7f5df1bc26` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.0005408426004578359} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/fb5b707505c04e0a3608` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.0004168023995589465} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/fbfad40fea167a2d787c` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.06273236679990077} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |

</details>

### compile

<details><summary>65 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0270b9d316f03d88b3dd` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.09765075899485964} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/0a1baadd71bcfb400a41` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.06455886899493635} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/0b94033c6431fdc7e99c` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.006517237998195924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/12ae728db6256512e501` | full_pipeline_vmap16.first_call_s | {"full_pipeline_vmap16.first_call_s": 8.274980072994367} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline_vmap16/first_call_s` |
| `measurement/13b80e24bc53b30b3ea4` | curvature_sparse_b256.lower_s | {"curvature_sparse_b256.lower_s": 0.1285992420016555} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b256/lower_s` |
| `measurement/180064933eebffbce2f7` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04474651599593926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/1abbf31e6e79ef5bc49b` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.01483608600392472} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/1ebf3aed46bb35d01e56` | curvature_sparse_fp32.lower_s | {"curvature_sparse_fp32.lower_s": 0.04335076399729587} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_fp32/lower_s` |
| `measurement/242bdd71df60c825db4d` | curvature_sparse_b64.lower_s | {"curvature_sparse_b64.lower_s": 0.1351346009905683} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b64/lower_s` |
| `measurement/2f0b3a8b961bebe56dee` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.0014583009906345978} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/322c9b2dc74825c51188` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.06387144399923272} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/35e91a2906fccb91b0e0` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.000874724006280303} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/39225854f2250251fc1e` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 2.533930195000721} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/3d329c63c1634156cca5` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.6946079409972299} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/3e831857fc717e273c12` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.006299960004980676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/3f441c5fe244e0510826` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.4340730679978151} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/4cd44615c37e89cc2bae` | log_dets.first_call_s | {"log_dets.first_call_s": 0.006597237996174954} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/51db6b275f55821d0c85` | curvature_sparse_fp32.compile_s | {"curvature_sparse_fp32.compile_s": 0.22183578900876455} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_fp32/compile_s` |
| `measurement/55555551cff42ab176d6` | curvature_sparse_b256.first_call_s | {"curvature_sparse_b256.first_call_s": 0.05430014400917571} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b256/first_call_s` |
| `measurement/5787d69b7d58ed3231c8` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.06468662800034508} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/58ac5754be0390559aed` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.04185616299218964} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/5ab1412da756b85e1435` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.008826552992104553} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/6256bb6d7da9f07f1a41` | curvature_sparse_fp32.first_call_s | {"curvature_sparse_fp32.first_call_s": 0.15668544299842324} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_fp32/first_call_s` |
| `measurement/63c2ab69589510dc9294` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.2740247570036445} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/641c20f57381be9b281a` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 4.105360405999818} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/672aabde9e08a589582d` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 0.8503480490035145} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/6dcebc3d3a89995c718a` | operator_setup | {"operator_setup": 3.1246244970097905} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/operator_build_s` |
| `measurement/7062ec6ba22fced1e578` | curvature_sparse_b256.compile_s | {"curvature_sparse_b256.compile_s": 0.21120181499281898} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b256/compile_s` |
| `measurement/721178272f2c8a830ed1` | log_dets.lower_s | {"log_dets.lower_s": 0.025494297995464876} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/log_dets/lower_s` |
| `measurement/74ec26ee5fb1a927a2f4` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.014829655992798507} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/77f20a1c658a2becf6cf` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 0.06716830399818718} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/7b866de09a6ed25d090c` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.011858583995490335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/80253c78d9032d4d03af` | log_dets.compile_s | {"log_dets.compile_s": 0.14317565799865406} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/log_dets/compile_s` |
| `measurement/8094e5d32de58cb17453` | full_pipeline_certified.first_call_s | {"full_pipeline_certified.first_call_s": 0.1384535289980704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline_certified/first_call_s` |
| `measurement/82147dfc91a9799d5c24` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.05430098400393035} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/832c790ff04a0f525d49` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.0010698520054575056} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/87504655ddb289c40b08` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.6888320159923751} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/88f7aae247fa918e7f30` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.21024279200355522} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/9ac393aa30345b24020b` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.014499658995191567} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/9dde9504b72a8cfb1046` | curvature_sparse_b64.compile_s | {"curvature_sparse_b64.compile_s": 0.22013465999043547} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b64/compile_s` |
| `measurement/a43899a3da29810c13fb` | full_pipeline_certified.lower_s | {"full_pipeline_certified.lower_s": 0.8720637399965199} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline_certified/lower_s` |
| `measurement/a76021b5618d0d91f429` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.0004915540048387} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/abf6c69f4ca4a545bcf5` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.015426052996190265} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/af8569c8bdf871f5041d` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.05610732300556265} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/b5f2508b27be329cfdec` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.13661063999461476} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/be42a4c23a5a05c0c652` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.052766724009416066} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/bfababb05d41460890ee` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.0685127930046292} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/c5a94bf0a076808b0eef` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.006554438004968688} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/cbee975b733a52d7bd19` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 2.5039165759953903} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/ccbad6a90ef13c084689` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.006123021012172103} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/ce24ec457c040c011349` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.6543954249937087} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/d0801bdf7aef077dc13b` | full_pipeline.compile_s | {"full_pipeline.compile_s": 4.144033859993215} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/d156210e9ea8da2d602d` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 0.15012438500707503} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/d1995491e64533632b05` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.5042833810002776} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/d34d56b7cc838dd2774d` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.001868248000391759} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/d7271111010e1a1cf175` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.0007840350008336827} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/dda01fab7bf191e7645e` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.5045492589997593} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/e91a198a596ae7192d8c` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 2.6340387980017113} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/ed5dc94c4689105936ff` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.7832994749915088} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/f1f889f95ed86e5c70e8` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.1304342499934137} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/f28398282d13e53ecc1f` | curvature_sparse_b64.first_call_s | {"curvature_sparse_b64.first_call_s": 0.06933583799400367} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/curvature_sparse_b64/first_call_s` |
| `measurement/f811712b1620e247e005` | full_pipeline_certified.compile_s | {"full_pipeline_certified.compile_s": 4.453787029997329} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/full_pipeline_certified/compile_s` |
| `measurement/f84459062014f15a1998` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.0009855329990386963} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/f85c027096a1ccb116fc` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.08856124700105283} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/fd36c85a9fd96f5f9c46` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.06274208100512624} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/jit_phases/reconstruction_pdip/lower_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/819b8f941540782ac077` | host_peak_rss | {"host_peak_rss": 2339.06640625} | MiB | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/14604f07a25fab330489` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 0.11920346550032264} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/full_pipeline/per_call_s` |
| `measurement/a58e70bc10f6043b72a2` | single_jit_block | {"single_jit_block": 0.11920346550032264} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/n_sweep/pixelization_hpc_a100_fp64_n4096.json) `/full_pipeline_single_jit` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 20975 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 20975 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
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

- [likelihood_breakdown](../../../../scripts/interferometer/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/rectangular/likelihood_runtime.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
