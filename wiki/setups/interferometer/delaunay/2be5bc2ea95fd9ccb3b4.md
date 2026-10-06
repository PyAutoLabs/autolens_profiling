<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/alma_high/c667b1ccd25fb1b565e2`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 12 | recorded |  |
| delaunay_vertices | 1500 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| hilbert_pixels | 1500 | recorded |  |
| host_load_avg_end | [5.93, 2.38, 0.89] | recorded |  |
| host_load_avg_start | [1.58, 0.38, 0.13] | recorded |  |
| image_mesh | "Hilbert(weight_power=1.0, weight_floor=0.0)" | recorded |  |
| image_pixels_masked | 20108 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 2.0 | recorded |  |
| mesh | "Delaunay" | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 25600 | recorded |  |
| operator_extent_shape | [160, 160] | recorded |  |
| operator_fft_shape | [320, 320] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.025 | recorded |  |
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
| transformer_chunk_size | 1000000 | recorded |  |
| transformer_chunk_size_preset | 1000000 | recorded |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 5000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>29 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0181dad9b0a5f32d1b70` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.005928221601061523} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/01d121c5983027e6b4ff` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.02532310939859599} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/0ebc2bbab3de8796e054` | full_pipeline_vmap16.steady_per_call_s | {"full_pipeline_vmap16.steady_per_call_s": 0.04005998923130392} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/full_pipeline_vmap16/steady_per_call_s` |
| `measurement/13cb99929922df868865` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.00015965600032359363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/2134354ca68241246585` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.00040912149706855415} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/2a956d27e1be572af703` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.006655128096463158} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/2f06726463143ef89d9e` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0003501608967781067} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/34091e752503f8205503` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.00040912149706855415} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/35d17a1872404035f9c1` | steps.Curvature matrix F = Aᵀ W~ A (12 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (12 blocks of 128)": 0.020735609700204806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Curvature matrix F = Aᵀ W~0 A (12 blocks of 128)` |
| `measurement/37351111f13c09946f52` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.02532310939859599} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Reconstruction (pdip)` |
| `measurement/54c3a2769f97fc49ac0e` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.009529180498793722} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/55af621889af4563a52b` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.0021120948949828744} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/6a21a7d6b1c79fadf2a6` | component_total | {"component_total": 0.056869532592827454} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/total_step_by_step` |
| `measurement/75c24f6264fac455a574` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.00627838249783963} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/885ce63b25b4ed616641` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.025069161097053438} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/8a06512cd9e7326f0cce` | steps.Ray-trace grids (data + mesh vertices) | {"steps.Ray-trace grids (data + mesh vertices)": 0.0003501608967781067} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Ray-trace grids (data + mesh vertices)` |
| `measurement/8a0cd095555974a0cda5` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.0021120948949828744} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/90473f0a46a750e6407b` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.0011262361018452797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Regularization matrix H` |
| `measurement/9053e3dfb5fe85b5d024` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.00034857690334320066} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/940f8485f222a659a92b` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.00037674559862352873} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/steps/Mapping matrix L` |
| `measurement/ae1f00a489ba464f764b` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.00015965600032359363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/c29695673893ee0668b9` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.00014908710145391523} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/c75f54da244c887d5e3a` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 0.05497387039940804} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/d0c3cea5dfc9f8875572` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.00034728689934127033} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/d12c9c547f86e3e03193` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.007781364198308438} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/d30bd572e245812b7ebe` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 0.020735609700204806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/d3803d29f98569c367d9` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.05410261179786176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/e82c1315b6e1a24e567f` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.001953702000901103} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/f2d893cca37d8c1d5cf1` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.00034857690334320066} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |

</details>

### compile

<details><summary>53 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/013a37c87a55efd6b0f2` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.05304940399946645} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/01e60b4840ecb54a3d7f` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.000901404011528939} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/06eda997ca993f54fe6c` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 0.0847950599854812} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/113bd5cb57496d67f24f` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 3.432007858995348} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/12e05789f69af0142432` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0008947550086304545} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/19818b24cb337cb66e41` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.2327317310264334} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/1d5b3c57ff77f89906da` | log_dets.first_call_s | {"log_dets.first_call_s": 0.0028782329754903913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/2275c3e7ede882004d66` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.04169101297156885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/251f66f7806a85a0e723` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.937733982980717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/25d43c443a7fc30fd309` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.01122278202092275} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/2e8e5594b8c7e4f405e8` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.810185685986653} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/2f6a60ebe9accc6dfbe3` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.015458095993380994} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/31285400d0b32ddb3fca` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.03252791002159938} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/32751626fad430b3cc68` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.027247381978668272} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/36d6a21c9dbd54c285a2` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 0.027347852010279894} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/37f1dc7f2e4e1fa959f4` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.09254027099814266} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/3a73e1aada44d5fcffb7` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.014083533955272287} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/3c19fa4a405689a7863f` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.562188939016778} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/3f1d887c9a82da04da0d` | log_dets.compile_s | {"log_dets.compile_s": 0.15260340203531086} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/log_dets/compile_s` |
| `measurement/3f5d8ee99add95941225` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.01484158803941682} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/559b8eb2176deff26991` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.8649297090014443} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/5e67a73b606af67acfdf` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.012951009965036064} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/5fe46128e856463dbdce` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.1371328390087001} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/61cd05fab9b2edb8cf1e` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.09559595299651846} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/650de90ccdeff161b82e` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.0007447350071743131} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/6a2c0cb2750c35daa8b5` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 2.0246910589630716} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/7898a3a11e37bda5abda` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.19224179000593722} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/7e23b76f0007429cdcfd` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.0272546429769136} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/7edf2ce2cf67040daa0f` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.26081387902377173} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/80958dfdd5da2e850d1e` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.010378705977927893} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/809e3ea5b630d6137dfd` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.0008798540220595896} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/832d086fe58e59f8f425` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 2.049270326970145} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/83d0a748f6cd21c77b91` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.8160809780238196} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/855bcc71079c6dfdc5d1` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.1609590009902604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/8bd591b8d9bd6cf40a8c` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.055636228004004806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/8dc3926a77b45d220bd9` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04569726000772789} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/97e0a9e698c83346ec1d` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 4.453029889962636} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/9d27f6ed1ae54371beeb` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.0014665820053778589} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/a72c07b0c8687c1592dc` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.06522014900110662} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/b1163dd8188b4674ec4c` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.7798261130228639} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/b9a2e7055593abb484b8` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0023585259914398193} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/bb67185c55e2f2594b37` | full_pipeline_vmap16.first_call_s | {"full_pipeline_vmap16.first_call_s": 8.000864817004185} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/full_pipeline_vmap16/first_call_s` |
| `measurement/bc43deff5dec6a29121a` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.006287860975135118} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/be2ad693f8ca0beb465c` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.010735563992056996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/c75450074a53e5936b69` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.012360354012344033} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/ca7b2273aafcb1cdeff8` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.06345382001018152} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/cd996dba773c2a32efea` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.5428183680051006} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/cfa7e6a2f36a92a29f8a` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.012234943977091461} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/d1fc41f91652bf4223ea` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.06269329501083121} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/e129b044e1aff2ae2920` | log_dets.lower_s | {"log_dets.lower_s": 0.026564767002128065} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/log_dets/lower_s` |
| `measurement/e54aff0d73e23f514f75` | full_pipeline.compile_s | {"full_pipeline.compile_s": 4.409705044992734} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/ea8a05b07a1eafd77c79` | operator_setup | {"operator_setup": 4.879738799994811} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/operator_build_s` |
| `measurement/f88a00d61384469808e3` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.2237635560450144} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/jit_phases/curvature_sparse/compile_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/23c14b0d41656693462e` | host_peak_rss | {"host_peak_rss": 3676.15625} | MiB | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/24fd44d9439705c616a6` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 0.05410261179786176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/full_pipeline/per_call_s` |
| `measurement/cf7402d5bf74385af409` | single_jit_block | {"single_jit_block": 0.05410261179786176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/alma_high/delaunay_hpc_a100_fp64_r2.0.json) `/full_pipeline_single_jit` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 33257 MiB, 81920 MiB",
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
    "measured_at": "2026-09-28T19:31:05Z",
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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 33257 MiB, 81920 MiB",
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
    "measured_at": "2026-09-28T19:31:05Z",
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
