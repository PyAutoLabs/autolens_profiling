<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/alma_high/dce9df841d7a5fc3a751`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 10 | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| host_load_avg_end | [5.88, 2.37, 0.89] | recorded |  |
| host_load_avg_start | [2.33, 0.65, 0.22] | recorded |  |
| image_pixels_masked | 61572 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 78400 | recorded |  |
| operator_extent_shape | [280, 280] | recorded |  |
| operator_fft_shape | [560, 560] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.025 | recorded |  |
| pixels_solved | 1250 | recorded |  |
| positive_only_solver_requested | "pdip" | recorded |  |
| positive_only_solver_used | "pdip" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_preset | {"adapt_image": "truth lensed source (lensed_source.fits), no S/N cap", "adapt_snr_cap": null, "dataset_type": "interferometer", "instance": "iid draw 0 of the preset sequence (seeded)", "instance_unit_vector": [0.499784489007424, 0.41402443684610074, 0.41327450881714656, 0.4200167345680767, 0.5789514047288555, 0.4702516857248635, 0.567334291975647, 0.47173759315010344, 0.5239626288064884, 0.5543072472409505], "instrument": "alma_high", "lens_light": null, "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 0, "over_sample_size_lp_rule": null, "over_sample_size_pixelization_rule": null, "positions_penalty": null, "preset": "interferometer_hst_source_pix_2_standard", "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| solve_subset_edge_zeroed | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 1000000 | recorded |  |
| transformer_chunk_size_preset | 1000000 | recorded |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 5000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>28 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/04280d200c03d973b401` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.00019582369714044034} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/05dfac02e5255cd1278d` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.00011847260175272823} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/0c14b9ecfcb5e562d81e` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.00633339190389961} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/1bca71172d2966f69907` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.006604179699206725} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/282cb642f6043dfc1a44` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.004969130299286917} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/29e9f2f64adf43ee27ab` | steps.Curvature matrix F = Aᵀ W~ A (10 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (10 blocks of 128)": 0.05353067349642515} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (10 blocks of 128)` |
| `measurement/316fc43623c1305a9fb9` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0003394709958229214} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/53bf781da6aeaa189841` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 0.08311265749507583} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/6c48a76eb834d57a58de` | component_total | {"component_total": 0.08472513110027649} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/total_step_by_step` |
| `measurement/8e769bb98d91786c17a6` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.0017804217000957578} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/9d01dc3b9233472e843e` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.006800003396347165} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/9e0fa7887a19584e5894` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.0017804217000957578} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/a819b75009357f53a555` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.02137662350432947} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/ab079fe8519225011cec` | steps.Ray-trace grids (data + mesh vertices) | {"steps.Ray-trace grids (data + mesh vertices)": 0.0003394709958229214} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Ray-trace grids (data + mesh vertices)` |
| `measurement/ad9aebab8c646293fa7b` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.0008844025025609881} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/b4c8c39d804f6a06f442` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.0002707877953071142} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/b4eaaca48af8d16bfe88` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 0.0832129640970379} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/ca34e90f2dd24d56105f` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.00011847260175272823} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/d03e6024bc5569d7ee93` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.0006030268035829067} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/e033ee535447a38d76fb` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.0003849853994324803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/e363bf8cc3dbd2692eea` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.005993920908076689} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/eb75a98b60885ffc6d26` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.021507547702640295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/f076167a3174d14d0d3d` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.00026724979979917405} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/f5322a605075439a304b` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.0006030268035829067} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/f7da86805d5a9b681e9d` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.021507547702640295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/fb5dc7cf9840deede7a8` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.0003849853994324803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/fc035bff477b8ae6857a` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 0.05353067349642515} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/ffaf64afee6cbf73f05f` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.006883538397960365} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |

</details>

### compile

<details><summary>52 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0b3195c5374dadc72e8a` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.7965971480007283} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/0e5cfba001bed394bd94` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.6395493429736234} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/19366b05750aa877dc8f` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 0.8515427180100232} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/1bfc63cc9b0b2c53afe2` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.014887760975398123} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/1ea6bcd0a5f40b307cf7` | log_dets.lower_s | {"log_dets.lower_s": 0.02598347800085321} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/22e1d4fbe3f554846c08` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.010751320980489254} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/2aea613430b1dd398173` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.08944121701642871} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/33a499b1b086adb09c1b` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.18575981998583302} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/37e0dfbe867e90c8903b` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.0727477929904126} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/383211d280b967854a62` | full_pipeline.compile_s | {"full_pipeline.compile_s": 3.448093041020911} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/4047517a2b5339213551` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.002736358030233532} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/413db02c28bc039f2a17` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.009485346032306552} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/4c36e1edbcf95940e9ee` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 1.5945785579970106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/4dea8dba9d2790c1b15d` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.059219539049081504} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/517db5a9336b76ea3e18` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.0731555109960027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/5984e0457f821924b365` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.02343131898669526} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/5e93e26bbfd1d3adedcb` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.04153048200532794} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/5fc6ecbb1fcd873e6754` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.0013532539596781135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/611fcb19380962b0968f` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.015797530009876937} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/656b73e2f0dbc9a0bd74` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.3601127869915217} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/6d57b69abe68aabf3fd3` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.0639643520116806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/6edd7442d94b455d32ec` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 2.9839244509930722} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/6ff60413de2785fe43c8` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 0.05925150797702372} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/702f3c6abe2d04bf32f0` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.006158715987112373} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/766db1732165770f4b6f` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.000846438982989639} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/7800ab06e62a02667807` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.01487183099379763} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/7af9509cb92d111f549b` | log_dets.compile_s | {"log_dets.compile_s": 0.1280202929629013} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/7d6463859fd8506c40d3` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.05214208399411291} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/87e898c8422cb36c5bcd` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 0.9818091249908321} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/880a77533d218d8cd1aa` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.012491020024754107} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/8a2bab74aba1cf541a9f` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.2375225389841944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/902d78ed0419e66da169` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.001296473026741296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/9530b70d919e1925a606` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 1.668331143038813} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/974af308465d3c9123a2` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 0.0933367499965243} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/98e2b084f5f35e9ff575` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.15855837700655684} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/9bf0facfe9138f7de433` | operator_setup | {"operator_setup": 4.32394121697871} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/operator_build_s` |
| `measurement/9cdf04cb77337e033e43` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.060893048997968435} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/a05fd3b9918606996701` | log_dets.first_call_s | {"log_dets.first_call_s": 0.0026043179677799344} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/aebafde7186cf59a5154` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.32700444501824677} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/b51d64c258e75f58f1aa` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.5469957530149259} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/bc5ea120a017890be135` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.010374795994721353} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/bfaa496574835b867cfb` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0008574890089221299} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/bfbb99b2d4b693e1fa8f` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.045988327998202294} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/c2fcc896b50a620b01f3` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.743601634982042} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/c3867c897fe8b4616b30` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.023242861032485962} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/c74aa7b0bb0c83fa075c` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.015445954981260002} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/ccc1405ad6ae7858cfcd` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 2.6633335609803908} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/ce70bcbf413083344cc8` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.1175591089995578} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/cf5b24042109260cdad7` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.008995802025310695} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/dd65ff0fdd7c0173e693` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 0.09416952898027375} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/e7e245e1ad08832b0462` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.00897296197945252} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/ed4410604d6c92b5606a` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.0014722010237164795} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/86fbf14a7dd33babc081` | host_peak_rss | {"host_peak_rss": 3380.6328125} | MiB | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>3 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0e6f5a207f1b48876692` | single_jit_median | {"single_jit_median": 0.08296770500601269} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/full_pipeline_single_jit_median` |
| `measurement/361207a46b65c5fa4160` | single_jit_block | {"single_jit_block": 0.0832129640970379} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/full_pipeline_single_jit` |
| `measurement/fbe2b403a12283ce66cd` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 0.0832129640970379} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_gpu_sparse_fp64_wtilde/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/full_pipeline/per_call_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 33257 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
      "autolens_profiling": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "autolens_profiling.library_revisions": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "finufft.dependency_versions": "2.5.1",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-28T13:51:53Z",
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
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 33257 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
      "autolens_profiling": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "autolens_profiling.library_revisions": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "finufft.dependency_versions": "2.5.1",
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
    "measured_at": "2026-10-28T13:51:53Z",
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
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 33257 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
      "autolens_profiling": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "autolens_profiling.library_revisions": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "finufft.dependency_versions": "2.5.1",
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
    "measured_at": "2026-10-28T13:51:53Z",
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
- [likelihood_runtime_numba](../../../../scripts/interferometer/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/interferometer/delaunay/quick_update.py)
- [_streaming](../../../../scripts/interferometer/delaunay/_streaming.py)
- [streaming_accumulate](../../../../scripts/interferometer/delaunay/streaming_accumulate.py)
- [streaming_in_memory](../../../../scripts/interferometer/delaunay/streaming_in_memory.py)
- [streaming_parity](../../../../scripts/interferometer/delaunay/streaming_parity.py)
- [streaming_plot_scaling](../../../../scripts/interferometer/delaunay/streaming_plot_scaling.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
