<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · jvla

[Model index](index.md)

Exact setup ID: `interferometer/rectangular/jvla/15ff09b9b2240ec7e323`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 10 | recorded |  |
| edge_zeroed_pixels | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| host_load_avg_end | [62.43, 62.03, 60.34] | recorded |  |
| host_load_avg_start | [63.98, 59.0, 53.64] | recorded |  |
| image_pixels_masked | 384852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 3 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 490000 | recorded |  |
| operator_extent_shape | [700, 700] | recorded |  |
| operator_fft_shape | [1400, 1400] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.01 | recorded |  |
| pixels_solved | 1089 | recorded |  |
| positive_only_solver_requested | "pdip" | recorded |  |
| positive_only_solver_used | "pdip" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_preset | {"adapt_image": "truth lensed source (lensed_source.fits), no S/N cap", "adapt_snr_cap": null, "dataset_type": "interferometer", "instance": "iid draw 0 of the preset sequence (seeded)", "instance_unit_vector": [0.499784489007424, 0.41402443684610074, 0.41327450881714656, 0.4200167345680767, 0.5789514047288555, 0.4702516857248635, 0.567334291975647, 0.47173759315010344, 0.5239626288064884, 0.5543072472409505, 0.5399346262699198, 0.5099305691842051], "instrument": "jvla", "lens_light": null, "memo": "off", "mesh": {"class": "RectangularBilinearAdaptImage", "free_weights": {"weight_floor": {"lower": 1e-05, "type": "LogUniform", "upper": 1.0}, "weight_power": {"lower": 0.0, "type": "Uniform", "upper": 10.0}}, "kind": "rectangular", "shape": [35, 35], "source_pixels": 1225}, "mesh_kind": "rectangular", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 0, "over_sample_size_lp_rule": null, "over_sample_size_pixelization_rule": null, "positions_penalty": null, "preset": "interferometer_hst_rect_adapt_standard", "provenance": "subhalo_validation/scripts/pipelines/rectangular_adapt.py (rect_adapt source_pix[2], job 342311): RectangularBilinearAdaptImage shape (32, 32) (MESH_PIXELS_YX :38), regularization Adapt; the rest of the stage (adapt S/N cap, over-sampling, MGE, threads) as hst_source_pix_2 \u2014 likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: RectangularBilinearAdaptImage 35x35, adapt regularization; borrows the hst preset's mesh and regularization only", "regularization": {"coefficient": 1.0, "free": false, "scheme": "adapt"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: RectangularBilinearAdaptImage 35x35, adapt regularization; borrows the hst preset's mesh and regularization only", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | {"coefficient": 1.0, "free": false, "scheme": "adapt"} | name |  |
| solve_subset_edge_zeroed | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1225 | count |  |
| source_pixels_requested | 1225 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 1000000 | recorded |  |
| transformer_chunk_size_preset | 1000000 | recorded |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 25000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>28 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0ce40228e5bca1189456` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 5.556296328994601} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/1d89bcbe7452086a8f8d` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.004940661330086489} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/28a99bfe88e11f5ba51c` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.2974266986711881} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/374ddc34f506ddc73e7d` | component_total | {"component_total": 59.70516431562525} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/total_step_by_step` |
| `measurement/38687dbd970326ce8f0c` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.5454064103153845} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/41fbe5e7fcfe87292cc8` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 57.812496958319876} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/56cfabf824c8de9923d9` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.4262037830000433} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/5cd83aafd46692ecbec2` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.05296668866261219} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/5f92328d68b88fb7ad75` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 1.5992274349943425} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/6bf34dc3c3d25d2ce173` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.014318392340404293} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/7ac0af1d1fbc5ead36c6` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.18294725365315875} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/850899a722bc771bc37c` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.004940661330086489} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/8584ea7fb19df6a6c875` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 56.96412700866737} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/907bd55ae88a75571545` | steps.Curvature matrix F = Aᵀ W~ A (10 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (10 blocks of 128)": 56.25278732966399} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (10 blocks of 128)` |
| `measurement/9e19df81aeaa9da3d609` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.4405221753404476} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/a0d0d396e3fad6c788fe` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.029916014328288536} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/a6808b6a7c006f18f625` | steps.Ray-trace grids (data) | {"steps.Ray-trace grids (data)": 0.014318392340404293} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Ray-trace grids (data)` |
| `measurement/a7c0738717b963346d8b` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 56.25278732966399} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/af2ac5a51c7b1daf98bb` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 1.5992274349943425} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/bacc822996ea4e2fa227` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.029916014328288536} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/cb7bed671279483cc3ff` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.04776869832615679} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/cfcbc95f361fa7166406` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.3304644813373064} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/e1ea152938450f0b2bcb` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.28044086432782933} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/e367e6b6bec6c4d94fd2` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.18294725365315875} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/e95096a0c229865ad197` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 1.2833552843270202} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/ecc1fd4fcec49a112dca` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.35199033732836443} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/ece8be1186ad09b457cf` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.35199033732836443} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/f047fcb598ba5db18df6` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.9859285856558321} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |

</details>

### compile

<details><summary>52 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/013c42ba27e4f3416217` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 1.0027005929732695} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/0999a877be50231dc349` | log_dets.compile_s | {"log_dets.compile_s": 0.10559702903265134} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/0be16e6bb9a28299fbc7` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 1.6039544219966047} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/0dfca797b5b1b55328f4` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.020341539988294244} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/1e2d759dc4d76b84d1d7` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 56.038226404984016} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/2d7c443ed59940d6fc71` | log_dets.lower_s | {"log_dets.lower_s": 0.027883510978426784} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/2e15eb466ca854348d3b` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.019069546018727124} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/34bdddd3b158eb55798c` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.07691354403505102} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/3ae8d8cb0a68fdc75272` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 0.9420867890003137} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/3b5aae45fac00fdf6f4f` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.02180629502981901} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/3baab1a289096f0883d1` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.09206228796392679} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/3cc35c411cd821dafb07` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 1.0337580280029215} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/484c9f4a75857bf6d0ee` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.48882785299792886} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/4a6cfda8b53f0c6e1d96` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.29966268403222784} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/4c2ba7e9185462b522de` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.4378986220108345} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/4e6896409897809371e8` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 2.862233214022126} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/52640f6474d345766772` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.4953603360336274} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/53077fe0b4ae1d8f23c7` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.10194498003693298} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/59656d3eea4e91df0769` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.5959346530144103} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/5a72b2742c08b7e0692d` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.01456223998684436} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/5a796db4ebddda9fc9d6` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.07062744902214035} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/5bffd4986743081b8c7b` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.8762825390440412} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/5c7263e7544a752d02b5` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.0518637279747054} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/5e6b7cb55a99426488da` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.2346969300415367} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/60dc69707ab9c271711c` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.4173122080392204} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/6cb3e5cadb31a3aaa7e7` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.011910742032341659} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |
| `measurement/744da4c38252ef8e7bb7` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 5.798094747995492} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/7a02bea3ed35f169fd36` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.39833099400857463} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/7cee32f6f2a56ace5bf0` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.9995289939688519} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/83de57eb6b78fb8ea804` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 1.3105402420042083} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/8c8eff874ff5c1ccac54` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.17028164700604975} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/94a7b8cb4ffcab25cd29` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.1165477960021235} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/97777f17e69d7874168e` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 52.745095122023486} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/a4997e460c84d5b987e7` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.004124857019633055} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/a4b4a1b63966108a3bd1` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.05006743501871824} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/b534e3627b7860587c46` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 1.564583542989567} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/b845b2bab9a0a650db9e` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.09485516999848187} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/bc79ba01c926e7a1e50f` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.0442367930081673} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/c6823cc5257f7cac5229` | full_pipeline.compile_s | {"full_pipeline.compile_s": 3.071220457961317} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/ca00ad60b464082b8697` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.3441941849887371} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/cef0d04e2717fc7752bd` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.5693424380151555} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/cf8daeacb55ce3111b52` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.07190025201998651} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/da84fa566b9c9b20474d` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.007276695047039539} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/dd5a73ba0d55e0b54247` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 60.6847237169859} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/e1d1987b97acbe4fed76` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.2413897350197658} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/e64c3bee7ea19a923b74` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.19942799699492753} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/eb780316999f877948e6` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.059149936016183347} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/ef25d2297ea550f77f6d` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 1.6135388709953986} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/f10a6b7af7948237db61` | operator_setup | {"operator_setup": 133.51837459398666} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/operator_build_s` |
| `measurement/f1269e727f55a33de9c4` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.07968452799832448} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/f5c4e5b01c2953e0ab4c` | log_dets.first_call_s | {"log_dets.first_call_s": 0.04559617000631988} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/f7a1688be2901fb77e02` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.019250758981797844} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/2a8f82e2c56878dcc672` | host_peak_rss | {"host_peak_rss": 15684.17578125} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0ee321ae13570bb4db19` | single_jit_block | {"single_jit_block": 57.812496958319876} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/full_pipeline_single_jit` |
| `measurement/612c285429d15a25ec8b` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 57.812496958319876} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/rectangular/jvla/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/full_pipeline/per_call_s` |

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
      "cpu_count": 252,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-1",
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
    "host": "euclid-ral-compute-1",
    "measured_at": "2026-10-28T14:16:22Z",
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
      "cpu_count": 252,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-1",
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
    "host": "euclid-ral-compute-1",
    "measured_at": "2026-10-28T14:16:22Z",
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

- [likelihood_breakdown](../../../../scripts/interferometer/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/rectangular/likelihood_runtime.py)
- [likelihood_runtime_matrixfree](../../../../scripts/interferometer/rectangular/likelihood_runtime_matrixfree.py)
- [likelihood_runtime_numba](../../../../scripts/interferometer/rectangular/likelihood_runtime_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
