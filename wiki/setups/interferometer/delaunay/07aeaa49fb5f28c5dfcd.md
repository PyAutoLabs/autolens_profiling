<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · sma

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/sma/8f53cceab87c0c05ed65`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| dense_transform | "library one-shot TransformerNUFFT.transform_mapping_matrix" | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| host_load_avg_end | [66.62, 56.36, 51.38] | recorded |  |
| host_load_avg_start | [47.01, 46.18, 47.47] | recorded |  |
| image_pixels_masked | 3852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerMapping" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.1 | recorded |  |
| pixels_solved | 1250 | recorded |  |
| positive_only_solver_requested | "pdip" | recorded |  |
| positive_only_solver_used | "pdip" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_preset | {"adapt_image": "truth lensed source (lensed_source.fits), no S/N cap", "adapt_snr_cap": null, "dataset_type": "interferometer", "instance": "iid draw 0 of the preset sequence (seeded)", "instance_unit_vector": [0.499784489007424, 0.41402443684610074, 0.41327450881714656, 0.4200167345680767, 0.5789514047288555, 0.4702516857248635, 0.567334291975647, 0.47173759315010344, 0.5239626288064884, 0.5543072472409505], "instrument": "sma", "lens_light": null, "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 0, "over_sample_size_lp_rule": null, "over_sample_size_pixelization_rule": null, "positions_penalty": null, "preset": "interferometer_hst_source_pix_2_standard", "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [256, 256] | recorded |  |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| solve_subset_edge_zeroed | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| transformer_chunk_size_preset | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| visibilities | 190 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>24 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/17f37adc2ba4a099a265` | steps.Ray-trace grids (data + mesh vertices) | {"steps.Ray-trace grids (data + mesh vertices)": 0.0009514486999250948} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace grids (data + mesh vertices)` |
| `measurement/1ac177094480f791f8fe` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 7.748133695597062} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/1bf66d1fd59647cbdef5` | transformed_mapping_matrix.steady_per_call_s | {"transformed_mapping_matrix.steady_per_call_s": 6.762609784898814} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transformed_mapping_matrix/steady_per_call_s` |
| `measurement/256a078cce53af8f0577` | curvature_dense.steady_per_call_s | {"curvature_dense.steady_per_call_s": 0.017649064201395958} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_dense/steady_per_call_s` |
| `measurement/3571f83a1c3c3e8c787b` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.001308479200815782} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/389d99682fa78fff970f` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.04674053459893912} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/39d43de014dc9d41a9b7` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.0675091079028789} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/3a16a335e82536993e1b` | steps.Data vector D (dense, Re + Im) | {"steps.Data vector D (dense, Re + Im)": 0.0011621966026723386} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Data vector D (dense, Re + Im)` |
| `measurement/3c8a8b241b5f139d5ed5` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.3756282224028837} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/5027c2290751294ed4e7` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.001308479200815782} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/5653893d23ffbb9058f6` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.3272460292035248} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/5954592569ca50c102af` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.06694065940100699} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/5ff364cf33fcb7b835d2` | data_vector_dense.steady_per_call_s | {"data_vector_dense.steady_per_call_s": 0.0011621966026723386} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_dense/steady_per_call_s` |
| `measurement/615a9276a5352a4ccd03` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.04769198329886422} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/63dad7981cea594ed6aa` | steps.Curvature matrix F (dense Tᵀ N⁻¹ T, Re + Im) | {"steps.Curvature matrix F (dense T\u1d40 N\u207b\u00b9 T, Re + Im)": 0.017649064201395958} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Curvature matrix F (dense Tᵀ N⁻¹ T, Re + Im)` |
| `measurement/6a3e57563a8c48001383` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 6.47760677569895} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/6dd87bcc32510144bc01` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.04243130779941566} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/92cb6772f377d6ee991e` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.0009514486999250948} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/a03258fe268574780c59` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.47216598450322633} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/a8973f7591bc8d81d97e` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.04243130779941566} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |
| `measurement/b723ccbf50ff6bb0b196` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.3756282224028837} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/c2abd0492d85083550e9` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.019817124604014677} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/cb5ddd562bd01515ecab` | component_total | {"component_total": 7.267729714507004} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/e95da8fd17819a790f0d` | steps.Transformed mapping matrix T (library one-shot nufft2d2) | {"steps.Transformed mapping matrix T (library one-shot nufft2d2)": 6.762609784898814} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/steps/Transformed mapping matrix T (library one-shot nufft2d2)` |

</details>

### compile

<details><summary>43 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/042edf2393fafe3eab83` | curvature_dense.lower_s | {"curvature_dense.lower_s": 0.022027848986908793} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_dense/lower_s` |
| `measurement/090c3942b01366fe6344` | data_vector_dense.lower_s | {"data_vector_dense.lower_s": 0.017372363014146686} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_dense/lower_s` |
| `measurement/098497906ff929fc69b3` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.48128256504423916} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/0d26c2a6b680e3236262` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.1229012290132232} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/170cad614d2eb6f3d7db` | transformed_mapping_matrix.lower_s | {"transformed_mapping_matrix.lower_s": 0.14013029698980972} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transformed_mapping_matrix/lower_s` |
| `measurement/1aa80d5455f2e307f25a` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.4353899930138141} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/20c84dbdf9db7e892b2c` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 1.6448846509447321} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/2231b2e3d686655be0d1` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.4039095100015402} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/22f85141ecd167cecf1b` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.002677159965969622} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/2bd85def3e2d3a09954d` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.49145174300065264} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/2be465771857cf45a655` | data_vector_dense.first_call_s | {"data_vector_dense.first_call_s": 0.00632228801259771} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_dense/first_call_s` |
| `measurement/43275916fb2ca58e4c7e` | curvature_dense.first_call_s | {"curvature_dense.first_call_s": 0.022580388991627842} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_dense/first_call_s` |
| `measurement/43dea92cc945a0583400` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.40116076898993924} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/44b348d36a426f8d91a5` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 1.335860461986158} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/46f26f697899f43d8d8c` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.792338973027654} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/47f85a4987e8122c0452` | curvature_dense.compile_s | {"curvature_dense.compile_s": 0.11479890794726089} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_dense/compile_s` |
| `measurement/526936f90270871e2ba5` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 2.110217610024847} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/5a39951acd67a1cf4582` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 1.3697079989942722} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/5c61e46c94fa9c60d5b9` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.6237810899619944} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/5cd96cd879317409fe17` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.2192969579482451} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/6240131bf33ef40e0be1` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.2914246270083822} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/631b8ecf8b767f3fae21` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 5.689152726030443} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/7489be9c9163bbd4a659` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.051597585959825665} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/75bb000495a526667a3d` | transformed_mapping_matrix.first_call_s | {"transformed_mapping_matrix.first_call_s": 7.588054038002156} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transformed_mapping_matrix/first_call_s` |
| `measurement/78ccbbd340f5f6afa681` | full_pipeline.compile_s | {"full_pipeline.compile_s": 2.1803150109481066} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/7b9044d4123fa8fabbc5` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 1.1472623840090819} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/7f9aa84af80fa8547743` | log_dets.first_call_s | {"log_dets.first_call_s": 0.0368297029635869} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/873e0edd975c56ab28bf` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 1.4128137890365906} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/89105b57971325238e5b` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.5004546699929051} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/9299848ba56e066c6816` | log_dets.compile_s | {"log_dets.compile_s": 0.11116211995249614} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/935879a24e31f6d8878b` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.2347575330059044} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/a70ebd4cd0cb9bae4c00` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 6.950842412014026} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/a9e1c9bf6050b32d0df5` | log_dets.lower_s | {"log_dets.lower_s": 0.04385866195661947} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/aa21f23274fae873c876` | data_vector_dense.compile_s | {"data_vector_dense.compile_s": 0.0757884859922342} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_dense/compile_s` |
| `measurement/ca46db7b68995c68f43c` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 1.2448249639710411} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/cc078bfe6e7e351280d5` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.003657307999674231} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/cddc8b102a9ee699aab6` | transformed_mapping_matrix.compile_s | {"transformed_mapping_matrix.compile_s": 0.40194133296608925} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/transformed_mapping_matrix/compile_s` |
| `measurement/dff5117b1ac6369c71a5` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.07750521705020219} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/eac83c491a52218a958f` | operator_setup | {"operator_setup": 0.0} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/operator_build_s` |
| `measurement/ec47c4d4b47587f0c921` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.10399290802888572} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/f189b2b106f9f8d0a57c` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.06179182598134503} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/f504377be5278034a3af` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.08996778901200742} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/ff12e39f85120faae3f8` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.029384805005975068} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/667f56d8d8cd2a0df6ca` | host_peak_rss | {"host_peak_rss": 11118.37890625} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/57c6334dd87d9c633a8c` | single_jit_block | {"single_jit_block": 6.47760677569895} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/full_pipeline_single_jit` |
| `measurement/65e6a46eb896b5618183` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 6.47760677569895} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/sma/jax_cpu_dense_fp64_nufft/hpc_ral_cpu/breakdown.json) `/full_pipeline/per_call_s` |

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
    "measured_at": "2026-10-28T13:53:37Z",
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
    "measured_at": "2026-10-28T13:53:37Z",
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
