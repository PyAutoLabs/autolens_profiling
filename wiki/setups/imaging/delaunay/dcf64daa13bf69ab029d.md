<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · euclid

[Model index](index.md)

Exact setup ID: `imaging/delaunay/euclid/be5290b8d1780fbc4f63`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | "subtracted S/N (truth tracer), floored" | recorded |  |
| adapt_minimum_percent | 0.01 | recorded |  |
| delaunay_vertices | 1280 | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| image_pixels_masked | 3841 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_source | "truth" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_adapt_snr_cap | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 4, 2]} | recorded |  |
| over_sample_size_pixelization_rule | {"pixels_above": 628, "pixels_below": 3213, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"} | recorded |  |
| over_sampled_pixels | 15664 | recorded |  |
| over_sampling_preset | "euclid_vis_pix_standard" | recorded |  |
| oversampled_pixels | 15664 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.1 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": null, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 4, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 628, "pixels_below": 3213, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "euclid_vis_pix_standard"} | recorded |  |
| production_preset | {"adapt_snr_cap": null, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_40_linear_2_basis", "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 20, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 4, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": {"factor": 3.0, "minimum_threshold": 0.2}, "preset": "euclid_vis_pix_standard", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "euclid_strong_lens_modeling_pipeline scripts/initial_lens_model.py:538-543 galaxy_name_image_dict_via_result_from (subtracted S/N, uncapped) for the Hilbert mesh; adapt_minimum_percent 0.01 (config/general.yaml:22)", "model_stand_in": "the dataset's truth tracer stands in for the previous stage's maximum-likelihood model", "over_sampling": "initial_lens_model.py:609-626: sub-size 4 where the source S/N > 3.0, else 2"}, "provenance": "euclid_strong_lens_modeling_pipeline/scripts/initial_lens_model.py (vis_pix stage, job 342301): mesh :613-615, image mesh :392-396, edge :448-455, regularization :617, over-sampling :498-511, sparse operator :364-365, positions :544-546, MGE :162-168, threads euclid_dr1_prelim/hpc/batch_cpu/submit_initial_lens_model_two_stage:48,161-166; light-profile radial bins util.py:931-937 ([4, 4, 2] since euclid_strong_lens_modeling_pipeline#56, 2026-09-08) \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; standard size (human ruling 2026-10-23): Hilbert 500 -> 1250 (+30 edge)", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; standard size (human ruling 2026-10-23): Hilbert 500 -> 1250 (+30 edge)", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1320 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>32 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00fa9a0cdd2c06459b45` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.011471077904570848} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/0e5a2f44d51fc9a5fea0` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0006230785977095366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/0ffd537965e80d998e5f` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00015462719602510333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/1d9930d77f52efb37e55` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.00032825320959091204} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/2c8d880a9b5f138f1334` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00012733250623568891} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/43ea7c2a7f00c8c77043` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0006230785977095366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/4b212f54aba0fc606760` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.028720414801500737} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/4be5ce488b8d261a0683` | component_total | {"component_total": 0.044848221412394196} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/6db49b16b22b3d6e5153` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00022157729836180806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/71e9e4b889520e1ae837` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00016493300208821892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/72d0a755f97d0bc5659f` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.000195759697817266} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/7c93f02f281db1c28baf` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00016493300208821892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/7f72991ac87fd06c20cc` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.0009500104933977127} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/80a54426e0660db3d4e6` | interpolator_prefix | {"interpolator_prefix": 0.006567775493022054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/interpolator_prefix_s` |
| `measurement/83fcaf9cf9f1b1cd3a37` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.006896028702612966} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/8e7abf11a0902e7aadef` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.006567775493022054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/99664f47682372833d89` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.0017309803981333971} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/9f167bc3e61747c16134` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.0011101868003606796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/aec9fb3c87a163fceacc` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00015462719602510333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/b446a11269969454995d` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.028729410492815077} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/be0a4077a8c8db144901` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.028720414801500737} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/c3a4729578e994de4914` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.011471077904570848} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/c4cd97b38d5cdbd6d834` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0009409566991962492} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/c9a8c9e568715ead7562` | steps.Data vector (D) | {"steps.Data vector (D)": 0.000195759697817266} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Data vector (D)` |
| `measurement/cfb944dcdd94db235af9` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.006896028702612966} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/cfdbc8fc0fd0370f6b34` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.001282376702874899} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/d05923e63206abd993ed` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0010001479997299612} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/dd30dca99788ed2cfa47` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0026587542030029} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/e36ed20e2d47ebbdfca7` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.0011101868003606796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/f12c82d65376f0f46258` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0017309803981333971} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Mapped recon + log evidence` |
| `measurement/fbe7ee9686b1ee96a5f7` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00022157729836180806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/fd1a90631c81727e0e0f` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00012733250623568891} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Profile-subtracted image` |

</details>

### compile

<details><summary>54 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0001c0d86a1b4cfe245b` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 6.753464355948381} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/00081d2950915e214006` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.025045880931429565} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/00992f80d923c25cd44d` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0002745359670370817} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/02b4e655ab1ff36eeede` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.03877802693750709} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/02cf5bae7c7e4298a0f2` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 3.2765708359656855} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/088baeb752842c26278e` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.012048115953803062} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/08b626fcb6aaeb042126` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.09254805301316082} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/12241f839edb9041db5a` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.08725496800616384} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/123cd1fd630ee76f6c55` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.004417886026203632} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/1559bd6fc86e2ba4c80b` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 8.479924872517586e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/171e8cd819c40a8d88af` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.013482128968462348} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/21c363bc10654cdaa17d` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.042557402048259974} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/261fb94edbba2d3ff571` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.13964156992733479} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/2cfda7a6b668705f8885` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.030584864085540175} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/2e241e42f1f7dfd23ce1` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.13118961197324097} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/321a794752f0893ec3e3` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 1.9699500040151179} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/3353b1ee3b451c7530ec` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006140240002423525} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/349d2a773112fc634c4f` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.43102897703647614} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/429da04d8a9111130013` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.04953984695021063} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/49761fb330613bc971cf` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.006681961007416248} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/56ffe2a93eec9ab5123e` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.003134232945740223} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/59bbcb2044ee11a9aab4` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.01877356506884098} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/5a6be789194238f82953` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.010508093982934952} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/770755111c4222d5362b` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.011761148925870657} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/7d2a5138111d9191903e` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0015187308890745044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/7d35cc788fc65a5222f7` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.10619024105835706} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/82aa374fe609de2c617f` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.007909656036645174} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/82f28cf4ddcd90673648` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.0781924860784784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/8852838ee3f92665c30b` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.030435527092777193} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/8bdf401aa77e4f257777` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.000846849987283349} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/8d720ab9bd746e0f28c3` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.7098769550211728} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/92a33cbcd9f3fe6777f8` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.4220219359267503} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/9743719e2fd0f1976962` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.49990976497065276} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/a767f73812c418fc68c8` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.02723742404486984} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/acc688a604e31f6ce284` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.00583801104221493} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/accb3b2dd64512bf80f4` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.07546317693777382} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/b165e66f7df9ab267553` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.40186827699653804} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/b39bc98a488a7bdb2c58` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.05020060995593667} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/b3c2c1f3d436636bd52d` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.019352568895556033} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/b92cca6dff9ce11c7f82` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.0018735280027613044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/bb8a5bab42176d293a92` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.001597240916453302} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/bd59af493eb45bd747f9` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 3.354422094998881} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/be855ce12637d242ba53` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.000754259992390871} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/c2798e6de902e63293e4` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.0028777059633284807} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/cce4e524c06ef7c0b18d` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.0023453510366380215} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/cd333a19e6e92b0af5db` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.0010226679733023047} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/cde92d84fb6f776302f7` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.0015585110522806644} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/d1fa7a34191c5cb05787` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.3376871030777693} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/d4b2f6c6838a112047a7` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.018105713999830186} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/d9749ea330ea76bcf7b9` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.08768321096431464} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/d9ab57c2f96a1626c8c1` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.3167648130329326} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/da8c40ab9c17418a58f1` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.030589553993195295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/e4d9be09c609e56ff141` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.001678570988588035} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/e577160f0637e5c66f27` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.3322830779943615} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 2549 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "8860c50c0dc924ee591bfd0157cde7946a80a25e",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-30T16:33:02Z",
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
