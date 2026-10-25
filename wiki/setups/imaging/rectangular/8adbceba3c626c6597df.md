<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · euclid

[Model index](index.md)

Exact setup ID: `imaging/rectangular/euclid/a985cef04d4b420ff342`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 3841 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_shape | [35, 35] | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "note": "Outer sub-size 1 retired repo-wide on 2026-09-08 (autolens_profiling#235): it leaves the outermost annulus un-over-sampled and causes gradient issues.", "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 15664 | recorded |  |
| oversampled_pixels | 15664 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.1 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": null, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 4, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 628, "pixels_below": 3213, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "euclid_rect_adapt_standard"} | recorded |  |
| production_preset | {"adapt_snr_cap": null, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_40_linear_2_basis", "memo": "off", "mesh": {"class": "RectangularBilinearAdaptImage", "free_weights": {"weight_floor": {"lower": 1e-05, "type": "LogUniform", "upper": 1.0}, "weight_power": {"lower": 0.0, "type": "Uniform", "upper": 10.0}}, "kind": "rectangular", "shape": [35, 35], "source_pixels": 1225}, "mesh_kind": "rectangular", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 20, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 4, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": {"factor": 3.0, "minimum_threshold": 0.2}, "preset": "euclid_rect_adapt_standard", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "euclid_strong_lens_modeling_pipeline scripts/initial_lens_model.py:538-543 galaxy_name_image_dict_via_result_from (subtracted S/N, uncapped) for the Hilbert mesh; adapt_minimum_percent 0.01 (config/general.yaml:22)", "model_stand_in": "the dataset's truth tracer stands in for the previous stage's maximum-likelihood model", "over_sampling": "initial_lens_model.py:609-626: sub-size 4 where the source S/N > 3.0, else 2"}, "provenance": "subhalo_validation/scripts/pipelines/rectangular_adapt.py (rect_adapt source_pix[2], job 342311): RectangularBilinearAdaptImage shape (32, 32) (MESH_PIXELS_YX :38), regularization Adapt; the rest of the stage (adapt S/N cap, over-sampling, MGE, threads) as hst_source_pix_2 \u2014 no Euclid rectangular production stage exists (the Euclid pipeline's pixelized stage is Delaunay only), so the Euclid rectangular row takes the rectangular mesh from this configuration and everything the stage owns (adapt S/N cap, light-profile radial bins, positions penalty, MGE) from the Euclid vis_pix stage, on the Euclid dataset \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; RectangularBilinearAdaptImage mesh; standard size (human ruling 2026-10-23): rectangular 32x32 -> 35x35", "regularization": {"coefficient": 1.0, "free": false, "scheme": "adapt"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; RectangularBilinearAdaptImage mesh; standard size (human ruling 2026-10-23): rectangular 32x32 -> 35x35", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "free": false, "scheme": "adapt"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1225 | count |  |
| source_pixels_requested | 1225 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 91600 | recorded |  |
| sparse_operator_build_s | 0.1938273390114773 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1265 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>38 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00910555d0cd2d337b0e` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.010479053799645043} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/0abf8c2a7c0679ad1619` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.00020397970220074058} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/1022854f61ede1579c93` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.0006214722001459449} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/1294049d433036175eae` | overlay_grid_jit.steady_per_call_s | {"overlay_grid_jit.steady_per_call_s": 0.0001421054999809712} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/overlay_grid_jit/steady_per_call_s` |
| `measurement/19e7ba72e3d36b8cc50c` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0007944702985696495} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/1b41bd8f10bc046933fb` | interpolator_prefix | {"interpolator_prefix": 0.0015247542993165552} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/interpolator_prefix_s` |
| `measurement/1d3f9f2a56f9caa8135c` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0026890926994383333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/1ea453b54a1258e60c2f` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.00026760709879454226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/218692da690ed55350ed` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.0012819291005143896} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/2578805f286ca63af101` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.010250729299150408} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/2c6351f940df8c2a814d` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.00022834370029158903} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/3a293bd6ed0ef02a2a9f` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.004578311098157428} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/4193b8073972037d67d9` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.0017530979996081442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/478e2581ed217857007c` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.0002340713981539011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/4a87c1197eaed5c204d6` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.02439170430006925} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/59c1399aee68988c2007` | ray_trace_jit.steady_per_call_s | {"ray_trace_jit.steady_per_call_s": 0.00017051419999916106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_jit/steady_per_call_s` |
| `measurement/5b4e069cad864a4b308e` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.0019022512016817928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/6b04f1c08ed18a1e5c10` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.0001500213984400034} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/767c488d6bbc6329af01` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00013400660245679318} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/78fcbc996b86d1aea78c` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.00017051419999916106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace grids` |
| `measurement/80fc781ca5f94a30e058` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.0019022512016817928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/85d89e89432099f9f013` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.0001520513993455097} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/913ba69cbdee21420b76` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.0017530979996081442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/97edc0a347a7827a9d95` | component_total | {"component_total": 0.04324914899771101} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/98c29c4f5dbf6f062f7e` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.02443396979942918} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/9f0e9d18f6840e9f2195` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.0001520513993455097} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/9f2b2a4b321cc7b87856` | steps.Inversion setup (sparse, steps 4-8 combined) | {"steps.Inversion setup (sparse, steps 4-8 combined)": 0.004578311098157428} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Inversion setup (sparse, steps 4-8 combined)` |
| `measurement/a974bf1e96f03bd796d7` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.000998990100924857} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/b31b0cfdaa67db2c2c86` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.02443396979942918} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/b40ecab506506af6b0bc` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0010273997992044314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/ba58b3c2606723437e0a` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.0032541944994591175} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/c3902d48a5bc0d226f4b` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.0009851773007540031} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/ccce736e3834830ff572` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00013400660245679318} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/ce25dcef53b6fe5ba049` | steps.Overlay grid (source pixel centres) | {"steps.Overlay grid (source pixel centres)": 0.0001421054999809712} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Overlay grid (source pixel centres)` |
| `measurement/d2278bd10b2689164f2e` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.0002340713981539011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/d3f5d5456ddd8fbf4829` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.0015247542993165552} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/f302ddaccb503a7bfd98` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0007944702985696495} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/fee6286d25a718590540` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.010479053799645043} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/01e767c57572198fe438` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.0047109480074141175} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/0c7d598437e290e4f3f4` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.13148808799451217} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/0cf42766ee11152a37aa` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.03029131898074411} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/162bd990dadcaac6ec41` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.30100317098549567} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/1930d4c5f9e90d56ce9e` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 6.533144542016089} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/1ba08135bb77dea33654` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.0849995305761695e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/2bfc18217d36aa78e468` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0010780490119941533} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/2cf55f3f2538a555deba` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.0074174689943902194} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/34e4db849b863a45e635` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 4.482982024986995} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/3a0ca5800383e3ed8ed8` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.0063795009918976575} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/4292d63b7bac030c90c2` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.08455566898919642} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/44038a46d80959141442` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.011440486006904393} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/44f482da8be723e88980` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.07494100299663842} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/4806eac7650e7245efd3` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.001079158013453707} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/4939bde6254ee029abe5` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.0009257089986931533} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/4f3e25c60181f6d67870` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.05562517401995137} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/532180d227a92528b6de` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.030605557025410235} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/58743accf1824b9079c3` | ray_trace_jit.first_call_s | {"ray_trace_jit.first_call_s": 0.0013762959861196578} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_jit/first_call_s` |
| `measurement/5896f5b2d4691f92c40e` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.21704705597949214} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/59fd5082360278bd562f` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.04065981600433588} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/6db2e89fb4b6c8015bd0` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.026160754990996793} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/704279303be203e08968` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.037993645993992686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/760954a4a2332abf8da2` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.0019729280029423535} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/76be3bb53563c7bf8b5c` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.008486587001243606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/7e6b339c4808b3bf33fc` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.034995508001884446} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/7f93cedb722fa268c557` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006821919814683497} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/83be5a0c96a01eae1894` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.3197560470143799} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/84d67d1460409041f24c` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.02488409000216052} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/87dd7a9a47b233f5cc56` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.32797078700968996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/87f8671aad4614bfb83f` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.11858210799982771} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/893f039b24ea00ec3bf2` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 3.2527880359848496} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/8c81cf5fd8709d7c4156` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.026046245999168605} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/8c91cedb8e00902f05bb` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.09983438299968839} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/8f57f4d9fc88bd17ef8c` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.006375860015396029} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/934c3014dc067aa95ded` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0005739429907407612} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/94ba17ae4de76a4fe351` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.03389608100405894} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/94c497b76626abc12e50` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.1633503900084179} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/9933329ec6cfabfac398` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.014980565989390016} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/9ae865fcab8c628b5d13` | ray_trace_jit.compile_s | {"ray_trace_jit.compile_s": 0.4201531129947398} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_jit/compile_s` |
| `measurement/9d2da807a141a7607257` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0063653399993199855} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/9d6294f7e48446e3db65` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.0017319110047537833} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/9d8aeffa9a40ad080a78` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.09391809601220302} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/a06aad880aaf531aa105` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.010972860007314011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/a21aad43edef1604bd52` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.018182240979513153} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/a2cecb23d65f8b64abb9` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0008036810031626374} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/a5fd73d96b30cf6d03db` | ray_trace_jit.lower_s | {"ray_trace_jit.lower_s": 0.093638928985456} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_jit/lower_s` |
| `measurement/aba68f7c96fa47b91251` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.235432755027432} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/abe6d17bae2105238bc2` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.3368742110033054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/ac8028e2de48f80e1da8` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.01789889601059258} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/b11f0a1628a17732a061` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.12457406299654394} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/b1a33369dbf112ae2dec` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.016748117981478572} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/bd5a9a0b03f41767c8fa` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.13167311600409448} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/be17f323b83d4ccf4298` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.004331892996560782} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/c1b84faa07b9ef26c25e` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.24240035901311785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/c1c51c35556a5fd66763` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.38495357701322064} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/c44a6ae4783010d48d7a` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.02655398999922909} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/cba42f196e73b0bfb767` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.0018274399917572737} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/cbc66346bd2fea33ec63` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.013497153006028384} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/ccff87d72c729bb1f9be` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.16767357400385663} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/cddc309fca6e3187bd29` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.00030847801826894283} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/cde3d3099b9f0d1e73f5` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.026032555993879214} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/d2518a68276d0e769a54` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.03445920499507338} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/d2e95c6cd05837fe820e` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 2.3445776400039904} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/d4e04aea3f520d9986b7` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.05417267998564057} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/dd5491a725575d034d1e` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.011183867987710983} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/e951c0d6a599d162e021` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 1.837072967988206} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/e9d549c9dbff22a05051` | overlay_grid_jit.compile_s | {"overlay_grid_jit.compile_s": 0.12654254102380946} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/overlay_grid_jit/compile_s` |
| `measurement/ea7958d1c42845ecac05` | overlay_grid_jit.lower_s | {"overlay_grid_jit.lower_s": 0.02021740999771282} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/overlay_grid_jit/lower_s` |
| `measurement/eb1d7b8230939c312223` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 2.254768877988681} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/efabb0a5406051b0f3cc` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.0008457310032099485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/f0e1f5b36ef3138634f9` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.050007675017695874} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/f8aa892c3ee3599ba381` | overlay_grid_jit.first_call_s | {"overlay_grid_jit.first_call_s": 0.0008283809875138104} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/overlay_grid_jit/first_call_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 761 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "d99ec2b30224ac021e39628b9904d01c44f33b29",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "d3a7721294652f81ec726b92931ab610ed27e3ac",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "e00b5e7d554d6958a6c8821edef6afdf01188e6d",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "f791ac77d136d664346fff21cc36079f8f9b1196",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "f116f70077e442db19a2a152f5ee78d57294c504",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "86c287d1fc90f7595b2106dbb6b55102c4d7c800",
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
    "measured_at": "2026-10-25T14:23:57Z",
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

- [hazards](../../../../scripts/imaging/rectangular/hazards.py)
- [likelihood_breakdown](../../../../scripts/imaging/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/rectangular/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/rectangular/likelihood_runtime_numba.py)
- [likelihood_runtime_numba_mge_mass](../../../../scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py)
- [parallel_scaling_numba](../../../../scripts/imaging/rectangular/parallel_scaling_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
