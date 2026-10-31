<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · euclid

[Model index](index.md)

Exact setup ID: `imaging/rectangular/euclid/efbb5020521b6dee611c`.

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
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1265 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>31 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/01f648ddce86e4389220` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00012867959740106016} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/122bcd8a5f18b1e0f96a` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0006777805014280602} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/17315fd9d11c7ee4cbbb` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00012087659852113574} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/22fc73b49088ac424aa5` | overlay_grid_jit.steady_per_call_s | {"overlay_grid_jit.steady_per_call_s": 0.00010749789944384248} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/steady_per_call_s` |
| `measurement/251071f332da73c8bdb4` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00012867959740106016} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/2d9105f59cda460d4ee4` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.02416911780019291} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/390c1368e28f0ebbc9e8` | steps.Data vector (D) | {"steps.Data vector (D)": 0.00027823589916806666} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Data vector (D)` |
| `measurement/47aa98e362ca84bd2fa7` | component_total | {"component_total": 0.0338299544993788} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/total_step_by_step` |
| `measurement/4a819befb0e72c6f3feb` | steps.Inversion setup (steps 4-8 combined) | {"steps.Inversion setup (steps 4-8 combined)": 0.006171078499755822} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Inversion setup (steps 4-8 combined)` |
| `measurement/4fa25f876ac81ada8f1d` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00012087659852113574} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/62afce7b3f87a9d355f7` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0028901115991175176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/6664519d8089a207aed2` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.0016771177004557102} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/6de9912ee413f8cee3b9` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0010112608986673876} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/732789a71e2a34878c19` | interpolator_prefix | {"interpolator_prefix": 0.0026422085997182877} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/interpolator_prefix_s` |
| `measurement/88d17a45b7414687c11e` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.00027823589916806666} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/8d794f7936714758523e` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.0010101040010340513} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/a0ac32ca1f60a823212f` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.0026422085997182877} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/a51b2013c28c55812fde` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.006171078499755822} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/a58d7fc1357fa9ff973c` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0006777805014280602} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/b38d82cf7fd116ca88d6` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.024045344101614317} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/b7e097c52b7dfc57c00f` | steps.Overlay grid (source pixel centres) | {"steps.Overlay grid (source pixel centres)": 0.00010749789944384248} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Overlay grid (source pixel centres)` |
| `measurement/baf56fe28fed48d0840a` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.0009757552004884928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/bd01970abbbf5051c995` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0009690005012089386} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/c50f719e5b14b30ee326` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.0021355088014388458} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/c60562dc3b980f796fe9` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.0016771177004557102} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/ce9679ff768d31828ff8` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.00015428829938173293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace grids` |
| `measurement/d9dad29702fc64324744` | ray_trace_jit.steady_per_call_s | {"ray_trace_jit.steady_per_call_s": 0.00015428829938173293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/steady_per_call_s` |
| `measurement/e94448baf7aaf4b53b62` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.024045344101614317} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/ef3330af987d79118724` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.00128260999917984} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/f56298a532a5669aae61` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.0009757552004884928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/fbe532b296840780e3e8` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0021355088014388458} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Mapped recon + log evidence` |

</details>

### compile

<details><summary>54 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/04ad20c16a6e11b26f25` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 2.3977176260086708} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/0a876533c203fb1b8e49` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.0017510710167698562} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/0abe9ecf53fd82c9446b` | ray_trace_jit.first_call_s | {"ray_trace_jit.first_call_s": 0.0016168619913514704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/first_call_s` |
| `measurement/0b641c6195b2a56acc5c` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.13557166399550624} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/102447e7af0c363b320e` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.19022897799732164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/1ad3db8f1ea82be020f7` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.315096837002784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/20e81e123281ce5f47f7` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.007087610982125625} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/2b7fc2effe4b9b969e7c` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.02478724002139643} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/2dae7e52d03b87aa2d3a` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.026066930004162714} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/33ec97d02a119e69366d` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.23978506700950675} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/39a8b9921967b3796acc` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.012162216997239739} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/3d44c4e055041d5cf8cc` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.0033798929944168776} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/3efaec95c8a7bffeac92` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.12431838901829906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/43efe2e7fe288f9116f2` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.028868464985862374} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/4eb3d506a542801e5124` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.007495656987885013} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/58b715475eb2ba60ea85` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.012126937974244356} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/5c5547e78e5571bd0277` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.046801859978586435} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/5c7290f68385202fea41` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.02572636998957023} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/5f2a81440f132257e096` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.00758119800593704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/6259001e10fd65220637` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.0010738190030679107} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/6c81431b862d92b932ae` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.0047559780068695545} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/706e11173db009776c3f` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 8.25000461190939e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/76f8059f13908279a743` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.001649682002607733} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/7be1af89f674a65c896d` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.27412969499710016} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/8134b56c362eff5535a4` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.000607152993325144} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/882f7b91e823424fd191` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.010214506997726858} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/8bb07b73d2db1aada7aa` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.02446256301482208} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/8da60a3d94e50dbbfc46` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.007695054984651506} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/990c614bf78e85934964` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.037563273013802245} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/9fd466148e06af58ba98` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.014456942997639999} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/a17bb8768390cc4b7a6e` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 2.677978660998633} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/a86854ef7ac48090cb68` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.0017144909943453968} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/acaa313c2019fd16b960` | overlay_grid_jit.first_call_s | {"overlay_grid_jit.first_call_s": 0.0007691909850109369} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/first_call_s` |
| `measurement/ae076b12173905415885` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.004870456003118306} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/af85f9a0f75ae759e2da` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.03387142301653512} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/b140c666a20240ec3fcb` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.026127274992177263} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/b6851ecfdb2593f34e97` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.2797592130082194} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/b89a9dad7568842a78a4` | ray_trace_jit.compile_s | {"ray_trace_jit.compile_s": 0.4201486329839099} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/compile_s` |
| `measurement/c7a2e7cc88196644f72a` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.08162582101067528} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/c9423f807e8add19021d` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0002491369959898293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/d05e388038b00e049729` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.3653307410131674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/d31e3f72c5396ffa25d0` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0006765830039512366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/d550c2f71bde4ab3ca50` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.07727982799406163} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/daa169345aa02b4f3b35` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.0016331819933839142} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/db71cc8bd6a4fcf58de3` | overlay_grid_jit.lower_s | {"overlay_grid_jit.lower_s": 0.014467969012912363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/lower_s` |
| `measurement/e13b557c9cc695dbd199` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.30081254299147986} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/e184ae149996135129a1` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 6.704542096005753} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/e4c93c9c51498a7e10d8` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 3.104048650013283} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/e91f786ef96e6c4eb388` | ray_trace_jit.lower_s | {"ray_trace_jit.lower_s": 0.09213452599942684} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/lower_s` |
| `measurement/f476b562363c17c85c35` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.0008927200105972588} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/f618c49f807540dfc26e` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.056034258013824} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/f729dfc120faa6023be2` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.08743470799527131} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/f86102d6846836811df7` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.024760719999903813} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/fa1b35f310d8785dda3a` | overlay_grid_jit.compile_s | {"overlay_grid_jit.compile_s": 0.13537523700506426} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/euclid/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/compile_s` |

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
    "measured_at": "2026-10-25T14:22:26Z",
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
