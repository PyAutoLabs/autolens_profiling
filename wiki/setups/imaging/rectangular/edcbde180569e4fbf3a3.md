<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · ao

[Model index](index.md)

Exact setup ID: `imaging/rectangular/ao/472682900af4fdda7b48`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 384753 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_shape | [35, 35] | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "note": "Outer sub-size 1 retired repo-wide on 2026-09-08 (autolens_profiling#235): it leaves the outermost annulus un-over-sampled and causes gradient issues.", "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 1542816 | recorded |  |
| oversampled_pixels | 1542816 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.01 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": 3.0, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 47431, "pixels_below": 337322, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "hst_rect_adapt_standard_ao"} | recorded |  |
| production_preset | {"adapt_snr_cap": 3.0, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_60_linear_2_basis", "memo": "off", "mesh": {"class": "RectangularBilinearAdaptImage", "free_weights": {"weight_floor": {"lower": 1e-05, "type": "LogUniform", "upper": 1.0}, "weight_power": {"lower": 0.0, "type": "Uniform", "upper": 10.0}}, "kind": "rectangular", "shape": [35, 35], "source_pixels": 1225}, "mesh_kind": "rectangular", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 30, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": null, "preset": "hst_rect_adapt_standard_ao", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus the other galaxies' model images, over the noise map), floored at adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max", "mesh_and_regularization_adapt_image": "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from (S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and rectangular_adapt.py:109", "model_stand_in": "no previous-stage fit exists in a profiling cell: the dataset's truth tracer (tracer.json) stands in for the maximum-likelihood model whose lens light is subtracted", "over_sampling": "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184", "rectangular_mesh": "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight priors config/priors/mesh/rectangular.yaml:43-61"}, "provenance": "subhalo_validation/scripts/pipelines/rectangular_adapt.py (rect_adapt source_pix[2], job 342311): RectangularBilinearAdaptImage shape (32, 32) (MESH_PIXELS_YX :38), regularization Adapt; the rest of the stage (adapt S/N cap, over-sampling, MGE, threads) as hst_source_pix_2 \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; RectangularBilinearAdaptImage mesh; no ao production reference: borrows the hst preset 'hst_rect_adapt' with the ao dataset; standard size (human ruling 2026-10-23): rectangular 32x32 -> 35x35", "regularization": {"coefficient": 1.0, "free": false, "scheme": "adapt"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; RectangularBilinearAdaptImage mesh; no ao production reference: borrows the hst preset 'hst_rect_adapt' with the ao dataset; standard size (human ruling 2026-10-23): rectangular 32x32 -> 35x35", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "free": false, "scheme": "adapt"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1225 | count |  |
| source_pixels_requested | 1225 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 8432736 | recorded |  |
| sparse_operator_build_s | 0.20330644299974665 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1285 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>38 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/049cea44d0addf5a4672` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.000970747901010327} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/0bc3c6a3decdcc2a2dc1` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.0001871897984528914} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/0c9889c51c3b6ab226c2` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.000970747901010327} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/12e7e69a910c4a127f2c` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.001348675400367938} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/131f526e9d570207ddc4` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.0053359348006779325} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/214c8790b9fd123d5aa2` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.0026524985994910823} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/27b49b179f7706c83d36` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.0026524985994910823} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/33c4271e73ea7b547592` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.033354553999379276} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/40282aeca1b6cbd3a286` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.0010086663998663426} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/4a17dcbe708735450e8d` | component_total | {"component_total": 0.988421823992394} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/total_step_by_step` |
| `measurement/4ca64acc0226d8e161de` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.0023708447988610715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/53460baa6062202d2a53` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.011175506701692938} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/53e2257c02ecebdbbdba` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.05472801740106661} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/5dc05528c224efa12615` | ray_trace_jit.steady_per_call_s | {"ray_trace_jit.steady_per_call_s": 0.00028896459843963385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/steady_per_call_s` |
| `measurement/64608968fa2aadd6199b` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.000687837100122124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/6d780766ef85764a6a68` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.007280888402601704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/6fbd3d18cd82dad85f2d` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00012888739875052124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/71b89758be583086879f` | steps.Inversion setup (sparse, steps 4-8 combined) | {"steps.Inversion setup (sparse, steps 4-8 combined)": 0.018380769799114204} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Inversion setup (sparse, steps 4-8 combined)` |
| `measurement/737a126fed33cd9ffaf8` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.0023708447988610715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/7f3511aa5ab1c6d96416` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.03350336519943085} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/879862e4fd7624c08a5e` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0009988334990339355} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/8bfc68f023a5a7dfc2d7` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.00028896459843963385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace grids` |
| `measurement/93ad19ae55bc7ea311db` | steps.Overlay grid (source pixel centres) | {"steps.Overlay grid (source pixel centres)": 0.00017901789979077876} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Overlay grid (source pixel centres)` |
| `measurement/9421b1608337f0eb7378` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.8658217165007954} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/9b8deabc2db4f044c81b` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.00043744989961851387} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/a05b1081eb4c2703a272` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.9258649196999613} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/a69a1fc09b425e979541` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.018380769799114204} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/b6971523f40750cbb89f` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0038946182990912343} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/bc6651fd2f5b37230d55` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.03350336519943085} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/ca51ef141afb928e4bb7` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.00105077090265695} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/d4387e4cc9aa4a6687fc` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.011175506701692938} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/dc815e2523b2a8d07b2c` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.9258649196999613} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/deaf61add683060df2de` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00012888739875052124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/df92f9a57d3c22fe4fe6` | interpolator_prefix | {"interpolator_prefix": 0.007280888402601704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/interpolator_prefix_s` |
| `measurement/e63df6530a97685a9420` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.0001871897984528914} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/eb1321ce2c53d634eb32` | overlay_grid_jit.steady_per_call_s | {"overlay_grid_jit.steady_per_call_s": 0.00017901789979077876} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/steady_per_call_s` |
| `measurement/ef662192ff1a6ca8d3cb` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0026829292008187624} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/f293c191fa6cf8c1897b` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.011629126401385293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/02a5ea1e659acf535707` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.9946569159801584} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/030e89f49222fdf9d747` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.0010907769901677966} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/09347926bffd8b4868a8` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.018772893003188074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/0ab160b746be25d3310e` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.0019199980015400797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/0d2ac028b3a87d90ae5d` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.010380031977547333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/16e07ea64e65ba87206a` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.024048693012446165} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/1756185ae572921ec379` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.009887716005323455} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/18dc7929f8a598bcb825` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.07877375499811023} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/1ba0ef467121991f96e8` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.026802140986546874} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/1dd9dd5684861845750d` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 3.925935572013259} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/1e2b383898d774702346` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0010288089979439974} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/23bd17d06511f77f3a5d` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.010756725998362526} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/25cebdf8716028d9e531` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.02994403601041995} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/280274b3b7e29f415fbf` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.0016387409996241331} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/2d598f4b47d6ddf08a0f` | overlay_grid_jit.compile_s | {"overlay_grid_jit.compile_s": 0.12305158699746244} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/compile_s` |
| `measurement/2e5e64fcb4cb2e3fc3e1` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.586879089998547} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/33211097f50ce53622d2` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.03454232300282456} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/3844af593989f9d60f03` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.08163721300661564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/3fafc4055f022a574fb8` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 13.09911631999421} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/448e5d1ed0abc7a0af57` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.0016607309808023274} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/4c6bf67d229ba801a299` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.004333710006903857} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/4f28257cc30ab65ad52d` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.014165177009999752} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/505c7aae42fa9fc9c14e` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.012823843018850312} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/52563d2dea872a64b5eb` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.5178071940026712} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/5682ccb1525a94889973` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.29307198300375603} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/5bc3a83bb64c71e8538d` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0007059320050757378} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/5f501bd5ad26ce7de7ee` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.13499894901178777} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/628245e14f9a858f51ea` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.025124862004304305} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/6b00dceac84c42207f13` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.343609473988181} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/72753d70dca94acc3c8a` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 9.139999747276306e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/859dad27964d3db6962f` | overlay_grid_jit.lower_s | {"overlay_grid_jit.lower_s": 0.013770061981631443} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/lower_s` |
| `measurement/8a92074a554484832670` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 8.649146065989044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/8e790f00580ae887ae2b` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.23831428299308755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/90393e4a9d8002aad8e9` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.06327050301479176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/905a8bda27d6e283bdb6` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.0013910749985370785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/91761f51b09480da4522` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.867922646983061} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/9abf0bdafff93364914d` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.8835483909933828} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/9bdd2ae8154abdc594e8` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.08915573699050583} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/9d5240bcd06903c8dd4f` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.02853091200813651} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/a121d8461183e3ca9903` | overlay_grid_jit.first_call_s | {"overlay_grid_jit.first_call_s": 0.0012262749951332808} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/overlay_grid_jit/first_call_s` |
| `measurement/a149f14b56ae9056768b` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0002937070094048977} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/a4e6fa88a9fa1bfa469c` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.060771821008529514} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/ac1ac7df03549702dcd1` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.00574540501111187} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/afe37db19256c7e0d450` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.03251280600670725} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/b14906e8c81bb30f855e` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0032349220127798617} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/b2b5d76c1ceaa60294ef` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.6044594180129934} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/bec68aee0ed81788070a` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.01607646598131396} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/c09f95fc8fa40251cedf` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.004220752016408369} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/c3e50c9f372819ae58b0` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.33767848199931905} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/c6b47242564f99b631eb` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.24104730199906044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/c805ba6b33e820aa25f3` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.08680663298582658} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/d4e75b4102e9328db23b` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.03499473899137229} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/db940a34c6e502b55773` | ray_trace_jit.lower_s | {"ray_trace_jit.lower_s": 0.091180371993687} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/lower_s` |
| `measurement/de8c72ab8b5b9d2682b0` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.00872364899259992} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/dfb83119a774e448db9b` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.05267680497490801} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/dfe8e0657947c0ef26b9` | ray_trace_jit.compile_s | {"ray_trace_jit.compile_s": 0.29294962697895244} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/compile_s` |
| `measurement/e5e92dcc4676041e4c07` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.02345581099507399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/e94521bac92a09718a69` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.03945540697895922} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/ec312fb4b0e0b5dc8863` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.09207558300113305} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/edc741cf348c34188f66` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 4.6271192689891905} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/f6398224c20cb51903bd` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.04181875000358559} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/f944331571769f65a896` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.037116652994882315} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/f9c9278cd0245ce1c2f4` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.23897638500784524} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/f9f635c9317fcf0a6ff9` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.2744382889941335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/faa0856de7c3f597f6c0` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.3359544609847944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/fac3fd11a855ca6198bc` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.2899254609947093} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/fbd7d23fcce6f0181ea2` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0007178419909905642} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/fd1cd2c0c342f91824ba` | ray_trace_jit.first_call_s | {"ray_trace_jit.first_call_s": 0.0015775119827594608} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_jit/first_call_s` |
| `measurement/fdcc98b01a38b0555332` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.011504177993629128} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/fe5e4e39253d321f34e1` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.006043921021046117} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/fe778084f1c28a123df4` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 2.8005090979859233} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/feb18c675a5d717b8c93` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.34006380499340594} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/ao/jax_gpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 17383 MiB, 81920 MiB",
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
    "measured_at": "2026-10-25T16:01:38Z",
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
