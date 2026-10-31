<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/789cd8cd17f390d7bc1b`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | "subtracted S/N (truth tracer), floored" | recorded |  |
| adapt_minimum_percent | 0.01 | recorded |  |
| delaunay_vertices | 1280 | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_source | "truth" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_adapt_snr_cap | 3.0 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sample_size_pixelization_rule | {"pixels_above": 2068, "pixels_below": 13293, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"} | recorded |  |
| over_sampled_pixels | 61600 | recorded |  |
| over_sampling_preset | "hst_source_pix_2_standard" | recorded |  |
| oversampled_pixels | 61600 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": 3.0, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 2068, "pixels_below": 13293, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "hst_source_pix_2_standard"} | recorded |  |
| production_preset | {"adapt_snr_cap": 3.0, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_60_linear_2_basis", "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 30, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": null, "preset": "hst_source_pix_2_standard", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus the other galaxies' model images, over the noise map), floored at adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max", "mesh_and_regularization_adapt_image": "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from (S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and rectangular_adapt.py:109", "model_stand_in": "no previous-stage fit exists in a profiling cell: the dataset's truth tracer (tracer.json) stands in for the maximum-likelihood model whose lens light is subtracted", "over_sampling": "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184", "rectangular_mesh": "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight priors config/priors/mesh/rectangular.yaml:43-61"}, "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 258780 | recorded |  |
| sparse_operator_build_s | 0.1566921900084708 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1340 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>38 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0028cca24fa5f059d85d` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.023035916598746552} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/01447a762541e73bfcd7` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.02726879270048812} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/05c95d72a25910a56d05` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.0007560147991171106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/099cbf193f72c343804f` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.7472523244010517} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/15a44fed30a41d9cbbab` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00026152940117754044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/15caa98a7918d172b90d` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.04430294860212598} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/372170be4087f7378d1b` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0010217117000138388} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/39589622f95a780837e7` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0010217117000138388} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/466a05b1c3e76dfac7c9` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 2.4454700178466737e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/48c56e926aab139a6458` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.7359063115989557} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/4b1be04a3c365d1248bd` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.02244326899817679} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/51f2677a458784156771` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.002645790300448425} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/54a7de27f43d793afe9f` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0030274955002823623} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/54f0d9d30c75a82c75a1` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 3.629349812399596e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/5f78f497be2b7ed0c69f` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 1.4706469308002852} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/61bbc2d5b26bad1d23b2` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.358364180100034} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/6bc797cde792c19387fe` | interpolator_prefix | {"interpolator_prefix": 0.358364180100034} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/interpolator_prefix_s` |
| `measurement/6e15694a7c1939a5b03b` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.03665121710218955} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/7574cb7fa4d13b7b43cf` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.3613916756003164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/862f8ce22a9748ccec8b` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.7359063115989557} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/93f899232da34614ff27` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.03665121710218955} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/96be21fa053fe50dddee` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.025694502799888143} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/9a40c2becbc6a4dd1ef1` | component_total | {"component_total": 2.645794239200768} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/total_step_by_step` |
| `measurement/9ef2c8c8e55c9adcfe10` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.023199882701737805} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/9ff5f86aacdfc76e7370` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.060308640499715696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/b20251fbe5c13c29adc5` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.3613916756003164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/bdaa330be34fb90aa0f2` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00026152940117754044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/c3549931bb87fa347ed1` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.0009280876984121278} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/ca65e35a7a9fbb90b163` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 2.4454700178466737e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/cb9be95716bd1c309b57` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 1.4706469308002852} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/d3590d5149f2b9d046c7` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0013225934002548457} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/d4dc7e6a2d6962d079c4` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.0007560147991171106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/e408418597cbe1e99f6b` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 1.4220594742015238} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/e7e6e7d126fdefadf161` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.3961396867001895} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/f23490a17e0c3fe6fc86` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 0.3961396867001895} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/f35f7ad9f781524e24ec` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.021301751901046374} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/fb1999c0c4570e9668ab` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 3.629349812399596e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/fec6bcc6b2582038ee18` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.0013225934002548457} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00b47c13ac9df878ff3a` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.01426648799679242} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/0ee4c2b6d48e58ce7acc` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.00955086000612937} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/1c662500eaa613b72107` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.02353688798029907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/1cb31e88cd6331bdcbb7` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 3.706070312007796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/20acb1b46db9b4e61e26` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.05853674400714226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/27fb3361a3f4730ca557` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 1.3973156200081576} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/3b54df5426387c2b167b` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0014477319782599807} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/3c0afbff38faf275646a` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.0893525870051235} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/3dd1c2df5bc8741be16b` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0003312969929538667} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/3f432f6a89dce9d94c65` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.099001383408904e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/4a5dee0f46cb8ed47678` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.049314889998640865} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/4e87e74a8c97cafbfdfa` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.002900116000091657} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/4e90cadec2bb86e5e626` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.0406465410196688} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/53046beb4c48c7a07dfc` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.40774384600808844} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/5568b6dd5cb48ba1f10e` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.06595993399969302} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/5aebe4c27cbf3a51d9f5` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 0.9545882889942732} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/5dec86bdbce558cdecd8` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.27010248697479255} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/60e837677d85ae1c109f` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.6259817910031416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/63fa043640c983d08743` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.027215134003199637} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/650c99f7b4ac4657e06a` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.05698477098485455} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/67af28f23923a5a8709b` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.18604357898584567} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/68358ca33ea4f6430825` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.18881153900292702} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/68a2d847c586d1f58b7e` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.005833283008541912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/691f589a7e3adf64bc44` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.08923705798224546} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/6e4dcaaf1367adbaddc9` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.02481435998925008} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/6ecdfc8ea1062470bc49` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.010294560983311385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/6fc43eb52e7e094806ba` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.010493743990082294} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/74749511b83a2a8ab183` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.023168751009507105} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/74a2d392067031be7ded` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.35720316000515595} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/77541e93796a1b1d498a` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.047764643997652456} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/795bbede0ad6add1aec7` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.3243867259880062} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/81c993e397335c304b65` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 1.4559126810054295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/83dbf8e259cce4fc56eb` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.7353516250150278} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/8405c77c10c95c7437aa` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.05941153599997051} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/84ee39bc6556e01e8872` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.08081349602434784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/85205c25f3c72cb19e70` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.02962155800196342} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/8824b0368abe5ff4260f` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.1682888870127499} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/88e9dede0a0517c3ff22` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.1603526640101336} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/89bebd96bb73645c33bb` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.030356238974491134} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/8d29f4478c1e8961718b` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.016663510003127158} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/979bc4d54c96c65b9ba1` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.07245443301508203} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/a41cacd683934bc75b35` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 1.1233346189837903} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/a444f0e870f6b29a6f5a` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0010081750224344432} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/a6fc670e1a232f82b4da` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.0159896069962997} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/a9df18d550e855f18922` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.36289246499654837} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/ab1ab538b864b5e6753e` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 6.60462518199347} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/ab328d253576830cb8cf` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.016338801011443138} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/ac7454891aaea322e15d` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.00025131599977612495} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/b2a3039c972e1841d54d` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.004095142998266965} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/b2df27560ad39cf1e512` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.031122284999582916} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/b7b769a97e3267729aa0` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.28969861002406105} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/bcf3e408c3db89e5c7e1` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.25497442099731416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/c0d116489d4dd5215b71` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.012797272007446736} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/c2fa1e8be5d290f8fa96` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.27970589499454945} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/c4188bba5615e8b2f2b3` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.026499994011828676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/c459620768de13ae6e02` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.002464605000568554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/c508b992bdc5610aaabb` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.043696064996765926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/c5be88c238c16f02de35` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0020432609890121967} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/c93af29efe348d0d6e96` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.730791798006976} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/d4552b7bfbb30de39c82` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.00019910800619982183} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/d483643c2acbce961e16` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.0597024789894931} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/d538f0fb39757633fed9` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 2.338105861999793} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/d557ff8156ca9ded1bee` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.023058421997120604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/e4bd56cfb1cea11ec3df` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0020348550169728696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/eeb61a2609cc489ab7c7` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.03079422700102441} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/f112c1797596b096151a` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.02267124698846601} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/f144f3c13157a0dfee2d` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.03249388499534689} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/f1a29322ee226109e1aa` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.13067990497802384} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/f1e526f360b0f54312e1` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.06812286700005643} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/fa76bb0c1860206ec90f` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 7.689834034012165} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/fd0dcdfd5f2b9543e896` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.045709447003901005} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/fda99753237ad0d7e606` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.08144242700655013} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
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
    "measured_at": "2026-10-25T14:14:13Z",
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
