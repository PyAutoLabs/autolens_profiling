<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · jwst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/jwst/5f41e8e3e1c467065680`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | "subtracted S/N (truth tracer), floored" | recorded |  |
| adapt_minimum_percent | 0.01 | recorded |  |
| delaunay_vertices | 1280 | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| image_pixels_masked | 42737 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_source | "truth" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_adapt_snr_cap | 3.0 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sample_size_pixelization_rule | {"pixels_above": 5407, "pixels_below": 37330, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"} | recorded |  |
| over_sampled_pixels | 171392 | recorded |  |
| over_sampling_preset | "hst_source_pix_2_standard_jwst" | recorded |  |
| oversampled_pixels | 171392 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.03 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": 3.0, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 5407, "pixels_below": 37330, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "hst_source_pix_2_standard_jwst"} | recorded |  |
| production_preset | {"adapt_snr_cap": 3.0, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_60_linear_2_basis", "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 30, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": null, "preset": "hst_source_pix_2_standard_jwst", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus the other galaxies' model images, over the noise map), floored at adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max", "mesh_and_regularization_adapt_image": "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from (S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and rectangular_adapt.py:109", "model_stand_in": "no previous-stage fit exists in a profiling cell: the dataset's truth tracer (tracer.json) stands in for the maximum-likelihood model whose lens light is subtracted", "over_sampling": "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184", "rectangular_mesh": "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight priors config/priors/mesh/rectangular.yaml:43-61"}, "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; no jwst production reference: borrows the hst preset 'hst_source_pix_2' with the jwst dataset", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; no jwst production reference: borrows the hst preset 'hst_source_pix_2' with the jwst dataset", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 707496 | recorded |  |
| sparse_operator_build_s | 0.1307372619630769 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1340 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>38 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0b1e2d6096349bd884ae` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0019324828987009822} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/156399273f77edf5957f` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.05278637079754844} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/169b3ef0c6dfbfd4bea7` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.8032890719012358} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/1901dcc82eb512c3758a` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.009439369197934866} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/1d658ce4ed92499d441f` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.0001770731061697006} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/264ac8fa88e7238b129e` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.030781954701524228} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/29a8f85f8904d0ea9983` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.03332913799677044} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/3286be64031b3aa734dc` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0306986419018358} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/4498184dd3dca3c7b2d2` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 8.06155032478273e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/4fab61817ee9c07c5202` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.04234345550648868} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/5a6c8c0aeca95340fe91` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.032694282999727875} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/5fc53565c1c48c9a5ad0` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.9290545752970502} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/65f7fe1f123f7585cd21` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.0026232742005959154} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/6a0ebb656c2679d84814` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.04041708710137755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/7574d04ead589e1fca7d` | interpolator_prefix | {"interpolator_prefix": 0.7938497027033009} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/interpolator_prefix_s` |
| `measurement/80b108ee2fea626058d1` | component_total | {"component_total": 6.609452997311018} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/8445731de1b20402a9cc` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.9290545752970502} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/8e5add58d824ecf31307` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.0017624698928557336} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/8ee588a9990b767a6bd3` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.07011473890161142} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/946bc829542e05436b05` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 0.8763386460021139} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/9ba2045fd7af08f6caac` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 8.06155032478273e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/9fe70dee13a010cc35cc` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0001770731061697006} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/a03eb4a088c6ae6ff58f` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0019324828987009822} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/a1eadd8629930166eb12` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.8032890719012358} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/a45547c308ac46ed577b` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0026232742005959154} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/a4df8c20c336d0b52b7c` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.8763386460021139} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/a502e63f7fb5d3d75c19` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 4.7475854922086} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/ad0537abb01f549be1f2` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.15216385530075058} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/b3da8a66f6feaaece00e` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.0017624698928557336} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/b6f54871f5cd5040a799` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.7938497027033009} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/b7251ed132db48213898` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.04041708710137755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/d0b76f293d652913d024` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 4.1911902371793985e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/d0e996781fec94c73c1c` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.002202159899752587} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/f466d4a51c93a6113b62` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 4.6376749984920025} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/f8d1be857296525fe80b` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 4.7475854922086} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/f957a3cd9667874f54f2` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.9481783014023677} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/fa3cf47f6391a0ef10e9` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 4.1911902371793985e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/fdd8f65807fdb2bd3e0d` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.007293848798144609} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/02999dce1ebd05854ccf` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.14490197494160384} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/03282bbeb11fd5379870` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.06501166091766208} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/061af801e2427d14bc9f` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.04090692906174809} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/0bb7f1cceebc2ec73d9b` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.2358456349465996} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/0cef7d8cba95ff7506c2` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.05088783404789865} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/0fb8506a53b726e85f12` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.2669181659584865} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/0ff4b21fa941c422b186` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.004446657025255263} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/123f4cf750088e13d07e` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.052249716012738645} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/13fb86014c00c665ff64` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.030405360972508788} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/163592d92f358b53c7e9` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.15327299002092332} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/1852f8153d69d89f2cc4` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.07157504104543477} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/1ced5ebbcb6b58c716b1` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.01082614902406931} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/231948466c0eda9a6bf6` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.04293051501736045} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/24b024356ab357b309a1` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 4.648575337021612} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/2643342f6f3122b2ff92` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0003189300186932087} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/2cbb2e7b1e8793f52e5c` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.257691067061387} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/2cd215f54a91443d5fed` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.05713559908326715} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/3b9229a11f17638fe626` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.8080205350415781} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/4198538e042bb16ea13e` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.03612905589397997} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/4b58d9975f5aad8d85d3` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.06983619998209178} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/5380fbf05acca1bc3da7` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.24809909099712968} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/5d8fddb5bf6934dee7f1` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 2.1679650590522215} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/5da452464869616f6560` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0037164600798860192} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/644c89fb5ba392aa90a7` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 3.0397372480947524} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/6593912dcf967e151073` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 3.4466116030234843} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/65beb8a47f8915905f5c` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.7894703530473635} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/66e0f8a8179376107e3d` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 0.9582287800731137} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/718976f94e8cabf907e8` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.360144337057136} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/72e5aa9c370a8db1845c` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.006656778044998646} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/76ea34ad612b16ace0a0` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.0028908089734613895} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/7a7f30c4150b819a734b` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.00033697602339088917} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/8708d93bf37af8f2dcc8` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.009238764992915094} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/8ac65b482adafdb5b0ed` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.14684999897144735} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/8c5ba097678539713a91` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0026474070036783814} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/8ccf678eb35ae3805ace` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.07527033402584493} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/8d04372bba7f379d6b6d` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.008207304985262454} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/8e238b5aecf08482b6fe` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.9279222899349406} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/92a2040a735bdd29ea4e` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.0059116369811818} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/9c0776e99cebedea3078` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.023001469089649618} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/9c763a5a6c5912c8338e` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.012861747993156314} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/aa7c376c9d6761b7f205` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.016636729007586837} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/adee984f12fff0711c21` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.15853086998686194} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/af03c0a38bf0dfe45e6b` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0003951339749619365} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/af390f9e7cf847562c31` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.029152269009500742} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/b07fcb80e23754f412b2` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0012434019008651376} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/b2f301a028e0976b4180` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.06334680400323123} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/b350bf2be83782db49b0` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.14824727806262672} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/b421bbceaf31608bd085` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.04820480803027749} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/b84a468499b7181a1117` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.024433650891296566} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/bbd311694165711fce09` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0038997610099613667} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/bd23a3c3040361f14e3b` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.044690652983263135} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/be6dd59f77202eb108ec` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.024730348959565163} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/c30778b6ce606ec3e5c8` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.006304366048425436} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/c6827b1142e95e2e9d34` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.8887950710486621} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/c89f7a64765a019160ba` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.002546799019910395} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/c8a24c2cd42c1681a106` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 1.3515649189939722} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/cead1bbc2076595bf11d` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.01713130797725171} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/d11bf7eb2931e7ba7dd3` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.06513993896078318} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/d75635fc2b23933f2b4f` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.0541989468038082e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/daadfbe26d0d500c992e` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.1064041790086776} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/df576674e2015a345fa3` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.92847923503723} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/e19590b36203e6caaadd` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.05494984798133373} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/e490772742c46d3d0208` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.010611339937895536} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/e80ffe996a22328b7548` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.035507673979736865} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/eb29bc66ec676d19aa51` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.04047734499908984} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/eb2f3415cea9c4934cdd` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.225981633993797} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/f14c1289d6c4e9bfa1f3` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 4.769568998948671} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/f620bdf083d77bb134d4` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.033185438020154834} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/f6618b5fe703b379b5ab` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.030973737011663616} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/f859954b9e390e5aa411` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.1005753269419074} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/fd21241843990078a814` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 4.772558444994502} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/ff726fbd66f7eb0d13c3` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.030218870961107314} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/jwst/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |

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
      "cpu_count": 236,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-10-5",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
    "host": "euclid-ral-compute-10-5",
    "measured_at": "2026-10-30T16:35:13Z",
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
