<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · euclid

[Model index](index.md)

Exact setup ID: `imaging/delaunay/euclid/3f58b1c0e0ec3499d566`.

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
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 68700 | recorded |  |
| sparse_operator_build_s | 0.15684576699277386 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1320 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>38 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/06b1f8956544b806c729` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.0002462306001689285} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/0939ef97ac2d933a9677` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.005407305702101439} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/0fdde885c2adc9821f1d` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.0300982496002689} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/10c08230a5d74cb1caab` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.0002524149022065103} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/111d930d87c66f2beea4` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 1.824169885367155e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/12cf47e5548d662f6621` | interpolator_prefix | {"interpolator_prefix": 0.11687719219771679} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/interpolator_prefix_s` |
| `measurement/37b936c9a4386b39cc44` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.5824048645998119} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/4afca80d1fbe90740fcf` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 1.9645699649117886e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/4f5327341787af7fcc99` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 1.9645699649117886e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/58cf7ea5827a5f18cfbd` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.1266225524013862} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/5ef7350eb4829b9c9d02` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.11914824340201449} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/62f55a9daeac7a4b67e9` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.02288535269908607} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/64c3769104adbf6691d0` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0002462306001689285} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/65cbe18f0deac512a52f` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.005943839301471599} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/7725e39415e57231f71d` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0022710512042977} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/81f123376e4f9e402c8d` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.0010920239001279697} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/848bf7cec4f268ba3f4d` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.008963603799929842} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/95e1837994e1f660b767` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.05572530719800852} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/a4bde7d420aa005cd9cf` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0006106504006311297} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/ad4bd89ddc0621011159` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.022632200800580905} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/afa787b26f4452c72ed9` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.0300982496002689} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/c07cba58e036acdbf548` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 1.824169885367155e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/c26f43e935ce729ab189` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.0003463001019554213} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/c3698383a6746f82e7d3` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.0004247500008204952} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/c705e1fc31c19e448505` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.02543134599982295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/c7ce621206a0a1db0c72` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0003463001019554213} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/cd18e7f8bf82950c8251` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0006106504006311297} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/cd456397a1f53bcf11fa` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.5777858404006111} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/cfabc4ef67d98bf1aa9e` | component_total | {"component_total": 1.1854304654087173} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/total_step_by_step` |
| `measurement/d21a8c404412528653aa` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.026725795399397612} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/d5b5f7ca14355ece1ebf` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.4425402641994879} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/e0031d601c5532627292` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.11687719219771679} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/e11969d6de70ae246e1d` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.4425402641994879} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/e5fadfbfb3950627cbd2` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.4362328264018288} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/ea8ad32698df455df4a0` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.11914824340201449} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/ed008b1a8f9416df1a30` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.0002524149022065103} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/f2c05b4990bb51e1a36f` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.5824048645998119} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/f6dc76d2573c64e2c5bb` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 0.1266225524013862} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/032e74d38b4c56a0b5c2` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.023696024989476427} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/04d2d0a06380419380af` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.06938105198787525} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/09cfb5c1120c812e2317` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.007674217980820686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/0b826d8a821ade3bf2d3` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.41319362199283205} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/11314c27945e0f4aa339` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.027870983991306275} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/1342af2d45004cf6cc63` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.12018665901268832} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/15cbd80d0a9c34103db7` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.008601976005593315} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/18d3b3f8653f7b16a41d` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.16643520997604355} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/1c2dc57b610d39e43adb` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.028683306998573244} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/211ed90498670b0dce22` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.41701839602319524} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/26097b04015855d49360` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.10787404800066724} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/2969f6b471a28f0143a4` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.16592567699262872} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/2b4ad822258b2aeec46f` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.007943323988001794} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/2bb533d996262577a367` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.134999911300838e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/2f6b129f0db4db001ef4` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.019753812986891717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/364763fdd1ecdb3c1c04` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.00025480700423941016} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/36f030c1b7692442384d` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.058219066995661706} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/37bcd1f9c6a50ec48840` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.005665951990522444} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/381b84a4a7fb63927541` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.05454070199630223} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/3937c047135fb3c0672a` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 2.8644532060134225} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/3f28f6f72148939d315c` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.26028467700234614} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/4244eda077687a731681` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.030974825989687815} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/4264e16e7cce1f034f57` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.02546554198488593} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/42936d49f00e37dadb53` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 1.6717957690125331} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/4403c8968ae06eab2a27` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.048320636997232214} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/482edcee93d6338bf267` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0011668429942801595} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/4a4acdce865d6a255e88` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.07127384899649769} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/4cef33453b31770625dd` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.005307835002895445} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/544ff6a3827bb0c51e1a` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.009059850999619812} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/5d1ec98a2a814ab010b7` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.25557906401809305} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/61bfb7082ed128d64943` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.1741687650210224} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/62fc727e0186f517dec4` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.01303444299264811} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/677c2ccb6bea538a5140` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.08311393301119097} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/691357ca8190738be4b0` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.026075835019582883} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/6c2b9ecf9f8416dc8d8a` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0002077370008919388} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/702f299ce999e8cf993c` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.07795370600069873} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/7890607841f26a8b05c3` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0016874669818207622} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/7aaa0d77c85afcd386de` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.045632900000782683} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/8ec47e760370f235319e` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0003052460087928921} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/94252e50011a4dc8b323` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.27886631299043074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/978af5c74b5b6a6de125` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.5732153400022071} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/9c9d44fcecb39db28fcc` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 0.9938525970210321} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/ac50553beda4efd76901` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.046312821010360494} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/ae36d7b21593e27b86f7` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.03196551400469616} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/aff0cd8a347ee1bf6889` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.22199859001557343} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/b2badf5c453c8606d6a6` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.050080015003914014} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/b37ee607ae73e3bddbfb` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.029268166981637478} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/b502022ce3f83d7fea96` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.012215531984111294} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/b8e492379109a7b4cd36` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.00115833300515078} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/ba832ce2685009c22b63` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.3330334849888459} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/bad3942853d645fb7519` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.5697859019855969} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/bcd69cfdc5d8a3982605` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 2.6916334140114486} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/bea1476850cb035723f7` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.1209758420009166} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/bec3505044e91b7450af` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0012212129950057715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/bfc9dfdb03cca0152541` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 1.2034197900211439} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/c0685a9c691334e82b35` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.07682859199121594} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/c0fe46f250d64e867169` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.037622701987857} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/c683cb53d95d1d870295` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.022974063002038747} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/ccd57ac5b8ae20805f31` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.0016253409849014133} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/cf5d7a1f696a63bd61f9` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.03191771201090887} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/d251543ca48917771de7` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.13024123100331053} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/d7959e87f396b143df5f` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.19074305301182903} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/dc59a9d36836b0adc1b7` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.07127437999588437} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/dde7d6e1b368d9c2d9c0` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.45304023000062443} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/dfd7d6550c7292d4ac1f` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.012661068001762033} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/e72d168f8e35ba791921` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0006820109847467393} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/f1a611033e08cfbf19a6` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.2494906380015891} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/f1da50ee74edc78c9bc1` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.09046587001648732} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/f9bb46ad11b7e63b88f3` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 3.8181536679912824} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/f9df5003b38c3ae9d52d` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.010167236992856488} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/fa0f258cdb8a6f4b5dbf` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.023166858998592943} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/ff9a03c51aaa2d0c4bd7` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.016636928980005905} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |

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
    "measured_at": "2026-10-25T14:13:32Z",
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
