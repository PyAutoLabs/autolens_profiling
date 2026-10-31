<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · euclid

[Model index](index.md)

Exact setup ID: `imaging/delaunay/euclid/268290c28f75b726cb37`.

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
| sparse_operator_build_s | 0.13708216208033264 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1320 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>38 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00e094f864363d70cd41` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.11549082649871707} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/02c4bdb991f9f139f783` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.023107678804080933} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/065d7c060d246810b6e7` | component_total | {"component_total": 1.179683963395655} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/09b729f97ca89e59c2d4` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.4289380489033647} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/20c3d6538550b829d8d2` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.02346673549618572} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/2b41ea093112657b7dbb` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.5733583937049843} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/2fd78b14c10c7e83348b` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.11912896400317549} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/32e55255917a6055ea28` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0003706626011990011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/34f3071625f1b7b43d64` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 1.732890959829092e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/473a026176236edd5baf` | interpolator_prefix | {"interpolator_prefix": 0.11549082649871707} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/interpolator_prefix_s` |
| `measurement/490c7c6d91a8aa236d31` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.4394993526977487} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/4a453d043c39545536b1` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0006094526965171098} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/4b74ebf3d28562e96202` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.0002742756973020732} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/533355401c8850eb5f0c` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.5767571251024493} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/5aba1b63734d00dc839b` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.5767571251024493} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/5d8ff9b56eda4b2d95a1` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.02521986359497532} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/6c5cb9e31cbc3f7c844e` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.0004119090037420392} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/7322b8e01004fa59dfbb` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.00025432889815419913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/7613740dae9ed191c335` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.031996202596928924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/7d3618cb8bff7d025439` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 0.126249471900519} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/818a52cc25fae5cd41a5` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0002742756973020732} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/91ff603911095887871f` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.009465222805738449} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/985f2a2b0dd0f915885d` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.003777239890769124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/9896a23250d01bdc3f2e` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 1.7624790780246256e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/9f45bb8745d64fb9ac34` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.0006996016949415207} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/bbbea242384ec2ef1fea` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 1.732890959829092e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/c2cd42b805b5f81a0335` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.00025432889815419913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/cf39d372fb6c29a4f34c` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.031996202596928924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/d281696866f01e81cd98` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 1.7624790780246256e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/db97369ff08b14f1a952` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.003638137504458419} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/e079f301fad44906accf` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.11912896400317549} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/e0f72c9df900b9e2bd78` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0006094526965171098} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/e1a81a2dea7939775cc5` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.00684628919698298} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/e8a2f6831f7fc6b51199` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.0003706626011990011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/eb8ee34e77f5e94e0fff` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.026162309304345398} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/fb24091fb08968e8bc76` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.054812236898578705} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/ff8c03b755ce05c4f845` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.126249471900519} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/ffbc5c62655d4b12f524` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.4394993526977487} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0051b98c70d2cf3c9d50` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.16070928901899606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/073778f2382f3c101512` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.08663078397512436} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/0be6b40fca8ad6d38588` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.1209767039399594} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/0e08a4db5a953db7ea26` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.00037957506719976664} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/0e99aa17794ea79ab6ab` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.006897747050970793} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/103dcd6d487c88dea929` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.0404820260591805} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/1387cf98ce2c656f30d5` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.028547559049911797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/15c0bd8798463c80e5b9` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.31634786899667233} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/1ea9ae63438b12c81732` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.023423339938744903} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/21037865739beee38e7f` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.0232582229655236} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/23253a253eb6f49d08ab` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.026039868011139333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/2a48ea4d5b70692e0dba` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.00030406599398702383} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/2b852fcc4195a9f7a0ae` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.4368244290817529} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/2c4431031dd5a5ce901d` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.033164364052936435} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/32d1c13fa772918af308` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.16212818201165646} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/342bf30af5251a48b208` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.03985655400902033} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/3522b2202f4c629cea3c` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.0009782080305740237} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/3a97b4e3101c0a15e426` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.013343670987524092} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/4268215b9723fa459875` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.012723986990749836} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/45efda7acc746dd3743b` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 2.682413366972469} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/4b97ec70be33480a53aa` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.04383644601330161} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/4ceb831a4047c1ae1a2d` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.2277893170248717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/50f58613c9ac9825dccc` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.09672788402531296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/5226ea993f6d4bd56fdd` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.018154863035306334} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/5700c403ddce5476cd50` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.1271918599959463} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/586adf1b15178ae71fb3` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.026058677933178842} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/5c530287646afe6e8151` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.007221632986329496} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/68e9b93fff6177688157` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0011868459405377507} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/6be870fde68f7bf76d4a` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.5739947989350185} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/6c01844d8d2936968aeb` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.06653874402400106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/6e6c59e9baecbcdc7c2e` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.1039082892239094e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/7cf0f36b31eb9248c889` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.09119823900982738} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/833fb12d769e4973abea` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.42838108900468796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/86db93e09af78a76e6a8` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.010228718048892915} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/8a70e9d842c9ae3c8259` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.02497116196900606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/8ff39debca55faf3a3f1` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.17290053400211036} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/95cc50b0f7ab9b9476dd` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.23226532305125147} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/97acefab764c8c2fbfc1` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 0.914441789034754} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/98a43e98b8a92f3d0f8f` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.0039976430125534534} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/990aa48a6df6f70ac05c` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.026187027106061578} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/9aed61a1217bf11aa823` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.5758773570414633} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/9c09318e5d8625f64587` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.005531145026907325} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/a101e14734df92093828` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.026046878076158464} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/a120b38be311eefeb5eb` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0003187459660694003} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/a137c2435f0c1451f030` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.028392520966008306} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/a5c6939ef601b82716a5` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.03229083400219679} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/a7e0e1fa2d424c678b9b` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 2.342593799927272} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/a93aa6b0210e1d171980` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0010942460503429174} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/b7fd5180bed48c7fff7b` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.24211818596813828} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/bc4bc3076b9a37d4b076` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 1.01036195398774} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/bf7b2b5942472adedbf9` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.42708289506845176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/c1a5408a94f204e42103` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 1.6252014839556068} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/c1e878d00283ff4389aa` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0025079200277104974} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/c2408b6f26ebffe33198` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.12063465896062553} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/c37f912c60d278ecc890` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.012965115020051599} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/c549d17dd0812e95b218` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.019074790994636714} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/c81032c0c838fb6358ab` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.007502299966290593} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/cdf88f49d2d216ab9266` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0012372239725664258} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/d6b419ea2b1c0f10214a` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.066929170046933} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/db0b7e0f33443168f850` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.05852544098161161} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/dd53001e916a09f96832` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.009393667918629944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/dd570aa700e365b4693c` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.008366719004698098} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/e6c2781c8060e9b4a8bc` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.044122793013229966} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/e715bf0d8487042fa6b2` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0006936019053682685} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/eb830ae3463b8fa272a0` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.05465045594610274} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/ebf0d287ee68e7cc8927` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.20476928201969713} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/f0577f7aa25cfc7b6cc1` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 3.137525398051366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/f14eba24c5b9de342e67` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.165628771064803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/f5e2428f72b8d0a12234` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.052349664038047194} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/fb420e15607d69401d99` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.06546398799400777} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/fe0c2d37b4e6dc2a4320` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.2407578220590949} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/fe72aed8c75b92af5eac` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.002675376948900521} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_sparse_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |

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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-30T16:30:29Z",
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
