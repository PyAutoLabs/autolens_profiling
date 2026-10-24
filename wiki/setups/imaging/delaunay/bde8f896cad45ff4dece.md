<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/a882de6431d0dbb4d901`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh | "delaunay" | recorded |  |
| mesh_shape | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pass_budget_max | 12 | recorded |  |
| pass_budgets | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | recorded |  |
| pins_mode | "fp64" | recorded |  |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| safe_budget | 7 | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| source_pixels_requested | 1500 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>29 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00515dd6647a00dd0700` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.002850952884182334} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/0605437ae032f7146812` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.016034441720694303} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/10a57112cd3e417bca61` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.004871608782559633} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/187384badd7c86f197ab` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.0042670876253396274} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/21318386f8a429b1c06c` | s3_active_set_masked_p2_vmap16.steady_per_call_s | {"s3_active_set_masked_p2_vmap16.steady_per_call_s": 0.0021618073922581972} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_vmap16/steady_per_call_s` |
| `measurement/229f5c1fe8f7cfdf04a6` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.0011310951318591832} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/258e577774bd479de133` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.02813937240280211} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/2d72a9f6a82c020ef5cb` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.005482046399265528} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/2f32731d4253c7d6cefa` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.0011303781997412444} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/2f413a34b6fbccdaf816` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.0011904349084943533} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/548938c6107d1f07ae0d` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.0011592781171202659} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/66d63ecb418f9844d924` | s3_nnls_pdip_vmap16.steady_per_call_s | {"s3_nnls_pdip_vmap16.steady_per_call_s": 0.016014735848875715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/6eec6e1580e910953d17` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.017333306884393097} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/796fd926182b87c63877` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 0.06533012646250427} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/a0b01949827541851b61` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.009463387820869685} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/a30a469a9eb61b9f7221` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.012110644904896616} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/b03e9ee64208b6d30e69` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.010762027092278003} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/b0dfd6c736a6aaa51a78` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.0014571722596883774} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/c57f357e2b39ee20a5b4` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.008242136985063552} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/c6db963454fd2083adac` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.003177676908671856} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/d649b7dc6a0750ef5a61` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.006853685295209288} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/dfef135a712e5c407045` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.014730889489874244} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/e261fc6ac6a55f849a95` | s0_nnls_pdip_vmap16.steady_per_call_s | {"s0_nnls_pdip_vmap16.steady_per_call_s": 0.021810196767910385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/e729636e96500c1c204c` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.004815887240692973} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/ea899d98eb0dfceb6883` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.003041195822879672} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/eba3ee7dbcd03aad0d4c` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.0015209368895739317} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/f10ba66af1c83bbc1e12` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 0.049442224111407994} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/f60c0a73d23b3fb60915` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.013678947696462274} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/f66482a19877e9ed33da` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.03755903411656618} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |

</details>

### compile

<details><summary>81 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0260fb7ff0cec19c9606` | s3_active_set_masked_p2_vmap16.first_call_s | {"s3_active_set_masked_p2_vmap16.first_call_s": 0.46121700713410974} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_vmap16/first_call_s` |
| `measurement/0857c56dd12bf0acfd5f` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.02965326327830553} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/08e2a972e53fd996d5e2` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.44429466873407364} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/0a51597d8550db229eda` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.47624959889799356} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/0c70eb43a8ebd89de6cc` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.4297546548768878} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/0cfcab6f35f521b134e5` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.03408303717151284} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/0d2554316d7989c2896b` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.009697170928120613} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/16dd8c5819e97168fc8f` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 7.680151611566544e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/1a0375ff1de13b8bfeba` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.044568613171577454} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/1a9927d75b18a769fbe7` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.03471676306799054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/1d81f6eb2effeffae74c` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.08277816418558359} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/1f07bf8d4b8c3de73da8` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 3.408255998045206} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/24c799fb01d5cb69ef20` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0009331949986517429} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/2814dc2aba2e30443c54` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.4498062557540834} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/2898beac64d9de569eba` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.0034375390969216824} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/2c6ec6c99edac411d2fa` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.03485652199015021} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/2f73619c15cd2fb0782b` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.033680859953165054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/32ab0b2ba7bc10dd0991` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.4408013988286257} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/3987fc44b6ea79a3f541` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.019774931948632002} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/3a2e1a7e02b6723f7650` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.01234034588560462} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/4087e6825d6b823a3fac` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.47412130096927285} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/41638edfed654f827bc1` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.48058667313307524} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/4825b30b39184da5861b` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.019241536036133766} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/4f030061f510b4eccde3` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.0012622023932635784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/51ad7b6306cec7cbf03f` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.06291332421824336} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/560002d171176815004e` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.039620062336325645} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/59c5fa8881c91b1b83a1` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.00020003877580165863} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/5b87acfb5542b81d3b73` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.012138298247009516} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/5c4fca5ac6ded53c197f` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.03392644692212343} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/6125d44f52b65df1ebb8` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.0017144600860774517} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/61c237214f1b34ec7347` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.03886224702000618} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/68432c63b81135523325` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.029391793999820948} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/68fc1069d50f5c537e54` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.4527524681761861} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/74645b4835874019b593` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.022099327761679888} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/7523ca4492d2eccc5eb1` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.44929229002445936} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/75ee371a46e4b3e3e87c` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.3300340427085757} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/78b74671e8aa96e79a5f` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.4775039008818567} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/79a3436e0b7e9de35f3b` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.013370330911129713} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/79fc59f9e0e4bd997851` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.0388007671572268} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/7c9020fbc9306c4ea391` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.46402528090402484} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/856c17c0fc4b63b0f403` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.03495109220966697} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/876e90e09d4141b61b03` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.4319576919078827} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/8c95116a44345a77c1c2` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.0073107569478452206} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/95878b078cd2eb28273a` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.008198041003197432} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/95be36d251e46d8ff9e0` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 0.05855502188205719} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/9f0ecf6dc6f5839b7d71` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.00029620807617902756} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/9f824b861641478f065d` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.004701981786638498} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/a1b693cce2a0cba3bfb2` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.0019886880181729794} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/a6f75ef0fc61d19b297e` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.4701402192004025} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/b4fe6cf5627c90008f82` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.016036193817853928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/b8523c18e3cf5683390d` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.432736377697438} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/b92ba793e2aef61bf80e` | s0_nnls_pdip_vmap16.first_call_s | {"s0_nnls_pdip_vmap16.first_call_s": 0.899213835131377} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_vmap16/first_call_s` |
| `measurement/bfea5acc569bb6057c0e` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.034736033994704485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/c313c933ba34515f4324` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.0012002130970358849} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/c5c8fa5c86438f21ec07` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.024894930887967348} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/c7d528912e8ec5231920` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.010703325737267733} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/c8ff26393e514bdc66ac` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.0061002131551504135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/cbb20b03ef2de5d5f658` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 13.921968382783234} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/cf2111e82c208e337f54` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.017279756721109152} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/cf74dc6027e08ef06521` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.014711821917444468} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/cfd441369040becb620d` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 0.16993959993124008} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/d35425d79bdc79d63df4` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.1810847450979054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/d51f92c61d67899642df` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.43510143272578716} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/d527adf4415d25892507` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.14703629817813635} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/d5d543af3e8e8c05cf36` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.137005165219307e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/d630cd0db66ab5ee5f82` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.033959228079766035} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/d9d6acea02923613f341` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.4420087309554219} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/dc6ce6d6d6522c6661b4` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.006882248912006617} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/de140887f5b0db9ba95b` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 4.097027704119682e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/e2828d561510f7091217` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.003825316671282053} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/e65fa40607477d983501` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.008094222284853458} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/e99795b7f7de9fc33cd0` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.03597896592691541} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/eac4d23a185bd3fdb65c` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.009497813880443573} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/f7523ac08e8596dde143` | s3_nnls_pdip_vmap16.first_call_s | {"s3_nnls_pdip_vmap16.first_call_s": 0.7818222432397306} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_vmap16/first_call_s` |
| `measurement/f8eaaa5c0e778440986a` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.03413543710485101} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/fad3dad9304bea27c423` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.020642176270484924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/fc7bba1b0abf01ad787f` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.004926899913698435} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/fcb40951dfe723ff2571` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.0386379100382328} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/fd6fa9aec1c5d314d59a` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.0702236988581717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/fe3f09f3ad61aac39f78` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.03566426783800125} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/fe73edb1d1b7026fdcca` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 4.0306265680119395} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n1500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |

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
      "hostname": "euclid-ral-gpu-2",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 4733 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoLens": "2026.8.17.1"
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
    "has_provenance": false,
    "host": "euclid-ral-gpu-2",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
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
