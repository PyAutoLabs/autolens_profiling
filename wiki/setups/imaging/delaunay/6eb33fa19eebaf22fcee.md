<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/14edab025de31e674faf`.

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
| source_pixels | 2500 | count |  |
| source_pixels_requested | 2500 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>29 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/07b7b1ffbf07e8a4df26` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.05519286380149424} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/17f6e74a6c687a941f42` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.0018021062947809695} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/4453b5f6a1c3c6ad529a` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.009410993102937937} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/4b993a2d93958b60d8bb` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.005006820196285844} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/4dc9a852a930d381e199` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0050207612104713915} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/55535eb4802ca5ad8708` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.00726223080419004} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/5619c7486b1755c26b39` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.02516682241111994} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/57b7df55cc5173f4ee90` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.0018273321446031332} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/5f0cfbd69847c7c2c06f` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.013844350585713983} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/69cd8af42325c4725e53` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.027556794183328747} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/74fb37773ea4ebc18b20` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.0229525834787637} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/765b947fbba5616e5d9e` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.029755833093076944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/7d278c2c7beca15391f0` | s0_nnls_pdip_vmap16.steady_per_call_s | {"s0_nnls_pdip_vmap16.steady_per_call_s": 0.05823536091193091} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/89439aab49c2c811d896` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.0018513149116188288} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/8a401d28d88e71f686bf` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.020588352670893074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/90a5db685a8fa956e9ea` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.012083561206236482} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/b9b8a111c1ec3f6ba5c4` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.004801731463521719} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/c32eb0c401a0299ec0e4` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.001816496206447482} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/c44c909b517217b0f9f6` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.018388715712353586} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/c45c6793ccd8f9aa3032` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 0.1114834533073008} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/cb2c3789076fd53f7cb5` | s3_nnls_pdip_vmap16.steady_per_call_s | {"s3_nnls_pdip_vmap16.steady_per_call_s": 0.04753639876726083} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/cccd41df10e394f9b328` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.01615615296177566} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/d1513e23cabffff5b641` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.06558148097246885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/d1cd57823dba02891a13` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.002404336677864194} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/eceacfd24635f1cf673b` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.012021067505702376} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/efba053940f615d1905c` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 0.0934074908029288} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/f243a7be554b7c66fd42` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.011709957383573055} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/f975224f7250acd8b94f` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.002504684031009674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/fcd9de850af0f1698919` | s3_active_set_masked_p1_vmap16.steady_per_call_s | {"s3_active_set_masked_p1_vmap16.steady_per_call_s": 0.004087267449358478} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_vmap16/steady_per_call_s` |

</details>

### compile

<details><summary>81 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/010bf6824d86a6b19efd` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 4.3156170938164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/14af2fea6f8162e05d38` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.015274819917976856} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/15aa3a88e6044b43eb60` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.0033431099727749825} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/19c0af8d92ea5b8d051f` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.45139724714681506} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/1a8430ee79586956e289` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.0499730706214905e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/1a9e0fa16080d8fc990c` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.0018702587112784386} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/1e806a451db47f47784d` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.013059591874480247} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/1f7b1e35fb15ff62add6` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.03521938994526863} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/2b2a318529e23ed011db` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.011794209945946932} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/2c5853d98e197a6159e3` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.005999873857945204} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/2d4edde9d61952b338cd` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.03520384058356285} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/2e24e5150372b69f8af6` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.46125277737155557} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/2f99d80cdd4f79004cbd` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.046452642884105444} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/33f4339e2f49a7fba606` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.00877849804237485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/3bba2c4552c2c8dfed54` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.05842005414888263} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/3d42d9e6caca015eddb7` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.019978141877800226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/3d91b4168960f3252efe` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.03518282109871507} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/466a664feaaff4134d8f` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.4559421190060675} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/46d792ad1929d183c66b` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.42740513011813164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/4bacdababf1b005fa293` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.03516518184915185} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/4bd10237efe96a381d2e` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.05858237110078335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/4cdc56b951e4d311d4a3` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.443954371381551} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/4d73e0e3821a953124fd` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.018699610140174627} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/4d83289b34dbd3cfbeab` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 7.139984518289566e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/4fabab438c5027fbcfe4` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.44475729623809457} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/5029a66fd094fc355e59` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.0025183153338730335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/57bbb7ac3702dcf92cc9` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.017629353795200586} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/5c40740375483fba7aef` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.006656960118561983} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/5ee5c6061cb4b770915d` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.00028793932870030403} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/5eebf413e9943c7d7f23` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.44367709290236235} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/688ebbca4e23c7d2444c` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.3388790637254715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/69302825d46f221dfbb6` | s3_nnls_pdip_vmap16.first_call_s | {"s3_nnls_pdip_vmap16.first_call_s": 1.300469839014113} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_vmap16/first_call_s` |
| `measurement/6e3acb487f5086a76659` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.450333661865443} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/7700368a9650ddfc6ce5` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.034872813150286674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/7793907ebf920147f415` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.05698681203648448} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/780863e948ac8cd994de` | s0_nnls_pdip_vmap16.first_call_s | {"s0_nnls_pdip_vmap16.first_call_s": 1.4662204729393125} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_vmap16/first_call_s` |
| `measurement/78508f141f2fc5e6b491` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 13.775403275154531} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/795f92321b30a41a953e` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.009770851116627455} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/79e6d752bdd08ea39b67` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.07652250491082668} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/7a7d737897243879c0b7` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.03125679399818182} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/7ad771d44b6d89c30477` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.00017408980056643486} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/7db62029f1d7f5d84210` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.010953205171972513} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/823697afff36a86770e9` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.03866131976246834} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/877adfefa7e5aba30d1b` | s3_active_set_masked_p1_vmap16.first_call_s | {"s3_active_set_masked_p1_vmap16.first_call_s": 0.3128836490213871} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_vmap16/first_call_s` |
| `measurement/89f22e53a008355db6fa` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.0018732091411948204} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/8b6105383c1175edea24` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.0352138108573854} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/8b6ac401e93817ff8265` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.06730168871581554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/8bc1145e8d2d138f6be9` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.08438639808446169} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/914318122b914e1ed24f` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 0.10217389278113842} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/9652de537d7e52ee15a4` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.3104961640201509} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/99de5bc590805b249278` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.006639889907091856} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/9af120f64b8be11ed89b` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.034830872900784016} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/9eabc02eec9c1683f018` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.03348962077870965} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/ab608067c2fda67f96d1` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.44197414303198457} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/ab9432ca605ff0bbd7d0` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0003073480911552906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/adca3aba4b8b9a31bd62` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 1.0750256478786469e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/af3d61eb78c50f311eef` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.4122149501927197} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/b1de1297a7ee2675a4fa` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 3.4394532428123057} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/b1fbdbf21709a97707d5` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.02663180185481906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/be1bd47254c50f771853` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.024595243856310844} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/c0010e9b17d780656190` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.31481555895879865} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/c010d6438e89057dc256` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.02217299910262227} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/c5ac8716e2195c90ca11` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.034766693133860826} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/ca8127e2cf453861ddc2` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.022821594960987568} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/cf385d58773d5dad5623` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.03594701597467065} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/d5c9abf66307d16a6eb7` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 0.21182043105363846} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/d61010346fce0b27e5d9` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.04199053067713976} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/d79146518ffba1456954` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.06327733397483826} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/d99481b2a953077b675c` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.022607445251196623} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/db3826153dda5b216ced` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.4264616062864661} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/df29cb45244cca55903c` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.46820365684106946} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/e4a00a2d57015c7fa900` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.03545164875686169} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/e4c932116c224010b88d` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.45241533033549786} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/eb723d720b1964fd329b` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.029076407197862864} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/ed8f801be5f4e204357f` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.4850381566211581} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/efa4b3cafdbe20a152d0` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.015576537232846022} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/f1e80c01ce01e1a15afb` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.4395304168574512} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/f2debc514b8653cf2486` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.012783183716237545} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/fb5882c615678641d888` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.035355569794774055} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/fddf10fe379390f72a9b` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.00983648095279932} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/fdf3081f5b67bc3dae49` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.4507108200341463} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/scans/n2500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 8957 MiB, 81920 MiB",
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
