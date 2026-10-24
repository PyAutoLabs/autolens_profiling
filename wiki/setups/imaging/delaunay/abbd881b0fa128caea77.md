<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/95008036ee25b35fb8a1`.

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
| source_pixels | 500 | count |  |
| source_pixels_requested | 500 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>26 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/087839c7580d921570f6` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.0021106100000906737} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/0a20d6fb4c3025e01db8` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 0.39117501999971865} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/19478e2a19b83668b2c2` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.005512490000182879} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/2750940d9dccaf2790f9` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.0020683200000348734} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/345d6250f58caecdd24f` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.005547080000178539} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/39f1eece32637d987fb2` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.002963679999811575} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/3a1a5cc792b10f7ea53e` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.031085520000124232} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/48f32dae7830ae10378b` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.006972959999984596} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/49cd5db34f67758c9d12` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.029297959999894374} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/4dc028d8b0cd313af239` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.009605999999985215} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/53917abd7fb183582ed6` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.06942323000002944} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/59da219d2a504956f51b` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.10611730999989959} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/6e32c6ad87b40a82b2d4` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.03484664000025077} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/6e8631035a3c21a49a64` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.0020663500003138325} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/89f6305d97f3a725a2b8` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 0.45282665000013367} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/9b7db301b7b02b06d5e0` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.0020298399998864626} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/afcfbcd2356e2fff78cc` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.08822950000030687} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/c1104efaf838d6939cb8` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.021233230000143523} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/c56d5527b875fbf0ec19` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.023702010000124573} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/df2bbd8a3ad83d4ce778` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.03319600000031642} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/e66220f2ad290cb13be7` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.015014190000147209} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/ea6a5c2412f8855a7089` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.026504600000043864} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/ed4563aa422ee9158bf8` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.01787984000002325} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/f13add2dd9af111542b9` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.037806959999943504} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/f2ab8b96e458c371a6ec` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.02166800000013609} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/f6b53bf97762c66b9bc1` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.005680819999906816} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |

</details>

### compile

<details><summary>78 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/049694f47b42695a5132` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.038509900001372444} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/0504c1ee4c5893325941` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.0063286999975389335} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/05b6c527bb711b8fef98` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.34051629999885336} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/08bb82bdf0c4f3092713` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.2293935000016063} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/0ce0f001817783f519c2` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0002865000024030451} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/0fb8a3d307cd3f14f156` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.039037300000927644} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/11be3cfab91abd6089ba` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.057618899998487905} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/130028952e23125aeac4` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.039603299999726005} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/156dde40c812485d864f` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.05516119999811053} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/17a0a17f7b2a7aea0f61` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.2843792999992729} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/19e58f055c7b81842da8` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.003627599999163067} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/1db480ce3d3f26004158` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.02399169999989681} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/26109209c066f3659ac7` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.04022400000030757} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/2b1ebe72d3e19d94ce71` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.36293139999907} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/318bf5a2126ff9c551fa` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.10807200000272132} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/326075ce60a23a9d63be` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.0038371000009647105} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/356289a57af73cab3fa6` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.04435369999919203} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/38ff52947f98037ab0ca` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.00030469999910565093} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/402bdc18ccdc9732af3c` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.02499800000077812} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/42c119ac455f61583e37` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.1999996786471456e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/432ceb3b1b10df26c0c3` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.00028450000172597356} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/469225fd63f97faef2cc` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.035318799997185124} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/4971868f8e20de9e8385` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.007519399998273002} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/513ced9f476383b10c0f` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.010336799998185597} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/55155bd96cd446cb6307` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.08564810000098078} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/552c5a29d7c1db7cd392` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.01666309999927762} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/595ee1596aeab7fd8a7d` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.023698199998761993} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/5c00127cb56ca80deddc` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 1.4057300999993458} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/6233a279bf13ba4cfd98` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.46630450000157} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/7085114fa6f2443e4b21` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.027880700003152015} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/72b999be4b316d100891` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.006499700000858866} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/74c4724ea9c5417adb84` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 0.5184521000010136} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/76ef21f091927e16632e` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.00707469999906607} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/78d6062bffd4c8a8e0e1` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.2614651000003505} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/79bbe32410c52008bb84` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.3417087999987416} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/82485c804a10b3ed909f` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.22255219999715337} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/82fd138900ccc80ce696` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.019540600002073916} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/8575e56da643d1968dd0` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.01555010000083712} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/8735ed678a33c43e37f0` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.28089770000224235} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/8755c823417b23b8ccb3` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.020060800001374446} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/8af57538f838dcd58a87` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.032767900000180816} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/8bc80345b76f3853a436` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.04456049999862444} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/91ce54ae408dd06b4e69` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.04614220000075875} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/94b2f83bb8d0c88de64f` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.046150199999829056} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/963aa774a7c73ffef3f6` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.3181575000016892} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/985b975a17f86d350863` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.02640419999806909} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/9882e308d28035709c77` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.03372080000190181} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/9cb8678ce19bead84b51` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.05213309999817284} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/9d7eff67878d6230d187` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.002178200000344077} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/a7f3aeaacb225a1ea1f8` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.32301210000150604} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/ac85afe5a0b558aeb315` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.04695779999747174} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/adcadb9cdd74613edcba` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 0.380692499998986} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/b0ba9515997ffae9e192` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.042054100002133055} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/b12b08c8aa21e63c9e12` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.006826600001659244} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/b1fed7ab0bb9daa5fa0c` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.036385300001711585} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/b6a0d1eac08a32075e76` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 7.098661799998808} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/bc8a25a32b97a6980ea8` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.08837520000088261} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/bd59c88e477513bd67cc` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.3107339000016509} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/c12eaca048a9f3c2adfa` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.016566999998758547} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/c1f8261c5c8687998ee4` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.05390840000109165} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/c37380aaf1f11c26d228` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.031340899997303495} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/c69791c9e82534202ec4` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.02319119999810937} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/c98ff8652f5bc83046fe` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.06719449999945937} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/d0b93540cfd948e7d0b8` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.2328641999993124} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/d1edb22c968f0734b137` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 1.1399999493733048e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/d4a11865c35ffdcf6072` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.11227940000026138} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/d75801094daf54b37d93` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.002319599996553734} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/d7c6aaa5a4305e152d4f` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.193843699998979} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/d924503f82b4f3213ba5` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.30900240000119084} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/df3c4d32c32f58d1fc41` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.2645046999969054} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/e47f6c1171eefe8e58cc` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.03868940000029397} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/e9bacc5b58003ada6b39` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.3074226999997336} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/ec48632ccb3036597957` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.05294590000266908} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/ec66d9ed396f5fac5828` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.04129289999764296} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/f13938d0f67e3dab552c` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 3.020641200000682} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/f6eabcdd2ed41093d203` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.0022888000021339394} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/f9c4100569088ee3719b` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 1.2099997547920793e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/fb55feb975e2716862da` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.04871310000089579} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n500/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |

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
      "cpu_count": 8,
      "device": "cpu:0",
      "hostname": "DESKTOP-H143S82",
      "omp_num_threads": "1",
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
    "host": "DESKTOP-H143S82",
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
