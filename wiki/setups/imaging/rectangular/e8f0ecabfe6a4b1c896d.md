<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/a90bdb0094c5d416997b`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh | "rectangular" | recorded |  |
| mesh_shape | [63, 63] | recorded |  |
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
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "scheme": "constant"} | name |  |
| safe_budget | 11 | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 3969 | count |  |
| source_pixels_requested | 4000 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>26 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/05e61d17307b6a7a5f04` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 1.0542557199998555} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/089a670853d01592d15c` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 4.637876019999749} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/1fb735270a17508f22c7` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 3.2044881800000438} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/205b951c80d770cf2608` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 4.706529569999839} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/2c3de2a61d786c4365c6` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.631008060000022} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/325f5414a72b50369d2e` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.471866880000016} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/3371668b8c3109be80ab` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 6.305193810000128} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/3412570422b3db0e178e` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 6.825235629999952} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/368809cc49260c5ec2e6` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 1.0594299800002773} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/434b7c496f186d3d0370` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 19.019159669999862} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/4a278e14c9c61402337e` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 8.333091559999957} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/6a7e676223c1980e72a3` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 1.08418838999969} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/88f4fbe5455c738f4c54` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 15.395898189999935} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/8b0277560fc725b010de` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.45035244999999124} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/929a16e0321f61a3f477` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 5.242843149999681} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/948b3f42305ca6c457fc` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 4.959472660000029} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/a047f1790b83485391a8` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 5.717783209999834} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/a09531cdbb5d605a8a02` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 4.174535240000114} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/b37af9ce794d1b2cecd3` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 2.689472419999947} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/d1d8724d24357f216955` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.5067597499997646} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/d39a129a383a2df42413` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 3.615338630000042} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/e7b393f38babfa7efbda` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 2.0936500100000557} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/eedf1fce8f82f6aa66d6` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.4765295699999115} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/efa472da0ca1fa822ec6` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 1.5825362899999162} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/f2520c53bea5246d833e` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 14.110172079999757} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/f8f43387b382c3695cbe` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.44921219000025303} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |

</details>

### compile

<details><summary>78 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/03219bd41c263165b55e` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.03567100000145729} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/0517a364e60dca7b26b3` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.1399999493733048e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/052c55c194573b7a2c9e` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.00035150000257999636} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/0752e3382a405085399d` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.025558200002706144} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/0769a7020685fb81816e` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.013061900001048343} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/0c0b8a2a8c9ebc182637` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.5852529000003415} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/0c16a1425c20df5f43b6` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.0413639999969746} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/0ff03706224eb337b915` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.00025860000096145086} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/10926315bd10e2940955` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.20863069999904837} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/10fcf3f246432bac3103` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.23618139999962295} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/14df070d0a69c7a59a6a` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 1.0177661000016087} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/155f4c2b15f0389cafac` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.208830499999749} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/1798eeff5674e6500e46` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 1.2197328000002017} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/1a2c9c6bb552f72240ec` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.3334985000001325} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/1c67c7ad1d50ec3db832` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.07252329999755602} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/1fdfb097b99937530836` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 14.19873070000176} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/246ac784c994064ddd39` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 3.7711077999992995} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/27f6a3ffd4985abff865` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.23444170000220765} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/2a57953ca28f2439a31e` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 15.345213500000682} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/2f88bd8db18226d43e7f` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.22671170000103302} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/2fe2a29efcfe53e35e1f` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 1.3199998647905886e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/3910d543aa8db4283980` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 8.354985600002692} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/3e81b8f9de340493e73b` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.00029599999834317714} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/40400757a952797531ec` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 6.786032399999385} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/40b9c9b304c3a504d87e` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.03964790000100038} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/41aff66323fa1d4ada6b` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 6.82123589999901} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/4588cc51fa5191bf31af` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.029421600000205217} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/45abe83027662d4222f2` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 4.183646500001487} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/468884d6e2bdd200f634` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.21455040000000736} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/4875aa75eda245b6360d` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.07055030000265106} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/4affe79fc6783e20a036` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 1.590000101714395e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/4cce1dcc3aebab8f5589` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.4590877000009641} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/4d02681a5a09c03f8613` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.2536389999986568} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/539c8d03b3968cf0b522` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 6.35883670000112} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/57cd84470683308e1c4d` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 4.66019850000157} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/5c6a29e374f4ee67f54c` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.029739699999481672} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/5f2eb0fdd75aaedec337` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.5195171000013943} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/5f7fabd917b39f9c47a6` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 1.5417235999993864} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/6147a968de2f79446e6c` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.22543369999766583} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/63f85c6689375f9cb081` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.05184889999873121} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/6663eb529e0e23a600b8` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.020294199999625562} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/6a97f2079d45ca566780` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.06530399999974179} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/6e3191e723c95f06c47b` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 3.246533100002125} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/73c2a43c7c6ea96c819a` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.014170099999319063} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/75c0fd9f3d79fcdd02e8` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.028571299997565802} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/778c1a2f9778119f569f` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.4275019999986398} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/77acc1fa8ffa0a568e08` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.2216164999990724} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/7ec664a0eda5ff913db3` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.05736290000277222} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/82c05af452a987ccc4b4` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.21364119999998366} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/862c5b0c8e0887875735` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.235139699998399} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/8e126227e506b32a2d81` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 1.1036411999994016} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/8e3455128ff160d39277` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.22563740000259713} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/91ac9fa3d657ffa8424c` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.03006420000019716} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/940771fe1182e685e462` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 5.595845100000588} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/953ff0f1ed327f76e3f7` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 5.18395210000017} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/9796d97053b206c71428` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.028541300001961645} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/9f0eed7ad5e63cce4539` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.9970370999981242} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/a4241a1db70a119f44ae` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.027836799999931827} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/a8da61cc7e72db21f2ab` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.027166399999259738} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/a9b69b544d46294270bd` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.22275950000039302} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/ac39b3f84750c68df7d0` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 1.1948054999993474} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/acb160d78d95f18a2958` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.4695322000006854} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/ad367ac5d4f44132220c` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 6.021133300000656} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/b937515325b738c97fe0` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.026983100000506965} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/bb1b059352af915f8afc` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 19.061746199997287} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/bce732e70bc7d2968a55` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.21923489999971935} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/c458f2757644223353fa` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.02805159999843454} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/c4d28f0f20ce5f9ad658` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.02874869999868679} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/d0fa3e31b8943f8070ad` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.20767010000054142} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/db1bfb552259c8572876` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 2.7963495000003604} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/dbbcaf2e945968a150a0` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.15929719999985537} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/e0749744877648263d4e` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.015369699998700526} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/e471225c3e79a1353d9d` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 3.7790253999992274} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/e88157662fcd881905d9` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.6826089000023785} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/ec330849c4e7c39d2e23` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.9178257000021404} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/ee6357ddd3d4973e0561` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.02041339999777847} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/f9de296911113b40a623` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 2.5703104000021995} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/fa5c9f3cff5199d285c4` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 2.073430800002825} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/scans/n3969/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |

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

- [hazards](../../../../scripts/imaging/rectangular/hazards.py)
- [likelihood_breakdown](../../../../scripts/imaging/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/rectangular/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/rectangular/likelihood_runtime_numba.py)
- [likelihood_runtime_numba_mge_mass](../../../../scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py)
- [parallel_scaling_numba](../../../../scripts/imaging/rectangular/parallel_scaling_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
