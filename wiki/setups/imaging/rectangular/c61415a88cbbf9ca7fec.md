<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/d4e157e0542ce84da78a`.

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
| mesh_shape | [32, 32] | recorded |  |
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
| source_pixels | 1024 | count |  |
| source_pixels_requested | 1000 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>26 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/05cfe3a38a00adcbd7e3` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.4228620199999568} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/066a59b7bcaa60755323` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.1689398300000903} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/174465851aeec4f8f60f` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 1.2481180599999788} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/17937e8fbf774f875c5b` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.0117553100000805} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/22c6dd1de18edf8025f5` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.03171092000011413} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/22efedcd4cba09b25fa5` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.04860392999980832} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/31a51789548938e4806e` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.10540337999991607} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/472b66bea66f44a151b0` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.012439920000178972} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/55c71fe06ca854989c58` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.08296904000017094} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/583f1246dcc0bac644ca` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.018495480000274254} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/58c428d40c9f9c4b878f` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.19773569999997562} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/63bb090b5a614304e950` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 1.1917065900001034} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/648ff3b08c2fe481c854` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.24011463999995614} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/80c8104ee8c37a10c7bf` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.06367124999997031} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/84b30ea90df982fecefd` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.1234976199997618} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/8766bdc0b5149691b394` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.011685909999869182} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/8bb76b461942e12713e7` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.011999359999754234} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/8c1bb8988bd4ea0bdac5` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.028900720000092407} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/8db789b0001908c01c76` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.15082538000024215} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/a0da2388d5d922ed46b4` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.14009480999993684} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/aef44177cc76db32ba42` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.02330653999997594} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/c0052ec3db450fe2a7d0` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.10825694000013755} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/c820ddf06d0b5f2119f2` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.35952562999991644} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/e351b83a0a9d0bb043be` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.5202048499999364} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/e69c993189ed3ac30e30` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.03738204000001133} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/f98fb1b666080188cdf3` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.18277090000010504} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |

</details>

### compile

<details><summary>78 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/02ab13b77576bb6b248e` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.1596927999999025} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/0a68bf57c7b61f426f5f` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.2312763999980234} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/0f4585b7c0faa45a0676` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.4161811000012676} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/135db09452d0995b59be` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.2962110999978904} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/1b314282f37ae5695a8b` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 1.324301700002252} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/1f0e6f86b69ea574d410` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.21631650000199443} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/1f279107db6ca25f73d0` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.033936899999389425} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/2010dd41fc9c593fa20f` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.03281779999815626} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/250f79772c2c17fa5134` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.00030139999944367446} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/25c05a288222dc98d3fb` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.3100001524435356e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/25c5b45e2ab5fe369cda` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.11452509999799076} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/2a9d1714a360942ddcee` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.012481300000217743} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/2dab59da7d30acebeb97` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.040389300000242656} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/3388a8c3ebccf06520cb` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.052584199998818804} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/339390cdd39ac0862c4d` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.2088489000016125} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/38cb2312812f4ab28287` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.033242199999222066} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/38ffcb44e7175418cdf9` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.034720799998467555} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/4420725f3aa6a54a0801` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.09973600000012084} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/4c72246f47ef935d5359` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.04494920000070124} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/4d0baa6e39f585bf84a1` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.06120830000145361} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/50dc12db7e683158db32` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.03333340000244789} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/596a0a96c31a40bf7d2a` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.37276319999728} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/59f837cd4e6d20247c33` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.01576819999900181} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/5beea28439420db2a147` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.02660460000333842} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/5d49f7d359e52fae5b88` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.032761099999333965} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/5e2641f8d6915878ff8f` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.03669409999929485} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/5f8e4f9bf0174fc765fb` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.23262489999979152} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/64d190f788a3cb6721ac` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 1.340000017080456e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/6e95d17709a3eb937fe7` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.22314680000272347} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/705f6ed611db0f9d22a5` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 3.045813500000804} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/793e76a790381a76da2b` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.06190019999849028} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/7d587f14e771ac49b884` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.022359099999448517} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/7f06e408425a8f8f9d34` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 1.2000000424450263e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/8226729929e8737a3478` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.03372699999817996} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/834d61f9bc111da1c7bc` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 1.3617033000009542} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/85bbde77f58b9655ff7f` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.15500519999841345} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/86a072f69137c955f3ec` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.24876019999646815} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/8715de11de46762c1397` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 1.467246599997452} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/8c6bba3b090c54191642` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.026051500000903616} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/8dd6781b7dc1bdcfc89d` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.033564499997737585} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/8f3d0da4044c6177644a` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.19343929999740794} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/8f5df4a42eeb7c1a9100` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.27354770000238204} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/90b3f15a651ec334131c` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.171367700000701} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/9388c54b519d37551aea` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.2741189999978815} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/97e31e5bc711167c19d6` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.012311300000874326} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/a293100b6ddf8ffab94e` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.11980140000014217} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/a7bb62676aa6d549f3e2` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.033917500000825385} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/a8608795fc96427520f2` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.11514790000001085} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/a9a8a9c86468e30a3736` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.00029279999944265} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/a9cd4ce0b3dbd9406e7c` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.137716500001261} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/acff7ca551583023c14f` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.3015110000014829} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/ad7c695c6d919609ae99` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.020878499999525957} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/b9c403be728c0345c6af` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.10137010000107693} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/bc31009c5fc0df629d34` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.011914499998965766} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/be19b6f5596bdc8507a6` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.07267309999951976} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/be857a0007f94091b521` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.5010145000014745} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/bea6537355730837e54c` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.2529523000011977} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/bfdf76bd0e459fa0dd44` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 7.593761200001609} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/c1fc1acc0bebee550ef1` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.02665029999843682} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/c7bea1bdcde10422f650` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.021094199997605756} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/c9c9ca22fd63cc7dee0d` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0003084999989368953} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/cae982f2dec251721a1d` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.21962129999883473} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/cd8b044c040ef50598c1` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.2234707999996317} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/cf2207fe090c4c935a8e` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.049617499997111736} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/d04d608dce25e25c5175` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.2475654999980179} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/d0e391db2b9267c5d62a` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.012447699999029282} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/d3a568e10189f41b89d7` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.009596200001396937} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/d59b221a6d419709f9df` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.22400650000054156} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/d6737d66bb58cdaff257` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.45435859999997774} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/e47b338987800e835b49` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.027281400001811562} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/e484365fb853c620a4de` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.07199490000130027} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/e5f552ff5338317105f8` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.23476360000131535} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/ec34f0f864a6f366d1ab` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.03357580000010785} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/ed358795469881ede384` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.09090590000050724} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/ed653ebce312bbc1d369` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.25759800000014366} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/ee75a2c677bf7302c222` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.030909299999621} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/f1b4dc394a27f3c6e7a9` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.13432190000094124} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/fb7064ed96f9bc7e1fc7` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.21639019999929587} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1024_local_cpu_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |

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
