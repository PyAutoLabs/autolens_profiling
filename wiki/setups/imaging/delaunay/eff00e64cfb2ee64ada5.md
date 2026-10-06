<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/08156a52e2400dfb778f`.

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
| pins_mode | "none" | recorded |  |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| safe_budget | 7 | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 2500 | count |  |
| source_pixels_requested | 2500 | recorded |  |
| tau_rel | 1.4774739905670691e-05 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | true | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>26 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/048fa157ffabc87ec48e` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.12477485999988858} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/0af40bf055e7b6ba1401` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.2221352000000479} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/12186fb1805935147a5d` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.08241448000007949} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/280f8b0115d717eb006f` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.5791322100001708} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/29c1af20529c7a3a4272` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.9699172499997075} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/29c2b3e948bb0d318207` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.36480045999996946} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/2f8014edc521a4e0ec5f` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.299142589999974} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/387d13363e82362601c4` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0870555699999386} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/4d08dbbb357284898e82` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 2.218954620000295} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/4e9dfe83b7865211bfc6` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.03815759999997681} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/6a998e3217424cc43d9b` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.07767765999997209} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/82cb4d6af31be6bc17d9` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.03660395000006247} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/ae621f953c7f17770216` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.04065594999992754} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/b1a99708aa6750659274` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.17354126000027464} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/c64fa6450c161d4e2674` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.041892189999998664} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/c719d8305b4f465f51c5` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.42693275000019637} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/c9f96960a8847e4effdf` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.38669329000003927} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/d095219c08961a9eab47` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.2742923599998903} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/d749dbda464bc464e599` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.972818130000087} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/de221483b39963c156d2` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.9948174199998903} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/dee70deb29f449520149` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.4987753599998541} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/e65d5e8ab9a82b2d8ded` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.5279804499998135} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/ede6f9b51a8c8b44dd6d` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.035485699999844654} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/f227e60f62875bb3100a` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.03872037000001001} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/f2d058970bfc593848e7` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 2.1493470800000067} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/fe1f41a2832c163a15b3` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 1.099427040000228} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |

</details>

### compile

<details><summary>78 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0459fd5b4910ada3a782` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.5499305999983335} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/0569ace2b4cabf7baf91` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.2971484999980021} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/0719bae2157a1730e60f` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.08806440000262228} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/076690009c43177039ba` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 35.412227099997835} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/092abd24b4c54199e350` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 2.7778995000007853} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/0b1e6262b8310871274e` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.1604062000005797} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/0f4348d3f8c0199367bf` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 1.5553081999969436} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/16870d931e45e1b14172` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.5473485000002256} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/195d157081bc5ae030bb` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.15083189999859314} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/1ad0d0fcc5c97d82ddc2` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.015979600000719074} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/1d3599af6c64654722eb` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.07962760000009439} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/205bec048ebf03cf399e` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 2.6324629000009736} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/2135b90cb225e11e4dc7` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 3.5000000934815034e-05} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/226fec4f8595b06c8c85` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.48623249999945983} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/2285647f0dcfaaa5c5ae` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.1516656999992847} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/2537b6a607fb3708a3bd` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.14852090000204043} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/2c2543729539b8c6fc67` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.14870730000257026} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/2d8c682e0e26e297d7ea` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.03641359999892302} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/30d472e32c49719edb0e` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.009686599998531165} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/33ce2e44afd597111179` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 1.4035418999992544} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/3572d906b93ac7075d8d` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.15334489999804646} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/3644d5e5ba99c1b1df23` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 1.1220171999993909} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/3d520879e34479ef580c` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.11601379999774508} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/3f9b4ec2f0c57e754cb3` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 4.3700001697288826e-05} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/4119354c2a77e60ead8c` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.04678830000193557} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/4c0859eb85d1ed0febd0` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.9690634999969916} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/4c14ea9bd491ded59fd9` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.5789285999999265} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/5164bca01d61969dc9cb` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 1.4791592000001401} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/53c1d6096de43414ce10` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.5187136999993527} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/556fbfb0a89869417b2c` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.0011688999984471593} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/55edf5d7590e1895cca5` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.04208859999926062} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/593ae989ebc3b2d5410c` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 1.0467704000002414} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/5b57e155a1bbe284b585` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.05310230000031879} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/634273f33e54519bb058` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 1.4588695999991614} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/67f443619e8a08704a49` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.03871320000325795} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/6e5b6e1448ec9e62fdc4` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.26421940000000177} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/77428e6b6a24f9f7110f` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.18895980000161217} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/7a0b30b98c273d0438cb` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.16048399999999674} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/7b6a1e87e397c33ed9f1` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 1.5724719999998342} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/7cc62a4a29248bddd3fb` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.3026843999978155} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/7f1bf7181c719b4bf224` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 7.126989099997445} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/82a3dd464707f07ab2f0` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.09711230000175419} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/86c40a7d2776a8130749` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.0398709000000963} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/874c1c5487f6255b3b11` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 1.311734199996863} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/98619f93316ea32c7f38` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.08340840000164462} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/98dbbe31584d0b4243ae` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 1.480924600000435} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/9c90ee337fe1c3ad6dc7` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.035572400000091875} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/9d54a797af99d3eb8803` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.20112280000103055} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/9f8b1220c4c67784f225` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 1.5984248999993724} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/a09b45233b909eaac553` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 1.5086628999997629} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/a565ecc1dc281108b563` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 1.4876461000021663} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/a5fe5007c9a6bb6727b8` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.03005240000129561} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/ae03ec8161e80dedde5f` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.5679958000000624} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/af0745e67aaee79eb433` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.1361715999992157} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/b07d08184e8c182e526c` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.08972649999850546} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/b68007eba4a92dd006c6` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.2085905999992974} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/bfa86324099922171deb` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.3589279000007082} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/c23c57fd9937629d456b` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.4070809000004374} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/c3b49fdcabb1e77061ea` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 2.0265050000016345} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/c41697b21cdb51b63de5` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 1.3687585000006948} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/c503895ffd6d04dbadf5` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.5780088999999862} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/c5483618fb01afd07be0` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.3810977000030107} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/cdbe77fc4622e2edd744` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.1517015999997966} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/ce0106fcd59810cf7ffb` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 4.819999958272092e-05} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/cef2ab6e2c494e42a178` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 19.41564500000095} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/cf9ce66c4fa1f0341397` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.6099266999990505} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/db5772930e919f240f17` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.1680056000004697} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/ddb304be8b9f53098d0f` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.15144560000044294} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/e0fb02df6284aefaaccf` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.23946799999976065} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/e1ef1c9fc45c870f954c` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.16080529999817372} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/e2f8f6879df57463133a` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.0008978000005299691} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/e4517935013bd15fc91d` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.17521839999972144} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/e5bcf760e44bf1eb7b30` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.886308300003293} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/e88ade4ceec2b725b826` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.001214200001413701} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/eb6f9da412427a6687e0` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 1.10872400000153} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/ed4bc536688e3d9cd844` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.018166499998187646} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/f696e43fe4f52237a436` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.5002195999986725} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/fbf19b5f5f66d5b4eef2` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.5743567000026815} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "gpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 8,
      "device": "cuda:0",
      "hostname": "DESKTOP-H143S82",
      "nvidia_smi": "NVIDIA GeForce RTX 2060 with Max-Q Design, 4776 MiB, 6144 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "mixed",
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
