<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/340c4efc3bf218f34258`.

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
| pins_mode | "none" | recorded |  |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "scheme": "constant"} | name |  |
| safe_budget | 11 | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 3969 | count |  |
| source_pixels_requested | 4000 | recorded |  |
| tau_rel | 1.4774739905670691e-05 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | true | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>26 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/033cadb5e73943b7106d` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.15344945999968332} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/035680e882b5d98cd1e1` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.15468997999996645} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/0b6d664b06e98bbb71c8` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 1.2356022400002984} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/2079919cc1dbaf623342` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 1.6985632899999472} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/22e06eaa992472322206` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 2.065732940000089} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/2509be3969e6bf14fbc2` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.13249332000013964} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/4201f1dd9b170b8b9427` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.8030521699998644} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/44c115bd5800645c71a8` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 2.8740407299999786} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/4a3f272cf5065d3e9c0e` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 2.5321006999998645} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/4c463b1c76761ecad41c` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 2.485498160000134} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/4e0f890cc2f77b5ad2df` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 6.075832549999904} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/506f3721e89f9208ff2b` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.9044151699999929} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/6f0b1fabd160c9af4552` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.1431042600001092} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/70c61c654f4a7bc923f4` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.3077861599998869} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/72afa6ae7152325f74dd` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.14240571000009367} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/7789a4dc790621a6055a` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 1.3986030600000958} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/92ccd642ec06b92c0829` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 1.9072988700001587} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/939697eadebe90a09081` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.604812630000015} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/c92a64b2d8858cbbf15c` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.3189109900002222} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/ccf25d38a2e0a97e6959` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.4717489999999088} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/d354c271029d26e6d044` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 5.3629404900002555} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/ed79fd72fba71d8f3f79` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.313215449999916} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/f3f064055fd58cfbe20f` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.15392180000017106} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/f5835e8db82e16a4e88f` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 1.5759748199998285} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/fdf5160e1c03719c4130` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 4.026984989999983} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/ff99f841545fbd0fd855` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 1.1204658800001197} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |

</details>

### compile

<details><summary>78 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/04937eac0fb03d2d3877` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 4.830000034417026e-05} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/0546236d51d35dcc956f` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 1.4148991999973077} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/069f8af66a54cb1060c6` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.11784419999821694} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/105a720587fd33d495f4` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.1030006000000867} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/142c5477f21bf086a520` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 2.863248300000123} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/158a986bb756b7db94e8` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.5355557000002591} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/171891a7529060038406` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.14569420000043465} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/198f6017335b52588c53` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 1.5876103999980842} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/19c8d36be29da912f512` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.1141274000001431} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/1d916def10c7750e1626` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.0011071999979321845} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/26a93da8be4baf09bed2` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 2.1451063999993494} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/271fabaa62327363d1cd` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.6267980000011448} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/28a1fde2b60f71abe750` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 5.3276587999971525} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/308d2cc8ece159edf5e7` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 1.9037845999991987} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/3100fca91c6e5bd8e5e2` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 1.3925085999981093} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/39646358cf9846ad7f62` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0011905999999726191} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/3bf0129ce38b79529fe4` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.15606620000107796} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/4492549d636b91454892` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 4.030000127386302e-05} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/4a669afcf61815da32a5` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.15427790000103414} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/51aa3ab7d2cf1f081914` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 4.190000254311599e-05} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/5328021a28ce9bb1d1fa` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 37.604737900001055} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/578c955e8c12c859ded4` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 1.51121289999719} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/58eb8116aa3f8b1b0b2e` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.15151439999681315} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/6619286bf08a747f6b32` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.15106359999845154} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/68e69038984a94a0950b` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.15223039999909815} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/6a4908b543b63ecddda6` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.20862050000141608} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/6afbd09ec0dc489867f0` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.013332999998965533} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/6dfde20508cef0a9d79b` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 1.3448738000006415} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/6f53c7b5677c6136e0ab` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.14981149999948684} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/773a31747dfbf6710640` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 2.0914082000017515} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/7db2077ac9849d9e899e` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.2020879999981844} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/7fdb765641e2b80b0c1c` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 2.640891400002147} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/88f246621a9c0e0f50f1` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 3.025040699998499} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/8ada3c4881b36837bd4b` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.3660426999995252} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/8f38202bc391a559c6bd` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 1.0409365999985312} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/90e4224074d8905866d7` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.14818530000047758} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/910c20b90e43274c3f29` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 3.5471051999993506} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/92b4a0db3bcb51a45887` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 5.780454499999905} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/93873086e8d5c429c244` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 18.752124800001184} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/9e4e2b8264b8a942bdca` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.1485806999990018} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/a10aa2f71382328d2e76` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.1008469000007608} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/a2e585708c58edfa21f9` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 1.18261739999798} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/a37c8b5e5ead32e73399` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 1.391407099999924} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/a854ffc5272d6577aeda` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.15561689999958617} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/a8a15b884989ff3ae6eb` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.16677680000066175} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/a9aa8f3ea7ce00d671e9` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 6.255690900001355} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/ab3b20a23a182f8c743c` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 1.321839500000351} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/acbce077a1149022683f` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 1.3716467000012926} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/b14e7c43389b3d415a67` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.15724240000054124} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/b25cfd8f16e64390901d` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.8330676000005042} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/b435f23bfecd3b4392d6` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.3181310999971174} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/bd6f1fad8e131522ca11` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.9283071999998356} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/bf1bf3215f7159da8953` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 1.5923718999983976} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/bf893e02547447d672d0` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.3401814999997441} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/c2859e578dc6957aa5ec` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 1.286124700000073} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/c577f0fe385e5ae13caa` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 1.8108786000011605} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/c5994b1950fa9fba3ae2` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.34624600000097416} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/ca62e2c5c6f9e4a0258d` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.14620379999905708} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/cfa28963ad9d36e403a3` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 1.4372245000013208} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/d267979d75222b511523` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.3330443999984709} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/d92be6f8f1418bd9c0fe` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 12.028985199998715} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/e0f69a22fe4a464ffd8c` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.5206743999988248} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/e243050f15978c7868f8` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 1.3488165999988269} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/e681e57c6c70bd780d0c` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.1519628999994893} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/e698b859e5639dda1083` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.20123719999901368} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/eb0b951e0131a7f3a627` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.3646757999995316} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/ecc26aeed2d5b12d4cbf` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 1.3433609999992768} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/f04c36f15980958dcc27` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 1.6183661000031861} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/f10e940043279ba07765` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.0010318999993614852} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/f1c9b3fba4b4ea19436e` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 1.5641391999997722} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/f47a92952d3c065dfc11` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 1.9382157000000007} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/f57be1c10742f786b46f` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 1.1743494999973336} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/f63a3c414a785dc6abc2` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 1.5762906999989355} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/f86a8dc230b45c7802b6` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.39904410000235657} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/fb41506275d550c988c8` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.04335019999780343} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/fc8b4f9327e0f289e13a` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.13409829999727663} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/fc8c73877864ed20395f` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.15944020000097225} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/fd03d5e894a5c5c05384` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.14128600000185543} | s | gpu / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_local_rtx2060_mp_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |

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
      "nvidia_smi": "NVIDIA GeForce RTX 2060 with Max-Q Design, 4774 MiB, 6144 MiB",
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

- [hazards](../../../../scripts/imaging/rectangular/hazards.py)
- [likelihood_breakdown](../../../../scripts/imaging/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/rectangular/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/rectangular/likelihood_runtime_numba.py)
- [likelihood_runtime_numba_mge_mass](../../../../scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py)
- [parallel_scaling_numba](../../../../scripts/imaging/rectangular/parallel_scaling_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
