<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/157ca4149e681501e932`.

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
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>26 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/15e7e7cf50d408d075ce` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.5251467499998398} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/1b17ef28472cf9d5a744` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.2923206700001174} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/22993df5e44de15653f4` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.3431884199999331} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/2f1912fbc743762abd2b` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.9712448099999165} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/321e8fc29024fdb09a11` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.48084140999999364} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/4bc1d44868f4c6d9b648` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.9913133400001243} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/4d6222d085224618a946` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.03940032999998948} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/551ac429dac2447b588d` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.038775810000151976} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/5cdb0eca1b82eaba6774` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.20922898999997414} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/5cee17ff57e809c5d024` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.03908646999989287} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/624fdf95c2536c8075ce` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0779912999998487} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/69894e9aeaf0f32881c5` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.867731129999811} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/7901b4aa279a7862dca0` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.03525996999997005} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/886c4a7bcc426198f0c6` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.08073217999990448} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/8aad5e7571efe095cc2c` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 2.2250345199998263} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/8f2957dec65abe9be982` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.25070795999999973} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/9a2990cd76a263c4fe78` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 2.2905949600000897} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/a3425a5f5df9c8c9339f` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.036598910000247994} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/a9e80b5b0ce8f46c00aa` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 1.0593176899998071} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/adf6666647b2599b0428` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.4466997799998353} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/b34185a20c1ba6fe3150` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.08392896000004839} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/b40d2f4c68d8428c82f1` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.03824418000003789} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/c1d93ec1158f837c3c31` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.12352552000011201} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/d7014b01b081cbd0b609` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.5628395900002943} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/dfed11d477a2efb436ab` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.17178079999976034} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/e948f3fe9c25d99b1346` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.402068109999891} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |

</details>

### compile

<details><summary>78 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0a4030ebbbed610d14c7` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 1.5327555000003485} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/0c3abeda36bed4da5323` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.15310479999970994} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/0ebf39a4fbbee7c0cfc8` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.0002826000018103514} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/0fbfdb0278b19dea9cf3` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.4907206000025326} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/1831152efd2f22a55e35` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.11279099999956088} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/1d5be437e3b735e6b035` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 2.7345494000001054} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/24e265fdfb5aa0b52f39` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.25623490000361926} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/265c9a59dcd243b507e3` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 4.73000000056345e-05} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/27fd4394acac9af58400` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.23216429999956745} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/29ca8f890d8fe259374c` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.42587510000157636} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/2d1bc311d165a4e10238` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 1.1373045000000275} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/3609ff69ea8be34a6c72` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.06208110000079614} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/386fcd27f8b62a5eb4a2` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.45040139999764506} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/39512c2c461b3d189a37` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 1.502108299999236} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/39ca4cd9c54b97a7e709` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 1.612703099999635} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/405b1ce750d98a5e8858` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 34.244430299997475} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/40bbd5766fe076e2f079` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.2197121999997762} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/41f6a71ec8322a1ba454` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 1.0415126999978384} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/42d795fb85bf9f330189` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 7.211426899997605} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/43d2764ffc5d76827270` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 1.116057200000796} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/45490c9b79c675ba6c3d` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.3551475999993272} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/4bff90557a65210eebd5` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.47778350000226055} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/4c96b6920101c8deb31d` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.0365793000019039} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/546311e9522ee77729b6` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.1692923000009614} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/55874f95cb7e852bb61d` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.1436982999985048} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/57ed23435cdb614b50e3` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.14169250000122702} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/5903947ae64b0154ae36` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.15377250000165077} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/61223ec7171f389137df` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0011345000020810403} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/64090d2c3a408e04647c` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.15963050000209478} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/65634a8f2cc667532e26` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.5116931999982626} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/6762389e57ec3c89efd7` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.03667039999709232} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/6cdea0b3d2fb93fa6149` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.09276210000098217} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/70ae025a7e6a9b243dad` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 2.050149800001236} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/77eef1d2aaef446243bc` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.05020959999819752} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/79f0ea87edd76490fb5c` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.26917169999796897} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/80239fd531f47bd55fdf` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 1.3582277000023169} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/8416c18e86852fc22972` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.5645847999985563} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/8704f186f9b8281fe96c` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.1546462000005704} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/89d053be566bd7b5b739` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.1542007999996713} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/8a482f485c4f62652905` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.28639119999934337} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/96692ba64bf5ea18a5d6` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.15218969999841647} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/9a4363bc8276cc7a5b0b` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.4678029999995488} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/a08ecf5ccdb57b143238` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.7303766000004543} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/a4c1d488093c619e514c` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 1.3951263000017207} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/a6bd18a566b1fa473f27` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.1539448999974411} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/a6cc53669611b27dacb0` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 19.519793600000412} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/ad3029a389f03f1ade06` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.039641900002607144} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/b18a67cee6447e58b72f` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 1.2403999000016483} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/b25c81ccac701d83cccc` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 1.3971740000015416} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/b457d4abc1d0008a340f` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.04134110000086366} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/bea8649e44784c74235b` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.026571599999442697} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/c1f854c1418de60b16fc` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 1.4511653000008664} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/c2445170685fccdf06a2` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 1.595684899999469} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/c268b295576c1e1c6f9d` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.08186009999917587} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/c29863ad0b120f62a948` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.21701620000021649} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/c63e11aa937027db10b4` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.049999991664663e-05} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/c94d7ba9f8a6659ba7a8` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.04584160000013071} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/ca505cc71b9661afbfea` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.08153630000015255} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/cc831b145de6a7c49e5b` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.012933800000610063} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/d1d90703a2168efbaa10` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.7165516000022762} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/d24b1210593e30a7ee5a` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.5421596999985923} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/d381cd91f2f45b229596` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.1509409999998752} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/d38f03db654d3b62c8f3` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 4.1200000850949436e-05} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/d8539d496b6b3c939283` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 1.6106141999989632} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/dc7ddcf38edf011b5620` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.19997839999996359} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/dd92d6dec7d0f647934e` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.0010914000013144687} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/e0095f95da4a453c1dcf` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 1.9882986999982677} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/e620c99be7abfa13d660` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 1.4150233999971533} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/e715c214e7c35f998590` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.9701045000001614} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/f48b4f297a0aa3982987` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.15247460000318824} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/f4c214b107b63316b555` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.3250182000010682} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/f5246253068aec228b9b` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.11878810000052908} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/f754e82d4a0ad24ae1b2` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.15842760000305134} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/f9dc85c9ee5bd06fadfb` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.3873988000013924} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/fa0f30ff047f901cd839` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.07588159999795607} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/fb5a9b5c095c86a308d4` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 1.397242399998504} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/fc2b3a73c05378b3a537` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.020083200000954093} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/fe246ab7414c4e6f66bd` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 2.815385099998821} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n2500_local_rtx2060_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |

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
      "nvidia_smi": "NVIDIA GeForce RTX 2060 with Max-Q Design, 4770 MiB, 6144 MiB",
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
