<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/20aca9926b0631e8ab19`.

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
| mesh_shape | [39, 39] | recorded |  |
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
| source_pixels | 1521 | count |  |
| source_pixels_requested | 1500 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>29 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/04723bec6d08d5ce9ce6` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.0057950963266193865} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/0b2d2786603c8a6a4e9b` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.0048802959267050024} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/0e448850e20efa72a0b1` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.004349830886349082} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/18d33d34f473689f7cb3` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0031587451230734587} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/22272db19415262c9946` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.001232158625498414} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/27dbf151d143258e081e` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 0.0386046598199755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/288118938aeeb69ce1c1` | s3_nnls_pdip_vmap16.steady_per_call_s | {"s3_nnls_pdip_vmap16.steady_per_call_s": 0.014906998467631638} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/42c9aa94392a92fa899e` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.00330264032818377} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/44dc16c6af383bc355e0` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.011214794870465995} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/4b542f72061d63c25aec` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.008464120281860232} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/4cd8d073a2825b76ecd6` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.005001410096883774} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/5c08fc7209c4eae0b742` | s0_nnls_pdip_vmap16.steady_per_call_s | {"s0_nnls_pdip_vmap16.steady_per_call_s": 0.021946110305725595} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/5eaad50e3204bceea81d` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.0011814008932560683} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/5ee7967aa9c4fc979833` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.009810627205297351} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/6100ec197228c8dc64a3` | s3_active_set_masked_p7_vmap16.steady_per_call_s | {"s3_active_set_masked_p7_vmap16.steady_per_call_s": 0.006106090676621534} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_vmap16/steady_per_call_s` |
| `measurement/6704424bd360217affb5` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.0015401607844978571} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/6a29bc2300eb95c144b8` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.026101973839104176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/70c6236c3bcba72208ec` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.0013513489160686732} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/7d5ff1602131ed6935f6` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.002970521105453372} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/90b05ad258b82ef88ade` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 0.051056401198729874} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/9ed260ad47688d97b86b` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.012487684190273286} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/af83665982c44a137931` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.017882655980065464} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/bcfaa1673a9bdb7f10ab` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.007100212527438999} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/bf6869b3048acd1a35fb` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.0015600906684994698} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/c483f6d6674e8f0a097a` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.037110149813815954} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/cabaa39e5440989d4b0d` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.015218189870938658} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/e150d43d80a4080076a1` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.013843996124342084} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/e40ad0aa2eb90288d1ac` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.016568983811885117} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/f226c44e7277239ec5ff` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.0012078216765075921} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |

</details>

### compile

<details><summary>81 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0050b2d3f6db7f27b859` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.45817419420927763} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/044893ee9bc9e7e2072e` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.44067438086494803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/05bb2df761cae46ae913` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.04103734390810132} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/0992c779fc35d097bc80` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.03426270466297865} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/0a6f5801c9b6699ee2e4` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.06908206595107913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/0b509fe140ff74d45a25` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.45406482089310884} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/0c4fd1c42f5ba15d3b19` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 3.2882504956796765} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/0ebce713508bc27e88eb` | s3_active_set_masked_p7_vmap16.first_call_s | {"s3_active_set_masked_p7_vmap16.first_call_s": 0.5572209530510008} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_vmap16/first_call_s` |
| `measurement/0f2cda36b28d116c7f2a` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.008228090591728687} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/11fd7315237b7994bc84` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 13.258701958693564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/1265aa095831513febb9` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.03412899514660239} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/14f1674f294a51b10ff7` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.013944035861641169} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/16b26f243211784d99ec` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.0047750407829880714} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/1a65c784b1d27e05d0c6` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.147195418830961} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/2c675566a968bd579285` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.08788248430937529} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/2e0b09b32273006f0949` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.001907847821712494} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/2e0e0c790534bf71c754` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.00203566811978817} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/2f2c5b40885a60ad1b26` | s0_nnls_pdip_vmap16.first_call_s | {"s0_nnls_pdip_vmap16.first_call_s": 0.891919108107686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_vmap16/first_call_s` |
| `measurement/2fd1cb3e724c498e6590` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.03508022101595998} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/3737f65f8771e1970034` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.04065670585259795} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/387390ea675d1ad03b8e` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.03406097600236535} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/457a9aeb793fe9ff575b` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.011172762606292963} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/4642f0bf3c7ec7b2bda9` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 1.1570286005735397e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/464b11a4c3b84f4ba6a0` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.000289158895611763} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/473c44bf1c46a73b7fa0` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.03445318480953574} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/542ca9bbf4432c405331` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.004835721105337143} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/5b777119a1c4c29fc772` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.1560041457414627e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/5be137ff9dfd0751f5d4` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.4116673138923943} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/5d74ff808d7c5f5bcd67` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.0013055517338216305} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/603964723b927bc566bd` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.4024440199136734} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/65015bb5566cb34812b8` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.45363842230290174} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/6802833307df90b5ea78` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.45004705525934696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/6af762319d00688730e1` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.3687261613085866} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/6b011fd2c9804330de62` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.01448143320158124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/6ebfc2cdec810ff5dba3` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.03398314630612731} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/70fadfc1651d8b01443c` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.005834425333887339} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/747d2026f93327b33d17` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.016864289063960314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/782c18b3ecfab6c675f5` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.3105549910105765} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/7a255063d02dc2fbe805` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.0071302782744169235} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/7b945bb610d5a394d5ae` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.00028905877843499184} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/83aaf1c636ac0c95aa55` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.027532003819942474} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/84dff327421fb870e019` | s3_nnls_pdip_vmap16.first_call_s | {"s3_nnls_pdip_vmap16.first_call_s": 0.7852952862158418} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_vmap16/first_call_s` |
| `measurement/85a1202a0a3b56c01a4c` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.008660119026899338} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/88b974deac4329622f27` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.42601393908262253} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/95fbab3c0f5c2d3ad735` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.17729517817497253} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/994ce3f37a420f6b0ac1` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.033735197968780994} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/99c32ff5838c81f165d5` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.040855005383491516} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/9b4c68ac5c836b052e7a` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.41048600105568767} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/a64110a78c3464bb7250` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 0.14957834361121058} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/af67feb3c50d833f02cc` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.02181989885866642} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/b20f1c666a69c92e4f1d` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.07272617518901825} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/b21bb3143d34bb13c172` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.03380182711407542} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/b387069d141b79bc21fc` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.0039618657901883125} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/b766ef3e63c03445ecf2` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.01941898325458169} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/b9ac08b17af638b9bb83` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.03381735598668456} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/bbe1eec2970b3051c7e0` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 0.04683588910847902} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/bc3f050fbda4013e95c8` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.0072523970156908035} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/bfe548f3d91529ae2439` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.45846069511026144} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/c659ba05779bd595740d` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.4469729820266366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/c6d939d4feb99e59e140` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.009947770740836859} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/cbccef3ec4d144abbdab` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.009771622251719236} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/cc8e99b5a41e9a2234c3` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.004166354890912771} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/cdc93b7fb33e12a65571` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.4287095433101058} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/d390c2786f85007bec9a` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.012735605239868164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/d5958aa71a6c0ddcf723` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.03355719894170761} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/d64217746787626da519` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.0153927281498909} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/d815628972e3d49403dd` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 3.2831485071219504} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/d920c9de393df7fc9678` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 1.0589603334665298e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/dc4c298b242f12596dfd` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.04473461117595434} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/de6d3f949e93479c9432` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.022957343142479658} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/e57e3ff09430e48e531a` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.03743936587125063} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/e8526367c350a940fffa` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.03429084504023194} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/ed1765dd9287e0e4f35d` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.4237911319360137} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/edc8ef5025e57abec3a2` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.46839805506169796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/efb1986ec9fb238fadc2` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.018052891362458467} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/f194627b77979c5af7c9` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.46647396590560675} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/f2469e7248b6a25b018e` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.01929987408220768} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/f4bf1e9fea8f5acd93c0` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.0003273589536547661} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/f8f9d1c29e477e9ac897` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.0012687030248343945} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/fa69e4dbebc3a9926c0c` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.012707894202321768} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/fef350ec99d60e56b046` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.027665024157613516} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n1521_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 4731 MiB, 81920 MiB",
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

- [hazards](../../../../scripts/imaging/rectangular/hazards.py)
- [likelihood_breakdown](../../../../scripts/imaging/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/rectangular/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/rectangular/likelihood_runtime_numba.py)
- [likelihood_runtime_numba_mge_mass](../../../../scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py)
- [parallel_scaling_numba](../../../../scripts/imaging/rectangular/parallel_scaling_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
