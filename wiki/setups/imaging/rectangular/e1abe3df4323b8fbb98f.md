<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/f042b99ffa98131d05e9`.

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
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>29 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0f11461db74ee3afc753` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.0033498610369861125} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/11318801eea7a6d896e1` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.00868767690844834} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/167030fed88360153973` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.05511699328199029} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/18e00ae1032acba43778` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.03849580842070281} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/1d0f636b4aebd67f565a` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.034499596292153004} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/23563bc773117c790073` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.012843210995197297} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/2b862f0f2a928c4dd33b` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.003241721633821726} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/3207362c08bdcec03d7a` | s0_nnls_pdip_vmap16.steady_per_call_s | {"s0_nnls_pdip_vmap16.steady_per_call_s": 0.17590615561348386} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/35f3a3d23a679d9a83df` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.1192529427818954} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/3939d15e1e0f463a4377` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.07329645799472928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/3d770dcdad363b31456b` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.04261506549082696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/45b7874154b262182973` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.02974983789026737} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/4661cbbcd1f93c3fa0b4` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.008815758209675551} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/554e9fa3ed38c8046a69` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.02139358068816364} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/56eb1051b79cf4a78c36` | s3_active_set_masked_p6_vmap16.steady_per_call_s | {"s3_active_set_masked_p6_vmap16.steady_per_call_s": 0.043968244671123105} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_vmap16/steady_per_call_s` |
| `measurement/871d05515601cd4dd673` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.003335388982668519} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/9e4a8b41ff268de02598` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.025658754305914043} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/adad252a503d92ea6cc1` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.017209492903202773} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/b0efb4cf7b5b540b1f36` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.004209870798513293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/bff7914fd6513b8e0562` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.004363254783675075} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/c1c9672e45754f2b2441` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 0.13427654169499875} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/c4b31cf95f4f4e3194f0` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.0031886759214103224} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/c833d52ddb15ce54a241` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.029979759408161045} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/cae8d9ca5f1c46097d57` | s3_nnls_pdip_vmap16.steady_per_call_s | {"s3_nnls_pdip_vmap16.steady_per_call_s": 0.10577759877778589} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/d362e03b73a873d5b218` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.034146266523748636} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/d5638e6b35b89bd0f93c` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.0468711503315717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/d934a08071b89f49809a` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 0.16249160799197854} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/e04f639c529be0233c8a` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.05103797959163785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/efb97e37605d3b2e64c0` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.00884639802388847} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |

</details>

### compile

<details><summary>81 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0087cb1e02b935690858` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.010860134847462177} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/0724da1985f1309490cd` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.035590245854109526} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/116d81cd2ebb3628436d` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.7180100600235164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/119146e2d2377882d57c` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.35790019296109676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/14f6e16483c65c3a08be` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.3140030205249786e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/16abe1e1d13aa46de3df` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.0037897969596087933} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/18a0a86ba273097dd15f` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.023670668713748455} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/1bef3e8d9d2d16c946b2` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 2.3249536752700806e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/1db04d80e753636e7faa` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.453198395203799} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/225162ed8b7c7dc1b9d4` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.03403639793395996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/25a1d932997455e7a396` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.4709629090502858} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/25b9df909ede881f01b5` | s3_nnls_pdip_vmap16.first_call_s | {"s3_nnls_pdip_vmap16.first_call_s": 2.2681490550749004} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_vmap16/first_call_s` |
| `measurement/27c07737d61b65a5d1bd` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.07474369229748845} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/28d31383fb1df1d349a5` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.0355359367094934} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/2b50a1a209128959cdc8` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.04032173007726669} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/2ec8ba6ce1ab8be0f53a` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.43702269392088056} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/30d063c0551a515dfa95` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 3.2470169756561518} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/36439e657c4fedaee0b1` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.02309695305302739} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/37cde9bcbb2111f65f42` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.4469005330465734} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/38bc41ab7e7e3f7c6fe6` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.048510781954973936} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/38ecdf60b78942d4eddc` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.03482914203777909} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/3bfc1f68fa80d5e65f76` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.01896123681217432} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/4189a333839dbb6377a3` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.4195664548315108} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/442828d474d70ac44f18` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 0.1440274240449071} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/4693ba3d38338517c700` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 3.2857412467710674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/4963cbde6f5d2e24fee1` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.3207887290045619} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/5a1eae7ab95f2c253f04` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.10572876688092947} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/64e2e5d329625edc4a2d` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.4185839518904686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/64f1778471b94ee937b1` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.01630367198958993} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/66470b7d121eaa0ea6c6` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.03472280316054821} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/66aecdaf592e19a3a50b` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.06403321959078312} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/6a92b3ff4022f0a9f516` | s3_active_set_masked_p6_vmap16.first_call_s | {"s3_active_set_masked_p6_vmap16.first_call_s": 1.2685704492032528} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_vmap16/first_call_s` |
| `measurement/6c69f2cce47ff9e6fd6c` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.022949503269046545} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/7318f3ccf134f37884ff` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.44046838115900755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/73f9ae364aec4c927ce5` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.44069685973227024} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/75118d8af2359a06ff3e` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.41988988406956196} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/77c7f4f5199b2eb538b6` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.7488272748887539} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/7c8357c0f965ada0a28e` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.010208128951489925} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/7ff17dd9d734855f6de9` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.0357973980717361} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/81695ea9a829cfa10fa4` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.45163710601627827} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/8242c514847f6f769727` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.013826927170157433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/856533f232fe40e1aaf1` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.03998918132856488} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/873beb8d21826af2cc38` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.08271807478740811} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/87799084cc33a68b92b1` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.4407335799187422} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/87df8621528228c977ce` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.004726413171738386} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/888fcc92b079c5faea1b` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.45180583465844393} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/89c1b4ac50a2f23cc986` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 1.0580290108919144e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/8e82132ac5b421bcd0b7` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.4596837470307946} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/9373359bfecfee7c0adb` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.027131076902151108} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/969f8706bb5283eee75a` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.03652579104527831} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/97d826af343055a2ae4d` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.04518814804032445} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/984b9aa6766e3b6dba2c` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.034512723330408335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/a293c8fd968638b3f4a1` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.43149552633985877} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/a5eadd63a2d4c537b82d` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0006415657699108124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/aa9d8c8e3ef1e62187ca` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.020882095210254192} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/aae606cd2894479e3dfe` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.012566934805363417} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/ab1a1cb3cb328541cbad` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.03137683169916272} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/ab58153e0cbce754ddd6` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.4279322363436222} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/ad02e1701b4b275a3d72` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.00034936796873807907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/b1e1bcf20517bc291329` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.08307506190612912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/c0344db56f2a5fad9671` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 12.87807126995176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/c0c05db4b2b932cf5f73` | s0_nnls_pdip_vmap16.first_call_s | {"s0_nnls_pdip_vmap16.first_call_s": 3.3544592382386327} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_vmap16/first_call_s` |
| `measurement/c128285704f4c8fcfa86` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.1207971558906138} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/c4406cbdffc5981966c2` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.08067371603101492} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/c4b65a9ca96b815d53a7` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.009917860850691795} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/c6e883da977d4bcd84dd` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.010787104722112417} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/cc8ae9ecb98795b9bd30` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 0.25534452171996236} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/cfecd1a6d547dd7936a8` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.03480113297700882} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/d1501e8fc1b914b57730` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.04390764841809869} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/d197245eb1091c63709b` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.0390090555883944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/d6297dd42d17982a802b` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.04607785400003195} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/dc4b8e036d39794a1d6f` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.0752315791323781} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/e1956269ddda8cfad3aa` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.003462758846580982} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/e36d884f271eda128b3a` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.037586345337331295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/e3dfab557e01926b274c` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.0002960977144539356} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/e50e32b255acf026c527` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.07069283816963434} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/e8e73a591028d43a0319` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.0035930280573666096} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/e9d8ffe2ee3c861700af` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.03513039089739323} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/fa3d5bc29ecacc0ca2e0` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.03450160287320614} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/fb84947e9a75089344e3` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.05227750865742564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/fd3d3a9281ce420c4ecc` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.014824571087956429} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_rectangular_n3969_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 16893 MiB, 81920 MiB",
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
