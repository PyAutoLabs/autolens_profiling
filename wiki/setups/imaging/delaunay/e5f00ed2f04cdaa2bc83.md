<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/da7f6773b122ae01c55b`.

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
| source_pixels | 4000 | count |  |
| source_pixels_requested | 4000 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>29 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/028b54be2a4a6a4c3580` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.008201108314096928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/04e46eea3d51c657d805` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.0041335225105285645} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/0b1d65e50cde97005321` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.03027283912524581} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/0ca56b400f65565ed9a3` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.020082165626809} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/0e4296d33289c73ef39f` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.028115056920796633} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/0fb3d4caa172d849bdcf` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 0.200373428221792} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/2afc3f3d05d5169d64ef` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.08728750897571444} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/2f1ccdc67e09288e74e8` | s0_nnls_pdip_vmap16.steady_per_call_s | {"s0_nnls_pdip_vmap16.steady_per_call_s": 0.17414112953993027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/30a1267fcfb434aa60a5` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.024108701711520554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/353778ac8b6381f4de9e` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.05210846927948296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/463eb2cb4bfe7922a909` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.016150908870622517} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/46f2b3ec7dfb5e8af134` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.11737183928489685} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/482b175708b96ab4d8e8` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.008347962377592921} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/4ba6327c9e12380749ae` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.0320880392100662} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/5c1bae09c56f91ab0c55` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.032394901383668184} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/7540ba66530fcb11a765` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.002956695482134819} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/988bb137a17047d4ac1f` | s3_active_set_masked_p2_vmap16.steady_per_call_s | {"s3_active_set_masked_p2_vmap16.steady_per_call_s": 0.018416102381888778} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_vmap16/steady_per_call_s` |
| `measurement/9deeabcbb14d2dabb790` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.0030140159651637077} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/a67cb9249b013d2d5db3` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.04414856848306954} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |
| `measurement/a6e8d8f38bb3c0da2e5f` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.03609533440321684} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/ba8e7ae10d551b835cc7` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 0.16520025287754833} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/c5af048af785f46f5eb9` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.003958427486941219} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/c8865fa6a9d2d5993385` | s3_nnls_pdip_vmap16.steady_per_call_s | {"s3_nnls_pdip_vmap16.steady_per_call_s": 0.12935541810002177} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/d3ddeae9e1c7aa191592` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.008453935710713267} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/ec0eb22cbf22b80ba621` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.012125998781993986} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/f1038c1db7e367dd4511` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.040140660386532544} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/f9fc9e57246e426366d1` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.0029768771957606075} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/fc0bef52a18d5e3e924f` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.048199445474892855} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/ff90a94848659fa92873` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.0029608632903546095} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |

</details>

### compile

<details><summary>81 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00113444981199ecf16d` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.01466453168541193} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/03d59960b379bab17c2d` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.004471343010663986} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/0582b1c4682646a9e831` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.03441016608849168} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/06988905b646fe7f3bfc` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 0.406848241109401} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/0884e1ca96c1f7fb307b` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.03748284699395299} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/0a1d11f20944683e4b4c` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 3.406120751053095} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/0d1b6bfbce0e53a3bac5` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.0347768641076982} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/0ea8f35563e5822f1a26` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.03485576203092933} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/10b0aceb3114c1978ff6` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 0.4175551268272102} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/1160b672d4e5ac519842` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.4052274120040238} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/1319c9580bfa01d82e3d` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.4476425787433982} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/171e3780d4e1d79e991d` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.02206218894571066} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/268b018e1cd1855e420c` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.04514015093445778} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/26ffb5a9bafef2c36bee` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.05323410173878074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/2d73659ffddf0edf69d9` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.04552586004137993} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/2f43b02e407abde5ddaf` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.06309355469420552} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/34b9c5388d9bfe697d85` | s3_nnls_pdip_vmap16.first_call_s | {"s3_nnls_pdip_vmap16.first_call_s": 2.6028704661875963} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_vmap16/first_call_s` |
| `measurement/36f48cdff18378ba03ec` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.03387778904289007} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/36f501b234cba74b008e` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.03476392384618521} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/3af51d2cc89b215b46c4` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.023583239875733852} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/40d9d30e3484a73dd5eb` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.003031961154192686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/480a43f317b89752e035` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.07970525603741407} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/4b12131561228c2843d8` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.00017243996262550354} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/4b5768896f988e9a8181` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.025631288066506386} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/4de662006142398de9a3` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.45627525681629777} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/4ee11ab46d49563565a3` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 3.8231134358793497} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/511cb9a547cc09221c07` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.013738338369876146} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/58922201c2f0783b6211` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.1350028216838837e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/5d39389068bedcf4ec6b` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.038097512908279896} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/5ecf8b3011eff08d9d0a` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.04067341797053814} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/64ea7d3f0c188d45d798` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.7060531228780746} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/66341e944e4177a06975` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.017626296263188124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/735809f672c4161b5f41` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.034782752860337496} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/79bbb1ecaab49015ff15` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.4622968123294413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/7c29e43522342831a1f3` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.003451309632509947} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/8366f2098c4654703dca` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.39351094141602516} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/83d3a923f86bd1c81b16` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.08566237986087799} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/846dd2c8b44c9a525a40` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.03354820143431425} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/87c22ae14b9af12ec262` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 0.40959322499111295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/888d626aa767204736bd` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.003122691996395588} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/8c17c25f7a4913653e7d` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 0.39870009990409017} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/8cc9661da77d458b20e9` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.08901748107746243} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/8fd6fea6e0eb5010c2db` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 0.17725619906559587} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/92f18b1a7aeb391a67cc` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.02176126092672348} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/987e226a78b220fbd437` | s3_active_set_masked_p2_vmap16.first_call_s | {"s3_active_set_masked_p2_vmap16.first_call_s": 0.7982697039842606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_vmap16/first_call_s` |
| `measurement/9a26f191be164695fc27` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.07961584720760584} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/9b0dceb2163e522c3085` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.03438993636518717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/9c301e114a8df7579fb7` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.04945195699110627} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/9e315c583844ffc557b5` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 0.2980269272811711} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/a0eb088b6a482f4cf659` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.11887669283896685} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/a1ccd1705ee361198e59` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.03408608725294471} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/a21c3390d87c457934e7` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.2835339941084385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/a5061031e8efa77f0c74` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.03443338489159942} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/a8476122d216a028aad4` | s0_nnls_pdip_vmap16.first_call_s | {"s0_nnls_pdip_vmap16.first_call_s": 3.3177827349863946} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_vmap16/first_call_s` |
| `measurement/a8d987f2ae7722929d02` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 1.3300217688083649e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/ae058842cf747fd005b4` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.44139784714207053} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/af9726aa5a4cde89ca40` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.0149216721765697} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/b09efaf6376048327bc5` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.748206852003932} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/b0aef590d35fff9fb847` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.03743912698701024} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/b464e8e7fde4a5ff8ad7` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.08210890181362629} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/c5be15fb1cc7c8ab03aa` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.03447846416383982} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/c731e0aabbe79f9a8b57` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.00030538812279701233} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |
| `measurement/cb5088048d9afb52fbf3` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.009619373362511396} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/cde53d7a4bbfa7b95a79` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.01409550616517663} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/d141639740ea57ad37d3` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 0.3863931931555271} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/d262f2f3e1cf68742192` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.029670273885130882} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/d809c23710bd54603fef` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 7.319729775190353e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/d96d1a429b7920a079c4` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.009859672281891108} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/dba73de0df75e0f934e0` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 13.52373067010194} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/dc59ae888c5b9f5e2bc0` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 0.39467067504301667} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/ddf4138dcb27c9a5fb24` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.009422414004802704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/de97c0368875dc3611b8` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 0.394546034745872} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/deaae1a9682f75890e14` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.03529416024684906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/e04f8d893205b4e7bfe9` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.021557450760155916} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/e05b3762e36502985aad` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.4717608271166682} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/e0de50931ed6d4338a31` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 0.3871852089650929} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/e8fd1944e7d6bbe9e60a` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.06957484595477581} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/f3a640eed35a96df191a` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.04150304291397333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/f88f1758ffb6634e6827` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.010069489944726229} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/f8e6f4d65ec715b68cd0` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 0.3963625431060791} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/faf67cf153b276406725` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.0002549877390265465} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_delaunay_n4000_hpc_a100_fp64_fixed_light.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 16895 MiB, 81920 MiB",
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
