<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/a69c7d89dc11f8aa9ad4`.

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
| source_pixels | 1000 | count |  |
| source_pixels_requested | 1000 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>26 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00e97b211502fa601782` | s3_active_set_masked_p12_jit.steady_per_call_s | {"s3_active_set_masked_p12_jit.steady_per_call_s": 0.07271570000011707} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/steady_per_call_s` |
| `measurement/028ffbf3d02356576544` | s3_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s3_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.010876619999908143} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/063c30ad45582da4d902` | s3_active_set_masked_p6_jit.steady_per_call_s | {"s3_active_set_masked_p6_jit.steady_per_call_s": 0.04806793000025209} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/steady_per_call_s` |
| `measurement/07f114f78d42f9ebbab8` | s3_active_set_masked_p3_jit.steady_per_call_s | {"s3_active_set_masked_p3_jit.steady_per_call_s": 0.021551439999893774} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/steady_per_call_s` |
| `measurement/0ba16af99a463f9801a2` | s3_active_set_masked_p9_jit.steady_per_call_s | {"s3_active_set_masked_p9_jit.steady_per_call_s": 0.06198327000020072} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/steady_per_call_s` |
| `measurement/0f37e0d9c5abb277e1d3` | s0_cholesky_solve_jit.steady_per_call_s | {"s0_cholesky_solve_jit.steady_per_call_s": 0.006240640000032727} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/steady_per_call_s` |
| `measurement/13f092720059553d1dea` | s0_nnls_pdip_jit.steady_per_call_s | {"s0_nnls_pdip_jit.steady_per_call_s": 0.1364294000002701} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/steady_per_call_s` |
| `measurement/24196d5f8dbcbe7e7561` | s3_curvature_reg_build_jit.steady_per_call_s | {"s3_curvature_reg_build_jit.steady_per_call_s": 0.18574426999985008} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/2d47973f3d40809fcfe3` | s0_log_det_curvature_reg_jit.steady_per_call_s | {"s0_log_det_curvature_reg_jit.steady_per_call_s": 0.005279190000146627} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/3db365ab140986d09e72` | s3_library_likelihood_jit.steady_per_call_s | {"s3_library_likelihood_jit.steady_per_call_s": 0.3778209799998876} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_library_likelihood_jit/steady_per_call_s` |
| `measurement/4c829740b98e1d96e362` | s3_active_set_masked_p8_jit.steady_per_call_s | {"s3_active_set_masked_p8_jit.steady_per_call_s": 0.05695407999992312} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/steady_per_call_s` |
| `measurement/4fd6af0bd5b6a8808678` | s3_active_set_masked_p11_jit.steady_per_call_s | {"s3_active_set_masked_p11_jit.steady_per_call_s": 0.06648110000023735} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/steady_per_call_s` |
| `measurement/516db66640dd47c80539` | s3_log_det_regularization_jit.steady_per_call_s | {"s3_log_det_regularization_jit.steady_per_call_s": 0.005136009999841917} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/steady_per_call_s` |
| `measurement/544136c50889264fde7e` | s3_nnls_pdip_jit.steady_per_call_s | {"s3_nnls_pdip_jit.steady_per_call_s": 0.09390031000002637} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/steady_per_call_s` |
| `measurement/556abf4f66507981b00f` | s3_active_set_masked_p1_jit.steady_per_call_s | {"s3_active_set_masked_p1_jit.steady_per_call_s": 0.01068459999987681} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/steady_per_call_s` |
| `measurement/6df607b248670ff99b98` | s0_curvature_reg_build_jit.steady_per_call_s | {"s0_curvature_reg_build_jit.steady_per_call_s": 0.18655765000003158} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/steady_per_call_s` |
| `measurement/727b6267d64878ae2b2f` | s3_active_set_masked_p7_jit.steady_per_call_s | {"s3_active_set_masked_p7_jit.steady_per_call_s": 0.05431091999998898} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/steady_per_call_s` |
| `measurement/7d166beeab2306482b92` | s3_active_set_masked_p4_jit.steady_per_call_s | {"s3_active_set_masked_p4_jit.steady_per_call_s": 0.042078590000164694} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/steady_per_call_s` |
| `measurement/82eb8f497f7781647e12` | s3_active_set_masked_p2_jit.steady_per_call_s | {"s3_active_set_masked_p2_jit.steady_per_call_s": 0.016014530000029482} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/steady_per_call_s` |
| `measurement/85bd3cf4ece556c78174` | s0_nnls_pdip_one_iteration_jit.steady_per_call_s | {"s0_nnls_pdip_one_iteration_jit.steady_per_call_s": 0.012078050000127405} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/a0d7aaa71d158d5e80b5` | s0_library_likelihood_jit.steady_per_call_s | {"s0_library_likelihood_jit.steady_per_call_s": 0.43712666999999783} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_library_likelihood_jit/steady_per_call_s` |
| `measurement/affae9466195e2df394d` | s0_log_det_regularization_jit.steady_per_call_s | {"s0_log_det_regularization_jit.steady_per_call_s": 0.005235019999963697} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/steady_per_call_s` |
| `measurement/b0cb81beb5141c591b35` | s3_active_set_masked_p5_jit.steady_per_call_s | {"s3_active_set_masked_p5_jit.steady_per_call_s": 0.05091654999996535} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/steady_per_call_s` |
| `measurement/cb03fee698571b4cdd0e` | s3_cholesky_solve_jit.steady_per_call_s | {"s3_cholesky_solve_jit.steady_per_call_s": 0.006369869999980437} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/steady_per_call_s` |
| `measurement/de1bcd4ce56858706ccf` | s3_log_det_curvature_reg_jit.steady_per_call_s | {"s3_log_det_curvature_reg_jit.steady_per_call_s": 0.005244800000218675} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/f8625ee5d6cbd02b22c0` | s3_active_set_masked_p10_jit.steady_per_call_s | {"s3_active_set_masked_p10_jit.steady_per_call_s": 0.06385971999989123} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/steady_per_call_s` |

</details>

### compile

<details><summary>78 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/002d948c4fcb48e9c9c8` | s3_log_det_regularization_jit.first_call_s | {"s3_log_det_regularization_jit.first_call_s": 0.00541729999895324} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/first_call_s` |
| `measurement/09bfa7aaeb6c8ff33b61` | s3_active_set_masked_p6_jit.first_call_s | {"s3_active_set_masked_p6_jit.first_call_s": 0.14615779999803635} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/first_call_s` |
| `measurement/0ff15a10aecd5cb9cc69` | s3_active_set_masked_p10_jit.lower_s | {"s3_active_set_masked_p10_jit.lower_s": 0.15583739999783575} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/lower_s` |
| `measurement/10f2007700d6c8328917` | s0_log_det_curvature_reg_jit.lower_s | {"s0_log_det_curvature_reg_jit.lower_s": 0.021675099997082725} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/lower_s` |
| `measurement/113c120a91bc87279903` | s3_active_set_masked_p8_jit.first_call_s | {"s3_active_set_masked_p8_jit.first_call_s": 0.21469820000129403} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/first_call_s` |
| `measurement/13b3e5ceea07c55bbc60` | s3_active_set_masked_p9_jit.first_call_s | {"s3_active_set_masked_p9_jit.first_call_s": 0.21561109999674954} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/first_call_s` |
| `measurement/14038c8b53b8be908fac` | s3_active_set_masked_p5_jit.compile_s | {"s3_active_set_masked_p5_jit.compile_s": 1.6504303999972763} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/compile_s` |
| `measurement/160cf82acaae845d504a` | s0_library_likelihood_jit.first_call_s | {"s0_library_likelihood_jit.first_call_s": 1.1593203999982507} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_library_likelihood_jit/first_call_s` |
| `measurement/18980046d5bd1cd966c1` | s3_active_set_masked_p1_jit.lower_s | {"s3_active_set_masked_p1_jit.lower_s": 0.05007869999826653} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/lower_s` |
| `measurement/1f55f462a99d14909e3e` | s3_active_set_masked_p7_jit.compile_s | {"s3_active_set_masked_p7_jit.compile_s": 1.683307600000262} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/compile_s` |
| `measurement/24dd7bd39b69d17f5ba4` | s3_active_set_masked_p1_jit.first_call_s | {"s3_active_set_masked_p1_jit.first_call_s": 0.015626000000338536} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/first_call_s` |
| `measurement/286d7ab9cb8dc8589754` | s3_active_set_masked_p4_jit.first_call_s | {"s3_active_set_masked_p4_jit.first_call_s": 0.11451489999672049} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/first_call_s` |
| `measurement/28f33d6c4cdc6e3f6a6e` | s0_nnls_pdip_one_iteration_jit.lower_s | {"s0_nnls_pdip_one_iteration_jit.lower_s": 0.031030899997858796} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/2ac2c7f9ec712b1d0d4c` | s3_active_set_masked_p3_jit.first_call_s | {"s3_active_set_masked_p3_jit.first_call_s": 0.03505429999859189} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/first_call_s` |
| `measurement/3625472ac5e8bd14af56` | s0_nnls_pdip_one_iteration_jit.compile_s | {"s0_nnls_pdip_one_iteration_jit.compile_s": 0.44205800000054296} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/3838287aa56a1843c423` | s0_curvature_reg_build_jit.compile_s | {"s0_curvature_reg_build_jit.compile_s": 0.17871260000174516} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/compile_s` |
| `measurement/387c1ae82bef8554e8ec` | s3_active_set_masked_p10_jit.compile_s | {"s3_active_set_masked_p10_jit.compile_s": 1.6131814000000304} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/compile_s` |
| `measurement/3ada2108c612259440d0` | s3_library_likelihood_jit.compile_s | {"s3_library_likelihood_jit.compile_s": 4.566847699999926} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_library_likelihood_jit/compile_s` |
| `measurement/3c32ddc905390a86dad3` | s3_active_set_masked_p12_jit.first_call_s | {"s3_active_set_masked_p12_jit.first_call_s": 0.24828849999903468} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/first_call_s` |
| `measurement/3d85a6317e453b262b69` | s3_active_set_masked_p6_jit.lower_s | {"s3_active_set_masked_p6_jit.lower_s": 0.14942850000079488} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/lower_s` |
| `measurement/3eb847a3b541ee49191a` | s3_cholesky_solve_jit.first_call_s | {"s3_cholesky_solve_jit.first_call_s": 0.007297699998161988} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/first_call_s` |
| `measurement/471c75b2462a90cf3d32` | s0_curvature_reg_build_jit.lower_s | {"s0_curvature_reg_build_jit.lower_s": 0.04066939999756869} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/lower_s` |
| `measurement/495d9cb69f2f8e6eef11` | s3_active_set_masked_p11_jit.lower_s | {"s3_active_set_masked_p11_jit.lower_s": 0.15054519999830518} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/lower_s` |
| `measurement/49c44b1d881bab6bf0f4` | s3_nnls_pdip_jit.lower_s | {"s3_nnls_pdip_jit.lower_s": 0.051789700002700556} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/lower_s` |
| `measurement/4b414acc80702d0c58e9` | s3_active_set_masked_p3_jit.compile_s | {"s3_active_set_masked_p3_jit.compile_s": 0.522865200000524} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/compile_s` |
| `measurement/4c82f0c91893e3f2f748` | s3_active_set_masked_p6_jit.compile_s | {"s3_active_set_masked_p6_jit.compile_s": 1.436945099998411} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p6_jit/compile_s` |
| `measurement/4daca86ea7237bf8736f` | s0_log_det_curvature_reg_jit.compile_s | {"s0_log_det_curvature_reg_jit.compile_s": 0.14014490000045043} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/compile_s` |
| `measurement/54f87d0392d1b4110dd9` | s3_active_set_masked_p2_jit.compile_s | {"s3_active_set_masked_p2_jit.compile_s": 0.5087571000003663} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/compile_s` |
| `measurement/60de0f86b10c85022718` | s3_active_set_masked_p5_jit.first_call_s | {"s3_active_set_masked_p5_jit.first_call_s": 0.17921479999859002} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/first_call_s` |
| `measurement/60fad5c8ed9893ca37e6` | s0_cholesky_solve_jit.compile_s | {"s0_cholesky_solve_jit.compile_s": 0.0949058000005607} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/compile_s` |
| `measurement/7105eafaa726fddf452a` | s3_active_set_masked_p9_jit.lower_s | {"s3_active_set_masked_p9_jit.lower_s": 0.16037449999930686} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/lower_s` |
| `measurement/73602a5acb2101c8b5c7` | s3_active_set_masked_p8_jit.lower_s | {"s3_active_set_masked_p8_jit.lower_s": 0.20042910000120173} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/lower_s` |
| `measurement/745b3e1d325ed6b4f3d1` | s0_nnls_pdip_jit.compile_s | {"s0_nnls_pdip_jit.compile_s": 0.4136508999981743} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/compile_s` |
| `measurement/76bbe638f5be2edc0a27` | s3_library_likelihood_jit.lower_s | {"s3_library_likelihood_jit.lower_s": 0.5464869000024919} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_library_likelihood_jit/lower_s` |
| `measurement/76f0ad54104249a79a3e` | s3_active_set_masked_p7_jit.first_call_s | {"s3_active_set_masked_p7_jit.first_call_s": 0.20880850000321516} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/first_call_s` |
| `measurement/77f3c9165b95dc8b8ee5` | s3_nnls_pdip_jit.compile_s | {"s3_nnls_pdip_jit.compile_s": 0.9629942999999912} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/compile_s` |
| `measurement/7a0db6f33a63f92e649e` | s3_active_set_masked_p5_jit.lower_s | {"s3_active_set_masked_p5_jit.lower_s": 0.1511127999983728} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p5_jit/lower_s` |
| `measurement/7a170e9b94002d8bfcd8` | s3_active_set_masked_p11_jit.compile_s | {"s3_active_set_masked_p11_jit.compile_s": 1.456014999999752} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/compile_s` |
| `measurement/7b17f68e07aac48f38b9` | s0_library_likelihood_jit.lower_s | {"s0_library_likelihood_jit.lower_s": 18.839743999997154} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_library_likelihood_jit/lower_s` |
| `measurement/7f99738ee7ef15587d01` | s3_log_det_regularization_jit.lower_s | {"s3_log_det_regularization_jit.lower_s": 0.0002884999994421378} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/lower_s` |
| `measurement/83813a49392bc4dbd28b` | s3_nnls_pdip_one_iteration_jit.first_call_s | {"s3_nnls_pdip_one_iteration_jit.first_call_s": 0.015946299998176983} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/886111be5568b80249d7` | s3_active_set_masked_p2_jit.lower_s | {"s3_active_set_masked_p2_jit.lower_s": 0.046039299999392824} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/lower_s` |
| `measurement/8c9b22915c36a7cc8793` | s3_curvature_reg_build_jit.first_call_s | {"s3_curvature_reg_build_jit.first_call_s": 0.18915320000087377} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/first_call_s` |
| `measurement/8e175115a7c9f2e41998` | s3_nnls_pdip_jit.first_call_s | {"s3_nnls_pdip_jit.first_call_s": 0.10322799999994459} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_jit/first_call_s` |
| `measurement/90a270f10a94ae8515ff` | s3_curvature_reg_build_jit.compile_s | {"s3_curvature_reg_build_jit.compile_s": 0.17734759999802918} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/compile_s` |
| `measurement/9767e8460b6f50e66db0` | s3_active_set_masked_p8_jit.compile_s | {"s3_active_set_masked_p8_jit.compile_s": 1.4459859999988112} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p8_jit/compile_s` |
| `measurement/a903ca8ae73a6a7d060d` | s0_log_det_curvature_reg_jit.first_call_s | {"s0_log_det_curvature_reg_jit.first_call_s": 0.006322100001852959} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_curvature_reg_jit/first_call_s` |
| `measurement/a9df5ab955e0e23bf1ab` | s3_active_set_masked_p12_jit.lower_s | {"s3_active_set_masked_p12_jit.lower_s": 0.14853260000018054} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/lower_s` |
| `measurement/ac909f78c6dd9502f97c` | s3_curvature_reg_build_jit.lower_s | {"s3_curvature_reg_build_jit.lower_s": 0.019377100001293} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_curvature_reg_build_jit/lower_s` |
| `measurement/acb345b88faa729db5d0` | s3_active_set_masked_p4_jit.compile_s | {"s3_active_set_masked_p4_jit.compile_s": 1.4186310000004596} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/compile_s` |
| `measurement/b06686345db48a2e6714` | s3_active_set_masked_p7_jit.lower_s | {"s3_active_set_masked_p7_jit.lower_s": 0.15624039999966044} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p7_jit/lower_s` |
| `measurement/b3389d7701f4f1dc4427` | s0_log_det_regularization_jit.compile_s | {"s0_log_det_regularization_jit.compile_s": 1.5899997379165143e-05} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/compile_s` |
| `measurement/b6d12560e329e3b1e1ac` | s0_nnls_pdip_jit.first_call_s | {"s0_nnls_pdip_jit.first_call_s": 0.1459620999994513} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/first_call_s` |
| `measurement/b791e3d0e42f02df6f01` | s3_active_set_masked_p11_jit.first_call_s | {"s3_active_set_masked_p11_jit.first_call_s": 0.23433859999931883} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p11_jit/first_call_s` |
| `measurement/bc68c6d6270b368cc1de` | s3_active_set_masked_p10_jit.first_call_s | {"s3_active_set_masked_p10_jit.first_call_s": 0.2370374999991327} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p10_jit/first_call_s` |
| `measurement/be4c09818025a8f87072` | s3_log_det_regularization_jit.compile_s | {"s3_log_det_regularization_jit.compile_s": 1.1899999663000926e-05} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_regularization_jit/compile_s` |
| `measurement/c0b2dc5cea055af5d923` | s0_log_det_regularization_jit.first_call_s | {"s0_log_det_regularization_jit.first_call_s": 0.00506649999806541} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/first_call_s` |
| `measurement/cc102772d34467db679f` | s3_cholesky_solve_jit.lower_s | {"s3_cholesky_solve_jit.lower_s": 0.013618600001791492} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/lower_s` |
| `measurement/cd8cf3db7ca2d10c1de6` | s3_active_set_masked_p9_jit.compile_s | {"s3_active_set_masked_p9_jit.compile_s": 1.4919174999995448} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p9_jit/compile_s` |
| `measurement/ce5cb84159a8757f28fa` | s0_library_likelihood_jit.compile_s | {"s0_library_likelihood_jit.compile_s": 18.519776799999818} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_library_likelihood_jit/compile_s` |
| `measurement/ceb9c6bd8ec5af27b130` | s3_cholesky_solve_jit.compile_s | {"s3_cholesky_solve_jit.compile_s": 0.12157200000001467} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_cholesky_solve_jit/compile_s` |
| `measurement/d3217a361ac8cd2c5369` | s3_nnls_pdip_one_iteration_jit.compile_s | {"s3_nnls_pdip_one_iteration_jit.compile_s": 0.46814159999848926} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/d4079a03fcd90d020f26` | s0_cholesky_solve_jit.lower_s | {"s0_cholesky_solve_jit.lower_s": 0.013708200000110082} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/lower_s` |
| `measurement/e29c50f8771ef8ba96ee` | s3_library_likelihood_jit.first_call_s | {"s3_library_likelihood_jit.first_call_s": 0.6330086999987543} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_library_likelihood_jit/first_call_s` |
| `measurement/e4f60a5999c9521e0346` | s3_active_set_masked_p2_jit.first_call_s | {"s3_active_set_masked_p2_jit.first_call_s": 0.024977799999760464} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p2_jit/first_call_s` |
| `measurement/e5dd800a773382178818` | s3_active_set_masked_p12_jit.compile_s | {"s3_active_set_masked_p12_jit.compile_s": 1.7239870999983395} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p12_jit/compile_s` |
| `measurement/ed5b2e3b29d286890438` | s3_active_set_masked_p4_jit.lower_s | {"s3_active_set_masked_p4_jit.lower_s": 0.15361510000002454} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p4_jit/lower_s` |
| `measurement/ef6941c0b291a0e55a52` | s0_curvature_reg_build_jit.first_call_s | {"s0_curvature_reg_build_jit.first_call_s": 0.29650289999699453} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_curvature_reg_build_jit/first_call_s` |
| `measurement/f109724eab33c1848a45` | s3_log_det_curvature_reg_jit.first_call_s | {"s3_log_det_curvature_reg_jit.first_call_s": 0.007579799999803072} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/first_call_s` |
| `measurement/f2642147b77fe5bae281` | s3_active_set_masked_p1_jit.compile_s | {"s3_active_set_masked_p1_jit.compile_s": 0.28445339999962016} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p1_jit/compile_s` |
| `measurement/f3da93e6587729591161` | s3_nnls_pdip_one_iteration_jit.lower_s | {"s3_nnls_pdip_one_iteration_jit.lower_s": 0.025142399998003384} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/f53a14ca9d4f238c138f` | s0_log_det_regularization_jit.lower_s | {"s0_log_det_regularization_jit.lower_s": 0.0003814999981841538} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_log_det_regularization_jit/lower_s` |
| `measurement/f5a9d2da06a353a4f637` | s0_nnls_pdip_jit.lower_s | {"s0_nnls_pdip_jit.lower_s": 0.15279159999772673} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_jit/lower_s` |
| `measurement/f654d66a2aa2a5c3eac2` | s3_log_det_curvature_reg_jit.compile_s | {"s3_log_det_curvature_reg_jit.compile_s": 1.790000169421546e-05} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/compile_s` |
| `measurement/f6def30f9960a6b32449` | s3_active_set_masked_p3_jit.lower_s | {"s3_active_set_masked_p3_jit.lower_s": 0.045830999999452615} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_active_set_masked_p3_jit/lower_s` |
| `measurement/f74af1e1542761ac287f` | s0_nnls_pdip_one_iteration_jit.first_call_s | {"s0_nnls_pdip_one_iteration_jit.first_call_s": 0.019412699999520555} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/f943d12898e5f80b8255` | s0_cholesky_solve_jit.first_call_s | {"s0_cholesky_solve_jit.first_call_s": 0.013742800001637079} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s0_cholesky_solve_jit/first_call_s` |
| `measurement/fb97957d9914cc315243` | s3_log_det_curvature_reg_jit.lower_s | {"s3_log_det_curvature_reg_jit.lower_s": 0.0003789999973378144} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/local_gpu/light_fixed/scans/n1000/breakdown.json) `/jit_phases/s3_log_det_curvature_reg_jit/lower_s` |

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
      "nvidia_smi": "NVIDIA GeForce RTX 2060 with Max-Q Design, 1634 MiB, 6144 MiB",
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
