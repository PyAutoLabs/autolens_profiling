<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/0564a315d9ad30a42059`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| delaunay_vertices | 5000 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "note": "Outer sub-size 1 retired repo-wide on 2026-09-08 (autolens_profiling#235): it leaves the outermost annulus un-over-sampled and causes gradient issues.", "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 5000 | count |  |
| source_pixels_requested | 5000 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 491552 | recorded |  |
| sparse_operator_build_s | 0.6493619587272406 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 5060 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>51 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/061e4168e92e683c51aa` | interpolator_prefix | {"interpolator_prefix": 0.04175031329505145} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/interpolator_prefix_s` |
| `measurement/06caeb12ee620ff77586` | nnls_pdip_vmap16.steady_per_call_s | {"nnls_pdip_vmap16.steady_per_call_s": 0.28745172840135635} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/07964817b8585802abcf` | setup_prefix_6s.steady_per_call_s | {"setup_prefix_6s.steady_per_call_s": 0.04036034247837961} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6s/steady_per_call_s` |
| `measurement/1359f821655a0a22f718` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.699096993543208e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/18021a03b0026938d134` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.1712560244835913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/199e3b617fb4e526ea17` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.0014765192288905382} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/1c1c81f0f7dde60ea6fe` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.03302116971462965} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/1c6718565ec8a64bf4d1` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.013548777205869555} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/1ca174dc199bdb8dda82` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.030107853189110755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/22b164ac291ef5e3b3b8` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.018995954032288865} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/24dccc09f290b215cd7a` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.04175031329505145} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/267a38cf8bb4b570b3b3` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.03302116971462965} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/26b7cfe7282b944b1977` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.010260104155167937} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/270e6f86f93934b731d0` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.005499590886756778} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/31d8e5eefdd311eca8b5` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.005074081476777792} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/37b69d68b77b8b2b6e5b` | setup_prefix_6s_vmap16.steady_per_call_s | {"setup_prefix_6s_vmap16.steady_per_call_s": 0.019046723801875488} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6s_vmap16/steady_per_call_s` |
| `measurement/4031950b7bbf15cc0d76` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00015898710116744042} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/4dcf1d1c9da0cf9822b3` | inversion_setup_vmap16.steady_per_call_s | {"inversion_setup_vmap16.steady_per_call_s": 0.019625374450697564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/inversion_setup_vmap16/steady_per_call_s` |
| `measurement/50ecb79e225305c7da92` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.1712560244835913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/532384a78431f3395654` | setup_prefix_11_vmap16.steady_per_call_s | {"setup_prefix_11_vmap16.steady_per_call_s": 0.021629824439878577} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_11_vmap16/steady_per_call_s` |
| `measurement/56081719f74603ef58dc` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.06564793870784343} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/5c74143354546be17b61` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.17108228257857264} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/6ab80f3c0a12e6ec8d1a` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.006611434230580926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/6dbdfa1030a30cf1cfef` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.002644790266640485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps_vmap_per_call/Regularization matrix (H)` |
| `measurement/7724ee877a7d7cd90e63` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.0007616204209625721} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/7ec849bef827761fb369` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.010260104155167937} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/8bdd4150d6bd0ad77671` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.005168610392138362} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/8f2c5f60b291afe19b0a` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.03591514928266406} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/8f6b0421d5ef0020cfc0` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 0.03591514928266406} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/9174d0c7783d41e3a0bd` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00015898710116744042} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/9bb65397531d11f3457d` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.00014496520161628724} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/9d289db80d06c9349388` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0008238081820309163} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/a34a7dc6952e46ed7117` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00015647802501916885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/a35aab94fafe478dcfbd` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0008238081820309163} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/a856d9b97b9ab34ce647` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.06564793870784343} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/b2fbefbbe9c8a82b0ba3` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.00042589139193296435} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/b77aaf8a14678f105d76` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00012625730596482754} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/b862498a8dc3075a1b82` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.018985034173238092} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/c803fdccc5834f153295` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00015647802501916885} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/c86609b0fc028b08e4e2` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.018985034173238092} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/cbcfe1a32fbf266ed30d` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.00020443582907319068} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/cce41d8f1ccdc6c75e71` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00014887512661516667} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/d8c5ea2d0c9ce68de7fa` | steps_vmap_per_call.Inversion setup (sparse, steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (sparse, steps 5-8 combined)": 0.019625374450697564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps_vmap_per_call/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/e12ca1c556dd6d824850` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.00020443582907319068} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/e8d4300ce546948deade` | component_total | {"component_total": 0.27596891461871564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/total_step_by_step` |
| `measurement/e94f49cb0fc1adec9dc1` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.05919058853760362} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/efbdae30f79d495b070e` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.005192507198080421} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/f05de8d450e68c3e25b5` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00012625730596482754} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/f5d4c5981bbca636281c` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.021629824439878577} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/f9195afe64150e791ed1` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00014887512661516667} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/fccd085e4d3211b5ed59` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.00018675080500543117} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |

</details>

### compile

<details><summary>88 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/05e4c118129fc5064fd0` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.48999981116503477} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/079f16a7ccf522554770` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.08792636217549443} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/0d9ca00ce4d226fc2582` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 3.3588173892349005} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/0f2c2e049e4162f8079c` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.037018368020653725} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/13cd677a2308081cb8b8` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 9.204988628160208} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/15b8bf695c7df3ee4057` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.04799019172787666} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/162f0dae82cc2d9e8b27` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.4180644559673965} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/1e0810a0b7e7d32b213d` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.005325587932020426} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/1fc4444ffef9198b6aa2` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.4872500039637089} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/22348762749ce3e604bd` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.1290686042048037} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/2351c52f46bf180d34de` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.02743293857201934} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/27d7673f45eeea826cd4` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.05528877815231681} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/28ada322af0019ee7135` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0010277638211846352} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/291b4e24e20e0a0c1a0d` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0011993232183158398} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/2aa96632a3fe12165608` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.08940685726702213} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/2ef5004a969727a07f24` | nnls_pdip_vmap16.first_call_s | {"nnls_pdip_vmap16.first_call_s": 5.201894846744835} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_vmap16/first_call_s` |
| `measurement/302e0b85ea4c90ca7dec` | setup_prefix_11_vmap16.first_call_s | {"setup_prefix_11_vmap16.first_call_s": 5.634406004101038} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_11_vmap16/first_call_s` |
| `measurement/30a90367747d82ce452a` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0003989781253039837} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/312b5183c9576a6212db` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.1729500419460237} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/33c20efe5a1b31956b95` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006346260197460651} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/34eee4d5304ccbc43bb6` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.16753974417224526} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/354c0851e7f0a7cb42bc` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.09364727977663279} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/39b218a2ce09856922b3` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.00082374457269907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/411426113e001e3425a8` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 4.41852388298139} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/461ba4862b4de61fcee0` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 3.479334832634777} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_7/compile_s` |
| `measurement/4ad01aa4d69d36fd66c0` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.01907297410070896} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/4b42cd86fd993c8640ad` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.000712796114385128} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/4f111590504e4e24b705` | setup_prefix_6s.compile_s | {"setup_prefix_6s.compile_s": 3.457658624276519} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6s/compile_s` |
| `measurement/528634883f9e963b6b06` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.016140829771757126} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/53084044b7289e510251` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.2364809880964458} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/57a29377c7681604cf92` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.06981366081163287} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/57eb3283f334623db371` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.6045636190101504} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/5865c966621593c4d4f9` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.4631702918559313} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/595ebcdb3b7093e42232` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 11.609173430595547} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/5a0c066aaeab4ab96598` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.404103373643011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/5cd1dc0a42edf83c7777` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.0650405571796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/625427d651ba6ad47434` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.3730570822954178} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/649592130514da2f8556` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.0061084432527422905} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/65095908f02d80704d46` | inversion_setup_vmap16.first_call_s | {"inversion_setup_vmap16.first_call_s": 20.88407161924988} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/inversion_setup_vmap16/first_call_s` |
| `measurement/6f4f9243b0f1cd00a539` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 2.11034335475415} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/70f2d030ab72ccc47be3` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.02521030930802226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/73a166babd481c4be125` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0008338857442140579} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/777844fd04862005f6ba` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.042509775608778} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/7a67dca5053ca2694bfb` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.13706516800448298} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/7ee17e6169820eb3c830` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.010916832834482193} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/8094ebf4ccc95b695cc1` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.004797812085598707} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/80a215a5f400eeecb1d0` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.3399869203567505e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/8839c4c5dc20c87d43b9` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.007223055232316256} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/89bc65e5201f391c0bad` | setup_prefix_6s.first_call_s | {"setup_prefix_6s.first_call_s": 0.04923727409914136} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6s/first_call_s` |
| `measurement/8ac877599b31c3c60964` | setup_prefix_6s.lower_s | {"setup_prefix_6s.lower_s": 0.8226500391028821} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6s/lower_s` |
| `measurement/8ca69af87e73eb296787` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 4.548458083998412} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/8f6f7879db88cb939318` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.04176656389608979} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/94a65a985517d52275a8` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.036932459101080894} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/9722f0ac30ddbcf4ab11` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.029833655804395676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/9a0f0b8815ec1ba28338` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.1704827663488686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/9b70a5b742372c3a44db` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 7.956896000076085} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/9c91e0b908aab83ef7b9` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.011745526921004057} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/a8debbc23b10c978d51c` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.5471345079131424} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/aa4d0a99d7633239ef8f` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 2.801927102264017} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/af1b819e6330333c2f5b` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.178574298042804} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/af259fe91f60d812ff1e` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.004145274870097637} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/b2d85a62a642b9930505` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.02507008472457528} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/b3d03f5616de34bacc0f` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.09283617092296481} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/b5603f6affdb04b71de8` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.17321248026564717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/b69984db5b14edd03135` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.015045239124447107} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/bea79bf56d12a6f3532d` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.005856123752892017} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/c3623ce4582189c4f4ff` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 1.257709976285696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/c4d069453705778b5ac3` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.0071149771101772785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/c79e676b879e5f5bc05b` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.04845422878861427} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/ceb9fa7c6ac212cb75bf` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.033829864114522934} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/d05e0a31213d026585f9` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.07806927990168333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/d0f3d146cf8f2647d586` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 0.9566090442240238} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/d2f7f06149ea0180f793` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.6084514851681888} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/d355c1e5081dd32d2ec9` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.024502709042280912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/d41b8c35e037f8221f86` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.0749408402480185} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/d42a6825023a05326deb` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.01790347322821617} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/d8e16182889678b36d08` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.3815340050496161} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/da888bb70e8ff5bbe71d` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.12609945703297853} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/dc40079b517f511e839c` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.15439623221755028} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/e093614e4a13a7a9c2b3` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.006663269829005003} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/ec20b639e2e60173d2ab` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.0009777327068150043} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/f0fc03cdb7e443753459` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.27760207233950496} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/f6fd452051d066fabd4e` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.016097323037683964} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/f85752cb3a93c7d0760b` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0036166482605040073} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/fa915b1937a90df00ca6` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.043353729881346226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/fba738881165988c3437` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0013400930911302567} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/fc5d3fa08b5cbffb3357` | setup_prefix_6s_vmap16.first_call_s | {"setup_prefix_6s_vmap16.first_call_s": 4.529544956050813} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/setup_prefix_6s_vmap16/first_call_s` |
| `measurement/ffeb9815a8e4a5b51f0f` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.01269200211390853} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n5000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 14883 MiB, 81920 MiB",
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay_nn/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay_nn/likelihood_runtime.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
