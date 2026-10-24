<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/ced81e7a17d0f343362f`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| delaunay_vertices | 3000 | recorded |  |
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
| source_pixels | 3000 | count |  |
| source_pixels_requested | 3000 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 46083 | recorded |  |
| sparse_operator_build_s | 0.7258026991039515 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 3060 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>49 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0295cc889551cb218fb6` | nnls_pdip_vmap16.steady_per_call_s | {"nnls_pdip_vmap16.steady_per_call_s": 0.08393650582584086} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/0be2e30b2f5b78f2f196` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.00596313769929111} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/0c0b4dbca7ed3a8da727` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.012575491797178984} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/0cf53c36f91b1a8be82b` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00013619828969240188} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/0d69307ed29febe1c57a` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.0029354576021432877} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/12cee282b5abbb747743` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.004147615795955062} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/1f439af1f89e61f3c270` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.318370869848878e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/2522a86f15f6c99dc830` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0012260461691766978} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/29b26ba369970a74cb5c` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.010237335489364341} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/2bae6319604dfa2bebee` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.03772853808477521} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/414d4c1ab83936b58a6b` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.012575491797178984} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/43076c82118bb921edf8` | interpolator_prefix | {"interpolator_prefix": 0.012261074502021074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/interpolator_prefix_s` |
| `measurement/44a81d286f661841e4ce` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.010170703299809247} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/4c0f8724172c6e2b27d5` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.0001948388759046793} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/5db84b593ce9f1d79b77` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.03279797579161823} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/6632ead126dd13feae13` | inversion_setup_vmap16.steady_per_call_s | {"inversion_setup_vmap16.steady_per_call_s": 0.010697312350384892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/inversion_setup_vmap16/steady_per_call_s` |
| `measurement/6a404a491cf745bacb74` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.00018001599237322808} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/6a905dcd6ccf00eb8f9b` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0001623400952666998} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/79f10e70f5406d37cd82` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.00018001599237322808} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/7a23f0e8f333486f202b` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.03772853808477521} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/7c8f8c85d18f27e30902` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.00042826458811759947} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/7d8eb554f99cbd778966` | setup_prefix_11_vmap16.steady_per_call_s | {"setup_prefix_11_vmap16.steady_per_call_s": 0.010237335489364341} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_11_vmap16/steady_per_call_s` |
| `measurement/8390e906c20f0c3b11b2` | component_total | {"component_total": 0.1396630886476487} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/total_step_by_step` |
| `measurement/83bf9065f295d43536b1` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0022462533786892893} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/89e562ad1871c5da0f27` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.07597824148833751} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/8c830b17cba2fcac0883` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.0001706650946289301} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/96227b8ee6001955ad83` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.005060691805556417} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/99266dcd625d85385976` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.0001623400952666998} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/a73f397c569b33dfbd22` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.012261074502021074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/b1b454cfe36fee937914` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.0022861262783408164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/b272fec3e62bf60d29e0` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0003144172951579094} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/b37333eaa36db5021198` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 0.018831002712249755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/c2b4a8dfa8ce8f34270c` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0001706650946289301} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/c6323af1235649d18213` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.004147615795955062} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/c7f7327c6ef39b3ffe8f` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.0010601680725812912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/c96874679c23ba7b39ab` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.0006915390957146883} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/d3986977295fd39d191a` | steps_vmap_per_call.Inversion setup (sparse, steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (sparse, steps 5-8 combined)": 0.010697312350384892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps_vmap_per_call/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/d5f7c1a4a0e6398b0720` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00013463529758155345} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/d96d3496f048ebc481d2` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00013619828969240188} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/dafe81d074c6efde9d02` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00013463529758155345} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/e502218beccad7c2e772` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.018831002712249755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/e99e0d11d43626724c85` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.010281979985302314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/ecebd85243f0a08bd0b4` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.012374300789088011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/ee7e7e793a3e4f6648d1` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0022356174886226654} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/f1555bed7d06694d5a40` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.07663161382079124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/f33d49664ab3acc56074` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0012260461691766978} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/f55d9c1e609492e5d2ab` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.07663161382079124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/fcd8c8cca28ece8acabf` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.00015805107541382312} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/fe85eb2e55a357fe4098` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.010281979985302314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/interpolator_prefix_vmap_per_call_s` |

</details>

### compile

<details><summary>84 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0cb9c1230b23749f8990` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.1379000493325293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/0fc54319668a39723813` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.0065004341304302216} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/1153f4e6678962512502` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.044802360236644745} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/17f89c7d786dd5daa9cd` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.3472155607305467} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/19977eb6c84e0f702d19` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 3.1061468049883842} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/19b0e2e5d45701287086` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.09035301487892866} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/1cc6e48b0280117f34d1` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.0066358838230371475} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/1e5e822ffcec25ac5fe0` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.04629439115524292} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/2123a8972ab8ad89bb2c` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.17203249875456095} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/2565bbf8e782ae887929` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.06955326022580266} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/2902a6b7d81cffa8150b` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.04544583475217223} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/298f0154ff5bc937738b` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0014723418280482292} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/2ad8d9bf5140ac26df71` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.48429418401792645} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/2ded6594ed3b8831eed5` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 2.5891024437732995} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/3021faf92fa400ada4d7` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 9.258100581355393} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/3281660c740dd00f694c` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.012102712877094746} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/332c05edf62973e3e876` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 11.04556147288531} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/39f6c9f27b56b95eee15` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.4629680663347244e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/3a4f791a88aa7fda6ae1` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.0028151636943221092} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/3c047ffaa8cde1c76956` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.030153211671859026} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/42863ba714da1f3494d8` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.012811298947781324} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/42cf64866091d7117e97` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.11651448998600245} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/4dac7802720e28dd476c` | nnls_pdip_vmap16.first_call_s | {"nnls_pdip_vmap16.first_call_s": 1.8949988610111177} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_vmap16/first_call_s` |
| `measurement/4fca41c46e955fa3e3ba` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.14410376409068704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/51dc206184926fe6f2f0` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.08369059301912785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/526cece92254d4bc45c1` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.016813755966722965} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/539e5c11fd580b0981af` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.025087629910558462} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/5576a4ba31f948e4bee4` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 0.9285251018591225} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/58c2e83da03458827318` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.0040884180925786495} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/60000e71e9b9ccd5f450` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.001170312985777855} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/634979d6c2327f5b63a4` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 5.911936108022928} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/648f2a9a85a468275d61` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.017672481946647167} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/67decb93b624b35ead98` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.0475771832279861} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/69909c3fce4b97d1c478` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.1933765998110175} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/6b0b1d721626d4078675` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.2357179126702249} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/6e536057691eb1edf9fc` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.0028080930933356285} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/6ee21656ff04039de751` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.20877753291279078} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/6faba632a41cb02ffb61` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.08451716834679246} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/7b91d936c1424ddaadfa` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.01177760399878025} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/7e60d9086b0202b3f774` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 1.7107101217843592} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/7ec36d45242dcc51bb89` | setup_prefix_11_vmap16.first_call_s | {"setup_prefix_11_vmap16.first_call_s": 3.7506970427930355} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_11_vmap16/first_call_s` |
| `measurement/7ec789a8f4440b344da5` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.012283580843359232} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/7ff9d27562ee45960208` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.04528613714501262} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/80dd6c8dd95f4c9af52a` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.48245608527213335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/8ccb5328d62cb714462a` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.007637506816536188} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/8de08a265eb54b310ae1` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.03050647908821702} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/902778b1776267a2b544` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.07767282612621784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/92731b07b5089dedb7bb` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0011375229805707932} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/9a16c875d101703779ef` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.42335754493251443} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/9c5ac34b2007963ad9a7` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 2.283809628803283} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/9e1f55cb487d54b4b2bc` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.25692329509183764} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/9f5eb627ff8542b49b02` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.018154769204556942} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/a1558eb281834933de1d` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.002816963940858841} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/a3f60845932473925ea6` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 1.8969902810640633} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/a78b6cbeae5821581f89` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.000833295751363039} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/a7c9b630ed3fa9986828` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 1.2136577190831304} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/a9e1004b33b7b9b52f1f` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 4.358416217844933} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/abc6a85bf5dad9c031b3` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.07769105583429337} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/ace953d65fbf4e51250d` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0014078407548367977} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/b9c4015179ee31e5396e` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.00035176798701286316} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/c4e720f48a17434e2599` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.03406037017703056} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/cf08cb71a04fd50fd1f6` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.4070678660646081} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/cf74f36a5fc868d360ac` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.0034675896167755127} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/d4c04195ad9602a5d792` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 1.962248055730015} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_7/compile_s` |
| `measurement/d88c92325786aedb71b1` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.43704316671937704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/dac3af078570cecb8928` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.6437187530100346} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/dbc4268a17a85287d4f9` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.014160779770463705} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/ddff3bd7b4303a22c3f0` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0006638760678470135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/dfde1435c160553c2b8c` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.026347812730818987} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/e38faa4b20b00fde990e` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.07801108388230205} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/e3c48c80be3ceeef137b` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.4767730748280883} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/e459cccd311f2f2e0332` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.10090708592906594} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/e85fc862bc1cc8c8d042` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.014675416983664036} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/e9602a15ba3895dfbb88` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.0008113449439406395} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/ea2320f0def7e3dd47a7` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.08677340531721711} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/ee3d61dbe46d42ac53e3` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.11522927694022655} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/ee72c24218866358cbb5` | inversion_setup_vmap16.first_call_s | {"inversion_setup_vmap16.first_call_s": 15.754710319917649} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/inversion_setup_vmap16/first_call_s` |
| `measurement/f0feee8c0c25d5c81aa4` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.005064371973276138} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/f2eccf4f7d97a5b6619b` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.005144352093338966} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/f309328ca6424bf0fabe` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.009082358796149492} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/f9a8b6f52ead3e33a988` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006562559865415096} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/fad67d8818d5c9a591c9` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.09515716787427664} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/fbe107c51fccfe294b18` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.02513520885258913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/fd0ae3eda0a5fbd78590` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.042484222911298275} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_sparse_fp64/hpc_a100/scans/n3000/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 7687 MiB, 81920 MiB",
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
