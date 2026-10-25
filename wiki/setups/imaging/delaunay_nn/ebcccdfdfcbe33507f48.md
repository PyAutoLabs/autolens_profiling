<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/a3c8a14f5374002af312`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| delaunay_vertices | 1500 | recorded |  |
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
| source_pixels | 1500 | count |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1560 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>41 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0299d6f7fd1d5b2f969d` | inversion_setup_vmap16.steady_per_call_s | {"inversion_setup_vmap16.steady_per_call_s": 0.015319496556185187} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/inversion_setup_vmap16/steady_per_call_s` |
| `measurement/084f76a619595073fcdd` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.515794375445693e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/0dfb9dae827e88f0286e` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.019414788484573363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/1dae1148a79d4f6148c9` | setup_prefix_8.steady_per_call_s | {"setup_prefix_8.steady_per_call_s": 0.02284741410985589} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_8/steady_per_call_s` |
| `measurement/235b5188ea46c1683a9b` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.0183509883005172} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/26885f4d6c19c956b93a` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.019414788484573363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/2c719c315116f316f599` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004829921806231141} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/2d8e230c3cb4fe81c45b` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.0010731908958405256} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/36f38e69881d52be2071` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.0003289210144430399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/3ce6b380075e6586cea9` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.027632087096571924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/4444018045b2e89f01ed` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.027632087096571924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/58dcf697f2193a846715` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.004829921806231141} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/5be295823ec4cc7034a8` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0022753710858523847} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Mapped recon + log evidence` |
| `measurement/7d690836f578ffc81206` | setup_prefix_8_vmap16.steady_per_call_s | {"setup_prefix_8_vmap16.steady_per_call_s": 0.0150407444438315} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_8_vmap16/steady_per_call_s` |
| `measurement/83a8427f8d9c0d5ac02a` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0009420246817171574} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/8bdeb2a934d4ffb16bd4` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00014607920311391354} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/8d0285ab13c3fabf6a9c` | component_total | {"component_total": 0.07490132919047027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/total_step_by_step` |
| `measurement/995fc0f9d52f3bd152d1` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0003289210144430399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Data vector (D)` |
| `measurement/9a23da8f3fa017574428` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00020104290451854466} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/9bc79d72d55cdb1a1e89` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.009070808687829413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/9e68790720411db87a4b` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.037309553404338655} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/a848f4a93d4d754f8f2a` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0008694520918652408} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/ad4b519d966740ea804b` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.009070808687829413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/b185a4ba945bc0ee0487` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.0022753710858523847} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/b3259bcd9c2b5279562d` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.037309553404338655} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/bb861da4478161edbdd6` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.018545336392708122} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/bdf4b4787e2be2f7ab01` | setup_prefix_11_vmap16.steady_per_call_s | {"setup_prefix_11_vmap16.steady_per_call_s": 0.009961265043239109} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_11_vmap16/steady_per_call_s` |
| `measurement/c1964795296b8cb1f928` | setup_prefix_6s_vmap16.steady_per_call_s | {"setup_prefix_6s_vmap16.steady_per_call_s": 0.006516138168808539} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6s_vmap16/steady_per_call_s` |
| `measurement/d47358770235df474d06` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0009420246817171574} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/d738c715d7e35ae339d6` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.00661381598765729} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/ddcf5dbd9a70a81507b6` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00020104290451854466} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/e638e25c06d6d6a7a92f` | interpolator_prefix | {"interpolator_prefix": 0.018545336392708122} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/interpolator_prefix_s` |
| `measurement/e82f08417cc49b73e75a` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00014401120133697987} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/e8d452753362771c1cbe` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00022286470048129558} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/eaa76f5ce513be4d7e6d` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00014401120133697987} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/ee2f0335f992cc1e6197` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00014607920311391354} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/ef381075943f0a4f0010` | setup_prefix_6s.steady_per_call_s | {"setup_prefix_6s.steady_per_call_s": 0.018413695110939442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6s/steady_per_call_s` |
| `measurement/f658e5274f09dfa09ca2` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.009961265043239109} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/f8baea966e1f146f86fb` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00022286470048129558} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/fdeddf3a41ceae0a6c20` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.015319496556185187} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/fff78d7daf9f008270d8` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.000890456355409696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/steps_vmap_per_call/Regularization matrix (H)` |

</details>

### compile

<details><summary>55 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0032e02d33264e03daa2` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 4.551712044049054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/01e3b5d07f223d332a80` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0016428709495812654} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/0230c626b83e40d0bbe4` | setup_prefix_8.lower_s | {"setup_prefix_8.lower_s": 0.6583214248530567} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_8/lower_s` |
| `measurement/0551c25fe3b46e20ee26` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.07891068398021162} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/070a8abb7f14e1d4a417` | setup_prefix_8_vmap16.first_call_s | {"setup_prefix_8_vmap16.first_call_s": 5.046458543045446} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_8_vmap16/first_call_s` |
| `measurement/13aa214bc9a7e9db911a` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 1.0480510059278458} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/174fe71843c3b01c818a` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.407774087972939} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/1e8dd32b973e10371244` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.0009199460037052631} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/203f3f47c0f64c88aa71` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 12.155205837916583} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/25729a77f09fac74ec58` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.06682160287164152} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/28b0a675b7cf5898048c` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 3.6639362249989063} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_7/compile_s` |
| `measurement/28b533efb457a55f430e` | setup_prefix_11_vmap16.first_call_s | {"setup_prefix_11_vmap16.first_call_s": 5.078157044015825} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_11_vmap16/first_call_s` |
| `measurement/38a221d81fa73f1efb15` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.015671771951019764} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/4294e7a7906b55dd260b` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 4.466043039225042} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/4f540e859dd5ce15724b` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 1.3687242651358247} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/51e5e304bc64b2374f8c` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.041200357023626566} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/533939904b108c398502` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0011242639739066362} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/57e373d21cafface5e83` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.0387755217961967} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/5893375e2d6931400e8f` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.4698414499871433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/60f344d194bf160b383c` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.005010301945731044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/62de555138fbf083f90b` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.029786571860313416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/63bab8ed506aca7a9098` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.23637450695969164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/6982da1c8cbc62d4263a` | inversion_setup_vmap16.first_call_s | {"inversion_setup_vmap16.first_call_s": 20.20731581095606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/inversion_setup_vmap16/first_call_s` |
| `measurement/6c83adb746d9ba282d0e` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.0062058051116764545} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/753ffe4605442db6a614` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.43200498213991523} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/7590a2cea255e4fd3914` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.09345522406511} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/7886d3c6c3b54500dcf3` | setup_prefix_6s.first_call_s | {"setup_prefix_6s.first_call_s": 0.028263490181416273} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6s/first_call_s` |
| `measurement/84f3a9d4f04c65e290d6` | setup_prefix_6s.compile_s | {"setup_prefix_6s.compile_s": 3.4501203619875014} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6s/compile_s` |
| `measurement/877328988f2a1b2c9730` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.17938449699431658} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/8b93d15a65f35517e805` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.06024231994524598} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/8d1112af1703c44c4165` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.17134287301450968} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/9806ec4685af40a98fd8` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.3994670279789716} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/98c905329fbdea266e0e` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.02757216291502118} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/a391798efca26170ee36` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.08539139898493886} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/a3eba7780bb798ee5113` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 4.7103982088156044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/a8dddcf9d8eea350550f` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.148027203977108} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/af8f246de006af7f5469` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0007882548961788416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/b65e4c4d93ebdc4795e7` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.005502039100974798} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/b712c57b561fa2236ade` | setup_prefix_6s_vmap16.first_call_s | {"setup_prefix_6s_vmap16.first_call_s": 4.976720307022333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6s_vmap16/first_call_s` |
| `measurement/be7e85acd9fa4db029ea` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.23017423111014068} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/c18fc2b07808eb505d0d` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.9342909478582442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/c4361b4717da8afb36a3` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.06104088597930968} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/cae0c1cd4e876ad8018a` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.09535822202451527} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/cbfff564c8b802039619` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 0.6463278131559491} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/d1abb3d749301e80e536` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.04608455998823047} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/d306ac05eac7c22c0bd7` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 3.7205863050185144} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/d4782c628b23940210f1` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.005116011016070843} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/d95e2b17fe78b4d5bdb8` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0038423589430749416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/daec4df4836fe21386bc` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.003316761925816536} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/db957741a216e9bf296e` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.02949865418486297} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/e0b0963a01bc61b18665` | setup_prefix_6s.lower_s | {"setup_prefix_6s.lower_s": 0.6241243479307741} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_6s/lower_s` |
| `measurement/e100d5289fb71ac10cae` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.02756226505152881} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/f10198c09ea751b23583` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006842261645942926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/f577d39818a99ae53b90` | setup_prefix_8.first_call_s | {"setup_prefix_8.first_call_s": 0.03064663801342249} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_8/first_call_s` |
| `measurement/fa32e1841008edab86af` | setup_prefix_8.compile_s | {"setup_prefix_8.compile_s": 3.546806424856186} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/history/2026.8.17.1_2026-09-10/breakdown.json) `/jit_phases/setup_prefix_8/compile_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 41495 MiB, 81920 MiB",
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
