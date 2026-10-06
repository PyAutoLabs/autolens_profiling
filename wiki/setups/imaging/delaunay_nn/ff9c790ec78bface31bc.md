<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/36fe6ab73f08ff6e0ffc`.

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

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>41 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00528d366fbfcae9f651` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.0183509883005172} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/0df679b1c18fde4da1af` | inversion_setup_vmap16.steady_per_call_s | {"inversion_setup_vmap16.steady_per_call_s": 0.015319496556185187} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/inversion_setup_vmap16/steady_per_call_s` |
| `measurement/12bd1c132017d0d4608b` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0008694520918652408} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Regularization matrix (H)` |
| `measurement/1c40d313b392025bdefb` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0009420246817171574} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/2961655614fa4943af23` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.018545336392708122} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/298ba683bbdaa4268a5a` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00014401120133697987} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/2b09ebaf3000bfec32fd` | setup_prefix_11_vmap16.steady_per_call_s | {"setup_prefix_11_vmap16.steady_per_call_s": 0.009961265043239109} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_11_vmap16/steady_per_call_s` |
| `measurement/2f39efb7b2b0f5dfd4c1` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00020104290451854466} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/35ff1e5da62421d14ef5` | setup_prefix_8_vmap16.steady_per_call_s | {"setup_prefix_8_vmap16.steady_per_call_s": 0.0150407444438315} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_8_vmap16/steady_per_call_s` |
| `measurement/421687653b79e72295ae` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.0010731908958405256} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/4f5640efe0edd05eb8cb` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.0003289210144430399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/54909f43c519f8446187` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.515794375445693e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/5d491557425ba2a0fc18` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.027632087096571924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/6007f5f54e5ac42c3736` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.009961265043239109} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/6e0c90f395d152f47ec5` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.000890456355409696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps_vmap_per_call/Regularization matrix (H)` |
| `measurement/6eafb1e67abd1af7ab12` | component_total | {"component_total": 0.07490132919047027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/total_step_by_step` |
| `measurement/72afeecaf460f36b6cf0` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00014607920311391354} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Profile-subtracted image` |
| `measurement/7b5ba3071cb98ad66a01` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00014401120133697987} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/7bc6dc476e35ec9a0574` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.009070808687829413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/82c669f2e934b4461167` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.009070808687829413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/93e13942fd894eca24d5` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.019414788484573363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/regularization_matrix_prefix_s` |
| `measurement/9468085468178931dbf1` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.00661381598765729} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/9878d2274703d862b5c1` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00014607920311391354} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/99098471e1fe7c6dd218` | setup_prefix_6s_vmap16.steady_per_call_s | {"setup_prefix_6s_vmap16.steady_per_call_s": 0.006516138168808539} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6s_vmap16/steady_per_call_s` |
| `measurement/9ff7ad75eba8e0d3d71e` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0022753710858523847} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Mapped recon + log evidence` |
| `measurement/a89036d7ec66bbe128f8` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.027632087096571924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/b9cd05e994130ca09be3` | setup_prefix_6s.steady_per_call_s | {"setup_prefix_6s.steady_per_call_s": 0.018413695110939442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6s/steady_per_call_s` |
| `measurement/bb849f859f8b5a2d900f` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0009420246817171574} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/bdff54a9de97272bf02b` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0003289210144430399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Data vector (D)` |
| `measurement/c39040baddfcc5a46d6c` | interpolator_prefix | {"interpolator_prefix": 0.018545336392708122} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/interpolator_prefix_s` |
| `measurement/ca2a08117d4fa06e26d4` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.037309553404338655} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Regularized reconstruction` |
| `measurement/cafba53f004e604d17a1` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.019414788484573363} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/cb01a20a1857b83f3eb1` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.004829921806231141} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/d4c28ebc8cffad6e6d44` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004829921806231141} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Curvature matrix (F)` |
| `measurement/d6f9e5eb41104d9096f7` | setup_prefix_8.steady_per_call_s | {"setup_prefix_8.steady_per_call_s": 0.02284741410985589} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_8/steady_per_call_s` |
| `measurement/ea013aa28f2507e58ed5` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.0022753710858523847} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/eac71844a1e9b2ea7361` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.015319496556185187} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/ec85f1c53c86219ef61c` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.037309553404338655} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/fb3627a7734283a47c3e` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00022286470048129558} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Ray-trace data grid` |
| `measurement/fdd2d02fa36e74203399` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00022286470048129558} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/ffaac59b131198c26ccd` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00020104290451854466} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/steps/Ray-trace mesh grid` |

</details>

### compile

<details><summary>55 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00ba70113aa5495489d1` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.0062058051116764545} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/0786dafd11f113e1667a` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.0387755217961967} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/0b45a2e280efe1c38b88` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.17938449699431658} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/0da294f299919ef2c00f` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.4698414499871433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/11586e25279f9afcd281` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.04608455998823047} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/18237569adcb1914d51a` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.23637450695969164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/1e84e2ada6dcb0a9c9a9` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.06682160287164152} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/1f0029e10f1a388a6371` | setup_prefix_8.lower_s | {"setup_prefix_8.lower_s": 0.6583214248530567} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_8/lower_s` |
| `measurement/25bc83b9cfea9fc9ec39` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0038423589430749416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/283107f933fc23e896b2` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.0009199460037052631} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/2ef642bbd8e1a3248407` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.08539139898493886} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/3d003c17389bbeed5302` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.029786571860313416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/3f5c2e3f28b41fe187ab` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.23017423111014068} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/4107275be045ac5f16d4` | setup_prefix_6s.lower_s | {"setup_prefix_6s.lower_s": 0.6241243479307741} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6s/lower_s` |
| `measurement/44be1d7a7a69c4a12a11` | setup_prefix_8.compile_s | {"setup_prefix_8.compile_s": 3.546806424856186} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_8/compile_s` |
| `measurement/4e2e12998e536a16dac9` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 4.7103982088156044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/4f580b300cbf6328932a` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0011242639739066362} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/50d16314a6a8d8b454ce` | setup_prefix_6s.compile_s | {"setup_prefix_6s.compile_s": 3.4501203619875014} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6s/compile_s` |
| `measurement/51e6426ec79a8528bd5c` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 4.551712044049054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/5d4c5d3571ea7a98a2d2` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.17134287301450968} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/6d96cf67c4e58b7fc16b` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.005502039100974798} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/6ec9d57ac151c8cea1b6` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.06104088597930968} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/729cbee796005770bba1` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.407774087972939} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/74f3e3e2e456fae614b7` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.3994670279789716} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/7821f4c6b68673ad54a5` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 3.6639362249989063} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_7/compile_s` |
| `measurement/78c21f1eea67596c5ba7` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0007882548961788416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/7a222c80ca43448c9114` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.015671771951019764} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/7bdc0da1b7de929c1502` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0016428709495812654} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/855a9e178cc227226125` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.005116011016070843} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/8820adb60efa7582cd98` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 1.0480510059278458} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/8b63a98cc411c32cd1ad` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 1.3687242651358247} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/9b685643fd6ccf68c5dd` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.09535822202451527} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/a714e50cacd7570da0b9` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.003316761925816536} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/b2464471ac0c33090bcd` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 3.7205863050185144} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/b53da025d96af152ad53` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006842261645942926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/b5d969f5b4642a2b27e6` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 12.155205837916583} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/b5e6a4182045fd67525d` | setup_prefix_6s_vmap16.first_call_s | {"setup_prefix_6s_vmap16.first_call_s": 4.976720307022333} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6s_vmap16/first_call_s` |
| `measurement/c3132f79e44292c620aa` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.02756226505152881} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/c331d7e8b48af919c66a` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 0.6463278131559491} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/c45045307744e02e4786` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.148027203977108} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/c64adf56a91e863e59c5` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.9342909478582442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/c692e98ccd23fe7cea54` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.07891068398021162} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/c72a28bad13f927c1867` | inversion_setup_vmap16.first_call_s | {"inversion_setup_vmap16.first_call_s": 20.20731581095606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/inversion_setup_vmap16/first_call_s` |
| `measurement/c89bbef622f2834903e2` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 4.466043039225042} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/cfaf7ad13b11e4e943c5` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.02757216291502118} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/d1f3ddeaa8e2f72eb5c9` | setup_prefix_6s.first_call_s | {"setup_prefix_6s.first_call_s": 0.028263490181416273} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6s/first_call_s` |
| `measurement/d311cdc16d6228b79a8d` | setup_prefix_8.first_call_s | {"setup_prefix_8.first_call_s": 0.03064663801342249} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_8/first_call_s` |
| `measurement/e02afd8969eb69f91851` | setup_prefix_8_vmap16.first_call_s | {"setup_prefix_8_vmap16.first_call_s": 5.046458543045446} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_8_vmap16/first_call_s` |
| `measurement/e15dc3b11dc2f69a99f2` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.09345522406511} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/e4925d85facd46d68322` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.041200357023626566} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/e933650230a6601da63b` | setup_prefix_11_vmap16.first_call_s | {"setup_prefix_11_vmap16.first_call_s": 5.078157044015825} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/setup_prefix_11_vmap16/first_call_s` |
| `measurement/f26cf18fbf9631fd03f5` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.06024231994524598} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/f6272d58bd955ee3968b` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.02949865418486297} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/fb41f845c4654f405dc2` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.005010301945731044} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/fc4e174de4798f86d7e3` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.43200498213991523} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.json) `/jit_phases/reconstruction_jit/compile_s` |

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
