<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/73e2b01fcd0f698af171`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_shape | [71, 71] | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "note": "Outer sub-size 1 retired repo-wide on 2026-09-08 (autolens_profiling#235): it leaves the outermost annulus un-over-sampled and causes gradient issues.", "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "scheme": "constant"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 5041 | count |  |
| source_pixels_requested | 5000 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 5101 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>40 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0223765c719d9f7b48e6` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.031799543742090465} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/0ab1d543328bb06701ec` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.0007698700443143025} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/26bae75f6ca1b9b18b58` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0006162083242088557} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Data vector (D)` |
| `measurement/2c5a63c14c43223529ec` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0008539759088307619} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/3c6a4af0cfe313e7e18b` | overlay_grid_jit.steady_per_call_s | {"overlay_grid_jit.steady_per_call_s": 0.00014538518153131008} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/overlay_grid_jit/steady_per_call_s` |
| `measurement/3ffe1e9715996e4c8da3` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.006899541011080146} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/4e411654560847649ced` | component_total | {"component_total": 0.25648605627939103} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/total_step_by_step` |
| `measurement/4e83a6ea8381653f53a4` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.005409828806295991} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/5134be3f0ddb42abf9d6` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00012021530419588089} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/5751efcda11c60d018b9` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.0001356333028525114} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Profile-subtracted image` |
| `measurement/599eaebfec7b0456bb03` | nnls_pdip_vmap16.steady_per_call_s | {"nnls_pdip_vmap16.steady_per_call_s": 0.26584853904496414} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/64f7138d635322a5f8e7` | steps.Overlay grid (source pixel centres) | {"steps.Overlay grid (source pixel centres)": 0.00014538518153131008} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Overlay grid (source pixel centres)` |
| `measurement/65c745d6a1633a290dc2` | setup_prefix_8.steady_per_call_s | {"setup_prefix_8.steady_per_call_s": 0.027654965315014123} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_8/steady_per_call_s` |
| `measurement/696eb9064980609d2b63` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.001770433411002159} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/regularization_matrix_prefix_s` |
| `measurement/69d911e1339553b902ec` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.288962613325566e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/7e40992d97c78f902101` | regularization_matrix_assembly_jit.steady_per_call_s | {"regularization_matrix_assembly_jit.steady_per_call_s": 0.0003506990149617195} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_assembly_jit/steady_per_call_s` |
| `measurement/8773ba3391f3a5feac9e` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.16401164466515183} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/8b2d2d26c7aa055c1447` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.005799228511750698} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/8bb6a88efaee2793496b` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.011134045571088791} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Mapped recon + log evidence` |
| `measurement/8cf2c1f4060d659d6f81` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.014071925217285753} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/967f7c666abe6995a368` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0008539759088307619} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/99253aa9f91e38eafdc0` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.16401164466515183} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Regularized reconstruction` |
| `measurement/9df38410072d9540ad51` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.04720861199311912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Curvature matrix (F)` |
| `measurement/9e2fdaa2c6d32de9e311` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.00014498038799501956} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/a72dc34386d2c6aa50b3` | ray_trace_jit.steady_per_call_s | {"ray_trace_jit.steady_per_call_s": 0.0001908149104565382} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/ray_trace_jit/steady_per_call_s` |
| `measurement/a7907159e7c2cc19d5b5` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.04720861199311912} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/aa4f575aa1c014368af7` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0002699773758649826} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Regularization matrix (H)` |
| `measurement/ac078376f07da310a31d` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.001770433411002159} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/af68eee6578abef9f2bc` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.0006162083242088557} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/b3238b9882f30e1cb827` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.011134045571088791} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/baa5951fa8dfbfea17b0` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.001047653891146183} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/cae34b29086f96913b5b` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.16295570600777864} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/d145c44cd13fac8355fd` | interpolator_prefix | {"interpolator_prefix": 0.0015004560351371764} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/interpolator_prefix_s` |
| `measurement/d1efd50380024f328e7d` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.0001356333028525114} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/d1f3c969c40c98b2da3f` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.0015004560351371764} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/dbf8ad8907265d27a363` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.0001908149104565382} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Ray-trace grids` |
| `measurement/e62447afd837d4494f06` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00012021530419588089} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/f4e715af746936a4d5e4` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.005476787313818931} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/f975f506269119ae33f8` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.0021318972110748293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/feb86103ac723817743b` | steps.Inversion setup (steps 4-8 combined) | {"steps.Inversion setup (steps 4-8 combined)": 0.031799543742090465} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/steps/Inversion setup (steps 4-8 combined)` |

</details>

### compile

<details><summary>70 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/021998c5ac03bf4972c3` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.0011590630747377872} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/076d964aa67b3b1c2620` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.015770966187119484} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/091eaddc03e41900bb9d` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.24807281233370304} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/0c783782e7ace61c3b81` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 2.739553981926292} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/13d4f84278f844edd660` | setup_prefix_8.lower_s | {"setup_prefix_8.lower_s": 0.16138814808800817} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_8/lower_s` |
| `measurement/17046b9b266ba6b1a71a` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.1394699988886714} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/17c21c9cd959bd97d14c` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.02635717298835516} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/1820aecb24d5d15029f6` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 11.496863184031099} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/1986f26db8d257400445` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.4310237020254135e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/1ad296fc0aeceea2ba9f` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.02821636199951172} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/1cfcfed91e6f08823e30` | overlay_grid_jit.first_call_s | {"overlay_grid_jit.first_call_s": 0.0010207542218267918} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/overlay_grid_jit/first_call_s` |
| `measurement/27bcc302712c6ee8db80` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 2.304461043328047} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_7/compile_s` |
| `measurement/299159a5b35a1112e7f0` | ray_trace_jit.lower_s | {"ray_trace_jit.lower_s": 0.12762511987239122} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/ray_trace_jit/lower_s` |
| `measurement/2ba3c29d7e3490adae23` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.08249359903857112} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/2e5dfb343f1748668b19` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.35345017723739147} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/33943d024291c0f4358e` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.012215947732329369} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/34652239a176522d3144` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.16510809678584337} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/36460707cd7cb30787fe` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.007434736005961895} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/3dd310db399cd2e7481b` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006291461177170277} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/4226df471b991a54188d` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.011490650940686464} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/4243c6cea13d89a2d011` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.011486231815069914} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/430398317d5fdffab809` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.4986134120263159} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/51678c0063625bdc5f82` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.00035300804302096367} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/5406bf0b0bbc41f5de87` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.005632635671645403} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/5d4c43acc47c8de59cf6` | setup_prefix_8.first_call_s | {"setup_prefix_8.first_call_s": 0.034159007016569376} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_8/first_call_s` |
| `measurement/5dd2a65901efebeadcdd` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.09504157397896051} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/6e9046dffe92be578340` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 0.14900015201419592} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/6f62ebc90310ad5cd970` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.0060059139505028725} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/75281166538185b266a2` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.007290756795555353} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/76905181fe36a47c669b` | ray_trace_jit.first_call_s | {"ray_trace_jit.first_call_s": 0.0016224500723183155} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/ray_trace_jit/first_call_s` |
| `measurement/787cb92da91c76e0b4e5` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.006096343044191599} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/7e1c0086646e0f5d2130` | regularization_matrix_assembly_jit.first_call_s | {"regularization_matrix_assembly_jit.first_call_s": 0.0009782039560377598} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_assembly_jit/first_call_s` |
| `measurement/8051dd166e4247369018` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.004573123063892126} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/81a2c278c32af44b9134` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.10063797095790505} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/82f894825023f3aa34ab` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.04415575694292784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/833afdd9fca62bb712e4` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.013251732103526592} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/85956a1b12720ebb24f7` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.054783354979008436} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/85f4253803311bc75d1a` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.09496078407391906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/866c030ef94acdec2945` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 2.0018573231063783} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/8a0088fbec1a1e86d979` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.06408272823318839} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/8ab913caa73b1fe25067` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0036200983449816704} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/8cf008017f818c0cf089` | overlay_grid_jit.lower_s | {"overlay_grid_jit.lower_s": 0.014414223842322826} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/overlay_grid_jit/lower_s` |
| `measurement/8f75bf49504ef29475ee` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 0.9099592640995979} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/8fa99ead9a7032af7db9` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 2.5441351272165775} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/953493ea60be1be2420b` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.023757438641041517} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/953a886a63b881124317` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.1494709807448089} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/9a3a180dae837ce08478` | setup_prefix_8.compile_s | {"setup_prefix_8.compile_s": 2.6996436011977494} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_8/compile_s` |
| `measurement/9c80d0420f58f7a63058` | nnls_pdip_vmap16.first_call_s | {"nnls_pdip_vmap16.first_call_s": 4.754805225413293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_vmap16/first_call_s` |
| `measurement/9ee8e489f543a01e4c0c` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.007342965807765722} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/a9951c7099d611053515` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.011688858736306429} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/aabbd951140bf15f9f52` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0006691357120871544} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/af2834a4af3096251d72` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 0.7122920099645853} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/b1eaed01dc35ed61393f` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 4.0748659539967775} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/b36c57f86129ed4f18fc` | regularization_matrix_assembly_jit.compile_s | {"regularization_matrix_assembly_jit.compile_s": 0.05820782296359539} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_assembly_jit/compile_s` |
| `measurement/bab6675cb36b6ea2b885` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.20558250602334738} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/baf3b13e4ebe875ca0d4` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.04007800109684467} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/bf6479d76c00d1943868` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.021600051317363977} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/c35bf8e44e5d8f226bf5` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.007915982976555824} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/ca052dd035211c9d1931` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.029016387183219194} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/cabe8a27caa4549d0fec` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.0064087617211043835} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/cd63d3ecdd6881889f5b` | ray_trace_jit.compile_s | {"ray_trace_jit.compile_s": 0.4224251350387931} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/ray_trace_jit/compile_s` |
| `measurement/db815bb21c4bd251f7e1` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 2.4009010791778564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/dba7fe841d4b1cd205f7` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.164378282148391} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/df8905e2d8c477d93d2a` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.09108768729493022} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/ef18cecf236db278bf87` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.06427383702248335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/f04a36b846ee05260fa2` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.4604464890435338} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/f4e5e0e8b59392201f23` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.4802960613742471} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/f80f916ad8b2c470f6d9` | overlay_grid_jit.compile_s | {"overlay_grid_jit.compile_s": 0.1592815830372274} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/overlay_grid_jit/compile_s` |
| `measurement/f8a110f4fe60d450cd2d` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.004620322957634926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/f9b7c656c153ce64098d` | regularization_matrix_assembly_jit.lower_s | {"regularization_matrix_assembly_jit.lower_s": 0.016981728840619326} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_n5000.json) `/jit_phases/regularization_matrix_assembly_jit/lower_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 33289 MiB, 81920 MiB",
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
