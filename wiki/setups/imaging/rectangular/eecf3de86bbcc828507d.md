<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/6d6bc226df0f14768a8e`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
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
| source_pixels | 1521 | count |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 61444 | recorded |  |
| sparse_operator_build_s | 0.6299648049753159 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1581 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>43 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0115cd5d2ff3794b6cd4` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.00014464749983744696} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/0420141d84ecc625c154` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.001230138004757464} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/05973eb741e959c70c2a` | steps.Inversion setup (sparse, steps 4-8 combined) | {"steps.Inversion setup (sparse, steps 4-8 combined)": 0.009487819392234087} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Inversion setup (sparse, steps 4-8 combined)` |
| `measurement/05a7b6958ff5cd63966b` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.020436532609164716} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/06a784c45741a36af68a` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.00022065378725528718} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/13e333748bf5dc6fa29a` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.00015576321166008711} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/1dd39ca8ef4f9f0c1fdc` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.00016952301375567914} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/26cd0362bf0a687bf734` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.005171496793627739} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/2f3958f12b15e5be4267` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.03665673309005797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/388b408abbe8c6510d93` | steps.Overlay grid (source pixel centres) | {"steps.Overlay grid (source pixel centres)": 0.00016412308905273677} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Overlay grid (source pixel centres)` |
| `measurement/3cbc78a2154f312cdfa0` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.001021576183848083} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/4801a38b9a4043771b09` | regularization_matrix_assembly_jit.steady_per_call_s | {"regularization_matrix_assembly_jit.steady_per_call_s": 0.00013231919147074222} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_assembly_jit/steady_per_call_s` |
| `measurement/49c57460f0a23dffaff4` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.001564703113399446} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/51c0bdf4dc2354b14539` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.0015396333998069166} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/regularization_matrix_prefix_s` |
| `measurement/53cac9a6d5e87cfc9222` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00014275519642978907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/5a46b731ed55df19f050` | ray_trace_jit.steady_per_call_s | {"ray_trace_jit.steady_per_call_s": 0.0001763120060786605} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_jit/steady_per_call_s` |
| `measurement/60cf1aa6c4d4e812c7f8` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.0022801331942901015} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/659d137b82d59cb65627` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.0001737358936225064} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/6984d8b15d6cd3e3ff9f` | overlay_grid_jit.steady_per_call_s | {"overlay_grid_jit.steady_per_call_s": 0.00016412308905273677} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/overlay_grid_jit/steady_per_call_s` |
| `measurement/699663660bb48480ad1c` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.0001763120060786605} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Ray-trace grids` |
| `measurement/702214f6cec4b8d92bd6` | inversion_setup_vmap16.steady_per_call_s | {"inversion_setup_vmap16.steady_per_call_s": 0.0007513893870054744} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_vmap16/steady_per_call_s` |
| `measurement/820d2aa793d87a31d501` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.0015396333998069166} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/84f62188e4f60df76937` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00014275519642978907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/8a9bad5c5c641a0fd8b7` | component_total | {"component_total": 0.07080817648675293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/total_step_by_step` |
| `measurement/99f28b6ca314c4b10ac3` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.001230138004757464} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/aa2ac1cc98f1a3435843` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.020436532609164716} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/abd78a1b12f3fd0f0945` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.00014282449992606417} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/acd155fba7a880a2ddd2` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.00022065378725528718} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/aea8bf5f47029ac5802c` | interpolator_prefix | {"interpolator_prefix": 0.001662615593522787} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/interpolator_prefix_s` |
| `measurement/b92193ce95e868567542` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00013595831114798784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/c566fa2a12169b401610` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.03665673309005797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Regularized reconstruction` |
| `measurement/c6c6bf700ec34b0caf6f` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.020826103491708638} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/c8f67a27a47a418934b7` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.00014282449992606417} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/ca06efa0afa10d27733d` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.001662615593522787} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/d825f437c191f74b8a7f` | setup_prefix_11_vmap16.steady_per_call_s | {"setup_prefix_11_vmap16.steady_per_call_s": 0.0001737358936225064} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_11_vmap16/steady_per_call_s` |
| `measurement/dcbeb6f1f12f21f4a4d9` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.457688723457978e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/e1b2646d0d7844585eae` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 3.091139369644223e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps_vmap_per_call/Regularization matrix (H)` |
| `measurement/eb6dff95d6b95297ad70` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.0010155112948268652} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/ebe26958af0063dfab28` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.0004434725036844611} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/f01c2b0066fd0d0bdbf8` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.009487819392234087} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/f264f6b805dd9e35ba53` | steps_vmap_per_call.Inversion setup (sparse, steps 4-8 combined) | {"steps_vmap_per_call.Inversion setup (sparse, steps 4-8 combined)": 0.0007513893870054744} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps_vmap_per_call/Inversion setup (sparse, steps 4-8 combined)` |
| `measurement/f7d1a83b4e03b7f83c45` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.0022801331942901015} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/fe836f02140447dcf663` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00013595831114798784} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/steps/Profile-subtracted image` |

</details>

### compile

<details><summary>68 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/030fa6c2321a57cbd9b5` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.0009203751105815172} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/046c8e4d9adc78258c5d` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.14068424585275352} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/08e68701002e107cf068` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.15827248711138964} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/0b2c2976d2f7d6e685e4` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 2.5341929490678012} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/0b76beec67e2f8cb1492` | regularization_matrix_assembly_jit.lower_s | {"regularization_matrix_assembly_jit.lower_s": 0.017094683134928346} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_assembly_jit/lower_s` |
| `measurement/0f1813e479246c1e9ac3` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.23093427694402635} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/10d84e2aed748dfab6d9` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.031152545008808374} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/1556a6c431e7d315ec5e` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.011507655959576368} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/160af6fb3bfa40f627fc` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.15296179708093405} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/1d84a0662e25240e9940` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.15394613100215793} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/20a4683feeb932674c93` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.029998810030519962} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/20e747acec581f2ba9cd` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.005227751098573208} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/2b2d7f2af19d07deb81b` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0007740359287708998} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/2f0c1de7f7686b04a44c` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.046182010089978576} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/3239ffaaa2f2c71c3500` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0007215070072561502} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/336b7605c090cdf241db` | ray_trace_jit.lower_s | {"ray_trace_jit.lower_s": 0.0869340191129595} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_jit/lower_s` |
| `measurement/34575d1710176b67e486` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 4.2527191520202905} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/3499f19d30d00151014d` | overlay_grid_jit.first_call_s | {"overlay_grid_jit.first_call_s": 0.0008375039324164391} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/overlay_grid_jit/first_call_s` |
| `measurement/3fdce9f5599e9f8a3d09` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.10935756284743547} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/46f8c3094be48338e2e1` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.029658632818609476} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/473523af8a461b9a1579` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.09186854097060859} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/4b75261e74192c9540e3` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.026033952832221985} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/520d5a07f10910fe090c` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.049173633102327585} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/52bd239c74dfaca7a85c` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0008838751818984747} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/5350bb40e1e648d8d0f7` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.012713928008452058} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/55230b7880a7e8e54398` | regularization_matrix_assembly_jit.first_call_s | {"regularization_matrix_assembly_jit.first_call_s": 0.00088261510245502} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_assembly_jit/first_call_s` |
| `measurement/5af3f3834f35114b79f2` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 0.7060533550102264} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/5b4a543b655918549e45` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 10.950751195894554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/5c3b65e45187ab1fe0d3` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.42646747292019427} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/62d604a943500d34504b` | setup_prefix_11_vmap16.first_call_s | {"setup_prefix_11_vmap16.first_call_s": 2.6815355489961803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_11_vmap16/first_call_s` |
| `measurement/64726b6a7d69d601b8d2` | overlay_grid_jit.lower_s | {"overlay_grid_jit.lower_s": 0.014418208971619606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/overlay_grid_jit/lower_s` |
| `measurement/6b5a6657934b8baefcc0` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.0008394650649279356} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/6cc49662df3150b66dca` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 2.477102512959391} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/compile_s` |
| `measurement/6fc090d04e18e77c0783` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.029924831120297313} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/744f12ed5e74029d1fc4` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.045298182871192694} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/86a76b0465b795b31f9a` | regularization_matrix_assembly_jit.compile_s | {"regularization_matrix_assembly_jit.compile_s": 0.05636985204182565} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_assembly_jit/compile_s` |
| `measurement/86c0d048c9b088eb30ad` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.006264834897592664} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/873658413b20db22cfa3` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.007236930076032877} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/881a98d0b5146bdd027e` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.05150728905573487} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/888a26df035217391744` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 2.899748417083174} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/89e4f18263dbaf5d364c` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.004384146071970463} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/9045dfe9201bb1e117f7` | overlay_grid_jit.compile_s | {"overlay_grid_jit.compile_s": 0.12817483814433217} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/overlay_grid_jit/compile_s` |
| `measurement/9441847e3fbe68bd5c96` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.5202393739018589} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/9c8f62a5607a8eb2f7f4` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.007719315821304917} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/9e883e37dbad23430a4b` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.008136644028127193} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/9e93632f3ab0ee638f1f` | ray_trace_jit.first_call_s | {"ray_trace_jit.first_call_s": 0.0017490200698375702} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_jit/first_call_s` |
| `measurement/a14dedfb4af0eaa3f1fa` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.03851717198267579} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/a1677f87938e5cedf9d3` | ray_trace_jit.compile_s | {"ray_trace_jit.compile_s": 0.3656455168966204} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_jit/compile_s` |
| `measurement/a7204689bbbfa81ba1b2` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.0045403840485960245} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/abb4977ccc1bee63ddc9` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 1.648633886827156} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/b09b33549db54201d7a3` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.1550134951248765} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/b1c28f86f010dc24b5d7` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.04469084716401994} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/b68bef6234b311703127` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.21967181004583836} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/b72c1e3487d9af9e9ff6` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.004368614871054888} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/c153a104e4fa1ab8f8d8` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.09608657704666257} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/c34580746664097cb949` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 2.6542884919326752} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/c8668c93f2ac3adb96b8` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.001146744005382061} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/cc13fc50e950a8037061` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.017304342007264495} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/d0a4770a872a50885cd0` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 2.266913478029892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/d2d5d81e461e23ef82a9` | inversion_setup_vmap16.first_call_s | {"inversion_setup_vmap16.first_call_s": 16.89872416993603} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_vmap16/first_call_s` |
| `measurement/d3f3ea6cae4be7f62d7d` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.15441880794242024} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/d4816593ce72bc516926` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 5.870532633038238} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/d67248756e514f02fa1d` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.3591663739643991} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/de44f9d01c2e4c773f5e` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 0.9063885658979416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/e7209627ef5082f4f6fe` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.006471122847869992} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/e754943bb495069da901` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.014334569917991757} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/f23aa741fe3e7d2b20a4` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.08891635807231069} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/fd0a22b5d7a9b8006066` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 11.74013446108438} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 1547 MiB, 81920 MiB",
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
