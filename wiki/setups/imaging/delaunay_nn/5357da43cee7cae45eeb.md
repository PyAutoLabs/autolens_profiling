<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/c0645d7650996d082d77`.

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
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 491552 | recorded |  |
| sparse_operator_build_s | 0.7330818329937756 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1560 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>45 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/035a440a50456c220df6` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 0.02016033630352467} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/0a766308b9b9d8f1b348` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00013165317941457033} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/0f4e5f8ca16f14030d49` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.00014862811658531428} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/1dfe3b108394e94eface` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.01599315369967371} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/2cca1769bea5765415aa` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.000127425417304039} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Profile-subtracted image` |
| `measurement/2e7ed6877cb5e3b3f5f6` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 0.01980142530519515} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/31f5df8900cd52fee9fc` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.037924851989373565} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Regularized reconstruction` |
| `measurement/33d826645635d6b4ac70` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00020275989081710576} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/352221f5a94e016dc915` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.014470315305516124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/393bda3d2daaa89e6099` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 0.01980142530519515} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/3fe306e9f4113425d433` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.002226222399622202} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Mapped recon + log evidence (sparse)` |
| `measurement/486eeca027441213ffa1` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.0004149175947532058} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/486f8d3878b09732c7e9` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.002226222399622202} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/4fc5793c79049bff5dea` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.037924851989373565} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/5ba7075c3263f7fcb45d` | setup_prefix_11_vmap16.steady_per_call_s | {"setup_prefix_11_vmap16.steady_per_call_s": 0.007375862319895532} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_11_vmap16/steady_per_call_s` |
| `measurement/65330b1bebb4bffe9f6f` | steps_vmap_per_call.Inversion setup (sparse, steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (sparse, steps 5-8 combined)": 0.007154459318553563} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps_vmap_per_call/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/66ff5918071183a6a6e1` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.017442190484143794} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/regularization_matrix_prefix_s` |
| `measurement/68697eee3f02fa8e07dd` | setup_prefix_6s.steady_per_call_s | {"setup_prefix_6s.steady_per_call_s": 0.014810967398807407} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6s/steady_per_call_s` |
| `measurement/7d9bc659eccb7ab9c8c0` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00013165317941457033} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/7df45817dbd6a6f4b827` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.60313802654855e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/86c6e464419b994666d6` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00019102590158581734} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/8e37c7e8f2083744b52f` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00019102590158581734} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Ray-trace mesh grid` |
| `measurement/94ecaf225dfb8af7c0a2` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.00013584729749709367} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/a61ed17e09be743bea50` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.000846414198167622} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/aeb9d7d9fcdb2c1789dc` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 0.019908329588361084} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/af6e7ef01a0cbc3babfd` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.000127425417304039} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/affe1eff64acd3b92e6f` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.006541533274867106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/b8107827a5d0cb2abb05` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.017442190484143794} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/b9843a73200bc6ee5eb1` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.007375862319895532} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/c609d84ce3d1519fe3e7` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 0.0007281798869371414} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/c72229a88749d8880b4f` | inversion_setup_vmap16.steady_per_call_s | {"inversion_setup_vmap16.steady_per_call_s": 0.007154459318553563} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_vmap16/steady_per_call_s` |
| `measurement/d70358f71925e51bcdc6` | setup_prefix_6s_vmap16.steady_per_call_s | {"setup_prefix_6s_vmap16.steady_per_call_s": 0.006568338874785695} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6s_vmap16/steady_per_call_s` |
| `measurement/e2960a26153ea468b756` | interpolator_prefix | {"interpolator_prefix": 0.014470315305516124} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/interpolator_prefix_s` |
| `measurement/e35001f84fad3ba64f48` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0029718751786276705} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Regularization matrix (H)` |
| `measurement/e3f5e7b7cb73eb7affdc` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00020275989081710576} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Ray-trace data grid` |
| `measurement/e681265e5685ac7740a1` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.00014862811658531428} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/eaa3ec80e8b3edbfe8be` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.000846414198167622} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/ecba28c96f2da95fd674` | component_total | {"component_total": 0.08473261788021773} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/total_step_by_step` |
| `measurement/ed00d1973c89a7a1bf86` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.006523548130644485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/f1a8e5a250835ad1d73e` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.006523548130644485} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/f8368c8bb201e72c42ba` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.0049959308002144095} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/f8794b7d45340408fe2f` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.0010690759168937801} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/f94150b4f6c89aae7d2c` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 0.02016033630352467} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/fa276300c4d5a04e0f04` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.00018017191905528307} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/faefde7b274987d7b428` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.000852314189251047} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/steps_vmap_per_call/Regularization matrix (H)` |

</details>

### compile

<details><summary>69 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/05bd8740c0d9a3403952` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 1.9235457049217075} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/18577695f3159a4a9ef2` | setup_prefix_6s.compile_s | {"setup_prefix_6s.compile_s": 3.312490755924955} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6s/compile_s` |
| `measurement/1bd1a18942b690730adc` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.37856477382592857} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/1ce9e4ef6f19fb50e9f4` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.12443648697808385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/215247d4af61167a8db4` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.3803017439786345} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/2234816953fbe102a779` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.020946242148056626} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/2671b403978e40ac4bff` | inversion_setup_vmap16.first_call_s | {"inversion_setup_vmap16.first_call_s": 18.444173629861325} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_vmap16/first_call_s` |
| `measurement/27133819d37e92cb58f2` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 1.5217755218036473} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/307a25b46e80c202e227` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.046454118099063635} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/3118e319e537178c717d` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.04326199600473046} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/339e550ef82258ee6c71` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.21776813105680048} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/3a0c41f2feb7b9dbf4d3` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 3.4683157480321825} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/compile_s` |
| `measurement/43dfa8c59fc9f2a3f14f` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0011953541543334723} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/458672165f97bddad088` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.09537963196635246} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/49c73d0452760a802d1b` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0008082061540335417} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/5072f67a2736bcab6ba6` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 0.02682447899132967} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/574a20bf3ed19294d4a0` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.06020940002053976} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/5b4b96d3d0d6d416147c` | setup_prefix_6s_vmap16.first_call_s | {"setup_prefix_6s_vmap16.first_call_s": 4.454989521065727} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6s_vmap16/first_call_s` |
| `measurement/5bc9dd68e96c6238b17d` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 8.666831583017483} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/5c5a5bc3767d64a5a909` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.0009103049524128437} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/5cd9adb921bba0b8e279` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.05258748307824135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/6210d19ae84ed952a797` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0008881660178303719} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/6659c7a8d9ab6ea34738` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0013652518391609192} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/66b4f10f6556d69ec9fd` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0010666740126907825} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/6d7b432a74365e25b958` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 0.0040575070306658745} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/6fb8841d92c3bda6d6dd` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.03969390597194433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/711357bc427f5f2d047b` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.006318933796137571} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/728330d40ed1df34d43e` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.030749825993552804} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/76e070606e156322c235` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.023449808126315475} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/7b86ecd7b5cf2d5904d4` | setup_prefix_6s.lower_s | {"setup_prefix_6s.lower_s": 0.41362183494493365} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6s/lower_s` |
| `measurement/7eb2949853069d85a81b` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 12.136092556873336} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/8195104b8850adb9283c` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.4046319159679115} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/860fdb13da38236e72f7` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.11405576602555811} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/8b25cfd52d9e81011926` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.04700875515118241} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/8b4008d2fed8b266322d` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0036357599310576916} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/91e6315ac4ade7705cd9` | setup_prefix_6s.first_call_s | {"setup_prefix_6s.first_call_s": 0.1808408098295331} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6s/first_call_s` |
| `measurement/9660f7f129ba142ee4d9` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.0008290649857372046} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/97d9f0f31dd83d344f96` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.7098673440050334} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/9c5ba75e979b0f6dbd61` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.13627814105711877} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/9e435b4e306413a36f65` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.25550779700279236} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/a297a0846d981ffb84bb` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.4837399909738451} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/a354ca8f9f804c5a362e` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.681222055805847} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/a59b67be484944cb5e1e` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.052322885021567345} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/a7b8398e194c3b37a3c5` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.018214827170595527} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/aaa1df01ab59933fc973` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 3.7843590939883143} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/ad3d4579cf39e341dd38` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 0.02492012013681233} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/b3d18b766d6bb9689ca8` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006203369703143835} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/b6f68b76b2d9fbda6a83` | setup_prefix_11_vmap16.first_call_s | {"setup_prefix_11_vmap16.first_call_s": 5.538552796002477} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_11_vmap16/first_call_s` |
| `measurement/bdc56fab2abbece16ae9` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 7.727826402056962} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/be793af88e13760923bc` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.4787751780822873} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/c1d4152b7a718bb4e31b` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 4.50749976397492} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/c8c4ff3c97edb415648f` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 2.8660925361327827} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/d434dd9ca88f0752ec56` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.2746635309886187} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/d5a4ef551992100d4afe` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.01179394288919866} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/d6c7688beb636288def4` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.004713153000921011} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/d9f17aa94c448311100c` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.05824201088398695} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/e2e33bf52f6408fb7211` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 0.9269043300300837} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/e484287393e34cb858a3` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.004670304944738746} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/e64996e260aad0f201d4` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.170081980060786} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/e6db460368c5c899148c` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.030633497051894665} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/ec907d1b9c223aeca82a` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.014367758994922042} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/f2dc59dbb72dd486a1e3` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 4.602100969990715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/f3dc78dace4d9c979a58` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 0.13943423284217715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/f4ce8380abfd0db4f77a` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.19516189908608794} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/f50eaa2bd7e99f12715a` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.027109545888379216} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/f563974cbf7c0b056f2e` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.12412822898477316} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/f68d9fcf3f670eba7c12` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.004657974932342768} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/f7749b7eeb86470617db` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.08692091098055243} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/fab3023742b898403650` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.011127387871965766} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_sparse.json) `/jit_phases/data_vector_sparse_jit/lower_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 2563 MiB, 81920 MiB",
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
