<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · alma_high

[Model index](index.md)

Exact setup ID: `interferometer/delaunay/alma_high/6be55b359af93be1ec5b`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| curvature_blocks | 10 | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| host_load_avg_end | [61.36, 57.21, 52.07] | recorded |  |
| host_load_avg_start | [55.5, 51.7, 49.51] | recorded |  |
| image_pixels_masked | 61572 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_note | "simulators/interferometer.py puts no lens emission in the visibilities; mass fixed near truth (tight Gaussian priors), no lens light." | recorded |  |
| mapper_class | "Mapper" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mixed_precision | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_repeats | 3 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| nnls_preconditioning | "jacobi" | recorded |  |
| nufftax_version | "0.6.1" | recorded |  |
| operator_M | 78400 | recorded |  |
| operator_extent_shape | [280, 280] | recorded |  |
| operator_fft_shape | [560, 560] | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.025 | recorded |  |
| pixels_solved | 1250 | recorded |  |
| positive_only_solver_requested | "pdip" | recorded |  |
| positive_only_solver_used | "pdip" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_preset | {"adapt_image": "truth lensed source (lensed_source.fits), no S/N cap", "adapt_snr_cap": null, "dataset_type": "interferometer", "instance": "iid draw 0 of the preset sequence (seeded)", "instance_unit_vector": [0.499784489007424, 0.41402443684610074, 0.41327450881714656, 0.4200167345680767, 0.5789514047288555, 0.4702516857248635, 0.567334291975647, 0.47173759315010344, 0.5239626288064884, 0.5543072472409505], "instrument": "alma_high", "lens_light": null, "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 0, "over_sample_size_lp_rule": null, "over_sample_size_pixelization_rule": null, "positions_penalty": null, "preset": "interferometer_hst_source_pix_2_standard", "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| solve_subset_edge_zeroed | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 1000000 | recorded |  |
| transformer_chunk_size_preset | 1000000 | recorded |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 5000000 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>28 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1c81ed8683e06c95b28a` | setup_prefix_1.steady_per_call_s | {"setup_prefix_1.steady_per_call_s": 0.002764217342094829} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/steady_per_call_s` |
| `measurement/22768b99c73c586d5aa8` | solver_ab_certified.steady_per_call_s | {"solver_ab_certified.steady_per_call_s": 0.15051972466365746} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/steady_per_call_s` |
| `measurement/25d04c29b87d002a74b9` | reconstruction_pdip.steady_per_call_s | {"reconstruction_pdip.steady_per_call_s": 0.3858543116754542} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/steady_per_call_s` |
| `measurement/37e3e5d683bd34999262` | kernel_fft_apply.steady_per_call_s | {"kernel_fft_apply.steady_per_call_s": 0.4875282206727813} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/steady_per_call_s` |
| `measurement/3fbe299b1354f9463cf6` | steps.Log-det terms (2 Choleskys) | {"steps.Log-det terms (2 Choleskys)": 0.031148510985076427} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Log-det terms (2 Choleskys)` |
| `measurement/4f2a6388b68a79364d15` | kernel_gather_segment_sum.steady_per_call_s | {"kernel_gather_segment_sum.steady_per_call_s": 0.02176860033068806} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/steady_per_call_s` |
| `measurement/660a02a8ed66336bf7c0` | sparse_triplets.steady_per_call_s | {"sparse_triplets.steady_per_call_s": 0.0005742273254630467} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/steady_per_call_s` |
| `measurement/6d87444f7d4645ab8ce2` | kernel_scatter.steady_per_call_s | {"kernel_scatter.steady_per_call_s": 0.006426528988716503} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/steady_per_call_s` |
| `measurement/6ebc0a3a03af0c5fa2df` | steps.Sparse triplets (extent grid) | {"steps.Sparse triplets (extent grid)": 0.0005742273254630467} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Sparse triplets (extent grid)` |
| `measurement/7c5a100c427cecfba049` | data_vector_sparse.steady_per_call_s | {"data_vector_sparse.steady_per_call_s": 0.28455442333749187} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/steady_per_call_s` |
| `measurement/7c849ccf16aa8ada8a26` | steps.Border relocation + mesh interpolation (mapper) | {"steps.Border relocation + mesh interpolation (mapper)": 0.25721630033028003} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Border relocation + mesh interpolation (mapper)` |
| `measurement/7ff77ef589f8d1f35524` | steps.Data vector D = Lᵀ d~ | {"steps.Data vector D = L\u1d40 d~": 0.28455442333749187} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Data vector D = Lᵀ d~0` |
| `measurement/962dbb5a94c344510587` | full_pipeline.steady_per_call_s | {"full_pipeline.steady_per_call_s": 5.859374109325775} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/steady_per_call_s` |
| `measurement/9a15d95ebd8dfffc354e` | steps.Reconstruction (pdip) | {"steps.Reconstruction (pdip)": 0.3858543116754542} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Reconstruction (pdip)` |
| `measurement/9c707204c78fefd26223` | solver_ab_pdip.steady_per_call_s | {"solver_ab_pdip.steady_per_call_s": 0.38383563433308154} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/steady_per_call_s` |
| `measurement/a02f7fb0ad0828855b15` | curvature_sparse.steady_per_call_s | {"curvature_sparse.steady_per_call_s": 5.241362416010816} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/steady_per_call_s` |
| `measurement/a0bbb445617bc1e172cd` | figure_of_merit.steady_per_call_s | {"figure_of_merit.steady_per_call_s": 0.03032841900130734} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/steady_per_call_s` |
| `measurement/a8dfe30d9b1a00606104` | steps.Mapping matrix L | {"steps.Mapping matrix L": 0.0889887569937855} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Mapping matrix L` |
| `measurement/ac1b45593fe086a8758d` | setup_prefix_3.steady_per_call_s | {"setup_prefix_3.steady_per_call_s": 0.34896927466616035} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/steady_per_call_s` |
| `measurement/af6a50567fb939cac464` | library_fit_figure_of_merit.steady_per_call_s | {"library_fit_figure_of_merit.steady_per_call_s": 5.710518055343225} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/steady_per_call_s` |
| `measurement/b38441c06dde7bdd5246` | setup_prefix_2.steady_per_call_s | {"setup_prefix_2.steady_per_call_s": 0.25998051767237484} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/steady_per_call_s` |
| `measurement/b924c53662ea9338d3ab` | steps.Fast chi-squared + figure of merit | {"steps.Fast chi-squared + figure of merit": 0.03032841900130734} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Fast chi-squared + figure of merit` |
| `measurement/c081181a816c402dc01c` | steps.Curvature matrix F = Aᵀ W~ A (10 blocks of 128) | {"steps.Curvature matrix F = A\u1d40 W~ A (10 blocks of 128)": 5.241362416010816} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Curvature matrix F = Aᵀ W~0 A (10 blocks of 128)` |
| `measurement/ca84a718ba2cec7ba418` | steps.Ray-trace grids (data + mesh vertices) | {"steps.Ray-trace grids (data + mesh vertices)": 0.002764217342094829} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Ray-trace grids (data + mesh vertices)` |
| `measurement/da3724ec24e3ad576ea6` | component_total | {"component_total": 6.327373500680551} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/total_step_by_step` |
| `measurement/dcce35b64cee62534fc3` | setup_prefix_4.steady_per_call_s | {"setup_prefix_4.steady_per_call_s": 0.3535511923449424} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/steady_per_call_s` |
| `measurement/e28dcd17b1ec2fc3c0da` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.004581917678782077} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/fb0077b32ce63b3fa5cc` | log_dets.steady_per_call_s | {"log_dets.steady_per_call_s": 0.031148510985076427} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/steady_per_call_s` |

</details>

### compile

<details><summary>52 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/06a11806f823cf1a49ac` | setup_prefix_2.first_call_s | {"setup_prefix_2.first_call_s": 0.25997155997902155} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/first_call_s` |
| `measurement/0afee6a9fdea89615e5d` | kernel_fft_apply.lower_s | {"kernel_fft_apply.lower_s": 0.018513392948079854} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/lower_s` |
| `measurement/0f0f3311c8749a150d76` | operator_setup | {"operator_setup": 13.086512398032937} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/operator_build_s` |
| `measurement/0ff9722695a612ef9ce7` | kernel_scatter.lower_s | {"kernel_scatter.lower_s": 0.018625344964675605} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/lower_s` |
| `measurement/121ff8f4cf8f0cab72ad` | solver_ab_pdip.first_call_s | {"solver_ab_pdip.first_call_s": 0.3794066560221836} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/first_call_s` |
| `measurement/1492f5329150dd964805` | curvature_sparse.compile_s | {"curvature_sparse.compile_s": 0.19114767597056925} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/compile_s` |
| `measurement/15a0ecebbbfe390af7c3` | setup_prefix_3.first_call_s | {"setup_prefix_3.first_call_s": 0.35398507403442636} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/first_call_s` |
| `measurement/1848f8d68ff93928ad25` | solver_ab_certified.lower_s | {"solver_ab_certified.lower_s": 0.11378556996351108} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/lower_s` |
| `measurement/1a8a337f98881589b424` | figure_of_merit.first_call_s | {"figure_of_merit.first_call_s": 0.03000039601465687} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/first_call_s` |
| `measurement/26f7c9204dd9abb040af` | curvature_sparse.lower_s | {"curvature_sparse.lower_s": 0.06238928699167445} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/lower_s` |
| `measurement/2ab179011783c5494b64` | kernel_gather_segment_sum.first_call_s | {"kernel_gather_segment_sum.first_call_s": 0.020221905026119202} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/first_call_s` |
| `measurement/2dd5b3e4881acd3552b1` | solver_ab_certified.compile_s | {"solver_ab_certified.compile_s": 0.5753361580427736} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/compile_s` |
| `measurement/2de9734e8f327269b994` | solver_ab_pdip.compile_s | {"solver_ab_pdip.compile_s": 0.34820894803851843} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/compile_s` |
| `measurement/3232335d0a5e20f124b7` | setup_prefix_3.lower_s | {"setup_prefix_3.lower_s": 0.9087485199561343} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/lower_s` |
| `measurement/3341d2b49df189af686f` | full_pipeline.compile_s | {"full_pipeline.compile_s": 1.695304396969732} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/compile_s` |
| `measurement/368aa93dee2381102acc` | sparse_triplets.compile_s | {"sparse_triplets.compile_s": 0.06837355700554326} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/compile_s` |
| `measurement/3fbbd49df29a629ada14` | full_pipeline.first_call_s | {"full_pipeline.first_call_s": 6.141623227042146} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/first_call_s` |
| `measurement/433f31b7dc3b9b962a78` | log_dets.first_call_s | {"log_dets.first_call_s": 0.03196486399974674} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/first_call_s` |
| `measurement/43c282acebcb593a93d9` | kernel_fft_apply.first_call_s | {"kernel_fft_apply.first_call_s": 0.47212417400442064} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/first_call_s` |
| `measurement/463d4a8996d517385453` | data_vector_sparse.compile_s | {"data_vector_sparse.compile_s": 0.07026002201018855} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/compile_s` |
| `measurement/53df37d75494d6ea8db2` | log_dets.lower_s | {"log_dets.lower_s": 0.032541453023441136} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/lower_s` |
| `measurement/57f7f00f41f186fae9b2` | kernel_fft_apply.compile_s | {"kernel_fft_apply.compile_s": 0.1612029999960214} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_fft_apply/compile_s` |
| `measurement/597be8c332ac53fc3bee` | solver_ab_certified.first_call_s | {"solver_ab_certified.first_call_s": 0.1496154849883169} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_certified/first_call_s` |
| `measurement/5ed1388dd42b84673d74` | setup_prefix_1.lower_s | {"setup_prefix_1.lower_s": 0.23746904399013147} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/lower_s` |
| `measurement/67e0546df7ae9557f851` | figure_of_merit.compile_s | {"figure_of_merit.compile_s": 0.10319162398809567} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/compile_s` |
| `measurement/7aafaa7c8c946b56dd64` | reconstruction_pdip.compile_s | {"reconstruction_pdip.compile_s": 0.3878326310077682} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/compile_s` |
| `measurement/814f7df7aa774892c125` | reconstruction_pdip.first_call_s | {"reconstruction_pdip.first_call_s": 0.40474100603023544} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/first_call_s` |
| `measurement/8150e1a430493a36ca28` | library_fit_figure_of_merit.compile_s | {"library_fit_figure_of_merit.compile_s": 1.7811504889978096} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/compile_s` |
| `measurement/85f48124d7e97c4c0210` | reconstruction_pdip.lower_s | {"reconstruction_pdip.lower_s": 0.08236117695923895} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/reconstruction_pdip/lower_s` |
| `measurement/8c93edaf0f3d1f6e4d17` | setup_prefix_4.lower_s | {"setup_prefix_4.lower_s": 1.0206877870368771} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/lower_s` |
| `measurement/94b1b10091480476e2a9` | solver_ab_pdip.lower_s | {"solver_ab_pdip.lower_s": 0.08207511203363538} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solver_ab_pdip/lower_s` |
| `measurement/979662eecddfef2776f2` | kernel_scatter.first_call_s | {"kernel_scatter.first_call_s": 0.007334815978538245} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/first_call_s` |
| `measurement/97b100f1ec2bac0c49f5` | setup_prefix_1.first_call_s | {"setup_prefix_1.first_call_s": 0.0037968489923514426} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/first_call_s` |
| `measurement/9c01bf313aaa564d881d` | setup_prefix_3.compile_s | {"setup_prefix_3.compile_s": 0.9101195159601048} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_3/compile_s` |
| `measurement/9c977c58e62d46b42589` | setup_prefix_1.compile_s | {"setup_prefix_1.compile_s": 0.4138212140533142} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_1/compile_s` |
| `measurement/9db7070af8995b8700ba` | full_pipeline.lower_s | {"full_pipeline.lower_s": 1.304043640033342} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_pipeline/lower_s` |
| `measurement/a6dc5acae2343f5c422f` | setup_prefix_4.compile_s | {"setup_prefix_4.compile_s": 1.285765431006439} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/compile_s` |
| `measurement/a967622c6d7e9bd74d4f` | kernel_scatter.compile_s | {"kernel_scatter.compile_s": 0.08845993998693302} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_scatter/compile_s` |
| `measurement/b304e20c68537032c498` | sparse_triplets.lower_s | {"sparse_triplets.lower_s": 0.01603431801777333} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/lower_s` |
| `measurement/ba553656d5c64a2a4aa7` | kernel_gather_segment_sum.compile_s | {"kernel_gather_segment_sum.compile_s": 0.07788260403322056} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/compile_s` |
| `measurement/c30f1ac81d022db9d94a` | curvature_sparse.first_call_s | {"curvature_sparse.first_call_s": 5.6190240969881415} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/curvature_sparse/first_call_s` |
| `measurement/c486c35f8b5da3c12c30` | data_vector_sparse.lower_s | {"data_vector_sparse.lower_s": 0.008048190036788583} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/lower_s` |
| `measurement/cfeaf7b9c8abe46dc70e` | figure_of_merit.lower_s | {"figure_of_merit.lower_s": 0.017631167022045702} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/figure_of_merit/lower_s` |
| `measurement/d67b7279c4b2ade88082` | log_dets.compile_s | {"log_dets.compile_s": 0.10405478900065646} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/log_dets/compile_s` |
| `measurement/da60e7e7201a3c983ca7` | sparse_triplets.first_call_s | {"sparse_triplets.first_call_s": 0.0011366799590177834} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/sparse_triplets/first_call_s` |
| `measurement/dd2854333f60f7c4748d` | library_fit_figure_of_merit.first_call_s | {"library_fit_figure_of_merit.first_call_s": 5.841603362991009} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/first_call_s` |
| `measurement/e24dbd8d7fc0a22f5bba` | library_fit_figure_of_merit.lower_s | {"library_fit_figure_of_merit.lower_s": 1.231511396006681} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/library_fit_figure_of_merit/lower_s` |
| `measurement/e5e12b38cfe85dcb7c5f` | setup_prefix_2.lower_s | {"setup_prefix_2.lower_s": 0.8638483890099451} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/lower_s` |
| `measurement/f2b5117ebe7ca6cde52d` | data_vector_sparse.first_call_s | {"data_vector_sparse.first_call_s": 0.2757179230102338} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/data_vector_sparse/first_call_s` |
| `measurement/f38afb5d7ef513bacd55` | setup_prefix_4.first_call_s | {"setup_prefix_4.first_call_s": 0.3619350509834476} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_4/first_call_s` |
| `measurement/fcded45282de22fa59ee` | setup_prefix_2.compile_s | {"setup_prefix_2.compile_s": 0.956531937001273} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/setup_prefix_2/compile_s` |
| `measurement/fe1eda9bf8972eead4f6` | kernel_gather_segment_sum.lower_s | {"kernel_gather_segment_sum.lower_s": 0.010328751988708973} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/kernel_gather_segment_sum/lower_s` |

</details>

### memory

<details><summary>1 recorded memory measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/bd65b1ae8aa2a49b51b1` | host_peak_rss | {"host_peak_rss": 11907.83984375} | MiB | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/peak_rss_mb` |

</details>

### runtime

<details><summary>2 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/241c17a8ad925175d76a` | full_pipeline.per_call_s | {"full_pipeline.per_call_s": 5.859374109325775} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/full_pipeline/per_call_s` |
| `measurement/a936d12a871c5c1664e4` | single_jit_block | {"single_jit_block": 5.859374109325775} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/interferometer/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/history/2026.10.25.1_2026-10-28/breakdown.json) `/full_pipeline_single_jit` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": true,
      "cpu_count": 252,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "autolens_profiling.library_revisions": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": null,
    "statistic": "peak",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "repetitions": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": true,
    "host": "euclid-ral-compute-1",
    "measured_at": "2026-10-28T13:54:50Z",
    "qualified": false,
    "unknowns": {}
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": true,
      "cpu_count": 252,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "autolens_profiling.library_revisions": "80bbb028284ad8eb36d34692a414319f881b1d1e",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-compute-1",
    "measured_at": "2026-10-28T13:54:50Z",
    "qualified": false,
    "unknowns": {}
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

- [likelihood_breakdown](../../../../scripts/interferometer/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/interferometer/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/interferometer/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/interferometer/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/interferometer/delaunay/quick_update.py)
- [_streaming](../../../../scripts/interferometer/delaunay/_streaming.py)
- [streaming_accumulate](../../../../scripts/interferometer/delaunay/streaming_accumulate.py)
- [streaming_in_memory](../../../../scripts/interferometer/delaunay/streaming_in_memory.py)
- [streaming_parity](../../../../scripts/interferometer/delaunay/streaming_parity.py)
- [streaming_plot_scaling](../../../../scripts/interferometer/delaunay/streaming_plot_scaling.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
