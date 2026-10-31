<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · ao

[Model index](index.md)

Exact setup ID: `imaging/delaunay/ao/8b13c38229801b63914c`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | "subtracted S/N (truth tracer), floored" | recorded |  |
| adapt_minimum_percent | 0.01 | recorded |  |
| delaunay_vertices | 1280 | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| image_pixels_masked | 384753 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_source | "truth" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_adapt_snr_cap | 3.0 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sample_size_pixelization_rule | {"pixels_above": 47431, "pixels_below": 337322, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"} | recorded |  |
| over_sampled_pixels | 1542816 | recorded |  |
| over_sampling_preset | "hst_source_pix_2_standard_ao" | recorded |  |
| oversampled_pixels | 1542816 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.01 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": 3.0, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 47431, "pixels_below": 337322, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "hst_source_pix_2_standard_ao"} | recorded |  |
| production_preset | {"adapt_snr_cap": 3.0, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_60_linear_2_basis", "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 30, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": null, "preset": "hst_source_pix_2_standard_ao", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus the other galaxies' model images, over the noise map), floored at adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max", "mesh_and_regularization_adapt_image": "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from (S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and rectangular_adapt.py:109", "model_stand_in": "no previous-stage fit exists in a profiling cell: the dataset's truth tracer (tracer.json) stands in for the maximum-likelihood model whose lens light is subtracted", "over_sampling": "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184", "rectangular_mesh": "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight priors config/priors/mesh/rectangular.yaml:43-61"}, "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; no ao production reference: borrows the hst preset 'hst_source_pix_2' with the ao dataset", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut; no ao production reference: borrows the hst preset 'hst_source_pix_2' with the ao dataset", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_nnz | 6324552 | recorded |  |
| sparse_operator_build_s | 0.28977922606281936 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1340 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>37 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/105d034541ce33fdd41f` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.029204119497444482} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/15487fe3398c15c66340` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.04672651740256697} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/1f3f2220a84ca3770bf4` | regularization_matrix_prefix | {"regularization_matrix_prefix": 6.690792411798611} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/279ed98565f3aec0990d` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00028253860073164106} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/3b4c99050dbfe3e61f89` | curvature_F_off_diag_jit.steady_per_call_s | {"curvature_F_off_diag_jit.steady_per_call_s": 2.2809957214049064} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/steady_per_call_s` |
| `measurement/435c1661a886b45936ba` | curvature_F_diag_jit.steady_per_call_s | {"curvature_F_diag_jit.steady_per_call_s": 69.2831170344958} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/steady_per_call_s` |
| `measurement/4546f2676386b980bde8` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.03875111630186438} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/4e6be8717c711cbebac5` | sparse_triplets_jit.steady_per_call_s | {"sparse_triplets_jit.steady_per_call_s": 0.029255446104798466} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/steady_per_call_s` |
| `measurement/51fa06f605fe1af3d7b5` | log_evidence_sparse_jit.steady_per_call_s | {"log_evidence_sparse_jit.steady_per_call_s": 0.08524210239993409} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/steady_per_call_s` |
| `measurement/586f498c75ef10001aad` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.021050591394305228} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/6d72cf4d4deeda8a189b` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.001420897094067186} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/773dd0e19a25d4f23fa6` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 1.1995678224950097} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/7bf0368db7d4d5a45a16` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.001420897094067186} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/7ed47cca174b6c1d13d4` | psf_weighted_data_jit.steady_per_call_s | {"psf_weighted_data_jit.steady_per_call_s": 0.4857924765907228} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/steady_per_call_s` |
| `measurement/80c209932aedc7ccba4f` | curvature_matrix_sparse_jit.steady_per_call_s | {"curvature_matrix_sparse_jit.steady_per_call_s": 83.76832478109282} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/steady_per_call_s` |
| `measurement/86df41663db5fded258f` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 1.0739649144001304} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/8ae6e231887ba794c5d8` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.06921172619331628} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/8bf9fd7c89a4d05bf7b7` | interpolator_prefix | {"interpolator_prefix": 6.823948057100642} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/interpolator_prefix_s` |
| `measurement/8e14d6858fc43a9bb6f8` | steps.Curvature matrix (F, w-tilde) | {"steps.Curvature matrix (F, w-tilde)": 83.76832478109282} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Curvature matrix (F, w-tilde)` |
| `measurement/9b135cc0ff5b167e10b2` | mge_operated_basis_jit.steady_per_call_s | {"mge_operated_basis_jit.steady_per_call_s": 0.5569187793997117} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/steady_per_call_s` |
| `measurement/a7b00c252ba2abf69fb3` | data_vector_sparse_jit.steady_per_call_s | {"data_vector_sparse_jit.steady_per_call_s": 0.02967172550270334} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/steady_per_call_s` |
| `measurement/a8680163b8fbce9deeeb` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 1.1995678224950097} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/ae6053112e9312d36c77` | component_total | {"component_total": 92.75621951018694} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/aecd7ef9519e4c303656` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00028253860073164106} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/b5543a2805b9dac00733` | steps.Data vector (D, w-tilde) | {"steps.Data vector (D, w-tilde)": 0.02967172550270334} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Data vector (D, w-tilde)` |
| `measurement/b67915a0edc7ad7b3bd9` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00021618540631607175} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/b8d07351e5b07d2da175` | steps.Inversion setup (sparse, steps 5-8 combined) | {"steps.Inversion setup (sparse, steps 5-8 combined)": 7.744847395201214} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Inversion setup (sparse, steps 5-8 combined)` |
| `measurement/bdbb6add4c4a909c5bf3` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 6.690792411798611} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/bdc9a7749e48a677133a` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.021050591394305228} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/c265aaa03018925b22b3` | inversion_setup_sparse_jit.steady_per_call_s | {"inversion_setup_sparse_jit.steady_per_call_s": 7.744847395201214} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/steady_per_call_s` |
| `measurement/c594f30c9a332112c38d` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 6.823948057100642} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/c7d9795864472f51c034` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00021618540631607175} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/cdaf5bca477a634440b3` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.05448521579382941} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/d2c15da35206db1dc44c` | curvature_F_func_func_jit.steady_per_call_s | {"curvature_F_func_func_jit.steady_per_call_s": 0.09823989080032333} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/steady_per_call_s` |
| `measurement/e36016b8628611ca7add` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.04438608959317207} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/f0ac851d7554a2c24ebb` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.03875111630186438} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/fdc2b406b3c6a00a5430` | steps.Mapped recon + log evidence (sparse) | {"steps.Mapped recon + log evidence (sparse)": 0.08524210239993409} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Mapped recon + log evidence (sparse)` |

</details>

### compile

<details><summary>72 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0191d9e8b2096db37850` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.04334111092612147} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/06a435dd6bf8d27fb208` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.006867260904982686} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/12547b5394d0422fd1de` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.00640690210275352} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/1c6f13efd396797fa169` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.43774115201085806} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/1d39f15a154c2eb0732a` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.0075215710094198585} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/28198726bf9345567788` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 4.955418668920174} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/2e733a820661d14cddef` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0022653129417449236} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/2fe17402e53af77db37c` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.056408185977488756} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/33012d34273bcca3d9b0` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.023327118018642068} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/35221ae031d8b3fd7796` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.07321488403249532} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/356f4e6ea35e5ff676f3` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 1.3349069519899786} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/3940c35d6eb6c98e153d` | curvature_F_off_diag_jit.compile_s | {"curvature_F_off_diag_jit.compile_s": 0.12315444101113826} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/compile_s` |
| `measurement/3b1bb132ca4428bcb2c0` | data_vector_sparse_jit.lower_s | {"data_vector_sparse_jit.lower_s": 0.0073816480580717325} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/lower_s` |
| `measurement/3b96a163178ed7606308` | psf_weighted_data_jit.compile_s | {"psf_weighted_data_jit.compile_s": 0.0726782779674977} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/compile_s` |
| `measurement/3d781eb4e575acdf8dc4` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.08330944797489792} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/445e631844b6d8962b2a` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 7.200850315042771} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/45b690833037c1269d80` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.25270590791478753} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/485f967396d2b6947b23` | curvature_F_func_func_jit.compile_s | {"curvature_F_func_func_jit.compile_s": 0.0461516990326345} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/compile_s` |
| `measurement/48655a76256950ebe4e0` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0008717050077393651} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/4d8c0c338ad02c3ec865` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.021525784046389163} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/4faa47c0f0ba0f318d21` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.07355765905231237} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/4fe2d10fb42386cfe4d2` | curvature_F_func_func_jit.lower_s | {"curvature_F_func_func_jit.lower_s": 0.002843859023414552} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/lower_s` |
| `measurement/5191a21c44fc01b0ff32` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.0681707999901846} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/5f4b84c1c0dfd29dfd24` | inversion_setup_sparse_jit.first_call_s | {"inversion_setup_sparse_jit.first_call_s": 7.235028996015899} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/first_call_s` |
| `measurement/61264b77aa9b6ccbfd87` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.2555967979133129} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/63dccc48a45bfacf0d6a` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.03820740198716521} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/65058711f301c2c6bb43` | curvature_F_diag_jit.lower_s | {"curvature_F_diag_jit.lower_s": 0.02929934300482273} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/lower_s` |
| `measurement/65716a89f98cdc0c67be` | curvature_F_off_diag_jit.first_call_s | {"curvature_F_off_diag_jit.first_call_s": 2.169249444035813} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/first_call_s` |
| `measurement/6655969f51e4b6ba9072` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.00046461098827421665} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/66b1c55fef5270d6302a` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.014171828981488943} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/6a0154b43b3e1f263071` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.9881233740597963} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/6e695371302c770ea72c` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.018454795936122537} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/6fe510eb3e1339f4c9a8` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.0935632559703663} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/75d01b56d029215b1257` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0016808080254122615} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/76fc916b8a6b63cc0999` | inversion_setup_sparse_jit.compile_s | {"inversion_setup_sparse_jit.compile_s": 12.407005375949666} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/compile_s` |
| `measurement/7b3495bc41e5cfcc3906` | mge_operated_basis_jit.first_call_s | {"mge_operated_basis_jit.first_call_s": 0.5317805480444804} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/first_call_s` |
| `measurement/7fdd4c36057f83ae8f7b` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.07414212089497596} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/82da642cc9eda1a53ae7` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.4911056496202946e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/87a810abfb402b54f00a` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 6.999849193030968} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/8cd715ab1b96bc7d6adc` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.18984860996715724} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/8dc91e97256605bb52e5` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.08495947206392884} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/91b03d6604d1311cebcd` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.2502256300067529} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/96dffa3063b88124970e` | inversion_setup_sparse_jit.lower_s | {"inversion_setup_sparse_jit.lower_s": 4.803322267020121} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/inversion_setup_sparse_jit/lower_s` |
| `measurement/996d73d735f7f983b3c1` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.1685264309635386} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/99a7e16964cbfa4fa2c4` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.02976353804115206} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/9f851e5cf2e0993801ff` | curvature_F_func_func_jit.first_call_s | {"curvature_F_func_func_jit.first_call_s": 0.099353595986031} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_func_func_jit/first_call_s` |
| `measurement/9f8b035e1e346a047649` | curvature_F_off_diag_jit.lower_s | {"curvature_F_off_diag_jit.lower_s": 0.014575714012607932} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_off_diag_jit/lower_s` |
| `measurement/9f9ac4ffac75632be534` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.03747075202409178} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/aa1f00dba0aa7e3cfd98` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.469646819983609} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/aab99e04e4e91e35045b` | curvature_F_diag_jit.first_call_s | {"curvature_F_diag_jit.first_call_s": 65.60807803203352} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/first_call_s` |
| `measurement/ad74b5a408fc7855acf4` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.017487613949924707} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/aeb0dc329922a69bf0ca` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.04552069993223995} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/b110a33ca9423e1b6e30` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.29556405497714877} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/b3dfbe2c308ad9bf78c4` | curvature_matrix_sparse_jit.compile_s | {"curvature_matrix_sparse_jit.compile_s": 0.36553436797112226} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/compile_s` |
| `measurement/c3742becb31bfee709f4` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.25487983401399106} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/c98cd7e1ea9997be6576` | mge_operated_basis_jit.lower_s | {"mge_operated_basis_jit.lower_s": 3.000318548991345} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/lower_s` |
| `measurement/cd445805159eda6bb36a` | curvature_matrix_sparse_jit.lower_s | {"curvature_matrix_sparse_jit.lower_s": 0.04233984800521284} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/lower_s` |
| `measurement/ce79228a784e9d684f9c` | mge_operated_basis_jit.compile_s | {"mge_operated_basis_jit.compile_s": 7.3567287139594555} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/mge_operated_basis_jit/compile_s` |
| `measurement/d0961510d67f3d94df0a` | sparse_triplets_jit.first_call_s | {"sparse_triplets_jit.first_call_s": 0.0355814469512552} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/first_call_s` |
| `measurement/d144a5f87fbabc35c542` | data_vector_sparse_jit.first_call_s | {"data_vector_sparse_jit.first_call_s": 0.0280642278958112} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/first_call_s` |
| `measurement/d1518a0cd9e6a43572d1` | curvature_matrix_sparse_jit.first_call_s | {"curvature_matrix_sparse_jit.first_call_s": 122.02803828706965} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_matrix_sparse_jit/first_call_s` |
| `measurement/d3927f21ef22e29de5ef` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.18574615300167352} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/d9303dc19890033c9686` | curvature_F_diag_jit.compile_s | {"curvature_F_diag_jit.compile_s": 0.1985228779958561} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/curvature_F_diag_jit/compile_s` |
| `measurement/d9facd99efb60196f3a6` | log_evidence_sparse_jit.compile_s | {"log_evidence_sparse_jit.compile_s": 0.2531531510176137} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/compile_s` |
| `measurement/daa85d7f8bf60b4ab58e` | psf_weighted_data_jit.lower_s | {"psf_weighted_data_jit.lower_s": 0.01088514900766313} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/lower_s` |
| `measurement/dffba333953bb5334309` | log_evidence_sparse_jit.lower_s | {"log_evidence_sparse_jit.lower_s": 0.034508849028497934} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/lower_s` |
| `measurement/e18b0dae2d8f5e49fb53` | data_vector_sparse_jit.compile_s | {"data_vector_sparse_jit.compile_s": 0.08149966096971184} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/data_vector_sparse_jit/compile_s` |
| `measurement/eaeda7ca4c2c174f4211` | sparse_triplets_jit.compile_s | {"sparse_triplets_jit.compile_s": 0.0924590879585594} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/compile_s` |
| `measurement/ec6d3cf99a69eee106c7` | sparse_triplets_jit.lower_s | {"sparse_triplets_jit.lower_s": 0.014886931981891394} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/sparse_triplets_jit/lower_s` |
| `measurement/efd61c3083cfad62479d` | log_evidence_sparse_jit.first_call_s | {"log_evidence_sparse_jit.first_call_s": 0.15591825393494219} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/log_evidence_sparse_jit/first_call_s` |
| `measurement/f381e762bc6902f2f0e4` | psf_weighted_data_jit.first_call_s | {"psf_weighted_data_jit.first_call_s": 0.5057016198989004} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/psf_weighted_data_jit/first_call_s` |
| `measurement/fceb58343e8b17af2e94` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.943756898981519} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |

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
      "cpu_count": 236,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-10-5",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "8860c50c0dc924ee591bfd0157cde7946a80a25e",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.11.2",
      "jaxlib.dependency_versions": "0.11.2",
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
    "host": "euclid-ral-compute-10-5",
    "measured_at": "2026-10-30T17:08:56Z",
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
