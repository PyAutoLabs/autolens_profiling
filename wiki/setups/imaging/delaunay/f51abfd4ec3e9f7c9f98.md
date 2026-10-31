<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · ao

[Model index](index.md)

Exact setup ID: `imaging/delaunay/ao/2760f599559ae4ffacd1`.

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
| standard_size | true | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1340 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>32 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/093d6096bfb692d88588` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.000134870401234366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/0e1b9e0e5829a2b8c9e7` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.03871844100067392} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/0ff5db994b8ef3d232cc` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0009773133002454414} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/10249d24bba6fbc33d06` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.039108686498366296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/12aab4eb2ade3ebce87c` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0035255555994808674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/197b8bfc12a5bd59991f` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.208575459299027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/1e587320310bd4baebf5` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.0013289580994751305} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/24f7d3f4760d5a20f760` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.00014762420032639058} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/2987d63f4e78aac127c6` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.03974364069872536} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/38f895b2692a3926704a` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0026928555977065115} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/432100ccdaf5a3268fbb` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00027334369951859117} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/4a3e42dd42a198b9339a` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0027658597013214604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Data vector (D)` |
| `measurement/4b60343c09455206e5c3` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.0027658597013214604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/4fd5fc037297291a42eb` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.0009742683003423735} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/53d0205520a1e7ffc48c` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.043269196298206226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/5af31f6527f43b62b0f0` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.10082852659979835} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/670acc89e278781ec5ea` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.10082852659979835} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/6afd258111b87eda6d9e` | interpolator_prefix | {"interpolator_prefix": 0.03974364069872536} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/interpolator_prefix_s` |
| `measurement/7a39b50a32f120e0ebb5` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0010302376991603523} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/829395da41fac852fbeb` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00014762420032639058} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/8a95536f967df053cfb4` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.043269196298206226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/9cadf9ca1149312e897a` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00027334369951859117} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/a4b0afe91aae0f0f2a57` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.208575459299027} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/a9740986c1108dc18ffa` | component_total | {"component_total": 0.3607076287007658} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/total_step_by_step` |
| `measurement/a978e24503df7ef43b5a` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00014082439884077758} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/b3c885ed18b29cbad41f` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00014082439884077758} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/b7463a805e50bec385da` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.000134870401234366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/c65a5692cfff206d7d29` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.004218618202139623} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Mapped recon + log evidence` |
| `measurement/c8afc9c03abf491f3619` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.039108686498366296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/d3f8021b0b31cd5b71d0` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0009882601007120684} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/d5297e0b738b2d464937` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.004218618202139623} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/fe7db781ef1456caf3f4` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0009882601007120684} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |

</details>

### compile

<details><summary>54 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/063d7e73bcb8b7419b94` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.19874624800286256} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/068c8c0cab409d0803bd` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 3.851586804987164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/11d38929009845cf383d` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.02532837601029314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/12f6a75fb82ee3da6ed3` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.06523233599727973} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/145a4045cf787089f3a2` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.0018215280142612755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/16e975ba924fc53fcd33` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.009229249000782147} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/1ea7d271820e16177702` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.00105089700082317} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/21b11631f1bc8341d0de` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.08179077500244603} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/2e6dfcf9032f7dbcb652` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.027332761004799977} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/2e75268ea9eb6586d286` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.29679466801462695} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/37d29bd5dc3cbf9205a2` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.016796509007690474} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/3ae9f7d223cfa916e89f` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.09366641400265507} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/3b7afc78840e72612ed4` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.34232169098686427} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/3cd01651fe7049bdf9df` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.022652048006420955} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/45bea1209481e80b1fc3` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.5210927590087522} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/474404f5992216a2f1c1` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0002853159967344254} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/492b7d77dcf1abc355d9` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.06697646400425583} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/53b1f9049692e283ab53` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.00424966900027357} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/57b132c36746345006d7` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.09189928398700431} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/591b3ef31f96b2a2e2ea` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.4955662769789342} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/6022e09945f34b800442` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.007916675007436424} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/641c70372b0daff1fce5` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.001537360018119216} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/663165ec863e96297529` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0006682020029984415} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/6d55c20f54d6afb5e5d7` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 8.670001989230514e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/793c48b16ea6e4017b8c` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 11.193271035008365} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/7b681022b34b6fe8fcef` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.0572171309904661} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/7c157958bb4ec8039c99` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.004243968985974789} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/7c8b2335c9bc9df55ad9` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.05133839100017212} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/7d1f0bb675c088a4989d` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.04048059199703857} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/7dad0cd32d7e90670995` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.3188111830095295} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/866af9a5c63b348d3d84` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.3006657919904683} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/8c2030dcfb0d8d3f33cb` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.355603791977046} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/920397416def72a065c2` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.0016359800065401942} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/92c22b6d4c2a3ac4cb4c` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0011060959950555116} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/93e2c3aa7564e1f5d856` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.12917076598387212} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/9c5c407bbb1e750c6d9f` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.03992725900025107} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/a6172a1a8f290cc5f687` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.4147998490079772} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/b6206f6dc5c64ee8ef15` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.06594425599905662} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/b6e6855d5e55d36280af` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.010543134005274624} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/b700e3bfbb06aaa6124a` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.022274222021223977} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/bab9e2eaf6b757956e9e` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.45538974000373855} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/bd4921b1bbb4a0578892` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0018376980151515454} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/c3bb23afdb7cd2dab351` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.27975374402012676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/c3c7027f79448416f74d` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0007071809959597886} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/cf2f1d3d770071e7a3f5` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.059049828007118776} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/d0daed89d85cd0d2df44` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.08186281498637982} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/d68154ab9b6e8e115a23` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.003252940980019048} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/d842cdda149477e718d3` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.009722882008645684} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/dbc1720601aa19012e20` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.039868310996098444} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/e2df1ef06496fd276c47` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.005754529993282631} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/eb8dfd7e919fd419b50f` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.4186126450076699} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/ec9bb8ade24498b7a74c` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.005227117013419047} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/fe9e570ab9d523bc7245` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 4.043088550999528} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/ff0326ef90f279a1f92e` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.011553280986845493} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/ao/jax_gpu_dense_fp64/hpc_a100/history/2026.10.23.1_2026-10-25/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |

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
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 61545 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "d99ec2b30224ac021e39628b9904d01c44f33b29",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "d3a7721294652f81ec726b92931ab610ed27e3ac",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "e00b5e7d554d6958a6c8821edef6afdf01188e6d",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "f791ac77d136d664346fff21cc36079f8f9b1196",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "f116f70077e442db19a2a152f5ee78d57294c504",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "65b17065a66486ed8f5661d720afc18c31c6afd8",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-26T11:38:23Z",
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
