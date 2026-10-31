<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/b3214bdea75531e0b948`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_image | "subtracted S/N (truth tracer), floored" | recorded |  |
| adapt_minimum_percent | 0.01 | recorded |  |
| delaunay_vertices | 1280 | recorded |  |
| edge_zeroed_pixels | 30 | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_light_source | "truth" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh_adapt_snr_cap | 3.0 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sample_size_pixelization_rule | {"pixels_above": 2068, "pixels_below": 13293, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"} | recorded |  |
| over_sampled_pixels | 61600 | recorded |  |
| over_sampling_preset | "hst_source_pix_2_standard" | recorded |  |
| oversampled_pixels | 61600 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": 3.0, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 2068, "pixels_below": 13293, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "hst_source_pix_2_standard"} | recorded |  |
| production_preset | {"adapt_snr_cap": 3.0, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_60_linear_2_basis", "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 30, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": null, "preset": "hst_source_pix_2_standard", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus the other galaxies' model images, over the noise map), floored at adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max", "mesh_and_regularization_adapt_image": "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from (S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and rectangular_adapt.py:109", "model_stand_in": "no previous-stage fit exists in a profiling cell: the dataset's truth tracer (tracer.json) stands in for the maximum-likelihood model whose lens light is subtracted", "over_sampling": "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184", "rectangular_mesh": "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight priors config/priors/mesh/rectangular.yaml:43-61"}, "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut", "variant": "production"} | recorded |  |
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

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

- `measurement/b76a3b08f11d576af3f4`: breakdown / component_total = {"component_total": 0.06300834277644753} s (unreviewed).
- `measurement/6e43a109ce313911c41a`: breakdown / regularization_matrix_prefix = {"regularization_matrix_prefix": 0.007708000903949142} s (unreviewed).
- `measurement/800d13eec636bbf00189`: breakdown / interpolator_prefix = {"interpolator_prefix": 0.007638170709833502} s (unreviewed).
- `measurement/40eede3b4d71076c6d61`: breakdown / steps.Ray-trace data grid = {"steps.Ray-trace data grid": 0.00018545680213719607} s (unreviewed).
- `measurement/792ec0756b0aa9d827fd`: breakdown / steps.Ray-trace mesh grid = {"steps.Ray-trace mesh grid": 0.0001947956974618137} s (unreviewed).
- `measurement/1f2326b5750a7e13b33e`: breakdown / steps.Lens light images (pre-PSF) = {"steps.Lens light images (pre-PSF)": 0.0001543450984172523} s (unreviewed).
- `measurement/f272f21689daa210181e`: breakdown / steps.Blurred image (PSF convolution) = {"steps.Blurred image (PSF convolution)": 0.0008470878005027771} s (unreviewed).
- `measurement/71c7dcbd0b28d7565381`: breakdown / steps.Profile-subtracted image = {"steps.Profile-subtracted image": 0.0001562231918796897} s (unreviewed).
- `measurement/e5203686b98b911c6417`: breakdown / steps.Inversion setup (steps 5-8 combined) = {"steps.Inversion setup (steps 5-8 combined)": 0.019119272497482596} s (unreviewed).
- `measurement/e78a3a5b0376305541f0`: breakdown / steps.Data vector (D) = {"steps.Data vector (D)": 0.0003122652997262776} s (unreviewed).
- `measurement/2286d695278e63a7720f`: breakdown / steps.Curvature matrix (F) = {"steps.Curvature matrix (F)": 0.003935846907552332} s (unreviewed).
- `measurement/8c37c8b36b99ef180f2e`: breakdown / steps.Regularization matrix (H) = {"steps.Regularization matrix (H)": 6.983019411563925e-05} s (unreviewed).
- `measurement/ef2e66754ebf874f33bb`: breakdown / steps.Regularized reconstruction = {"steps.Regularized reconstruction": 0.03616273469524458} s (unreviewed).
- `measurement/e9e5b1be13f642e35fb9`: breakdown / steps.Mapped recon + log evidence = {"steps.Mapped recon + log evidence": 0.0018704845919273795} s (unreviewed).
- `measurement/46ff08fb0f2f8f4da4f4`: compile / ray_trace_data_jit.lower_s = {"ray_trace_data_jit.lower_s": 0.0890261649619788} s (unreviewed).
- `measurement/e6de03cefa8d53e77943`: compile / ray_trace_data_jit.compile_s = {"ray_trace_data_jit.compile_s": 0.3012845799094066} s (unreviewed).
- `measurement/d9b44fca70668c7eba68`: compile / ray_trace_data_jit.first_call_s = {"ray_trace_data_jit.first_call_s": 0.002027156064286828} s (unreviewed).
- `measurement/4f4b7ee15fffb985c368`: breakdown / ray_trace_data_jit.steady_per_call_s = {"ray_trace_data_jit.steady_per_call_s": 0.00018545680213719607} s (unreviewed).
- `measurement/6aa6f192d8b7071fbe26`: compile / ray_trace_mesh_jit.lower_s = {"ray_trace_mesh_jit.lower_s": 0.08484641509130597} s (unreviewed).
- `measurement/dc92214320695dc22d83`: compile / ray_trace_mesh_jit.compile_s = {"ray_trace_mesh_jit.compile_s": 0.41407021996565163} s (unreviewed).
- `measurement/b112114272a1b99cc794`: compile / ray_trace_mesh_jit.first_call_s = {"ray_trace_mesh_jit.first_call_s": 0.0014536429662257433} s (unreviewed).
- `measurement/4eb7d830f65cdb761229`: breakdown / ray_trace_mesh_jit.steady_per_call_s = {"ray_trace_mesh_jit.steady_per_call_s": 0.0001947956974618137} s (unreviewed).
- `measurement/ed9306ec96c560eef08c`: compile / lens_image_jit.lower_s = {"lens_image_jit.lower_s": 0.06545290793292224} s (unreviewed).
- `measurement/1bfb66587544ba8ba940`: compile / lens_image_jit.compile_s = {"lens_image_jit.compile_s": 0.042944825952872634} s (unreviewed).
- `measurement/5472b0ef069b16c82f33`: compile / lens_image_jit.first_call_s = {"lens_image_jit.first_call_s": 0.0006900830194354057} s (unreviewed).
- `measurement/553d51cb012d30d0b6ed`: breakdown / lens_image_jit.steady_per_call_s = {"lens_image_jit.steady_per_call_s": 0.0001543450984172523} s (unreviewed).
- `measurement/72805142afd7b63c43a5`: compile / blurred_image_jit.lower_s = {"blurred_image_jit.lower_s": 0.0608523819828406} s (unreviewed).
- `measurement/04cd62591c50ef8706f2`: compile / blurred_image_jit.compile_s = {"blurred_image_jit.compile_s": 0.15195632295217365} s (unreviewed).
- `measurement/2ccf1be82e9e80db48cb`: compile / blurred_image_jit.first_call_s = {"blurred_image_jit.first_call_s": 0.01791176595725119} s (unreviewed).
- `measurement/4e92b3e435a1c01a8fbe`: breakdown / blurred_image_jit.steady_per_call_s = {"blurred_image_jit.steady_per_call_s": 0.0008470878005027771} s (unreviewed).
- `measurement/54fc4571a6b3909b21a8`: compile / profile_subtract_jit.lower_s = {"profile_subtract_jit.lower_s": 0.001513330964371562} s (unreviewed).
- `measurement/e32ed3c63e7fd3998d8a`: compile / profile_subtract_jit.compile_s = {"profile_subtract_jit.compile_s": 0.030017400975339115} s (unreviewed).
- `measurement/5e6037751188f0eb6de6`: compile / profile_subtract_jit.first_call_s = {"profile_subtract_jit.first_call_s": 0.0005963339935988188} s (unreviewed).
- `measurement/ad97093272c2ab03fdfe`: breakdown / profile_subtract_jit.steady_per_call_s = {"profile_subtract_jit.steady_per_call_s": 0.0001562231918796897} s (unreviewed).
- `measurement/4c783c0d47269945ee3a`: compile / inversion_setup_jit.lower_s = {"inversion_setup_jit.lower_s": 3.851136868004687} s (unreviewed).
- `measurement/e46c2dbca6784b41842c`: compile / inversion_setup_jit.compile_s = {"inversion_setup_jit.compile_s": 11.215620680944994} s (unreviewed).
- `measurement/2f3e2dfc4f006ebfd034`: compile / inversion_setup_jit.first_call_s = {"inversion_setup_jit.first_call_s": 0.26976936601568013} s (unreviewed).
- `measurement/7f26f97e34a1856a0cd5`: breakdown / inversion_setup_jit.steady_per_call_s = {"inversion_setup_jit.steady_per_call_s": 0.019119272497482596} s (unreviewed).
- `measurement/d07ed8a9358bf5924979`: compile / setup_prefix_6.lower_s = {"setup_prefix_6.lower_s": 0.3201956719858572} s (unreviewed).
- `measurement/a4d3596309ffae6c676d`: compile / setup_prefix_6.compile_s = {"setup_prefix_6.compile_s": 1.9390947319334373} s (unreviewed).
- `measurement/d5cec7a156ce42160ec6`: compile / setup_prefix_6.first_call_s = {"setup_prefix_6.first_call_s": 0.013195122010074556} s (unreviewed).
- `measurement/0fa44b32933f75f79d7b`: breakdown / setup_prefix_6.steady_per_call_s = {"setup_prefix_6.steady_per_call_s": 0.007638170709833502} s (unreviewed).
- `measurement/238c1eff6b3f0274b999`: compile / data_vector_jit.lower_s = {"data_vector_jit.lower_s": 0.002164063975214958} s (unreviewed).
- `measurement/df290b577c2ee6fc3e8a`: compile / data_vector_jit.compile_s = {"data_vector_jit.compile_s": 0.06319025496486574} s (unreviewed).
- `measurement/b65855938339937cfc2a`: compile / data_vector_jit.first_call_s = {"data_vector_jit.first_call_s": 0.0008535090601071715} s (unreviewed).
- `measurement/70b79da44a1d3b530cbf`: breakdown / data_vector_jit.steady_per_call_s = {"data_vector_jit.steady_per_call_s": 0.0003122652997262776} s (unreviewed).
- `measurement/44f5b435f46285ac268d`: compile / curvature_matrix_jit.lower_s = {"curvature_matrix_jit.lower_s": 0.009870382025837898} s (unreviewed).
- `measurement/3a2c8421b1c867136d7d`: compile / curvature_matrix_jit.compile_s = {"curvature_matrix_jit.compile_s": 0.052767059998586774} s (unreviewed).
- `measurement/774db02bdda32c87e573`: compile / curvature_matrix_jit.first_call_s = {"curvature_matrix_jit.first_call_s": 0.004481215961277485} s (unreviewed).
- `measurement/901d96e838b4e49055c7`: breakdown / curvature_matrix_jit.steady_per_call_s = {"curvature_matrix_jit.steady_per_call_s": 0.003935846907552332} s (unreviewed).
- `measurement/3b5761d7ecbe6b07025d`: compile / regularization_matrix_jit.lower_s = {"regularization_matrix_jit.lower_s": 0.5965617490001023} s (unreviewed).
- `measurement/7fc9742f5e1704a16e61`: compile / regularization_matrix_jit.compile_s = {"regularization_matrix_jit.compile_s": 3.336488628992811} s (unreviewed).
- `measurement/efbca38b0ea8c662ef94`: compile / regularization_matrix_jit.first_call_s = {"regularization_matrix_jit.first_call_s": 0.014666134025901556} s (unreviewed).
- `measurement/c748807fc7ad516ad4f7`: breakdown / regularization_matrix_jit.steady_per_call_s = {"regularization_matrix_jit.steady_per_call_s": 0.007708000903949142} s (unreviewed).
- `measurement/9a49f5f4479987577586`: compile / reconstruction_jit.lower_s = {"reconstruction_jit.lower_s": 0.024412797996774316} s (unreviewed).
- `measurement/049c1d88c0497528dbe9`: compile / reconstruction_jit.compile_s = {"reconstruction_jit.compile_s": 0.31503048492595553} s (unreviewed).
- `measurement/8393bc3d3ccbef8f2b7e`: compile / reconstruction_jit.first_call_s = {"reconstruction_jit.first_call_s": 0.03765062999445945} s (unreviewed).
- `measurement/2fd95fe375cbd67461de`: breakdown / reconstruction_jit.steady_per_call_s = {"reconstruction_jit.steady_per_call_s": 0.03616273469524458} s (unreviewed).
- `measurement/265d723848aad1370c89`: compile / cholesky_curvature_reg_jit.lower_s = {"cholesky_curvature_reg_jit.lower_s": 0.005806451081298292} s (unreviewed).
- `measurement/0979cb31ad69ef9068f5`: compile / cholesky_curvature_reg_jit.compile_s = {"cholesky_curvature_reg_jit.compile_s": 0.0945683300960809} s (unreviewed).
- `measurement/7555a703e6fe989e853b`: compile / cholesky_curvature_reg_jit.first_call_s = {"cholesky_curvature_reg_jit.first_call_s": 0.0017582880100235343} s (unreviewed).
- `measurement/f716a052cae56f2c57bb`: breakdown / cholesky_curvature_reg_jit.steady_per_call_s = {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0010708393063396215} s (unreviewed).
- `measurement/52e2090e4eb959e81cb8`: compile / cholesky_solve_jit.lower_s = {"cholesky_solve_jit.lower_s": 0.006375364027917385} s (unreviewed).
- `measurement/c773a206bdefaae03b6a`: compile / cholesky_solve_jit.compile_s = {"cholesky_solve_jit.compile_s": 0.06787921895738691} s (unreviewed).
- `measurement/1fb160c5ea65525df2b6`: compile / cholesky_solve_jit.first_call_s = {"cholesky_solve_jit.first_call_s": 0.0019049079855903983} s (unreviewed).
- `measurement/5f2eab637166c6884a62`: breakdown / cholesky_solve_jit.steady_per_call_s = {"cholesky_solve_jit.steady_per_call_s": 0.0013241222011856734} s (unreviewed).
- `measurement/e60853ab954f36cb8a3a`: compile / nnls_pdip_jit.lower_s = {"nnls_pdip_jit.lower_s": 0.01886876509524882} s (unreviewed).
- `measurement/92fb88825669bdeb7783`: compile / nnls_pdip_jit.compile_s = {"nnls_pdip_jit.compile_s": 0.3087045500287786} s (unreviewed).
- `measurement/5ed7f1de4c114ea14dc4`: compile / nnls_pdip_jit.first_call_s = {"nnls_pdip_jit.first_call_s": 0.03773183899465948} s (unreviewed).
- `measurement/8ff9911257d88d723117`: breakdown / nnls_pdip_jit.steady_per_call_s = {"nnls_pdip_jit.steady_per_call_s": 0.03610947229899466} s (unreviewed).
- `measurement/7b7f5c2962e6bcf19bba`: compile / nnls_pdip_one_iteration_jit.lower_s = {"nnls_pdip_one_iteration_jit.lower_s": 0.01922804897185415} s (unreviewed).
- `measurement/5dfaeb4a75b62a2f255c`: compile / nnls_pdip_one_iteration_jit.compile_s = {"nnls_pdip_one_iteration_jit.compile_s": 0.3351968339411542} s (unreviewed).
- `measurement/c67cb948c6316420e19e`: compile / nnls_pdip_one_iteration_jit.first_call_s = {"nnls_pdip_one_iteration_jit.first_call_s": 0.004596225102432072} s (unreviewed).
- `measurement/c2be0e42a757142433d6`: breakdown / nnls_pdip_one_iteration_jit.steady_per_call_s = {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0027550181024707855} s (unreviewed).
- `measurement/800a5c37784c17f5dabc`: compile / log_evidence_jit.lower_s = {"log_evidence_jit.lower_s": 0.013079214026220143} s (unreviewed).
- `measurement/5b34296133059d637325`: compile / log_evidence_jit.compile_s = {"log_evidence_jit.compile_s": 0.1433959660353139} s (unreviewed).
- `measurement/760c9f4fd4025ec7bf3e`: compile / log_evidence_jit.first_call_s = {"log_evidence_jit.first_call_s": 0.002981264959089458} s (unreviewed).
- `measurement/a773b447ad31e2ea1842`: breakdown / log_evidence_jit.steady_per_call_s = {"log_evidence_jit.steady_per_call_s": 0.0018704845919273795} s (unreviewed).
- `measurement/c663c1534645af14bd11`: compile / log_det_curvature_reg_jit.lower_s = {"log_det_curvature_reg_jit.lower_s": 0.008602897985838354} s (unreviewed).
- `measurement/f8818c4bf274e205dbea`: compile / log_det_curvature_reg_jit.compile_s = {"log_det_curvature_reg_jit.compile_s": 0.08733175497036427} s (unreviewed).
- `measurement/8a4e80256d1665b1bece`: compile / log_det_curvature_reg_jit.first_call_s = {"log_det_curvature_reg_jit.first_call_s": 0.0015501920133829117} s (unreviewed).
- `measurement/accbc377757861c58f8b`: breakdown / log_det_curvature_reg_jit.steady_per_call_s = {"log_det_curvature_reg_jit.steady_per_call_s": 0.0009514276054687798} s (unreviewed).
- `measurement/23dac8a902fb7204ad41`: compile / log_det_regularization_jit.lower_s = {"log_det_regularization_jit.lower_s": 0.0003176559694111347} s (unreviewed).
- `measurement/3607ee697f31a48d0f9c`: compile / log_det_regularization_jit.compile_s = {"log_det_regularization_jit.compile_s": 1.0060030035674572e-05} s (unreviewed).
- `measurement/e2d08f1f3328359c1c34`: compile / log_det_regularization_jit.first_call_s = {"log_det_regularization_jit.first_call_s": 0.0011239469749853015} s (unreviewed).
- `measurement/6541c1e8a276f56e55c6`: breakdown / log_det_regularization_jit.steady_per_call_s = {"log_det_regularization_jit.steady_per_call_s": 0.0009678893955424428} s (unreviewed).

### breakdown

<details><summary>32 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0fa44b32933f75f79d7b` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.007638170709833502} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/1f2326b5750a7e13b33e` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.0001543450984172523} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/2286d695278e63a7720f` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.003935846907552332} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/2fd95fe375cbd67461de` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.03616273469524458} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/40eede3b4d71076c6d61` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00018545680213719607} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/4e92b3e435a1c01a8fbe` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0008470878005027771} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/4eb7d830f65cdb761229` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.0001947956974618137} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/4f4b7ee15fffb985c368` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00018545680213719607} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/553d51cb012d30d0b6ed` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.0001543450984172523} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/5f2eab637166c6884a62` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.0013241222011856734} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/6541c1e8a276f56e55c6` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.0009678893955424428} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/6e43a109ce313911c41a` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.007708000903949142} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/70b79da44a1d3b530cbf` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.0003122652997262776} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/71c7dcbd0b28d7565381` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.0001562231918796897} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/792ec0756b0aa9d827fd` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0001947956974618137} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/7f26f97e34a1856a0cd5` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.019119272497482596} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/800d13eec636bbf00189` | interpolator_prefix | {"interpolator_prefix": 0.007638170709833502} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/interpolator_prefix_s` |
| `measurement/8c37c8b36b99ef180f2e` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 6.983019411563925e-05} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/8ff9911257d88d723117` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.03610947229899466} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/901d96e838b4e49055c7` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.003935846907552332} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/a773b447ad31e2ea1842` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.0018704845919273795} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/accbc377757861c58f8b` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0009514276054687798} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/ad97093272c2ab03fdfe` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.0001562231918796897} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/b76a3b08f11d576af3f4` | component_total | {"component_total": 0.06300834277644753} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/c2be0e42a757142433d6` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0027550181024707855} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/c748807fc7ad516ad4f7` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.007708000903949142} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |
| `measurement/e5203686b98b911c6417` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.019119272497482596} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/e78a3a5b0376305541f0` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0003122652997262776} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Data vector (D)` |
| `measurement/e9e5b1be13f642e35fb9` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0018704845919273795} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Mapped recon + log evidence` |
| `measurement/ef2e66754ebf874f33bb` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.03616273469524458} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/f272f21689daa210181e` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0008470878005027771} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/f716a052cae56f2c57bb` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0010708393063396215} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |

</details>

### compile

<details><summary>54 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/049c1d88c0497528dbe9` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.31503048492595553} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/04cd62591c50ef8706f2` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.15195632295217365} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/0979cb31ad69ef9068f5` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.0945683300960809} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/1bfb66587544ba8ba940` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.042944825952872634} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/1fb160c5ea65525df2b6` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.0019049079855903983} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/238c1eff6b3f0274b999` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.002164063975214958} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/23dac8a902fb7204ad41` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0003176559694111347} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/265d723848aad1370c89` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.005806451081298292} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/2ccf1be82e9e80db48cb` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.01791176595725119} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/2f3e2dfc4f006ebfd034` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.26976936601568013} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/3607ee697f31a48d0f9c` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.0060030035674572e-05} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/3a2c8421b1c867136d7d` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.052767059998586774} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/3b5761d7ecbe6b07025d` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.5965617490001023} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/44f5b435f46285ac268d` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.009870382025837898} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/46ff08fb0f2f8f4da4f4` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.0890261649619788} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/4c783c0d47269945ee3a` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 3.851136868004687} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/52e2090e4eb959e81cb8` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.006375364027917385} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/5472b0ef069b16c82f33` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0006900830194354057} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/54fc4571a6b3909b21a8` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.001513330964371562} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/5b34296133059d637325` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.1433959660353139} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/5dfaeb4a75b62a2f255c` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.3351968339411542} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/5e6037751188f0eb6de6` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.0005963339935988188} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/5ed7f1de4c114ea14dc4` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.03773183899465948} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/6aa6f192d8b7071fbe26` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.08484641509130597} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/72805142afd7b63c43a5` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.0608523819828406} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/7555a703e6fe989e853b` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.0017582880100235343} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/760c9f4fd4025ec7bf3e` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.002981264959089458} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/774db02bdda32c87e573` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.004481215961277485} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/7b7f5c2962e6bcf19bba` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.01922804897185415} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/7fc9742f5e1704a16e61` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 3.336488628992811} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/800a5c37784c17f5dabc` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.013079214026220143} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/8393bc3d3ccbef8f2b7e` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.03765062999445945} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/8a4e80256d1665b1bece` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.0015501920133829117} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/92fb88825669bdeb7783` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.3087045500287786} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/9a49f5f4479987577586` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.024412797996774316} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/a4d3596309ffae6c676d` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 1.9390947319334373} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/b112114272a1b99cc794` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0014536429662257433} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/b65855938339937cfc2a` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.0008535090601071715} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/c663c1534645af14bd11` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.008602897985838354} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/c67cb948c6316420e19e` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.004596225102432072} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/c773a206bdefaae03b6a` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.06787921895738691} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/d07ed8a9358bf5924979` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.3201956719858572} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/d5cec7a156ce42160ec6` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.013195122010074556} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/d9b44fca70668c7eba68` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.002027156064286828} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/dc92214320695dc22d83` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.41407021996565163} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/df290b577c2ee6fc3e8a` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.06319025496486574} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/e2d08f1f3328359c1c34` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.0011239469749853015} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/e32ed3c63e7fd3998d8a` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.030017400975339115} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/e46c2dbca6784b41842c` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 11.215620680944994} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/e60853ab954f36cb8a3a` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.01886876509524882} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/e6de03cefa8d53e77943` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.3012845799094066} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/ed9306ec96c560eef08c` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.06545290793292224} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/efbca38b0ea8c662ef94` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.014666134025901556} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/f8818c4bf274e205dbea` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.08733175497036427} | s | a100 / float64 | selected reference; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 4635 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-30T16:34:25Z",
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
