<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `multi_dataset/delaunay/hst/e49030d530732b5fa494`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| adapt_snr_cap | 3.0 | recorded |  |
| dataset_scales | [1.0, 0.5] | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_light | "mge_60_linear_2_basis" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "off" | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": "0", "note": "Both gates are set for memo=off: the Settings flag and the AUTOARRAY_NNLS_WARM_START env kill-switch (autoarray/inversion/inversion/nnls_memo.py:63). settings_nnls_warm_start_memo is read back off the constructed Settings, so it records what took effect. The library default is true and both production projects leave it unset.", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| mesh | {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30} | recorded |  |
| mesh_class | "delaunay" | recorded |  |
| mesh_kind | "delaunay" | recorded |  |
| mesh_shape | [1280] | recorded |  |
| mge_gaussian_per_basis | 2 | recorded |  |
| mge_total_gaussians | 30 | recorded |  |
| multi_dataset_definition | "decision D6: 2a's hst imaging standard cell in every dataset (shared mask, PSF, over-sampling, adapt image, mesh and lens model), dataset e's data and noise map times s_e (same S/N); FactorGraphModel of AnalysisImaging (shared_preloads=False)" | recorded |  |
| n_datasets | 2 | recorded |  |
| n_instances | 20 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| omp_num_threads | "1" | recorded |  |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sample_size_pixelization_rule | {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2} | recorded |  |
| over_sampled_pixels | 61600 | recorded |  |
| oversampled_pixels | 61600 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| positions_penalty | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| preset | "hst_source_pix_2_standard" | recorded |  |
| production_data | {"enabled": true, "lens_light_source": "truth"} | recorded |  |
| production_data_record | {"adapt_image": "subtracted S/N (truth tracer), floored", "adapt_minimum_percent": 0.01, "lens_light_source": "truth", "mesh_adapt_snr_cap": 3.0, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"pixels_above": 2068, "pixels_below": 13293, "source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2, "threshold_map": "uncapped subtracted S/N (production)"}, "over_sampling_preset": "hst_source_pix_2_standard"} | recorded |  |
| production_preset | {"adapt_snr_cap": 3.0, "instance": "iid draw 0 of the preset sequence (seeded)", "lens_light": "mge_60_linear_2_basis", "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 30, "over_sample_size_lp_rule": {"centre": [0.0, 0.0], "radial_list": [0.1, 0.3], "sub_size_list": [4, 2, 2]}, "over_sample_size_pixelization_rule": {"source_snr_cut": 3.0, "sub_size_above": 4, "sub_size_below": 2}, "positions_penalty": null, "preset": "hst_source_pix_2_standard", "production_data": {"enabled": true, "lens_light_source": "truth"}, "production_source": {"adapt_image": "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus the other galaxies' model images, over the noise map), floored at adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max", "mesh_and_regularization_adapt_image": "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from (S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and rectangular_adapt.py:109", "model_stand_in": "no previous-stage fit exists in a profiling cell: the dataset's truth tracer (tracer.json) stands in for the maximum-likelihood model whose lens light is subtracted", "over_sampling": "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184", "rectangular_mesh": "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight priors config/priors/mesh/rectangular.yaml:43-61"}, "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut", "variant": "production"} | recorded |  |
| production_source | {"adapt_image": "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus the other galaxies' model images, over the noise map), floored at adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max", "mesh_and_regularization_adapt_image": "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from (S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and rectangular_adapt.py:109", "model_stand_in": "no previous-stage fit exists in a profiling cell: the dataset's truth tracer (tracer.json) stands in for the maximum-likelihood model whose lens light is subtracted", "over_sampling": "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184", "rectangular_mesh": "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight priors config/priors/mesh/rectangular.yaml:43-61"} | recorded |  |
| provenance | "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut" | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| sequence | {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4} | recorded |  |
| shared_preloads | false | recorded |  |
| size_override | "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped over-sampling cut" | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| standard_size | true | recorded |  |
| thread_env | {"n_threads": 1, "note": "Set by _production_config.pin_thread_env before numpy import; mirrors the production SLURM submit scripts.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": "1", "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_jax | false | recorded |  |
| variant | "production" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>25 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/05648aa5420b4bd16ea5` | steps.Mapper sparse triplets | {"steps.Mapper sparse triplets": 0.036038672922586557} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Mapper sparse triplets` |
| `measurement/0de05c4b3fea2af4bc4e` | steps.MGE operated mapping matrix [residual: dict assembly] | {"steps.MGE operated mapping matrix [residual: dict assembly]": 0.007279872377694119} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/MGE operated mapping matrix [residual: dict assembly]` |
| `measurement/14cf8eac553d03d26c46` | component_total | {"component_total": 2.6690022670227336} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/18a34eaad293630d341f` | steps.Log evidence (figure of merit) | {"steps.Log evidence (figure of merit)": 0.06702338704053545} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Log evidence (figure of merit)` |
| `measurement/1939a8d65bd6c491eaf6` | steps.MGE: batched PSF convolution [real-space np] | {"steps.MGE: batched PSF convolution [real-space np]": 0.06718153898691526} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/MGE: batched PSF convolution [real-space np]` |
| `measurement/1bcc6550f80b58656d79` | steps.log det (F+H) [Cholesky] | {"steps.log det (F+H) [Cholesky]": 0.04623549857205944} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/log det (F+H) [Cholesky]` |
| `measurement/37c2d6f6368367a38c6f` | steps.Blurred image (FFT convolve) | {"steps.Blurred image (FFT convolve)": 0.012258080540050287} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Blurred image (FFT convolve)` |
| `measurement/3f6ddb5d517de1500c7c` | steps.Mapped reconstruction [numba+FFT] | {"steps.Mapped reconstruction [numba+FFT]": 0.005012682944652624} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Mapped reconstruction [numba+FFT]` |
| `measurement/4b681809a3ae6c684568` | steps.Curvature matrix F [residual: mirror + diag-add] | {"steps.Curvature matrix F [residual: mirror + diag-add]": 0.001104469585698098} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Curvature matrix F [residual: mirror + diag-add]` |
| `measurement/50fcc4c3357cf13db6a6` | steps.Regularization term s'Hs | {"steps.Regularization term s'Hs": 0.0007367070065811276} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Regularization term s'Hs` |
| `measurement/5838e88ab76648e1a900` | steps.Profile subtracted image | {"steps.Profile subtracted image": 6.033718818798661e-05} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Profile subtracted image` |
| `measurement/5d8ee503fbf041d228bc` | steps.Reconstruction solve (BLAS) | {"steps.Reconstruction solve (BLAS)": 0.7218084729538532} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Reconstruction solve (BLAS)` |
| `measurement/69c293ed2aab6ee9effa` | steps.PSF-weighted data [numba] | {"steps.PSF-weighted data [numba]": 0.008107158057100605} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/PSF-weighted data [numba]` |
| `measurement/6e866a9b2f2aa98f310e` | steps.F: mapper×mapper block [sparse-op] | {"steps.F: mapper\u00d7mapper block [sparse-op]": 0.14630937537003774} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/F: mapper×mapper block [sparse-op]` |
| `measurement/7f723f1f62ff11a25143` | steps.F + H | {"steps.F + H": 0.004389630208606832} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/F + H` |
| `measurement/8c540f757ff3b49bec0d` | steps.F: mapper×linear-func block [FFT conv + scatter] | {"steps.F: mapper\u00d7linear-func block [FFT conv + scatter]": 0.07056618102069478} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/F: mapper×linear-func block [FFT conv + scatter]` |
| `measurement/93ecd5fcfa8b879b2411` | steps.MGE: image_2d_from(blurring_grid) x60 [numpy] | {"steps.MGE: image_2d_from(blurring_grid) x60 [numpy]": 0.00963456294266507} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/MGE: image_2d_from(blurring_grid) x60 [numpy]` |
| `measurement/9b2939f13f5398b1b80f` | steps.log det H [Cholesky] | {"steps.log det H [Cholesky]": 0.020798981364350766} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/log det H [Cholesky]` |
| `measurement/a1df5daf2ab54106840b` | steps.FitImaging construct | {"steps.FitImaging construct": 0.00010987168207066134} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/FitImaging construct` |
| `measurement/b8adab423f74a2098aef` | steps.Data vector D [numba] | {"steps.Data vector D [numba]": 0.008005812698684167} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Data vector D [numba]` |
| `measurement/c32f2141730ff9b2d3e2` | steps.F: linear-func×linear-func block [BLAS] | {"steps.F: linear-func\u00d7linear-func block [BLAS]": 0.010183702521317173} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/F: linear-func×linear-func block [BLAS]` |
| `measurement/d497bac68833bb7697a2` | steps.MGE: image_2d_from(grid) x60 [numpy] | {"steps.MGE: image_2d_from(grid) x60 [numpy]": 0.04441789174597943} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/MGE: image_2d_from(grid) x60 [numpy]` |
| `measurement/e123b1260d27b5bd5050` | steps.Inversion build (trace+Delaunay+mapper) | {"steps.Inversion build (trace+Delaunay+mapper)": 0.03493113536387682} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Inversion build (trace+Delaunay+mapper)` |
| `measurement/e1326980756c29e22813` | steps.Regularization matrix H | {"steps.Regularization matrix H": 0.007548034991486929} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Regularization matrix H` |
| `measurement/fed531dcdacd83e8208b` | steps.Model data + chi^2 | {"steps.Model data + chi^2": 0.004759075425681658} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/steps/Model data + chi^2` |

</details>

### runtime

<details><summary>1 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/d228421be965a3754d4c` | direct_call | {"direct_call": 1.2117082115146331} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_dataset/delaunay/hst/numba_cpu_sparse_fp64/hpc_ral_cpu/breakdown.json) `/direct_log_likelihood_function_per_call` |

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
      "autolens_profiling.library_revisions": "830314e9bcafafd469f0985f9d1c9629cca830ff",
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
    "measured_at": "2026-10-30T11:51:22Z",
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

- [likelihood_breakdown](../../../../scripts/multi_dataset/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/multi_dataset/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/multi_dataset/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/multi_dataset/delaunay/likelihood_runtime_numba.py)
- [likelihood_runtime_shared_preloads](../../../../scripts/multi_dataset/delaunay/likelihood_runtime_shared_preloads.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
