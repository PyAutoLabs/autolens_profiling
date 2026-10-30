<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · alma_high

[Model index](index.md)

Exact setup ID: `datacube/delaunay/alma_high/49a10458f5e4b9a5ad09`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| channel_line_sigma_fraction | 0.5 | recorded |  |
| channel_scales | [1.0, 0.9982713919606268, 0.9931034757050423, 0.984549665976605, 0.972697965315163, 0.957669456847167, 0.9396162450475305, 0.9187188934430776, 0.8951834196045215, 0.869237917326145, 0.8411288833576084, 0.811117331244534, 0.7794747776490636, 0.746479186937245, 0.7124109578975517, 0.677549032331932, 0.6421671991447901, 0.6065306597126334, 0.5708929110610674, 0.5354929930529173, 0.5005531347669072, 0.46627682389041203, 0.4328473116228034, 0.40042655462623894, 0.36915458527363604, 0.3391492920920331, 0.3105065840975546, 0.28330090582633616, 0.257586064389264, 0.23339632586750292, 0.2107477358184072, 0.18963961752737007, 0.1700562018282703, 0.1519683436961909] | recorded |  |
| cube_definition | "decision D2: the 2b interferometer standard cell's dataset in every channel (shared uv_wavelengths, noise map, mask and lens model), channel c's visibilities scaled by a_c (Gaussian emission line, a_0 = 1); FactorGraphModel of AnalysisInterferometer(shared_preloads=True)" | recorded |  |
| host_load_avg_end | [90.11, 84.64, 93.35] | recorded |  |
| host_load_avg_start | [93.41, 90.63, 100.14] | recorded |  |
| image_pixels_masked | 61572 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| inversion_class | "InversionInterferometerSparse" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": null, "note": "Both gates are set for memo=off: the Settings flag and the AUTOARRAY_NNLS_WARM_START env kill-switch (autoarray/inversion/inversion/nnls_memo.py:63). settings_nnls_warm_start_memo is read back off the constructed Settings, so it records what took effect. The library default is true and both production projects leave it unset.", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| n_channels | 34 | recorded |  |
| n_repeats | 3 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.025 | recorded |  |
| preload | {"cache": "off", "channels_s": 60.49240633496083, "note": "W~ built once (shared uv coverage and noise map) and handed to every channel; each channel computes its own dirty image. One-off, never a per-evaluation step.", "operator_build_s": 15.891899110982195} | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| production_preset | {"adapt_image": "truth lensed source (lensed_source.fits), no S/N cap", "adapt_snr_cap": null, "dataset_type": "interferometer", "instance": "iid draw 0 of the preset sequence (seeded)", "instance_unit_vector": [0.499784489007424, 0.41402443684610074, 0.41327450881714656, 0.4200167345680767, 0.5789514047288555, 0.4702516857248635, 0.567334291975647, 0.47173759315010344, 0.5239626288064884, 0.5543072472409505], "instrument": "alma_high", "lens_light": null, "memo": "off", "mesh": {"edge_pixels": 30, "hilbert_pixels": 1250, "hilbert_weight_floor": 0.01, "hilbert_weight_power": 3.5, "kind": "delaunay", "vertices": 1280, "zeroed_pixels": 30}, "mesh_kind": "delaunay", "mge_gaussian_per_basis": 2, "mge_total_gaussians": 0, "over_sample_size_lp_rule": null, "over_sample_size_pixelization_rule": null, "positions_penalty": null, "preset": "interferometer_hst_source_pix_2_standard", "provenance": "subhalo_validation/scripts/imaging.py + scripts/pipelines/delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, image mesh :37-38,:125-129, regularization :151, adapt S/N cap pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator :252-253,:807, no positions penalty :384-388, MGE :273-278, threads hpc/batch_cpu/submit_source_pix:5,51-55 \u2014 likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "regularization": {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"}, "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "size_override": "likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and regularization at the standard size, no lens light, the truth lensed-source adapt image with no S/N cap: Hilbert 1250 + 30 zeroed edge points, adapt_split regularization; borrows the hst preset's mesh and regularization only", "variant": "production"} | recorded |  |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| regularization | {"free": true, "priors": {"inner_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "outer_coefficient": {"lower": 1e-06, "type": "LogUniform", "upper": 1000000.0}, "signal_scale": {"lower": 0.0, "type": "Uniform", "upper": 1.0}}, "scheme": "adapt_split"} | name |  |
| shared_preloads | true | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1280 | count |  |
| source_pixels_requested | 1250 | recorded |  |
| sparse_batch_size | 128 | recorded |  |
| sparse_operator_method | "nufft" | recorded |  |
| standard_size | true | recorded |  |
| transformer | "TransformerNUFFT" | name |  |
| transformer_chunk_size | 1000000 | recorded |  |
| transformer_chunk_size_preset | 1000000 | recorded |  |
| use_mixed_precision | false | recorded |  |
| visibilities | 5000000 | recorded |  |
| vmap_batch_size | 3 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>2 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/50edbf0eca33451be9bb` | full_pipeline_cube_vmap3.steady_per_call_s | {"full_pipeline_cube_vmap3.steady_per_call_s": 22.050956281005508} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/jit_phases/full_pipeline_cube_vmap3/steady_per_call_s` |
| `measurement/b0e9e7b0d8106b29044a` | full_pipeline_cube.steady_per_call_s | {"full_pipeline_cube.steady_per_call_s": 22.453960400656797} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/jit_phases/full_pipeline_cube/steady_per_call_s` |

</details>

### compile

<details><summary>4 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/8292569c3c4c69edc293` | full_pipeline_cube_vmap3.first_call_s | {"full_pipeline_cube_vmap3.first_call_s": 81.58554725290742} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/jit_phases/full_pipeline_cube_vmap3/first_call_s` |
| `measurement/99976112e3559bcb7e99` | full_pipeline_cube.compile_s | {"full_pipeline_cube.compile_s": 7.9310911250067875} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/jit_phases/full_pipeline_cube/compile_s` |
| `measurement/baa3d62107f0337eb4eb` | full_pipeline_cube.first_call_s | {"full_pipeline_cube.first_call_s": 22.307866697083227} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/jit_phases/full_pipeline_cube/first_call_s` |
| `measurement/bdcc13f7a01552197623` | full_pipeline_cube.lower_s | {"full_pipeline_cube.lower_s": 13.339399524964392} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/jit_phases/full_pipeline_cube/lower_s` |

</details>

### runtime

<details><summary>3 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1f81085989688bfc8041` | vmap.batch_time | {"vmap.batch_time": 66.15286884301652} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/vmap/batch_time` |
| `measurement/d1f72b8c474c98651f45` | cube_single_jit | {"cube_single_jit": 22.453960400656797} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/full_pipeline_cube_single_jit` |
| `measurement/f16248e68f2634814591` | vmap.per_call | {"vmap.per_call": 22.050956281005508} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/datacube/delaunay/alma_high/jax_cpu_sparse_fp64_wtilde/hpc_ral_cpu/runtime.json) `/vmap/per_call` |

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
    "measured_at": "2026-10-30T12:01:12Z",
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

- [likelihood_breakdown_numba](../../../../scripts/datacube/delaunay/likelihood_breakdown_numba.py)
- [parity_reference](../../../../scripts/datacube/delaunay/parity_reference.py)
- [likelihood_runtime_numba](../../../../scripts/datacube/delaunay/likelihood_runtime_numba.py)
- [likelihood_breakdown](../../../../scripts/datacube/delaunay/likelihood_breakdown.py)
- [inversion_setup_decompose](../../../../scripts/datacube/delaunay/inversion_setup_decompose.py)
- [likelihood_runtime](../../../../scripts/datacube/delaunay/likelihood_runtime.py)
- [likelihood_runtime_shared_preloads](../../../../scripts/datacube/delaunay/likelihood_runtime_shared_preloads.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
