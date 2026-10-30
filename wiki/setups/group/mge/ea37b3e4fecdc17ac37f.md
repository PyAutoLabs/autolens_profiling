<!-- generated: build_setup_wiki.py; do not edit -->
# mge · euclid

[Model index](index.md)

Exact setup ID: `group/mge/euclid/80626e7d6017a05fb020`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| dataset_type | "group" | recorded |  |
| fast | false | recorded |  |
| galaxy_group_cell | {"dataset": "autolens_workspace scripts/group/simulator.py (grid 250x250 at 0.1\", galaxies, exposure 300 s, sky 0.1) and modeling.py (7.5\" mask, radial-bin over-sampling on every galaxy); autolens_visualization true_models/group.py; PSF from instruments/imaging.py euclid (21x21, sigma 0.1\")", "dataset_type": "group", "extra_galaxies": 2, "instrument": "euclid", "lens_light": "workspace MGE: 20 linear Gaussians per main lens, 10 per extra galaxy (centre fixed)", "main_lenses": 1, "mask_radius_arcsec": 7.5, "noise_scaled_region": null, "ruling": "likelihood-lifecycle 2d draft (#473), decisions D1-D9 on the campaign page", "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "source": "MGE, 20 linear Gaussians (one basis)"} | recorded |  |
| image_pixels_masked | 17692 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance | "iid draw 0 of the seeded sequence" | recorded |  |
| instance_unit_vector | [0.499784489007424, 0.41402443684610074, 0.41327450881714656, 0.4200167345680767, 0.5789514047288555, 0.4702516857248635, 0.567334291975647, 0.47173759315010344, 0.5239626288064884, 0.5543072472409505, 0.5399346262699198, 0.5099305691842051, 0.5974626164364525, 0.4381789532715836, 0.5552315204832754, 0.5058238713539893, 0.45735024341537395, 0.5870995335054062, 0.509803788830344, 0.41707028300618887, 0.5825033050631517] | recorded |  |
| inversion_class | "InversionImagingMapping" | recorded |  |
| mask_radius_arcsec | 7.5 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": null, "note": "Both gates are set for memo=off: the Settings flag and the AUTOARRAY_NNLS_WARM_START env kill-switch (autoarray/inversion/inversion/nnls_memo.py:63). settings_nnls_warm_start_memo is read back off the constructed Settings, so it records what took effect. The library default is true and both production projects leave it unset.", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 71920 | recorded |  |
| oversampled_pixels | 71920 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.1 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| sparse_batch_size | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| standard_size | true | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| xla_cpu_workaround | null | recorded | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>14 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0309403f20ce73281a49` | prefix_5.steady_per_call_s | {"prefix_5.steady_per_call_s": 0.004547590599395334} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_5/steady_per_call_s` |
| `measurement/1fa4268c9e71989c9a62` | steps.Data vector D (mapping + PSF) | {"steps.Data vector D (mapping + PSF)": 0.0038914264994673433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Data vector D (mapping + PSF)` |
| `measurement/32cc330c4db74eb5af33` | prefix_4.steady_per_call_s | {"prefix_4.steady_per_call_s": 0.004710717604029924} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_4/steady_per_call_s` |
| `measurement/37130076e2da914da597` | prefix_3.steady_per_call_s | {"prefix_3.steady_per_call_s": 0.0042669438989833} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_3/steady_per_call_s` |
| `measurement/47e1b5fbde314e567238` | prefix_7.steady_per_call_s | {"prefix_7.steady_per_call_s": 0.007752838300075382} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_7/steady_per_call_s` |
| `measurement/5ddb517c95151fbf5399` | steps.Blurred image (non-linear light) | {"steps.Blurred image (non-linear light)": 6.923300679773092e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Blurred image (non-linear light)` |
| `measurement/8e75948337bc4b66aaaf` | prefix_6.steady_per_call_s | {"prefix_6.steady_per_call_s": 0.007330130401533097} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_6/steady_per_call_s` |
| `measurement/942c3f591e003540bc0a` | steps.Ray-trace grids (tracer) | {"steps.Ray-trace grids (tracer)": 0.0003062843927182257} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace grids (tracer)` |
| `measurement/9b67d0e54ba9d47fc487` | prefix_2.steady_per_call_s | {"prefix_2.steady_per_call_s": 0.00037551739951595663} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_2/steady_per_call_s` |
| `measurement/9c85d50c9074a5cd0ad7` | steps.Curvature matrix F | {"steps.Curvature matrix F": 0.0004437737050466241} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Curvature matrix F` |
| `measurement/aa949c8ccf44b54aee20` | steps.Mapped reconstruction, chi-squared & log evidence | {"steps.Mapped reconstruction, chi-squared & log evidence": 0.00042270789854228445} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Mapped reconstruction, chi-squared & log evidence` |
| `measurement/ab9c8060ee4dff8f489d` | steps.Reconstruction (NNLS) | {"steps.Reconstruction (NNLS)": 0.002782539802137763} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Reconstruction (NNLS)` |
| `measurement/bf595f0086194d28a5f2` | prefix_1.steady_per_call_s | {"prefix_1.steady_per_call_s": 0.0003062843927182257} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_1/steady_per_call_s` |
| `measurement/e8983811d2a7f8cb9972` | component_total | {"component_total": 0.007752838300075382} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/total_step_by_step` |

</details>

### compile

<details><summary>21 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0eb4225e8edf5b976761` | prefix_1.first_call_s | {"prefix_1.first_call_s": 0.0026151579804718494} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_1/first_call_s` |
| `measurement/25202d112d4ad928101e` | prefix_2.first_call_s | {"prefix_2.first_call_s": 0.025349426083266735} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_2/first_call_s` |
| `measurement/3e0e579f1f1f17d72a4c` | prefix_6.first_call_s | {"prefix_6.first_call_s": 0.24166834901552647} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_6/first_call_s` |
| `measurement/4ade47bfc8f5209e42a4` | prefix_3.lower_s | {"prefix_3.lower_s": 3.740488652023487} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_3/lower_s` |
| `measurement/5123b3a5ba1c0ef8fcac` | prefix_7.compile_s | {"prefix_7.compile_s": 12.268698311992921} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_7/compile_s` |
| `measurement/5878b6108a94436ef547` | prefix_6.lower_s | {"prefix_6.lower_s": 3.820560054969974} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_6/lower_s` |
| `measurement/5a178377151440f2ef88` | prefix_3.compile_s | {"prefix_3.compile_s": 12.080427492968738} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_3/compile_s` |
| `measurement/5bd5673facfe1350bc3d` | prefix_4.first_call_s | {"prefix_4.first_call_s": 0.2628917869878933} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_4/first_call_s` |
| `measurement/700c93c2a7545d347e7d` | prefix_1.lower_s | {"prefix_1.lower_s": 0.5889100659405813} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_1/lower_s` |
| `measurement/7fb21fbf15ebbd2ab7eb` | prefix_2.compile_s | {"prefix_2.compile_s": 0.6166001949459314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_2/compile_s` |
| `measurement/8ed1ad995787e36f3ca0` | prefix_4.lower_s | {"prefix_4.lower_s": 3.8628086999524385} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_4/lower_s` |
| `measurement/a9d8907cfe9949cd3adb` | prefix_7.first_call_s | {"prefix_7.first_call_s": 0.32143507502041757} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_7/first_call_s` |
| `measurement/aa745258a7325cd467ad` | prefix_5.first_call_s | {"prefix_5.first_call_s": 0.2431177319958806} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_5/first_call_s` |
| `measurement/aab7eca02353aa26473a` | prefix_5.lower_s | {"prefix_5.lower_s": 3.8366071430500597} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_5/lower_s` |
| `measurement/adc263efcfcf7f815570` | prefix_6.compile_s | {"prefix_6.compile_s": 11.25767875299789} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_6/compile_s` |
| `measurement/b125338b375270b4a9e9` | prefix_5.compile_s | {"prefix_5.compile_s": 10.769549221964553} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_5/compile_s` |
| `measurement/bac30df33efd512f0e14` | prefix_3.first_call_s | {"prefix_3.first_call_s": 0.24292822496499866} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_3/first_call_s` |
| `measurement/cf3744ae40a647258b81` | prefix_4.compile_s | {"prefix_4.compile_s": 10.870406424975954} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_4/compile_s` |
| `measurement/e548fc25940180e55777` | prefix_7.lower_s | {"prefix_7.lower_s": 3.8184739189455286} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_7/lower_s` |
| `measurement/ee081c9e23c35fffba09` | prefix_1.compile_s | {"prefix_1.compile_s": 0.4520909229759127} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_1/compile_s` |
| `measurement/f74fe2b5e9956e513c31` | prefix_2.lower_s | {"prefix_2.lower_s": 1.6836366539355367} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/group/mge/euclid/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/jit_phases/prefix_2/lower_s` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 589 MiB, 81920 MiB",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-30T11:02:39Z",
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

No canonical script explicitly registered for this model.

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
