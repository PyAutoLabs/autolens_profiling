<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular_adapt_image_32x32 · sma

[Model index](index.md)

Exact setup ID: `interferometer/rectangular_adapt_image_32x32/sma/c9ef93446be717a141ba`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| formalism | "w_tilde_numba" | recorded |  |
| image_pixels_masked | 3852 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| jax_platforms | "cpu" | recorded |  |
| kernel | "direct_conv" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh | "rectangular_adapt_image_32x32" | recorded |  |
| mesh_shape | [32, 32] | recorded |  |
| mkl_num_threads | "1" | recorded |  |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| numba_num_threads | "1" | recorded |  |
| omp_num_threads | "1" | recorded |  |
| openblas_num_threads | "1" | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.1 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [256, 256] | recorded |  |
| rect_mesh | "bilinear" | recorded |  |
| regularization | "constant" | name |  |
| regularization_coefficient | 1.0 | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1024 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_jax | false | recorded |  |
| visibilities | 190 | recorded |  |
| xla_flags | "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0" | recorded |  |

## Evidence

[Original setup artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>15 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/036dcf1ab2e483d26987` | component_total | {"component_total": 0.18448875999893058} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/total_step_by_step` |
| `measurement/16b24e840099039fc4ea` | steps.log det H [Cholesky] | {"steps.log det H [Cholesky]": 0.015719380001246463} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/log det H [Cholesky]` |
| `measurement/1ad397d6ebc22ee9cafa` | steps.Reconstruction solve | {"steps.Reconstruction solve": 0.029371289999835425} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Reconstruction solve` |
| `measurement/2696c43433308bf3b2df` | steps.Data vector D [numba] | {"steps.Data vector D [numba]": 0.008142510000470792} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Data vector D [numba]` |
| `measurement/284c8731f7c7f9d4bfab` | steps.log det (F+H) [Cholesky] | {"steps.log det (F+H) [Cholesky]": 0.01685435999825131} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/log det (F+H) [Cholesky]` |
| `measurement/2eeb4a6e41bd0e4bcada` | steps.Fast chi^2 | {"steps.Fast chi^2": 0.00521632999952999} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Fast chi^2` |
| `measurement/3308b2ae1850876ee30d` | steps.Log evidence (figure of merit) | {"steps.Log evidence (figure of merit)": 4.6980000479379666e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Log evidence (figure of merit)` |
| `measurement/39d36968ee85a2db5287` | steps.Inversion build (trace+mesh+mapper) | {"steps.Inversion build (trace+mesh+mapper)": 0.003940740000689402} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Inversion build (trace+mesh+mapper)` |
| `measurement/4b2024470760656ca3d1` | steps.FitInterferometer construct | {"steps.FitInterferometer construct": 7.771000091452151e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/FitInterferometer construct` |
| `measurement/5fb94b3ad5493838c67e` | steps.Regularization matrix H (Constant) | {"steps.Regularization matrix H (Constant)": 0.002449809999961872} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Regularization matrix H (Constant)` |
| `measurement/694abbb63b7d7edace33` | steps.Curvature matrix F [residual: mirror + diag-add] | {"steps.Curvature matrix F [residual: mirror + diag-add]": 2.402999816695228e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Curvature matrix F [residual: mirror + diag-add]` |
| `measurement/74448df08e5e345ac7dc` | steps.F + H | {"steps.F + H": 0.001194799999211682} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/F + H` |
| `measurement/926f5d456626b84d2fff` | steps.Regularization term s'Hs | {"steps.Regularization term s'Hs": 0.0004962199986039195} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Regularization term s'Hs` |
| `measurement/bf2e074932d2a0cdb8c4` | steps.Mapper index/weight arrays | {"steps.Mapper index/weight arrays": 0.001209230000677053} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/Mapper index~1weight arrays` |
| `measurement/c916460870a8e66fe85d` | steps.F: mapper×mapper [numba preload scatter] | {"steps.F: mapper\u00d7mapper [numba preload scatter]": 0.09974537000089186} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/steps/F: mapper×mapper [numba preload scatter]` |

</details>

### runtime

<details><summary>1 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/db3066d48d41df35505f` | direct_call | {"direct_call": 0.1736454899997625} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/08/hostless_summaries/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_sma_v2026.8.17.1.json) `/direct_log_likelihood_function_per_call` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "backend": "cpu",
      "cpu_count": 8,
      "device": "cpu:0",
      "omp_num_threads": "1",
      "xla_flags": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0"
    },
    "library_version": "2026.8.17.1",
    "precision": null,
    "software": {
      "PyAutoLens": "2026.8.17.1"
    },
    "unknowns": {
      "precision": "Not recorded in this legacy evidence; no current default substituted."
    }
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
    "host": null,
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "host": "Not recorded in this legacy evidence; no current default substituted.",
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

No canonical script explicitly registered for this model.

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
