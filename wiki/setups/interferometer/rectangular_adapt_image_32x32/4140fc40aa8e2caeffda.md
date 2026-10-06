<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular_adapt_image_32x32 · alma

[Model index](index.md)

Exact setup ID: `interferometer/rectangular_adapt_image_32x32/alma/bc715730df444efa09f9`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| formalism | "w_tilde_numba" | recorded |  |
| image_pixels_masked | 15380 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| jax_platforms | "cpu" | recorded |  |
| kernel | "direct_conv" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh | "rectangular_adapt_image_32x32" | recorded |  |
| mesh_shape | [32, 32] | recorded |  |
| mkl_num_threads | "1" | recorded |  |
| n_repeats | 3 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| numba_num_threads | "1" | recorded |  |
| omp_num_threads | "1" | recorded |  |
| openblas_num_threads | "1" | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| real_space_shape | [800, 800] | recorded |  |
| rect_mesh | "bilinear" | recorded |  |
| regularization | "constant" | name |  |
| regularization_coefficient | 1.0 | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1024 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_jax | false | recorded |  |
| visibilities | 1000000 | recorded |  |
| xla_flags | "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>15 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0529363f614b060ebb53` | steps.F: mapper×mapper [numba preload scatter] | {"steps.F: mapper\u00d7mapper [numba preload scatter]": 1.4630309333345697} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/F: mapper×mapper [numba preload scatter]` |
| `measurement/384fa567b387f37c244b` | steps.Data vector D [numba] | {"steps.Data vector D [numba]": 0.05761279999812056} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Data vector D [numba]` |
| `measurement/3904306966a3c1c6f650` | steps.log det (F+H) [Cholesky] | {"steps.log det (F+H) [Cholesky]": 0.02161859999857067} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/log det (F+H) [Cholesky]` |
| `measurement/4fe4249d2c535ff3f16e` | steps.Reconstruction solve | {"steps.Reconstruction solve": 0.11139316666716088} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Reconstruction solve` |
| `measurement/53d11a706de6a090f0d6` | steps.Regularization term s'Hs | {"steps.Regularization term s'Hs": 0.0005044000014701547} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Regularization term s'Hs` |
| `measurement/7870e1a5e50cbdf8b23e` | steps.F + H | {"steps.F + H": 0.0015047000003202509} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/F + H` |
| `measurement/7ea4c046c86474a699b4` | component_total | {"component_total": 1.755069800002578} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/total_step_by_step` |
| `measurement/993a08ca1d652cde47bb` | steps.Regularization matrix H (Constant) | {"steps.Regularization matrix H (Constant)": 0.0032834999971479797} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Regularization matrix H (Constant)` |
| `measurement/aba4dd53df568c9ccd51` | steps.log det H [Cholesky] | {"steps.log det H [Cholesky]": 0.02118506666496008} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/log det H [Cholesky]` |
| `measurement/c0a5ca364543c6ac699e` | steps.Fast chi^2 | {"steps.Fast chi^2": 0.03223470000132996} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Fast chi^2` |
| `measurement/cc0ef05ecc12afd5f0fb` | steps.Inversion build (trace+mesh+mapper) | {"steps.Inversion build (trace+mesh+mapper)": 0.022316566668450832} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Inversion build (trace+mesh+mapper)` |
| `measurement/d7984f2373a0d4beb87f` | steps.Curvature matrix F [residual: mirror + diag-add] | {"steps.Curvature matrix F [residual: mirror + diag-add]": 2.706666903880735e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Curvature matrix F [residual: mirror + diag-add]` |
| `measurement/da9296ae1907beb90dcf` | steps.FitInterferometer construct | {"steps.FitInterferometer construct": 0.00011619999956261988} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/FitInterferometer construct` |
| `measurement/de2595db25a5c97b180c` | steps.Log evidence (figure of merit) | {"steps.Log evidence (figure of merit)": 0.014853266666856749} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Log evidence (figure of merit)` |
| `measurement/e7c63ed61b43a1d43ef7` | steps.Mapper index/weight arrays | {"steps.Mapper index/weight arrays": 0.005388833335018717} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/steps/Mapper index~1weight arrays` |

</details>

### runtime

<details><summary>1 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/5e5357eb05ce4f22b0d1` | direct_call | {"direct_call": 1.8367402000003494} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/pixelization_numba_direct_conv_breakdown_alma_v2026.8.17.1.json) `/direct_log_likelihood_function_per_call` |

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
