<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_hilbert_1500 · alma

[Model index](index.md)

Exact setup ID: `interferometer/delaunay_hilbert_1500/alma/fa13b2107c798db20f11`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| delaunay_vertices | 1500 | recorded |  |
| formalism | "w_tilde_fft_jax" | recorded |  |
| hilbert_pixels | 1500 | recorded |  |
| image_pixels_masked | 15380 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| jax_platforms | "cpu" | recorded |  |
| kernel | "jax" | recorded |  |
| lens_light | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh | "delaunay_hilbert_1500" | recorded |  |
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
| regularization | "constant_split" | name |  |
| regularization_coefficient | 1.0 | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_jax | true | recorded |  |
| visibilities | 1000000 | recorded |  |
| xla_flags | "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 --xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>7 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0ed61379979dfdacf21b` | steps.Mapper sparse triplets [extent-flat] | {"steps.Mapper sparse triplets [extent-flat]": 0.022639833332505077} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/steps/Mapper sparse triplets [extent-flat]` |
| `measurement/258714b64a3833876d5b` | steps.Regularization matrix H (ConstantSplit) | {"steps.Regularization matrix H (ConstantSplit)": 0.09591739999935574} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/steps/Regularization matrix H (ConstantSplit)` |
| `measurement/2a7689fc9433944b21c1` | component_total | {"component_total": 3.28098303333051} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/total_step_by_step` |
| `measurement/37f1eccd6d6c36a4c90f` | steps.FitInterferometer construct | {"steps.FitInterferometer construct": 9.876666687584172e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/steps/FitInterferometer construct` |
| `measurement/a0ea4a27ceec2c95634b` | steps.Data vector D [sparse-op] | {"steps.Data vector D [sparse-op]": 0.04735493333525179} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/steps/Data vector D [sparse-op]` |
| `measurement/fd7fe063384847e7b595` | steps.Inversion build (trace+Delaunay+mapper) | {"steps.Inversion build (trace+Delaunay+mapper)": 0.020249933331797365} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/steps/Inversion build (trace+Delaunay+mapper)` |
| `measurement/fe770843cc8271a761ae` | steps.F: mapper×mapper [sparse-op FFT] | {"steps.F: mapper\u00d7mapper [sparse-op FFT]": 3.0947221666647238} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/steps/F: mapper×mapper [sparse-op FFT]` |

</details>

### runtime

<details><summary>1 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/7677b8514e309bca7bf1` | direct_call | {"direct_call": 8.104913233333113} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/interferometer/delaunay_numba_jax_breakdown_alma_v2026.8.17.1.json) `/direct_log_likelihood_function_per_call` |

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
