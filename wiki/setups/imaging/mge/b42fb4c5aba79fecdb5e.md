<!-- generated: build_setup_wiki.py; do not edit -->
# mge · hst

[Model index](index.md)

Exact setup ID: `imaging/mge/hst/05aca978db135516bfcb`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| free_parameters | 4 | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_light | "mge_30_lmp_linear_fixed_geometry" | recorded |  |
| lens_mass | "mge_basis + NFWSph + ExternalShear" | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh_shape | [28, 28] | recorded |  |
| n_repeats | 3 | recorded |  |
| n_steady | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| omp_num_threads | "1" | recorded |  |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 784 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_jax | true | recorded |  |
| vmap_batch_size | 3 | recorded |  |
| xla_flags | "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/runtime/imaging/mge_mass_jax/mge_mass_jax_likelihood_summary_hst_v2026.8.17.1.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### runtime

<details><summary>1 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/9afa2cce95f95adf92eb` | single_jit_block | {"single_jit_block": 3.35121137999995} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/mge_mass_jax/mge_mass_jax_likelihood_summary_hst_v2026.8.17.1.json) `/full_pipeline_single_jit` |

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
      "cpu_count": 8,
      "device": "cpu:0",
      "hostname": "DESKTOP-H143S82",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
    "host": "DESKTOP-H143S82",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
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

- [hazards_nnls_capture](../../../../scripts/imaging/mge/hazards_nnls_capture.py)
- [likelihood_breakdown](../../../../scripts/imaging/mge/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/imaging/mge/likelihood_runtime.py)
- [quick_update](../../../../scripts/imaging/mge/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
