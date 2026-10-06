<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/f39450ca6df2a9a6a80f`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1521 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch_size | 3 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/runtime/imaging/pixelization/pixelization_likelihood_summary_hst_v2026.8.17.1.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### runtime

<details><summary>3 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/3bfe8df49f83ba2a4c42` | vmap.batch_time | {"vmap.batch_time": 13.868701229999896} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/pixelization/pixelization_likelihood_summary_hst_v2026.8.17.1.json) `/vmap/batch_time` |
| `measurement/9c3fb58441a20b13b2c2` | vmap.per_call | {"vmap.per_call": 4.622900409999965} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/pixelization/pixelization_likelihood_summary_hst_v2026.8.17.1.json) `/vmap/per_call` |
| `measurement/e69297edf7a8c7ba9a29` | single_jit_block | {"single_jit_block": 3.1860993599999348} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/pixelization/pixelization_likelihood_summary_hst_v2026.8.17.1.json) `/full_pipeline_single_jit` |

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

- [hazards](../../../../scripts/imaging/rectangular/hazards.py)
- [likelihood_breakdown](../../../../scripts/imaging/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/rectangular/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/rectangular/likelihood_runtime_numba.py)
- [likelihood_runtime_numba_mge_mass](../../../../scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py)
- [parallel_scaling_numba](../../../../scripts/imaging/rectangular/parallel_scaling_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
