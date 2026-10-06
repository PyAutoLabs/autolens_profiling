<!-- generated: build_setup_wiki.py; do not edit -->
# other · jwst

[Model index](index.md)

Exact setup ID: `imaging/other/jwst/a29fc82883ba71233165`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 42737 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 50312 | recorded |  |
| oversampled_pixels | 50312 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.03 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1521 | count |  |
| sweep_config | "local_cpu_fp64" | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch_size | 3 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/runtime/imaging/pixelization/jwst/comparison.json); JSON pointer: `/configs/local_cpu_fp64`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/116b49b17c0d62fff927` | single_call | {"single_call": 21.778462779999973} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/pixelization/jwst/comparison.json) `/configs/local_cpu_fp64/full_pipeline_per_call` |
| `measurement/36194cd1b5da915f3e71` | vmap.per_call | {"vmap.per_call": 28.32046649666663} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/pixelization/jwst/comparison.json) `/configs/local_cpu_fp64/vmap/per_call` |
| `measurement/d3dbd3a9b285fa0cea6b` | vmap.batch_time | {"vmap.batch_time": 84.96139948999989} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/pixelization/jwst/comparison.json) `/configs/local_cpu_fp64/vmap/batch_time` |
| `measurement/f393414e14ca395c4984` | single_jit_block | {"single_jit_block": 21.778462779999973} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/pixelization/jwst/comparison.json) `/configs/local_cpu_fp64/full_pipeline_single_jit` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "backend": "cpu",
      "device": "TFRT_CPU_0"
    },
    "library_version": "2026.7.6.649",
    "precision": "float64",
    "software": {
      "PyAutoLens": "2026.7.6.649"
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
