<!-- generated: build_setup_wiki.py; do not edit -->
# mge · ao

[Model index](index.md)

Exact setup ID: `imaging/mge/ao/ca51ea50e9e2c44cf087`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 384753 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| linear_gaussians | 0 | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 452472 | recorded |  |
| oversampled_pixels | 452472 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.01 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch_size | 3 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/runtime/imaging/mge/ao/mge_local_cpu_fp64.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### runtime

<details><summary>3 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/2f1141eaf23b86fbfa1d` | single_jit_block | {"single_jit_block": 3.108424629999354} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/mge/ao/mge_local_cpu_fp64.json) `/full_pipeline_single_jit` |
| `measurement/af7d283b1dd4a46a1173` | vmap.per_call | {"vmap.per_call": 4.922238283333475} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/mge/ao/mge_local_cpu_fp64.json) `/vmap/per_call` |
| `measurement/fc5668b58528e0935655` | vmap.batch_time | {"vmap.batch_time": 14.766714850000426} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/runtime/imaging/mge/ao/mge_local_cpu_fp64.json) `/vmap/batch_time` |

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

- [hazards_nnls_capture](../../../../scripts/imaging/mge/hazards_nnls_capture.py)
- [likelihood_breakdown](../../../../scripts/imaging/mge/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/imaging/mge/likelihood_runtime.py)
- [quick_update](../../../../scripts/imaging/mge/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
