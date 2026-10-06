<!-- generated: build_setup_wiki.py; do not edit -->
# mge · hst

[Model index](index.md)

Exact setup ID: `multi_dataset/mge/hst/bc0565f0cabf46b896cc`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| batch_size | 4 | recorded |  |
| image_pixels_masked | null | count | Not recorded in this legacy evidence; no current default substituted. |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mixed_precision | false | recorded |  |
| n_batch | 16 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| ndim | 15 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| transform | "pyloop_vag" | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../scripts/misc/jax_compile/results/local_gpu_NVIDIA_GeForce_RTX_2060_with_Max-Q_Design/mge.json); JSON pointer: `/0`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>1 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/69e84a1628c6b7de2c6f` | steady_call | {"steady_call": 0.7287} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_gpu_NVIDIA_GeForce_RTX_2060_with_Max-Q_Design/mge.json) `/0/steady_s` |

</details>

### compile

<details><summary>3 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1dc82936d8734329c390` | compile | {"compile": 117.869} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_gpu_NVIDIA_GeForce_RTX_2060_with_Max-Q_Design/mge.json) `/0/compile_s` |
| `measurement/4d0ba340db1896817111` | first_call | {"first_call": 1.115} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_gpu_NVIDIA_GeForce_RTX_2060_with_Max-Q_Design/mge.json) `/0/first_s` |
| `measurement/5a97c15016f6d21f1d2f` | trace | {"trace": 102.548} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_gpu_NVIDIA_GeForce_RTX_2060_with_Max-Q_Design/mge.json) `/0/trace_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "gpu",
    "hardware_details": {},
    "library_version": null,
    "precision": "float64",
    "software": {
      "jax": "0.10.2"
    },
    "unknowns": {
      "library_version": "Not recorded in this legacy evidence; no current default substituted."
    }
  },
  "method": {
    "cache_state": "unknown",
    "repetitions": null,
    "statistic": null,
    "synchronization": null,
    "unknowns": {
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

No canonical script explicitly registered for this model.

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
