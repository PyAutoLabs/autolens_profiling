<!-- generated: build_setup_wiki.py; do not edit -->
# mge · hst

[Model index](index.md)

Exact setup ID: `imaging/mge/hst/a1e86b3edc04064347b0`.

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
| transform | "vag" | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../scripts/misc/jax_compile/results/local_cpu/mge.json); JSON pointer: `/14`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>1 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/fb9b4cabb3e337d999fc` | steady_call | {"steady_call": 0.3375} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/mge.json) `/14/steady_s` |

</details>

### compile

<details><summary>3 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1fd754b2fdfb57a8b683` | compile | {"compile": 229.441} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/mge.json) `/14/compile_s` |
| `measurement/83e3d5fa17440effc23f` | trace | {"trace": 22.109} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/mge.json) `/14/trace_s` |
| `measurement/d9c9db8a3372b9466114` | first_call | {"first_call": 0.881} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/mge.json) `/14/first_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
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
    "cache_state": "cold",
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

- [hazards_nnls_capture](../../../../scripts/imaging/mge/hazards_nnls_capture.py)
- [likelihood_breakdown](../../../../scripts/imaging/mge/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/imaging/mge/likelihood_runtime.py)
- [quick_update](../../../../scripts/imaging/mge/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
