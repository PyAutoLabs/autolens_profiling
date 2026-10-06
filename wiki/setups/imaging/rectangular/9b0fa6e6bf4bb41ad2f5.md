<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/d2516e55763620175da0`.

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
| ndim | 12 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| transform | "vag" | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json); JSON pointer: `/1`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>1 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/79794a141e0449073d20` | steady_call | {"steady_call": 10.3919} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/1/steady_s` |

</details>

### compile

<details><summary>3 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/81a0cb650f5510722cb7` | first_call | {"first_call": 9.395} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/1/first_s` |
| `measurement/9d98b23213a0f142f651` | compile | {"compile": 30.729} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/1/compile_s` |
| `measurement/a70acc5711958bb6846d` | trace | {"trace": 7.323} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/1/trace_s` |

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
    "cache_state": "none",
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

- [hazards](../../../../scripts/imaging/rectangular/hazards.py)
- [likelihood_breakdown](../../../../scripts/imaging/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/rectangular/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/rectangular/likelihood_runtime_numba.py)
- [likelihood_runtime_numba_mge_mass](../../../../scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py)
- [parallel_scaling_numba](../../../../scripts/imaging/rectangular/parallel_scaling_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
