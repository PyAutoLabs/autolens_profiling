<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/d3aaac9a2e31b35fef5f`.

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
| transform | "laxmap_vag" | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json); JSON pointer: `/13`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>1 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/96a4a202448de40a4345` | steady_call | {"steady_call": 310.7151} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/13/steady_s` |

</details>

### compile

<details><summary>3 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0461b7c4d5a44598f08d` | first_call | {"first_call": 310.408} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/13/first_s` |
| `measurement/24f159bca5bfbfd70d56` | compile | {"compile": 1.157} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/13/compile_s` |
| `measurement/685f9632c595072bfae0` | trace | {"trace": 10.072} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../scripts/misc/jax_compile/results/local_cpu/pixelization.json) `/13/trace_s` |

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
    "cache_state": "warm",
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
    "host": "euclid-ral-compute-22",
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
