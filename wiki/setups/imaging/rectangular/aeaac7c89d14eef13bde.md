<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/755d9a67bb3abb171e70`.

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
| over_sampled_pixels | 17980 | recorded |  |
| oversampled_pixels | 17980 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1521 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>12 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/07cea828b894100d4534` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.1941322399998171} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Mapped recon + log evidence` |
| `measurement/2c961392bd9c9f32f4c2` | component_total | {"component_total": 8.648970759999793} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/total_step_by_step` |
| `measurement/524f3ec9d791c0e7bfee` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.002827370000159135} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Ray-trace grids` |
| `measurement/649479b45dcfdbd4de44` | steps.Inversion setup (steps 4-8 combined) | {"steps.Inversion setup (steps 4-8 combined)": 2.5466561199998976} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Inversion setup (steps 4-8 combined)` |
| `measurement/6db48c37746c7c4e7101` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 1.705186699999831} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Regularized reconstruction` |
| `measurement/742ee3e54424d8c3f1b1` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00016214999996009283} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Profile-subtracted image` |
| `measurement/7c6b303b8ac9ad214b3b` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0029215799997473367} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Regularization matrix (H)` |
| `measurement/8800ebad6c61e2570dd7` | steps.Data vector (D) | {"steps.Data vector (D)": 0.022497959999964224} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Data vector (D)` |
| `measurement/a498cced6cb836621d54` | steps.Overlay grid (source pixel centres) | {"steps.Overlay grid (source pixel centres)": 0.0003320800002256874} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Overlay grid (source pixel centres)` |
| `measurement/bd5704a45edf7ea800b2` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 6.536000000778586e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/d64d1101cd5e8ea4786c` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 4.172744299999977} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Curvature matrix (F)` |
| `measurement/db89bd36049e973de107` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.001444900000205962} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/pixelization_breakdown_hst_v2026.7.6.649.json) `/steps/Blurred image (PSF convolution)` |

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
      "device": "TFRT_CPU_0",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding"
    },
    "library_version": "2026.7.6.649",
    "precision": null,
    "software": {
      "PyAutoLens": "2026.7.6.649"
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
