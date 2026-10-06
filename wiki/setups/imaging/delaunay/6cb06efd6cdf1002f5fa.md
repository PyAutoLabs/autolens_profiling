<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/59e13d6a49ce12e0e03f`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| delaunay_vertices | 1500 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 17980 | recorded |  |
| oversampled_pixels | 17980 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>12 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/18b46f122ec091730e93` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.05636719999995421} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Regularization matrix (H)` |
| `measurement/1bf1b91d4d9fe1bc7997` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 2.962679700000001} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Regularized reconstruction` |
| `measurement/29c176d0abfa3e057396` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0011853399999949942} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/3b23e7406bcf88c20aae` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.002464989999998579} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Ray-trace data grid` |
| `measurement/3cffddbc0070da144a5c` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00020464000000401937} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Profile-subtracted image` |
| `measurement/67755ef4d4d5bb7e2f70` | component_total | {"component_total": 11.28559362999995} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/total_step_by_step` |
| `measurement/a4e78587043eff06f7e1` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 6.031315970000003} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Curvature matrix (F)` |
| `measurement/b5331cf02d2dd3d52d69` | steps.Data vector (D) | {"steps.Data vector (D)": 0.018807160000005752} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Data vector (D)` |
| `measurement/c1976a81575b071fdd82` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 3.0259999994086685e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/cc83b09e9d94ab5076a3` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0004549300000007861} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Ray-trace mesh grid` |
| `measurement/d92ce0c2a404f4cb507c` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 2.067606509999996} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/ef62e5cf9eb1e067b119` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.14447692999999617} | s | cpu / None | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_breakdown_hst_v2026.5.29.4.json) `/steps/Mapped recon + log evidence` |

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
    "library_version": "2026.5.29.4",
    "precision": null,
    "software": {
      "PyAutoLens": "2026.5.29.4"
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
