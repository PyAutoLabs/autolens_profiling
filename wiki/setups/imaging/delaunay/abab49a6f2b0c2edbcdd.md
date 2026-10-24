<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/ffdb16fa7b1b0e808a69`.

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

[Original setup artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>12 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/295c4ca44e2f2c8e5885` | steps.Data vector (D) | {"steps.Data vector (D)": 0.05481417000009969} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Data vector (D)` |
| `measurement/59d357e5b1cfe7ed748e` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.002923889999874518} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Ray-trace data grid` |
| `measurement/908d86ca20d7056b91bc` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0024762500001088484} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/92508bafd0aff8471d05` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.13277568000012252} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Mapped recon + log evidence` |
| `measurement/a5e3691c361f74523c74` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0470290000012028} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Regularization matrix (H)` |
| `measurement/a5fcc84201d1969068f2` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 1.683930590000091} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Regularized reconstruction` |
| `measurement/b3364336e5d80597e2f4` | component_total | {"component_total": 10.07246866000096} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/total_step_by_step` |
| `measurement/b4bba8b060ee6024a159` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 4.233989429999928} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Curvature matrix (F)` |
| `measurement/cadde9e0a25b3031275c` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00022619999981543515} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Profile-subtracted image` |
| `measurement/debc7cb00fd3f6c8f8d8` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.000754400000005262} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Ray-trace mesh grid` |
| `measurement/f12038c190a196ca9bfc` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 4.809999991266523e-05} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/fb7d88b5242be3999227` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 3.913500949999798} | s | cpu / None | archive support; unreviewed | [artifact](../../../../archive/2026/07/hostless_summaries/breakdown/imaging/delaunay_breakdown_hst_v2026.7.6.649.json) `/steps/Inversion setup (steps 5-8 combined)` |

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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
