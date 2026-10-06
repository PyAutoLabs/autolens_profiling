<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/5f2d87257124c1d9c84f`.

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

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>12 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0bf6a5134ab1c163e6b3` | component_total | {"component_total": 0.09684501230021852} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/total_step_by_step` |
| `measurement/0cc48474443f1681850e` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.002271496799949091} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Mapped recon + log evidence` |
| `measurement/0d682c515a057abb1f03` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.001022397699944122} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/499b1f6e1004a29f0c23` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.0001505960999566014} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Profile-subtracted image` |
| `measurement/574deba5707fc9ad7eb9` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004787345900058426} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Curvature matrix (F)` |
| `measurement/5d352f32208d6ccf0e66` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.03255944599995928} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Regularized reconstruction` |
| `measurement/61f7f59fe962c84286c0` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.03988879589996941} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/8da83d406ece8b010a1a` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00015739800001028924} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Ray-trace mesh grid` |
| `measurement/98dfccc21f9826e1f7de` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 9.903330001179712e-05} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Ray-trace data grid` |
| `measurement/b731909cc04d7f2e23f8` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 8.48646000122244e-05} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/cad0955f30868eba0240` | steps.Data vector (D) | {"steps.Data vector (D)": 0.00032698599998184366} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Data vector (D)` |
| `measurement/f45894de14c4eadb5f61` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.01549665200036543} | s | a100 / mixed | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_mp.json) `/steps/Regularization matrix (H)` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "backend": "gpu",
      "cpu_count": 124,
      "device": "cuda:0",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 4653 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding"
    },
    "library_version": "2026.7.6.649",
    "precision": "mixed",
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
