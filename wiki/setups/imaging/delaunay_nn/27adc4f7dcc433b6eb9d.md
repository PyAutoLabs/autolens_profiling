<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/7b852b625618f6f9af4d`.

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
| vmap_batch | 64 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>14 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0049a41f876adcd68016` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.002220816700719297} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Mapped recon + log evidence` |
| `measurement/08760ef8a132f6e908f1` | steps.Data vector (D) | {"steps.Data vector (D)": 0.00030764220282435416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Data vector (D)` |
| `measurement/0a4303b3428c1db1e789` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0001475970959290862} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Ray-trace mesh grid` |
| `measurement/30b44c20082163d41388` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.025735294190235437} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Curvature matrix (F)` |
| `measurement/372d58baf31623d2dc3c` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0008283608127385378} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/4841a78d8705918a0472` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0001519351964816451} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Ray-trace data grid` |
| `measurement/677284838badc6ece0a2` | component_total | {"component_total": 0.25166065434459595} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/total_step_by_step` |
| `measurement/6b7551851ec82fd8d2a4` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.17926478509325533} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/regularization_matrix_prefix_s` |
| `measurement/6f2b584a6e2b73ce3698` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00011879049707204103} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Profile-subtracted image` |
| `measurement/721fb6db7db71923e2eb` | interpolator_prefix | {"interpolator_prefix": 0.124452282814309} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/interpolator_prefix_s` |
| `measurement/a3817a0c7112f2adcdcb` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00012075048871338368} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/a95946dc83b657119438` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.030630071088671683} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Regularized reconstruction` |
| `measurement/b68a850959436f23639a` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.13658689379226416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/c5bd238f7a22da592c62` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.05481250227894632} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.json) `/steps/Regularization matrix (H)` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 25111 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoLens": "2026.8.17.1"
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay_nn/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay_nn/likelihood_runtime.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
