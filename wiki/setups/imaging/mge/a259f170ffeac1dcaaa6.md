<!-- generated: build_setup_wiki.py; do not edit -->
# mge · hst

[Model index](index.md)

Exact setup ID: `imaging/mge/hst/0644314a8f3ae8363f94`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| linear_gaussians | 0 | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>9 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/191c4580f41445499f1b` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.0003957606997573748} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/30612c4ed1a676fa44bf` | steps.Chi-squared & log likelihood | {"steps.Chi-squared & log likelihood": 0.00013455549778882415} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Chi-squared & log likelihood` |
| `measurement/3b19cf9a628e4c8f0816` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.00017925009888131171} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Ray-trace grids` |
| `measurement/6015cb9507b377796dad` | steps.Reconstruction (NNLS) | {"steps.Reconstruction (NNLS)": 0.003026946997852065} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Reconstruction (NNLS)` |
| `measurement/98fb002843bd4173cd44` | steps.Data vector (D) | {"steps.Data vector (D)": 0.00013244540023151785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Data vector (D)` |
| `measurement/9ccd4afec3b235d47887` | steps.Blurred mapping matrix | {"steps.Blurred mapping matrix": 0.0036562363005941735} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Blurred mapping matrix` |
| `measurement/a39c04e83c05dbce86c0` | steps.Mapped reconstructed image | {"steps.Mapped reconstructed image": 0.00014079450047574938} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Mapped reconstructed image` |
| `measurement/bb910461b542bce1e9d3` | component_total | {"component_total": 0.008957591495709493} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/ec221d1d76651085cf1d` | steps.Mapping matrix | {"steps.Mapping matrix": 0.0012916020001284777} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/hst/jax_gpu_dense_fp64/hpc_a100/breakdown.json) `/steps/Mapping matrix` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 559 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "d99ec2b30224ac021e39628b9904d01c44f33b29",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "d3a7721294652f81ec726b92931ab610ed27e3ac",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "e00b5e7d554d6958a6c8821edef6afdf01188e6d",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "f791ac77d136d664346fff21cc36079f8f9b1196",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "f116f70077e442db19a2a152f5ee78d57294c504",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "86c287d1fc90f7595b2106dbb6b55102c4d7c800",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
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
    "has_provenance": true,
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-25T14:38:38Z",
    "qualified": false,
    "unknowns": {}
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
