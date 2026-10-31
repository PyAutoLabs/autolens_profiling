<!-- generated: build_setup_wiki.py; do not edit -->
# mge · jwst

[Model index](index.md)

Exact setup ID: `imaging/mge/jwst/82bd551d946745b69247`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| image_pixels_masked | 42737 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| linear_gaussians | 0 | recorded |  |
| mask_radius_arcsec | 3.5 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 174752 | recorded |  |
| oversampled_pixels | 174752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.03 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>9 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/35113fc155399e6e7a79` | steps.Mapping matrix | {"steps.Mapping matrix": 0.021202400501351804} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Mapping matrix` |
| `measurement/4eacdfd406bcbe384613` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0004965586005710066} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Data vector (D)` |
| `measurement/4ece1c026bf0d61217f7` | steps.Ray-trace grids | {"steps.Ray-trace grids": 0.0026940373005345463} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace grids` |
| `measurement/83a0b81f79bb247ca119` | steps.Blurred mapping matrix | {"steps.Blurred mapping matrix": 0.04461472650291398} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Blurred mapping matrix` |
| `measurement/bfb754867330de651bd3` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.0038021015003323554} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/d1bdd54b4908d80e3ab3` | steps.Mapped reconstructed image | {"steps.Mapped reconstructed image": 0.00036075410898774864} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Mapped reconstructed image` |
| `measurement/d57cb9f7d4f6aac73ae2` | component_total | {"component_total": 0.07361966961761937} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/e238aaa087b699320c87` | steps.Chi-squared & log likelihood | {"steps.Chi-squared & log likelihood": 0.00013486840762197972} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Chi-squared & log likelihood` |
| `measurement/ef7de87cd9556ac76dd4` | steps.Reconstruction (NNLS) | {"steps.Reconstruction (NNLS)": 0.0003142226953059435} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/mge/jwst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Reconstruction (NNLS)` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": true,
      "cpu_count": 236,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-10-5",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "8860c50c0dc924ee591bfd0157cde7946a80a25e",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.11.2",
      "jaxlib.dependency_versions": "0.11.2",
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
    "host": "euclid-ral-compute-10-5",
    "measured_at": "2026-10-30T16:33:07Z",
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
