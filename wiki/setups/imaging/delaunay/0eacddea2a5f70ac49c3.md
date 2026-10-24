<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/163491f11f7c20d61b6f`.

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
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>18 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/04dd99941359c2d9c71e` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0008541680872440338} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/052d3780108eb23cb9c8` | steps.Data vector (D) | {"steps.Data vector (D)": 0.00032735608983784913} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Data vector (D)` |
| `measurement/1d106b9d26691ab6c3a0` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0002006817841902375} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/464eedd6dba9caa42adf` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00012256728950887918} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/6f43de0ed4125af9a525` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00015373211354017258} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/6f949eaf68f1182496cc` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.017932213038147893} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/71bd53ca67871ea633b9` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.04157442429568618} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/835b929e63af12784181` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.006686944056127686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/954c08ba6d6fbb30247f` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0022130414145067334} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Mapped recon + log evidence` |
| `measurement/b5865c689fc7dc4ee2a4` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.007555240982037503} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/b8a5d626579adce4c4f9` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0001707729883491993} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/c04e3628a8097c9cf3da` | interpolator_prefix | {"interpolator_prefix": 0.03147897180169821} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/interpolator_prefix_s` |
| `measurement/c8917ca1ce7ccad6ee48` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.03258446650579572} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/c8f783a771e229e14daf` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.010095452493987973} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/ca39e17bb6cc267ab20f` | component_total | {"component_total": 0.10112465219572186} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/total_step_by_step` |
| `measurement/e6d65ea11c12fa6121c6` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004901893110945821} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/f940ae3540d25df17e83` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.049500520317815246} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/ffe7ca52dcefc8eb1fa4` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.0008682969259098169} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/hst/jax_gpu_dense_fp64/hpc_a100/experiments/autotune4/breakdown.json) `/steps_vmap_per_call/Regularization matrix (H)` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 41527 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=4"
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
