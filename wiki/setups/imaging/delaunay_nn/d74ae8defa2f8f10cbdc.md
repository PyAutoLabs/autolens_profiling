<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/fc52dc092c573894ff91`.

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
| regularization | {"coefficient": 1.0, "scheme": "constant_split"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>17 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/10d2949b82dab0b4ced2` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00016295909881591797} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/153d104b7b3f5fc59e52` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.007249501612386666} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/19b36b189af1cfc768b9` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0008487703045830131} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/31d7c3427f0747ed5c4f` | component_total | {"component_total": 0.06793114787433296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/total_step_by_step` |
| `measurement/328034b93f4f2a040b50` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.0001565270824357867} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/470f673e44dcbffd19c4` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.008857690813601948} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/639ad481b29df87206ad` | steps.Data vector (D) | {"steps.Data vector (D)": 0.00031771629583090546} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Data vector (D)` |
| `measurement/7de79958aaa039e501d9` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00013823620975017548} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/982c86c85fe9ec384ec7` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00014115821104496718} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/a3317de45a4f8080015a` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.015288548404350876} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/a42bda8a69b72d384d21` | interpolator_prefix | {"interpolator_prefix": 0.014521126518957317} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/interpolator_prefix_s` |
| `measurement/a8563e1710b76cccae37` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0007674218853935596} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/c6437c921f054c720b96` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.030882706004194915} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/e4825784ef444acac089` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004847314488142729} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/f14b50ea6f825f259b29` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.015196861825825182} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/f82fefa90627e3567132` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.0273892231984064} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/f8d845d9898ff606d184` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0022791150957345964} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/constant_split/breakdown.json) `/steps/Mapped recon + log evidence` |

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
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 41495 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
