<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/82ddfad90e5c99855dfd`.

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

[Original setup artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>18 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/115768b739d72c447fb9` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.00749125627335161} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps_vmap_per_call/Regularization matrix (H)` |
| `measurement/328a40017bcbdfed4690` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004811750981025398} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Curvature matrix (F)` |
| `measurement/342f87c46da3d0b041f0` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.002190651698037982} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Mapped recon + log evidence` |
| `measurement/355abcba4761994c1595` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.028212246601469815} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/regularization_matrix_prefix_s` |
| `measurement/3d9380f4b0383c59e504` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00015950507950037717} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Ray-trace mesh grid` |
| `measurement/4f306d16719812b66434` | component_total | {"component_total": 0.08071557465009391} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/total_step_by_step` |
| `measurement/5f611634cafb8caef400` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.02748184888623655} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/6a591ccf07c660c0f7fa` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00016866999212652444} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Ray-trace data grid` |
| `measurement/70245433625e05f4a2c1` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0003359609981998801} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Data vector (D)` |
| `measurement/91fe9efc51faba617642` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.008943521181936375} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/9915e29c4ba7f1869487` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.03073617338668555} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Regularized reconstruction` |
| `measurement/a190926f36c4ea8030ae` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.013768400601111352} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Regularization matrix (H)` |
| `measurement/ced2c7302c8afd85706a` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.016434777455287986} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/d4a70be742b01660ea86` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00012276130728423597} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Profile-subtracted image` |
| `measurement/db92119c6ddc65db4631` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00013269931077957152} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/ddc9cfa76e45ec9ae345` | interpolator_prefix | {"interpolator_prefix": 0.014443846000358463} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/interpolator_prefix_s` |
| `measurement/e006c7bbe770203dcdfc` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.017673922318499534} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/e92b7d29c3c5d22fefd7` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.000807152409106493} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay_nn/hst/jax_gpu_dense_fp64/hpc_a100/experiments/launch_latency/breakdown.json) `/steps/Blurred image (PSF convolution)` |

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
