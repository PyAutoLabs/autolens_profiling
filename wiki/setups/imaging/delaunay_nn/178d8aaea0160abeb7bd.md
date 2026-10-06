<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/786c5eff4cbfa503500c`.

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

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>18 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/03aa91af75b009a8dbb4` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.01912680489331251} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/04de69e3d0a3254e2964` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0008371551986783743} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/058c96b3d3e2ea2e01bf` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00017858201172202826} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Ray-trace data grid` |
| `measurement/1267545cb2706056a206` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00015230320859700442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Profile-subtracted image` |
| `measurement/2b2a10fc7ce87baabf2c` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.002280585188418627} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Mapped recon + log evidence` |
| `measurement/41964962a792b888b16e` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.02721772431395948} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/regularization_matrix_prefix_s` |
| `measurement/4b1930fc1eb901143d3f` | interpolator_prefix | {"interpolator_prefix": 0.01701050759293139} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/interpolator_prefix_s` |
| `measurement/52da9e7f4f86572402e0` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004843506799079478} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Curvature matrix (F)` |
| `measurement/60c42efca9ac4d4821e4` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00016824111808091402} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Ray-trace mesh grid` |
| `measurement/64cdf27e6e8e8b2b79e3` | steps.Data vector (D) | {"steps.Data vector (D)": 0.00034742509014904497} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Data vector (D)` |
| `measurement/6ba65d562c90e36c5557` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.017888259218307213} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/73e50c3a86b29f5b94fc` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00014161630533635617} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/9a8f578ba17e2fe14024` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.030364763713441788} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/a3a74038612c180c405f` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.01252487623714842} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps_vmap_per_call/Regularization matrix (H)` |
| `measurement/f434d0122b31c7ad74e6` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.01020721672102809} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Regularization matrix (H)` |
| `measurement/f86f046175b3a363b3a9` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.006601928656164091} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/fa41679a1b2403b9072d` | component_total | {"component_total": 0.08038702416233719} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/total_step_by_step` |
| `measurement/fa587e0adca6ffef9dd1` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.030865628807805478} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_chunk2048.json) `/steps/Regularized reconstruction` |

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
