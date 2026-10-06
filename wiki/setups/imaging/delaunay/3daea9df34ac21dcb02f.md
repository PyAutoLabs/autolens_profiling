<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay/hst/a7c5b64e93d234498312`.

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

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>18 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/11e0e718740a317447ce` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00015273820608854293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Profile-subtracted image` |
| `measurement/130f6e341334eb6e89bb` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.00022505368106067163} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Regularization matrix (H)` |
| `measurement/14b1269cd1e1bab1b8e9` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.013805706561834085} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/1e6e2f6991a5c68b545e` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.020429736003279686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/3a6f78e05a2b7f050533` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00013617018703371286} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/3b4f6ab11c8c7fffe979` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0003239401150494814} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Data vector (D)` |
| `measurement/3da4cadd6c5be9091b89` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.0051385009312070904} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/5e70599125f46fd5e4f4` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.005068995762849226} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/6475f46db91ebb3b4a0b` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00018229398410767316} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Ray-trace data grid` |
| `measurement/76ca589f375c387f9055` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.007227654289454222} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/regularization_matrix_prefix_s` |
| `measurement/86a375b3e5c9b419dfdb` | interpolator_prefix | {"interpolator_prefix": 0.00700260060839355} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/interpolator_prefix_s` |
| `measurement/86d6617d5c04e39af5e4` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004797589988447726} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Curvature matrix (F)` |
| `measurement/891ed4fff05fd744e943` | component_total | {"component_total": 0.06204461364541203} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/total_step_by_step` |
| `measurement/8f160aefda53345c153e` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.0022666632896289228} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Mapped recon + log evidence` |
| `measurement/ef1d90e36c535eae21aa` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.032512539788149296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Regularized reconstruction` |
| `measurement/f157ceb525fe205c937a` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.00017150109633803367} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Ray-trace mesh grid` |
| `measurement/f697dec7bf3857877292` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 6.95051683578642e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps_vmap_per_call/Regularization matrix (H)` |
| `measurement/fd518aa8c27d3c76ec8d` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.00084638730622828} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.json) `/steps/Blurred image (PSF convolution)` |

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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
