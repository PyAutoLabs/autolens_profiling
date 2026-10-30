<!-- generated: build_setup_wiki.py; do not edit -->
# source_plane · simple

[Model index](index.md)

Exact setup ID: `cluster/source_plane/simple/a7eec237667c43d921a8`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| headline_arm | "solved" | recorded |  |
| image_pixels_masked | null | count | Not recorded in this legacy evidence; no current default substituted. |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| jax_enable_x64 | true | recorded |  |
| likelihood | "source_plane (FitPositionsSourceSolved)" | recorded |  |
| n_images_total | 6 | recorded |  |
| n_mass_profiles | 13 | recorded |  |
| n_planes | 3 | recorded |  |
| n_systems | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| standard_size | true | recorded |  |
| sub_row_arm | "plain" | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |

## Evidence

[Original setup artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>17 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/126a048f741990b7a587` | steps.P3.0 plain: weighted chi2 about Point centre point_0 | {"steps.P3.0 plain: weighted chi2 about Point centre point_0": 2.122779842466116e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/P3.0 plain: weighted chi2 about Point centre point_0` |
| `measurement/2add8cd2a142385b6990` | steps.P3.1 plain: weighted chi2 about Point centre point_1 | {"steps.P3.1 plain: weighted chi2 about Point centre point_1": 2.6701693423092364e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/P3.1 plain: weighted chi2 about Point centre point_1` |
| `measurement/3c5327f86f62c705c281` | steps.6.point_1 FitPositionsSourceSolved (fit total, analytic β*) | {"steps.6.point_1 FitPositionsSourceSolved (fit total, analytic \u03b2*)": 0.044589965022169054} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/6.point_1 FitPositionsSourceSolved (fit total, analytic β*)` |
| `measurement/45b6fd59dfe1e7be73cf` | steps.5 log-likelihood assembly (+ marginalisation) | {"steps.5 log-likelihood assembly (+ marginalisation)": 1.5279802028089762e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/5 log-likelihood assembly (+ marginalisation)` |
| `measurement/48a6e13c5cd42ec2c46d` | component_total | {"component_total": 0.0016673987847752867} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/total_step_by_step` |
| `measurement/4cd4f1d280a846248240` | steps.1.0 ray-trace point_0 (3 img) | {"steps.1.0 ray-trace point_0 (3 img)": 0.00035385770024731755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/1.0 ray-trace point_0 (3 img)` |
| `measurement/62f7a7e9b4697dbc3277` | steps.4.0 weighted chi2 about beta* point_0 | {"steps.4.0 weighted chi2 about beta* point_0": 1.9609695300459862e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/4.0 weighted chi2 about beta* point_0` |
| `measurement/64c73c64fe6dfe4472bd` | steps.3.0 solved source centre beta* point_0 | {"steps.3.0 solved source centre beta* point_0": 2.7210800908505917e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/3.0 solved source centre beta* point_0` |
| `measurement/7ac46fb1c955ea76aaef` | steps.7.point_0 FitPositionsSource (fit total) | {"steps.7.point_0 FitPositionsSource (fit total)": 0.01211272506043315} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/7.point_0 FitPositionsSource (fit total)` |
| `measurement/84c027b2400e959f7ec7` | steps.6.point_0 FitPositionsSourceSolved (fit total, analytic β*) | {"steps.6.point_0 FitPositionsSourceSolved (fit total, analytic \u03b2*)": 0.04946443694643676} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/6.point_0 FitPositionsSourceSolved (fit total, analytic β*)` |
| `measurement/8ffa2fbba7c21c2f3460` | steps.1.1 ray-trace point_1 (3 img) | {"steps.1.1 ray-trace point_1 (3 img)": 0.0003594005946069956} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/1.1 ray-trace point_1 (3 img)` |
| `measurement/c267a3e5cca6dd2dea14` | steps.3.1 solved source centre beta* point_1 | {"steps.3.1 solved source centre beta* point_1": 3.0416599474847317e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/3.1 solved source centre beta* point_1` |
| `measurement/cb8c56dc4deda4369174` | steps.2.1 lensing jacobian (hessian) point_1 | {"steps.2.1 lensing jacobian (hessian) point_1": 0.00048230619868263604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/2.1 lensing jacobian (hessian) point_1` |
| `measurement/d5a7dfc976b31cb61313` | steps.2.0 lensing jacobian (hessian) point_0 | {"steps.2.0 lensing jacobian (hessian) point_0": 0.00036477959947660564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/2.0 lensing jacobian (hessian) point_0` |
| `measurement/d8466992efd7b2138f31` | steps.P4 plain: log-likelihood assembly | {"steps.P4 plain: log-likelihood assembly": 1.4441902749240398e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/P4 plain: log-likelihood assembly` |
| `measurement/e63bd11771a3eb206f0f` | steps.7.point_1 FitPositionsSource (fit total) | {"steps.7.point_1 FitPositionsSource (fit total)": 0.009735703002661467} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/7.point_1 FitPositionsSource (fit total)` |
| `measurement/f9223f26640e25fa689e` | steps.4.1 weighted chi2 about beta* point_1 | {"steps.4.1 weighted chi2 about beta* point_1": 1.4537794049829244e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/cluster/source_plane/simple/jax_cpu_na_fp64/hpc_a100/breakdown.json) `/steps/4.1 weighted chi2 about beta* point_1` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cpu:0",
      "hostname": "euclid-ral-gpu-1",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling": "830314e9bcafafd469f0985f9d1c9629cca830ff",
      "autolens_profiling.library_revisions": "830314e9bcafafd469f0985f9d1c9629cca830ff",
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
    "host": "euclid-ral-gpu-1",
    "measured_at": "2026-10-30T10:58:15Z",
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

- [likelihood_breakdown](../../../../scripts/cluster/source_plane/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/cluster/source_plane/likelihood_runtime.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
