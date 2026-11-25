<!-- generated: build_setup_wiki.py; do not edit -->
# source_plane · simple

[Model index](index.md)

Exact setup ID: `experiments/source_plane/simple/2bbe5ccd829aa50222e6`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| free_parameters | 13 | recorded |  |
| headline_arm | "solved" | recorded |  |
| image_pixels_masked | null | count | Not recorded in this legacy evidence; no current default substituted. |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance | "prior medians of tight Gaussians on the truth (cluster_standard.model)" | recorded |  |
| jax_enable_x64 | true | recorded |  |
| likelihood | "source_plane (FitPositionsSourceSolved)" | recorded |  |
| n_images_total | 6 | recorded |  |
| n_mass_profiles | 13 | recorded |  |
| n_planes | 3 | recorded |  |
| n_repeats | 10 | recorded |  |
| n_systems | 2 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| point_centres | [[0.3, 0.5], [-0.8, 1.2]] | recorded |  |
| point_centres_source | "plain sub-row only: the simulator's true source centres" | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solved_source_centres | [[0.29996638175783463, 0.4999650707624289], [-0.7999404110613716, 1.199972152235472]] | recorded |  |
| solved_source_centres_source | "the headline fit's analytic beta* (source_plane_coordinate) per system, eager" | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| standard_size | true | recorded |  |
| sub_row_arm | "plain" | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| vmap_batch_size | 3 | recorded |  |

## Evidence

[Original setup artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>3 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/33d653dbb6f501878727` | full_pipeline_vmap3.steady_per_call_s | {"full_pipeline_vmap3.steady_per_call_s": 0.00032998343334232535} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_vmap3/steady_per_call_s` |
| `measurement/46c2758774f6482f1792` | full_pipeline_plain.steady_per_call_s | {"full_pipeline_plain.steady_per_call_s": 0.0010001799999372451} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_plain/steady_per_call_s` |
| `measurement/78271dcf39b6d5d767dc` | full_pipeline_solved.steady_per_call_s | {"full_pipeline_solved.steady_per_call_s": 0.0011369465999450767} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_solved/steady_per_call_s` |

</details>

### compile

<details><summary>7 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00ddff605db5fde03dca` | full_pipeline_solved.lower_s | {"full_pipeline_solved.lower_s": 7.609620328999881} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_solved/lower_s` |
| `measurement/03ebc38e489475d4f7ed` | full_pipeline_solved.compile_s | {"full_pipeline_solved.compile_s": 3.3073549769997044} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_solved/compile_s` |
| `measurement/18db967f8f8e6b7353e7` | full_pipeline_plain.compile_s | {"full_pipeline_plain.compile_s": 1.565768799999205} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_plain/compile_s` |
| `measurement/566d004a708f207e5b81` | full_pipeline_plain.first_call_s | {"full_pipeline_plain.first_call_s": 0.002368560000832076} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_plain/first_call_s` |
| `measurement/97cfaa8b817ea4257bb2` | full_pipeline_solved.first_call_s | {"full_pipeline_solved.first_call_s": 0.003717901001436985} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_solved/first_call_s` |
| `measurement/c0aaeecdd4f48270f992` | full_pipeline_plain.lower_s | {"full_pipeline_plain.lower_s": 2.649941645000581} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_plain/lower_s` |
| `measurement/c55a6ce1edb5392c63e3` | full_pipeline_vmap3.first_call_s | {"full_pipeline_vmap3.first_call_s": 13.71304340899951} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/jit_phases/full_pipeline_vmap3/first_call_s` |

</details>

### runtime

<details><summary>4 recorded runtime measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0e202ef5b38eb300f52d` | vmap.per_call | {"vmap.per_call": 0.00032998343334232535} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/vmap/per_call` |
| `measurement/10c8d193ad406cc57b22` | single_jit_block | {"single_jit_block": 0.0011369465999450767} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/full_pipeline_single_jit` |
| `measurement/7def598d10796e89a6ff` | vmap.batch_time | {"vmap.batch_time": 0.000989950300026976} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/vmap/batch_time` |
| `measurement/843eb75aa8eede951036` | single_jit_median | {"single_jit_median": 0.0007603220001328737} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../archive/2026/11/simple_b0_six_images/source_plane/runtime.json) `/full_pipeline_single_jit_median` |

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
      "cpu_count": 8,
      "device": "cpu:0",
      "hostname": "DESKTOP-H143S82",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "5240c7e88618d4bb0d9845e07e3ddf4e18bef1d7",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "cc764bf70bcca8015fb6eb2e99cbb1a226a073ad",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "9756fa9c1615d18b3e7d21fcb2e2c8e3b5290c63",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "930de76be28fc5f708a5c05efe9c8104ef9c0b89",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "3af6de8d6f9262289bd5e9dd2d78e191403332f1",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "72939f36ed145f9d73817dccb3b2edef09f7b815",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.62.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": null,
    "statistic": "median",
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "repetitions": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": true,
    "host": "DESKTOP-H143S82",
    "measured_at": "2026-11-25T13:15:34Z",
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
      "cpu_count": 8,
      "device": "cpu:0",
      "hostname": "DESKTOP-H143S82",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "5240c7e88618d4bb0d9845e07e3ddf4e18bef1d7",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "cc764bf70bcca8015fb6eb2e99cbb1a226a073ad",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "9756fa9c1615d18b3e7d21fcb2e2c8e3b5290c63",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "930de76be28fc5f708a5c05efe9c8104ef9c0b89",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "3af6de8d6f9262289bd5e9dd2d78e191403332f1",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "72939f36ed145f9d73817dccb3b2edef09f7b815",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.62.1",
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
    "host": "DESKTOP-H143S82",
    "measured_at": "2026-11-25T13:15:34Z",
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

No canonical script explicitly registered for this model.

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
