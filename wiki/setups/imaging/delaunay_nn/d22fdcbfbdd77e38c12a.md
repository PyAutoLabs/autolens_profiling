<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/22655c52b646cf711ed4`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| fallback_row_budget | 1 | recorded |  |
| fallback_rows | true | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh | "delaunay_nn" | recorded |  |
| mesh_shape | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pass_budget | 2 | recorded |  |
| pass_budget_max_diagnostics | 3 | recorded |  |
| pins_mode | "fp64" | recorded |  |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "VECLIB_MAXIMUM_THREADS": "8"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>6 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1f11d29471bda585cba6` | c_s3_positive_negative_library_likelihood_jit.steady_per_call_s | {"c_s3_positive_negative_library_likelihood_jit.steady_per_call_s": 2.995153989999926} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/steady_per_call_s` |
| `measurement/8176e1277716160e95ec` | b_s3_pdip_library_likelihood_jit.steady_per_call_s | {"b_s3_pdip_library_likelihood_jit.steady_per_call_s": 3.321693459999915} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/9c29ceba180058bf92fb` | d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s | {"d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s": 2.7480938900000185} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/steady_per_call_s` |
| `measurement/bf0e8c422c0e4970bca4` | d0_s3_certified_no_fallback_library_likelihood_jit.steady_per_call_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.steady_per_call_s": 2.957391909999933} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/steady_per_call_s` |
| `measurement/c0a7dc024563f1bdbce4` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.steady_per_call_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.steady_per_call_s": 2.9855960400000185} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/steady_per_call_s` |
| `measurement/c91a178e59f8dcfd040b` | a_s0_pdip_library_likelihood_jit.steady_per_call_s | {"a_s0_pdip_library_likelihood_jit.steady_per_call_s": 3.6972847099999853} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/steady_per_call_s` |

</details>

### compile

<details><summary>18 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00729e38a711ade61149` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.first_call_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.first_call_s": 2.7723501000000397} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/first_call_s` |
| `measurement/0d57a71b3bbc4b91fec0` | a_s0_pdip_library_likelihood_jit.compile_s | {"a_s0_pdip_library_likelihood_jit.compile_s": 10.133192499999495} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/compile_s` |
| `measurement/0f94a960dd3b8a1d7e92` | b_s3_pdip_library_likelihood_jit.lower_s | {"b_s3_pdip_library_likelihood_jit.lower_s": 0.8001478000005591} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/lower_s` |
| `measurement/11c1514e7a2f45cfd366` | a_s0_pdip_library_likelihood_jit.first_call_s | {"a_s0_pdip_library_likelihood_jit.first_call_s": 3.453942799998913} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/first_call_s` |
| `measurement/1299350e684ea87093c8` | c_s3_positive_negative_library_likelihood_jit.lower_s | {"c_s3_positive_negative_library_likelihood_jit.lower_s": 0.6198976999985462} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/lower_s` |
| `measurement/23c42ca71590ba7612f8` | d_s3_certified_fallback_library_likelihood_jit.compile_s | {"d_s3_certified_fallback_library_likelihood_jit.compile_s": 2.51908419999927} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/compile_s` |
| `measurement/28ad9cc07a1004adae0c` | d0_s3_certified_no_fallback_library_likelihood_jit.compile_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.compile_s": 2.401405299999169} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/compile_s` |
| `measurement/2b8a760359f287958bac` | b_s3_pdip_library_likelihood_jit.first_call_s | {"b_s3_pdip_library_likelihood_jit.first_call_s": 3.3575447000002896} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/first_call_s` |
| `measurement/3435f4e3b8e3c6de55d3` | d_s3_certified_fallback_library_likelihood_jit.first_call_s | {"d_s3_certified_fallback_library_likelihood_jit.first_call_s": 2.506721200001266} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/first_call_s` |
| `measurement/47f76f08d15966fb5fcf` | a_s0_pdip_library_likelihood_jit.lower_s | {"a_s0_pdip_library_likelihood_jit.lower_s": 4.154058399999485} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/lower_s` |
| `measurement/54cbd68f22eb0186c080` | d0_s3_certified_no_fallback_library_likelihood_jit.first_call_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.first_call_s": 2.258699300000444} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/first_call_s` |
| `measurement/567e4cdf2435cabc2f70` | d0_s3_certified_no_fallback_library_likelihood_jit.lower_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.lower_s": 0.8472639999999956} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/lower_s` |
| `measurement/698c54eea1a0b5a81885` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.compile_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.compile_s": 2.0609652000002825} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/compile_s` |
| `measurement/8e7758625329ae093923` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.lower_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.lower_s": 0.7808459000007133} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/lower_s` |
| `measurement/9446b421bf7e9a18a3a7` | c_s3_positive_negative_library_likelihood_jit.first_call_s | {"c_s3_positive_negative_library_likelihood_jit.first_call_s": 2.5558216000008542} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/first_call_s` |
| `measurement/de820245727ddfe6cd14` | b_s3_pdip_library_likelihood_jit.compile_s | {"b_s3_pdip_library_likelihood_jit.compile_s": 1.9718207999994775} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/compile_s` |
| `measurement/e81e6d5dcf22474b0db1` | d_s3_certified_fallback_library_likelihood_jit.lower_s | {"d_s3_certified_fallback_library_likelihood_jit.lower_s": 0.9040416000007099} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/lower_s` |
| `measurement/fd58d7d315fd6d74df6b` | c_s3_positive_negative_library_likelihood_jit.compile_s | {"c_s3_positive_negative_library_likelihood_jit.compile_s": 1.7119686999994883} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_nn_local_cpu_fp64_fixed_light_library_tall.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/compile_s` |

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
      "omp_num_threads": "8",
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
    "host": "DESKTOP-H143S82",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
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
