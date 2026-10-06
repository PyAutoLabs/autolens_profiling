<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · euclid

[Model index](index.md)

Exact setup ID: `imaging/delaunay/euclid/a674f7eb5d05c5f3ab94`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| dataset | "euclid" | recorded |  |
| fallback_row_budget | 1 | recorded |  |
| fallback_rows | true | recorded |  |
| image_pixels_masked | 3841 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| mesh | "delaunay" | recorded |  |
| mesh_shape | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 15664 | recorded |  |
| oversampled_pixels | 15664 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pass_budget | 7 | recorded |  |
| pass_budget_max_diagnostics | 12 | recorded |  |
| pass_budget_mode | "auto -> phase 3's production safe budget for delaunay (7); the smallest certifying budget at this configuration is measured and recorded, not used" | recorded |  |
| pins_mode | "fp64" | recorded |  |
| pixel_scale_arcsec | 0.1 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| routes_selected | ["a", "b", "c", "d"] | recorded |  |
| safe_budget | 7 | recorded |  |
| safe_budget_basis | "phase 3 (autolens_profiling#255): the smallest budget with ZERO fallback over a graded 41-model draw set, confirmed N-independent from 484 to 3969 source pixels by phase 4 (#257). This is the budget a production run fixes; the smallest budget that certifies at this configuration is recorded under certification, never used." | recorded |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 500 | count |  |
| source_pixels_requested | 500 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>4 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/09fc6ee5ee89c0bb85c9` | b_s3_pdip_library_likelihood_jit.steady_per_call_s | {"b_s3_pdip_library_likelihood_jit.steady_per_call_s": 0.0752934299998742} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/1feb34863c9cd7508256` | a_s0_pdip_library_likelihood_jit.steady_per_call_s | {"a_s0_pdip_library_likelihood_jit.steady_per_call_s": 0.10584225999991759} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/9eab9cbf04c87b2b5a3b` | d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s | {"d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s": 0.05578320000022359} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/steady_per_call_s` |
| `measurement/b8dd9179a30e192c318a` | c_s3_positive_negative_library_likelihood_jit.steady_per_call_s | {"c_s3_positive_negative_library_likelihood_jit.steady_per_call_s": 0.04523805000026186} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/steady_per_call_s` |

</details>

### compile

<details><summary>12 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/11ae9b2b28168dac5cc6` | d_s3_certified_fallback_library_likelihood_jit.compile_s | {"d_s3_certified_fallback_library_likelihood_jit.compile_s": 3.295096099998773} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/compile_s` |
| `measurement/2020b78b38ef3226d96b` | b_s3_pdip_library_likelihood_jit.compile_s | {"b_s3_pdip_library_likelihood_jit.compile_s": 2.775541599999997} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/compile_s` |
| `measurement/41cf8d84c259d8aedcd7` | b_s3_pdip_library_likelihood_jit.lower_s | {"b_s3_pdip_library_likelihood_jit.lower_s": 0.47926229999939096} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/lower_s` |
| `measurement/60b3195ab388d61e3ba3` | d_s3_certified_fallback_library_likelihood_jit.lower_s | {"d_s3_certified_fallback_library_likelihood_jit.lower_s": 0.5841533999991952} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/lower_s` |
| `measurement/6e28445d98342c82a174` | a_s0_pdip_library_likelihood_jit.compile_s | {"a_s0_pdip_library_likelihood_jit.compile_s": 7.9773716000017885} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/compile_s` |
| `measurement/72d1ad86cf73476d8a8c` | c_s3_positive_negative_library_likelihood_jit.lower_s | {"c_s3_positive_negative_library_likelihood_jit.lower_s": 0.44257310000102734} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/lower_s` |
| `measurement/7830405dd264baf03fe6` | c_s3_positive_negative_library_likelihood_jit.compile_s | {"c_s3_positive_negative_library_likelihood_jit.compile_s": 2.849409399997967} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/compile_s` |
| `measurement/824600b75d37ecc9b7b0` | b_s3_pdip_library_likelihood_jit.first_call_s | {"b_s3_pdip_library_likelihood_jit.first_call_s": 0.2835586000001058} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/first_call_s` |
| `measurement/94a74f80b508bc137528` | a_s0_pdip_library_likelihood_jit.first_call_s | {"a_s0_pdip_library_likelihood_jit.first_call_s": 0.7190057000007073} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/first_call_s` |
| `measurement/e173c46c747e970be9f7` | d_s3_certified_fallback_library_likelihood_jit.first_call_s | {"d_s3_certified_fallback_library_likelihood_jit.first_call_s": 0.2877846999981557} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/first_call_s` |
| `measurement/e9b3c2df2c04431ab21e` | a_s0_pdip_library_likelihood_jit.lower_s | {"a_s0_pdip_library_likelihood_jit.lower_s": 3.4411317000012787} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/lower_s` |
| `measurement/f1fe03ef661a9270bb99` | c_s3_positive_negative_library_likelihood_jit.first_call_s | {"c_s3_positive_negative_library_likelihood_jit.first_call_s": 0.23405510000156937} | s | gpu / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/fixed_light_library_delaunay_euclid_n500_local_rtx2060_fp64_fixed_light_library.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/first_call_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "gpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 8,
      "device": "cuda:0",
      "hostname": "DESKTOP-H143S82",
      "nvidia_smi": "NVIDIA GeForce RTX 2060 with Max-Q Design, 426 MiB, 6144 MiB",
      "omp_num_threads": "1",
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/delaunay/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/delaunay/likelihood_runtime_numba.py)
- [quick_update](../../../../scripts/imaging/delaunay/quick_update.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
