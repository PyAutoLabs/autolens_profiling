<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay · euclid

[Model index](index.md)

Exact setup ID: `imaging/delaunay/euclid/75c7e669e2b463249a4a`.

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
| source_pixels | 2500 | count |  |
| source_pixels_requested | 2500 | recorded |  |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>4 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/466561d5c0abde86e4fb` | d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s | {"d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s": 2.4710297500001617} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/steady_per_call_s` |
| `measurement/87dbe483ab82e43523d5` | a_s0_pdip_library_likelihood_jit.steady_per_call_s | {"a_s0_pdip_library_likelihood_jit.steady_per_call_s": 5.642305280000437} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/ac5a4d70916dc62785d1` | c_s3_positive_negative_library_likelihood_jit.steady_per_call_s | {"c_s3_positive_negative_library_likelihood_jit.steady_per_call_s": 1.5364398199999414} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/steady_per_call_s` |
| `measurement/af1a799994c204803ccb` | b_s3_pdip_library_likelihood_jit.steady_per_call_s | {"b_s3_pdip_library_likelihood_jit.steady_per_call_s": 4.900897399999667} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/steady_per_call_s` |

</details>

### compile

<details><summary>12 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/27c5e66ce16850294c10` | d_s3_certified_fallback_library_likelihood_jit.first_call_s | {"d_s3_certified_fallback_library_likelihood_jit.first_call_s": 2.6902401999977883} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/first_call_s` |
| `measurement/446a7cee8919765750f4` | d_s3_certified_fallback_library_likelihood_jit.lower_s | {"d_s3_certified_fallback_library_likelihood_jit.lower_s": 0.5214055000033113} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/lower_s` |
| `measurement/4d0ad242634e188da75f` | a_s0_pdip_library_likelihood_jit.compile_s | {"a_s0_pdip_library_likelihood_jit.compile_s": 5.081660499999998} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/compile_s` |
| `measurement/510cdc1baf68ca271dc4` | b_s3_pdip_library_likelihood_jit.first_call_s | {"b_s3_pdip_library_likelihood_jit.first_call_s": 4.486424700000498} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/first_call_s` |
| `measurement/692c3b2c7fdf917fba12` | c_s3_positive_negative_library_likelihood_jit.compile_s | {"c_s3_positive_negative_library_likelihood_jit.compile_s": 1.1555568000039784} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/compile_s` |
| `measurement/6957f31f5efe783934e9` | b_s3_pdip_library_likelihood_jit.compile_s | {"b_s3_pdip_library_likelihood_jit.compile_s": 1.3044249000013224} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/compile_s` |
| `measurement/936df5959aa8971dea09` | c_s3_positive_negative_library_likelihood_jit.lower_s | {"c_s3_positive_negative_library_likelihood_jit.lower_s": 0.47959830000036163} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/lower_s` |
| `measurement/a1cb10d0d52a39e4636e` | b_s3_pdip_library_likelihood_jit.lower_s | {"b_s3_pdip_library_likelihood_jit.lower_s": 0.44371370000590105} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/lower_s` |
| `measurement/c7b9e2b6efdf8b4a39f9` | a_s0_pdip_library_likelihood_jit.first_call_s | {"a_s0_pdip_library_likelihood_jit.first_call_s": 5.5431618000002345} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/first_call_s` |
| `measurement/cb072ca284e249e20b2d` | a_s0_pdip_library_likelihood_jit.lower_s | {"a_s0_pdip_library_likelihood_jit.lower_s": 3.148255000000063} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/lower_s` |
| `measurement/ce882902bbacf8051132` | d_s3_certified_fallback_library_likelihood_jit.compile_s | {"d_s3_certified_fallback_library_likelihood_jit.compile_s": 1.5439323999962653} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/compile_s` |
| `measurement/e17c49dfa2b0821288a0` | c_s3_positive_negative_library_likelihood_jit.first_call_s | {"c_s3_positive_negative_library_likelihood_jit.first_call_s": 1.784330599999521} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/delaunay/euclid/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_n2500_t8/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/first_call_s` |

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
