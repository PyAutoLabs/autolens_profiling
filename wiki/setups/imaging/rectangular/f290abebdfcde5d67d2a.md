<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/eb5b405aa2e5377113c5`.

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
| mesh | "rectangular" | recorded |  |
| mesh_shape | [39, 39] | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pass_budget | 7 | recorded |  |
| pass_budget_max_diagnostics | 8 | recorded |  |
| pins_mode | "fp64" | recorded |  |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "scheme": "constant"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1521 | count |  |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| tau_rel | 1e-09 | recorded |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>6 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/286eda15dff8dc1f4639` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.steady_per_call_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.steady_per_call_s": 5.391288130000066} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/steady_per_call_s` |
| `measurement/34ad618a1eeb9a623417` | a_s0_pdip_library_likelihood_jit.steady_per_call_s | {"a_s0_pdip_library_likelihood_jit.steady_per_call_s": 5.529184180000084} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/4d48da2ddaf8b1a7b3ce` | c_s3_positive_negative_library_likelihood_jit.steady_per_call_s | {"c_s3_positive_negative_library_likelihood_jit.steady_per_call_s": 4.325817029999962} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/steady_per_call_s` |
| `measurement/709562c212304e578cf8` | d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s | {"d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s": 4.702977079999982} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/steady_per_call_s` |
| `measurement/8ca51956e604ed5fe2c7` | b_s3_pdip_library_likelihood_jit.steady_per_call_s | {"b_s3_pdip_library_likelihood_jit.steady_per_call_s": 4.929599159999998} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/fcdb8d375a1603afa45d` | d0_s3_certified_no_fallback_library_likelihood_jit.steady_per_call_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.steady_per_call_s": 4.652552800000012} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/steady_per_call_s` |

</details>

### compile

<details><summary>18 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00866abcd1f2e42147bc` | a_s0_pdip_library_likelihood_jit.compile_s | {"a_s0_pdip_library_likelihood_jit.compile_s": 8.755736800001614} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/compile_s` |
| `measurement/1125787bbcc880b39701` | d_s3_certified_fallback_library_likelihood_jit.compile_s | {"d_s3_certified_fallback_library_likelihood_jit.compile_s": 1.6060545000000275} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/compile_s` |
| `measurement/1995c4388328fe94146e` | d_s3_certified_fallback_library_likelihood_jit.lower_s | {"d_s3_certified_fallback_library_likelihood_jit.lower_s": 0.52485119999983} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/lower_s` |
| `measurement/1d0acc91c9bbf0c4140b` | d0_s3_certified_no_fallback_library_likelihood_jit.compile_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.compile_s": 1.2186352999997325} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/compile_s` |
| `measurement/2645146fc79cb059987c` | b_s3_pdip_library_likelihood_jit.compile_s | {"b_s3_pdip_library_likelihood_jit.compile_s": 1.272444500000347} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/compile_s` |
| `measurement/31067ac4d74a939bc7de` | b_s3_pdip_library_likelihood_jit.lower_s | {"b_s3_pdip_library_likelihood_jit.lower_s": 0.38509650000014517} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/lower_s` |
| `measurement/4457c52ba2664419dc41` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.lower_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.lower_s": 0.4920047000014165} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/lower_s` |
| `measurement/4821a23395a43ce2c9e7` | a_s0_pdip_library_likelihood_jit.lower_s | {"a_s0_pdip_library_likelihood_jit.lower_s": 3.788551700001335} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/lower_s` |
| `measurement/4966b0dcaf8e8e7f94f6` | d_s3_certified_fallback_library_likelihood_jit.first_call_s | {"d_s3_certified_fallback_library_likelihood_jit.first_call_s": 4.813172900001518} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/first_call_s` |
| `measurement/76c218aaea874bb39105` | b_s3_pdip_library_likelihood_jit.first_call_s | {"b_s3_pdip_library_likelihood_jit.first_call_s": 4.990050500000507} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/first_call_s` |
| `measurement/8c514ac0793b1a8e8b65` | d0_s3_certified_no_fallback_library_likelihood_jit.first_call_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.first_call_s": 4.690304900001138} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/first_call_s` |
| `measurement/8cfb48439aec37e5733b` | d0_s3_certified_no_fallback_library_likelihood_jit.lower_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.lower_s": 0.3473446000007243} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/lower_s` |
| `measurement/b87ae929d8fe0f7c44c1` | a_s0_pdip_library_likelihood_jit.first_call_s | {"a_s0_pdip_library_likelihood_jit.first_call_s": 5.495068700000047} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/first_call_s` |
| `measurement/c793d10a4699e4b57d52` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.compile_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.compile_s": 1.5059152000012546} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/compile_s` |
| `measurement/c9bcf3161dad834638ab` | c_s3_positive_negative_library_likelihood_jit.first_call_s | {"c_s3_positive_negative_library_likelihood_jit.first_call_s": 4.65835960000004} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/first_call_s` |
| `measurement/dde1d47e4ca62fda5ba9` | c_s3_positive_negative_library_likelihood_jit.lower_s | {"c_s3_positive_negative_library_likelihood_jit.lower_s": 0.3386569000012969} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/lower_s` |
| `measurement/e9c37d587f6d436c9b2c` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.first_call_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.first_call_s": 5.794291800000792} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/first_call_s` |
| `measurement/eba61edf2d4c870f1320` | c_s3_positive_negative_library_likelihood_jit.compile_s | {"c_s3_positive_negative_library_likelihood_jit.compile_s": 0.9355221000005258} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_cpu_dense_fp64/local_laptop/light_fixed/experiments/library_t1/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/compile_s` |

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

- [hazards](../../../../scripts/imaging/rectangular/hazards.py)
- [likelihood_breakdown](../../../../scripts/imaging/rectangular/likelihood_breakdown.py)
- [likelihood_breakdown_numba](../../../../scripts/imaging/rectangular/likelihood_breakdown_numba.py)
- [likelihood_runtime](../../../../scripts/imaging/rectangular/likelihood_runtime.py)
- [likelihood_runtime_numba](../../../../scripts/imaging/rectangular/likelihood_runtime_numba.py)
- [likelihood_runtime_numba_mge_mass](../../../../scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py)
- [parallel_scaling_numba](../../../../scripts/imaging/rectangular/parallel_scaling_numba.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
