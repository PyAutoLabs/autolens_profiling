<!-- generated: build_setup_wiki.py; do not edit -->
# rectangular · hst

[Model index](index.md)

Exact setup ID: `imaging/rectangular/hst/e902fb05ffad139587ba`.

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
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| rect_mesh | "bilinear" | recorded |  |
| regularization | {"coefficient": 1.0, "scheme": "constant"} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1521 | count |  |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>12 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/0fde808ba04c1e75c01c` | b_s3_pdip_library_likelihood_jit.steady_per_call_s | {"b_s3_pdip_library_likelihood_jit.steady_per_call_s": 0.03865242092870176} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/1c8297748a0a1c4071ee` | d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s | {"d_s3_certified_fallback_library_likelihood_jit.steady_per_call_s": 0.025103542441502213} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/steady_per_call_s` |
| `measurement/288274201cefafd50c39` | e_s3_certified_budget1_fallback_fires_vmap16.steady_per_call_s | {"e_s3_certified_budget1_fallback_fires_vmap16.steady_per_call_s": 0.03028988981968723} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_vmap16/steady_per_call_s` |
| `measurement/3845591453400775b417` | d0_s3_certified_no_fallback_vmap16.steady_per_call_s | {"d0_s3_certified_no_fallback_vmap16.steady_per_call_s": 0.021351748943561687} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_vmap16/steady_per_call_s` |
| `measurement/4624aa51c0384db3faff` | d0_s3_certified_no_fallback_library_likelihood_jit.steady_per_call_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.steady_per_call_s": 0.02540718447417021} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/steady_per_call_s` |
| `measurement/7cebded2bfcfc0d3b260` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.steady_per_call_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.steady_per_call_s": 0.04145228499546647} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/steady_per_call_s` |
| `measurement/844aa6136b2408a5733d` | a_s0_pdip_library_likelihood_jit.steady_per_call_s | {"a_s0_pdip_library_likelihood_jit.steady_per_call_s": 0.05096764843910932} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/steady_per_call_s` |
| `measurement/96986b92dba4594119c3` | d_s3_certified_fallback_vmap16.steady_per_call_s | {"d_s3_certified_fallback_vmap16.steady_per_call_s": 0.03384811292635277} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d_s3_certified_fallback_vmap16/steady_per_call_s` |
| `measurement/a8ea8cf34799cc943680` | b_s3_pdip_vmap16.steady_per_call_s | {"b_s3_pdip_vmap16.steady_per_call_s": 0.028801840657251887} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/b_s3_pdip_vmap16/steady_per_call_s` |
| `measurement/bb19d5345c8fc5b4e9ca` | c_s3_positive_negative_library_likelihood_jit.steady_per_call_s | {"c_s3_positive_negative_library_likelihood_jit.steady_per_call_s": 0.02112751668319106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/steady_per_call_s` |
| `measurement/eb765231a3f0af9c7eda` | a_s0_pdip_vmap16.steady_per_call_s | {"a_s0_pdip_vmap16.steady_per_call_s": 0.032128060067771004} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/a_s0_pdip_vmap16/steady_per_call_s` |
| `measurement/fc893b34bc23e2e0c179` | c_s3_positive_negative_vmap16.steady_per_call_s | {"c_s3_positive_negative_vmap16.steady_per_call_s": 0.023057016576058232} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/c_s3_positive_negative_vmap16/steady_per_call_s` |

</details>

### compile

<details><summary>24 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/065c4d9f28fa599d9986` | d_s3_certified_fallback_vmap16.first_call_s | {"d_s3_certified_fallback_vmap16.first_call_s": 5.4974286472424865} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d_s3_certified_fallback_vmap16/first_call_s` |
| `measurement/1026d87a47379d685fbd` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.compile_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.compile_s": 3.769688887987286} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/compile_s` |
| `measurement/182e4a0d358e68615bad` | a_s0_pdip_library_likelihood_jit.first_call_s | {"a_s0_pdip_library_likelihood_jit.first_call_s": 0.5713863358832896} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/first_call_s` |
| `measurement/2f45d4a4a0177865c335` | d0_s3_certified_no_fallback_library_likelihood_jit.lower_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.lower_s": 0.47899644589051604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/lower_s` |
| `measurement/31881670e5f0a52b82f7` | d0_s3_certified_no_fallback_vmap16.first_call_s | {"d0_s3_certified_no_fallback_vmap16.first_call_s": 6.334332124795765} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_vmap16/first_call_s` |
| `measurement/37344679605bfded066d` | a_s0_pdip_vmap16.first_call_s | {"a_s0_pdip_vmap16.first_call_s": 20.408882540185004} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/a_s0_pdip_vmap16/first_call_s` |
| `measurement/46acbf3a5113a10e4772` | b_s3_pdip_library_likelihood_jit.lower_s | {"b_s3_pdip_library_likelihood_jit.lower_s": 0.36524866009131074} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/lower_s` |
| `measurement/6bce4c0f102c7490c587` | a_s0_pdip_library_likelihood_jit.lower_s | {"a_s0_pdip_library_likelihood_jit.lower_s": 3.333508708048612} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/lower_s` |
| `measurement/703245200a4ff318e727` | c_s3_positive_negative_library_likelihood_jit.first_call_s | {"c_s3_positive_negative_library_likelihood_jit.first_call_s": 0.08083438500761986} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/first_call_s` |
| `measurement/869b965ba86334f1f91f` | b_s3_pdip_library_likelihood_jit.first_call_s | {"b_s3_pdip_library_likelihood_jit.first_call_s": 0.04688119608908892} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/first_call_s` |
| `measurement/9de11f1ce9417e341374` | b_s3_pdip_vmap16.first_call_s | {"b_s3_pdip_vmap16.first_call_s": 5.498564231209457} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/b_s3_pdip_vmap16/first_call_s` |
| `measurement/9e0c265e662b234300ca` | d0_s3_certified_no_fallback_library_likelihood_jit.compile_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.compile_s": 3.9376447796821594} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/compile_s` |
| `measurement/acfdf0452c8e94325c1d` | d0_s3_certified_no_fallback_library_likelihood_jit.first_call_s | {"d0_s3_certified_no_fallback_library_likelihood_jit.first_call_s": 0.03347264090552926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d0_s3_certified_no_fallback_library_likelihood_jit/first_call_s` |
| `measurement/b177f4cf4dae25c12a39` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.lower_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.lower_s": 0.37076465832069516} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/lower_s` |
| `measurement/b3276c0f56805c114795` | a_s0_pdip_library_likelihood_jit.compile_s | {"a_s0_pdip_library_likelihood_jit.compile_s": 13.442297104280442} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/a_s0_pdip_library_likelihood_jit/compile_s` |
| `measurement/b8da5934e4d0a08737da` | e_s3_certified_budget1_fallback_fires_vmap16.first_call_s | {"e_s3_certified_budget1_fallback_fires_vmap16.first_call_s": 7.976106060203165} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_vmap16/first_call_s` |
| `measurement/c149f08442d35ca943c0` | d_s3_certified_fallback_library_likelihood_jit.first_call_s | {"d_s3_certified_fallback_library_likelihood_jit.first_call_s": 0.03440657630562782} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/first_call_s` |
| `measurement/cc7047cfb7b3801022a2` | e_s3_certified_budget1_fallback_fires_library_likelihood_jit.first_call_s | {"e_s3_certified_budget1_fallback_fires_library_likelihood_jit.first_call_s": 0.05316325882449746} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/e_s3_certified_budget1_fallback_fires_library_likelihood_jit/first_call_s` |
| `measurement/ce4786d86087dab40078` | b_s3_pdip_library_likelihood_jit.compile_s | {"b_s3_pdip_library_likelihood_jit.compile_s": 3.269226770848036} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/b_s3_pdip_library_likelihood_jit/compile_s` |
| `measurement/cec012e775419c701c29` | c_s3_positive_negative_library_likelihood_jit.compile_s | {"c_s3_positive_negative_library_likelihood_jit.compile_s": 3.0983610549010336} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/compile_s` |
| `measurement/d6d0e73d03ccfaa226f7` | c_s3_positive_negative_library_likelihood_jit.lower_s | {"c_s3_positive_negative_library_likelihood_jit.lower_s": 0.3212789259850979} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/c_s3_positive_negative_library_likelihood_jit/lower_s` |
| `measurement/e33d1f6858b5b2140287` | d_s3_certified_fallback_library_likelihood_jit.compile_s | {"d_s3_certified_fallback_library_likelihood_jit.compile_s": 3.5181696456857026} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/compile_s` |
| `measurement/e59a22de37d610d113d3` | c_s3_positive_negative_vmap16.first_call_s | {"c_s3_positive_negative_vmap16.first_call_s": 4.171506438869983} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/c_s3_positive_negative_vmap16/first_call_s` |
| `measurement/f751e2fc11d387898a74` | d_s3_certified_fallback_library_likelihood_jit.lower_s | {"d_s3_certified_fallback_library_likelihood_jit.lower_s": 0.3630145210772753} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/imaging/rectangular/hst/jax_gpu_dense_fp64/hpc_a100/light_fixed/experiments/library/breakdown.json) `/jit_phases/d_s3_certified_fallback_library_likelihood_jit/lower_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-2",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 35461 MiB, 81920 MiB",
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
    "host": "euclid-ral-gpu-2",
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
