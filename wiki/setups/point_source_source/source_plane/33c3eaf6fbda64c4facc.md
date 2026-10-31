<!-- generated: build_setup_wiki.py; do not edit -->
# source_plane · simple

[Model index](index.md)

Exact setup ID: `point_source_source/source_plane/simple/5e3c615681acf328bcef`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| fit_positions_cls | {"plain_control": "FitPositionsSource", "solved": "FitPositionsSourceSolved"} | recorded |  |
| free_parameters | {"plain_control": 8, "solved": 5} | recorded |  |
| hessian_method | "jacfwd" | recorded |  |
| image_pixels_masked | null | count | Not recorded in this legacy evidence; no current default substituted. |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| lens_redshift | 0.5 | recorded |  |
| likelihood | "source_plane_solved (FitPositionsSourceSolved + PointSolved)" | recorded |  |
| n_positions | 4 | recorded |  |
| n_repeats | 500 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| plain_control | "source_plane (FitPositionsSource + PointFlux)" | recorded |  |
| positions_noise_sigma | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| source_redshift | 1.0 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| weighting | {"plain_control": "jacobian", "solved": "jacobian"} | recorded |  |

## Evidence

[Original setup artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>27 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/105b0951b06a04de0508` | cse_magnifications_twice.steady_per_call_s | {"cse_magnifications_twice.steady_per_call_s": 0.00025664604804478584} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_twice/steady_per_call_s` |
| `measurement/233e27ae8006b2674995` | plain_magnifications_at_positions.steady_per_call_s | {"plain_magnifications_at_positions.steady_per_call_s": 0.00025734920194372534} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_magnifications_at_positions/steady_per_call_s` |
| `measurement/2a4e2b16ad77bdffb8f2` | floor_params_pytree_plain.steady_per_call_s | {"floor_params_pytree_plain.steady_per_call_s": 4.87223919481039e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_plain/steady_per_call_s` |
| `measurement/3cb328bfc86108b698d4` | floor_scalar_array.steady_per_call_s | {"floor_scalar_array.steady_per_call_s": 1.062917395029217e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_scalar_array/steady_per_call_s` |
| `measurement/3f2c1ab1818358ffef8c` | component_total | {"component_total": 0.0002605619956739247} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/total_step_by_step` |
| `measurement/5097cb84d50cb170865b` | steps.Fused-wrapper residual (fused - last prefix) | {"steps.Fused-wrapper residual (fused - last prefix)": 5.855108611285686e-07} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Fused-wrapper residual (fused - last prefix)` |
| `measurement/6e2f6d8cefced7bf5bd9` | plain_log_likelihood.steady_per_call_s | {"plain_log_likelihood.steady_per_call_s": 0.00026031512196641413} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_log_likelihood/steady_per_call_s` |
| `measurement/794959913573f7a0b78c` | steps.Precision tensor W_i (jacfwd Hessian -> A^-T Theta A^-1) | {"steps.Precision tensor W_i (jacfwd Hessian -> A^-T Theta A^-1)": 9.100011084228754e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Precision tensor W_i (jacfwd Hessian -> A^-T Theta A^-1)` |
| `measurement/8004d6a80ec9e612fc10` | steps.Marginalisation term (rebuilds W_i) | {"steps.Marginalisation term (rebuilds W_i)": 7.880007615312934e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Marginalisation term (rebuilds W_i)` |
| `measurement/8bdf592c44629f73745f` | solved_beta_hat.steady_per_call_s | {"solved_beta_hat.steady_per_call_s": 0.0002487269419943914} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_beta_hat/steady_per_call_s` |
| `measurement/993847596adf7c7bf625` | solved_chi_squared.steady_per_call_s | {"solved_chi_squared.steady_per_call_s": 0.0002586620239308104} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_chi_squared/steady_per_call_s` |
| `measurement/9c7dbfe58c849556c3a2` | steps.Solved centre beta* (rebuilds W_i and _beta_hat) | {"steps.Solved centre beta* (rebuilds W_i and _beta_hat)": 6.105517968535423e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Solved centre beta* (rebuilds W_i and _beta_hat)` |
| `measurement/b3f5a143fbaedafdd1f0` | solved_marginalization_term.steady_per_call_s | {"solved_marginalization_term.steady_per_call_s": 0.0002586529420223087} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_marginalization_term/steady_per_call_s` |
| `measurement/b6f537b6782de633491b` | solved_precision_tensor.steady_per_call_s | {"solved_precision_tensor.steady_per_call_s": 0.00025180758605711164} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_precision_tensor/steady_per_call_s` |
| `measurement/b7ab8b855655462e800c` | solved_source_plane_coordinate.steady_per_call_s | {"solved_source_plane_coordinate.steady_per_call_s": 0.00026169258798472585} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_source_plane_coordinate/steady_per_call_s` |
| `measurement/bc1247991a39a72f2ba7` | full_plain_control.steady_per_call_s | {"full_plain_control.steady_per_call_s": 0.0002570110419765115} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_plain_control/steady_per_call_s` |
| `measurement/bc542d826bc42ef7963a` | cse_precision_thrice.steady_per_call_s | {"cse_precision_thrice.steady_per_call_s": 0.00025797575199976565} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_thrice/steady_per_call_s` |
| `measurement/c344ab49677e2c653ce8` | plain_chi_squared_map.steady_per_call_s | {"plain_chi_squared_map.steady_per_call_s": 0.0002595610920106992} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_chi_squared_map/steady_per_call_s` |
| `measurement/cbc80b25f19a00b37b38` | steps.Ray trace: _beta_hat (deflections at observed positions) | {"steps.Ray trace: _beta_hat (deflections at observed positions)": 0.00024800197570584714} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/Ray trace: _beta_hat (deflections at observed positions)` |
| `measurement/d5d31cee537cdfaed063` | value_and_grad_solved.steady_per_call_s | {"value_and_grad_solved.steady_per_call_s": 0.0008275767880259082} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/value_and_grad_solved/steady_per_call_s` |
| `measurement/d8b05cbd719377b47900` | cse_precision_once.steady_per_call_s | {"cse_precision_once.steady_per_call_s": 0.0002586509600514546} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_once/steady_per_call_s` |
| `measurement/d9f063f2e44ed8595b94` | steps.fit.log_likelihood (re-evaluates every term) | {"steps.fit.log_likelihood (re-evaluates every term)": 1.104490365833044e-06} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/steps/fit.log_likelihood (re-evaluates every term)` |
| `measurement/dd2f095479d6f60e319e` | solved_log_likelihood.steady_per_call_s | {"solved_log_likelihood.steady_per_call_s": 0.0002620795820839703} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_log_likelihood/steady_per_call_s` |
| `measurement/e7f4f01349ec32c445c5` | cse_magnifications_once.steady_per_call_s | {"cse_magnifications_once.steady_per_call_s": 0.0002577722939895466} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_once/steady_per_call_s` |
| `measurement/eb7a1afa5356753f68d9` | floor_params_pytree_solved.steady_per_call_s | {"floor_params_pytree_solved.steady_per_call_s": 4.60168260615319e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_solved/steady_per_call_s` |
| `measurement/f2971051e852f90eeb6c` | full_solved.steady_per_call_s | {"full_solved.steady_per_call_s": 0.00027183834603056314} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_solved/steady_per_call_s` |
| `measurement/fcd83977fa381da8772c` | plain_model_data.steady_per_call_s | {"plain_model_data.steady_per_call_s": 0.0002557086399756372} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_model_data/steady_per_call_s` |

</details>

### compile

<details><summary>60 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00fe140d474b681b8991` | plain_log_likelihood.compile_s | {"plain_log_likelihood.compile_s": 0.6019244969938882} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_log_likelihood/compile_s` |
| `measurement/08be1158129356e2e037` | solved_marginalization_term.first_call_s | {"solved_marginalization_term.first_call_s": 0.0013481040368787944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_marginalization_term/first_call_s` |
| `measurement/08f788dfa2a3da7231b4` | cse_magnifications_once.lower_s | {"cse_magnifications_once.lower_s": 0.322375044983346} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_once/lower_s` |
| `measurement/094ad9a200e67bf02f28` | cse_precision_thrice.lower_s | {"cse_precision_thrice.lower_s": 0.6108753189910203} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_thrice/lower_s` |
| `measurement/0f0d693a8c5a7ecb9af5` | plain_model_data.compile_s | {"plain_model_data.compile_s": 0.17723637400195003} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_model_data/compile_s` |
| `measurement/0f885559b04906750ef9` | floor_params_pytree_plain.compile_s | {"floor_params_pytree_plain.compile_s": 0.02739519003080204} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_plain/compile_s` |
| `measurement/10b1818476aa5931c81d` | plain_magnifications_at_positions.lower_s | {"plain_magnifications_at_positions.lower_s": 0.2830662190099247} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_magnifications_at_positions/lower_s` |
| `measurement/14a9308f99512f58bc75` | solved_precision_tensor.first_call_s | {"solved_precision_tensor.first_call_s": 0.0010899470071308315} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_precision_tensor/first_call_s` |
| `measurement/187cd795f54516e4b95e` | cse_magnifications_twice.lower_s | {"cse_magnifications_twice.lower_s": 0.4156999199767597} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_twice/lower_s` |
| `measurement/1908e7fd4727bb564949` | cse_magnifications_once.compile_s | {"cse_magnifications_once.compile_s": 0.232657436048612} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_once/compile_s` |
| `measurement/1977aac919356d7800ff` | solved_source_plane_coordinate.compile_s | {"solved_source_plane_coordinate.compile_s": 0.46679570502601564} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_source_plane_coordinate/compile_s` |
| `measurement/1fa380a155b85a7f6431` | cse_precision_once.first_call_s | {"cse_precision_once.first_call_s": 0.0010235479567199945} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_once/first_call_s` |
| `measurement/27f8242c43c4cc154730` | floor_scalar_array.first_call_s | {"floor_scalar_array.first_call_s": 0.0002816370106302202} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_scalar_array/first_call_s` |
| `measurement/30a59d2a9ba34b7956dd` | solved_log_likelihood.first_call_s | {"solved_log_likelihood.first_call_s": 0.001086656004190445} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_log_likelihood/first_call_s` |
| `measurement/3990cad7eb62d74730d0` | cse_precision_once.compile_s | {"cse_precision_once.compile_s": 0.24378449301002547} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_once/compile_s` |
| `measurement/3acad0aa5479cc645828` | solved_precision_tensor.compile_s | {"solved_precision_tensor.compile_s": 0.3237502090050839} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_precision_tensor/compile_s` |
| `measurement/44e0917b7822147a5187` | floor_scalar_array.compile_s | {"floor_scalar_array.compile_s": 0.02225983195239678} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_scalar_array/compile_s` |
| `measurement/454b9c340a028e9dee9f` | floor_params_pytree_solved.first_call_s | {"floor_params_pytree_solved.first_call_s": 0.0004533850005827844} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_solved/first_call_s` |
| `measurement/482fa3779eeae1b38f01` | solved_beta_hat.first_call_s | {"solved_beta_hat.first_call_s": 0.0011317969765514135} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_beta_hat/first_call_s` |
| `measurement/4b2f57c8e3563a99d590` | solved_beta_hat.compile_s | {"solved_beta_hat.compile_s": 0.1746673349989578} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_beta_hat/compile_s` |
| `measurement/5450965a9fda033b8e52` | floor_params_pytree_solved.lower_s | {"floor_params_pytree_solved.lower_s": 0.006740328040905297} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_solved/lower_s` |
| `measurement/5819d48754abc14fe160` | value_and_grad_solved.lower_s | {"value_and_grad_solved.lower_s": 1.6794189019710757} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/value_and_grad_solved/lower_s` |
| `measurement/686d9b28fb8e5301e2d7` | solved_source_plane_coordinate.first_call_s | {"solved_source_plane_coordinate.first_call_s": 0.0010920059867203236} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_source_plane_coordinate/first_call_s` |
| `measurement/6be54786721fd7941eb5` | solved_chi_squared.lower_s | {"solved_chi_squared.lower_s": 1.145415956969373} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_chi_squared/lower_s` |
| `measurement/6c5f12b4a2f301802090` | plain_magnifications_at_positions.first_call_s | {"plain_magnifications_at_positions.first_call_s": 0.0011619459837675095} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_magnifications_at_positions/first_call_s` |
| `measurement/73fefe5df6fc0d3e2a43` | plain_chi_squared_map.lower_s | {"plain_chi_squared_map.lower_s": 0.5259890710003674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_chi_squared_map/lower_s` |
| `measurement/74e13923ddfd2de3eb8f` | solved_source_plane_coordinate.lower_s | {"solved_source_plane_coordinate.lower_s": 0.5360950899776071} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_source_plane_coordinate/lower_s` |
| `measurement/77c683736cbfa698bfcb` | plain_log_likelihood.first_call_s | {"plain_log_likelihood.first_call_s": 0.0011609360226429999} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_log_likelihood/first_call_s` |
| `measurement/791c7ed13a221ca5cbab` | value_and_grad_solved.compile_s | {"value_and_grad_solved.compile_s": 2.4805578180239536} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/value_and_grad_solved/compile_s` |
| `measurement/842a81776e908b1e1932` | solved_marginalization_term.compile_s | {"solved_marginalization_term.compile_s": 0.8585422750329599} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_marginalization_term/compile_s` |
| `measurement/8649194a2841df908194` | cse_magnifications_once.first_call_s | {"cse_magnifications_once.first_call_s": 0.0011700670002028346} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_once/first_call_s` |
| `measurement/8d4804fe020b4e75425f` | full_plain_control.lower_s | {"full_plain_control.lower_s": 0.2916628460516222} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_plain_control/lower_s` |
| `measurement/8d4cdd8f036b318cfb0d` | cse_precision_once.lower_s | {"cse_precision_once.lower_s": 0.22512698697391897} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_once/lower_s` |
| `measurement/94923ad044ab18c091f9` | value_and_grad_solved.first_call_s | {"value_and_grad_solved.first_call_s": 0.0020040959934704006} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/value_and_grad_solved/first_call_s` |
| `measurement/950f5f7abd38fa7239fa` | solved_log_likelihood.compile_s | {"solved_log_likelihood.compile_s": 1.2586279429960996} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_log_likelihood/compile_s` |
| `measurement/975e7bbb090c6c19da6e` | full_solved.first_call_s | {"full_solved.first_call_s": 0.003399818961042911} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_solved/first_call_s` |
| `measurement/99cb191b53d18a2190c2` | solved_beta_hat.lower_s | {"solved_beta_hat.lower_s": 0.07210907101398334} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_beta_hat/lower_s` |
| `measurement/a3accab164510373133f` | solved_chi_squared.first_call_s | {"solved_chi_squared.first_call_s": 0.0010963270324282348} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_chi_squared/first_call_s` |
| `measurement/a670ac8a194ad795dbd3` | full_solved.lower_s | {"full_solved.lower_s": 1.022978614026215} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_solved/lower_s` |
| `measurement/ad3021f1a2e2fea984d8` | plain_model_data.lower_s | {"plain_model_data.lower_s": 0.06736333796288818} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_model_data/lower_s` |
| `measurement/b984e8d1a0516816c310` | cse_precision_thrice.compile_s | {"cse_precision_thrice.compile_s": 0.384184620983433} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_thrice/compile_s` |
| `measurement/bcd5b3423b9fbd504bc9` | full_plain_control.first_call_s | {"full_plain_control.first_call_s": 0.00112548703327775} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_plain_control/first_call_s` |
| `measurement/bee9c158b766d3d65de5` | solved_log_likelihood.lower_s | {"solved_log_likelihood.lower_s": 2.200530442991294} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_log_likelihood/lower_s` |
| `measurement/c00958eeb99284c38143` | plain_chi_squared_map.first_call_s | {"plain_chi_squared_map.first_call_s": 0.0010322870220988989} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_chi_squared_map/first_call_s` |
| `measurement/c2106ac288b27a5d9adc` | floor_params_pytree_plain.lower_s | {"floor_params_pytree_plain.lower_s": 0.005828930006828159} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_plain/lower_s` |
| `measurement/c2bc0a6601e4c98effcd` | plain_chi_squared_map.compile_s | {"plain_chi_squared_map.compile_s": 0.4585897049983032} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_chi_squared_map/compile_s` |
| `measurement/c6f970629794d3e8c909` | floor_scalar_array.lower_s | {"floor_scalar_array.lower_s": 0.0038072150200605392} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_scalar_array/lower_s` |
| `measurement/c9195dbc58b0c0666f8c` | solved_marginalization_term.lower_s | {"solved_marginalization_term.lower_s": 1.3588913750136271} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_marginalization_term/lower_s` |
| `measurement/cac6903fca653f03e687` | plain_magnifications_at_positions.compile_s | {"plain_magnifications_at_positions.compile_s": 0.32118478999473155} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_magnifications_at_positions/compile_s` |
| `measurement/d0ac247e3b38c9d70255` | full_solved.compile_s | {"full_solved.compile_s": 0.5699472409905866} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_solved/compile_s` |
| `measurement/da0f5b2e01e26651f7e8` | plain_model_data.first_call_s | {"plain_model_data.first_call_s": 0.0013482249923981726} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_model_data/first_call_s` |
| `measurement/e463748bbcea15a49b77` | cse_magnifications_twice.first_call_s | {"cse_magnifications_twice.first_call_s": 0.0011792650329880416} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_twice/first_call_s` |
| `measurement/e5039e3a5f47262c1ec1` | cse_precision_thrice.first_call_s | {"cse_precision_thrice.first_call_s": 0.0010501659708097577} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_precision_thrice/first_call_s` |
| `measurement/e53af1693e288484b470` | plain_log_likelihood.lower_s | {"plain_log_likelihood.lower_s": 0.8854039899888448} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/plain_log_likelihood/lower_s` |
| `measurement/e63f022d35e18ee15eef` | cse_magnifications_twice.compile_s | {"cse_magnifications_twice.compile_s": 0.31679308298043907} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/cse_magnifications_twice/compile_s` |
| `measurement/e6c178da9bb4ceecbf2c` | full_plain_control.compile_s | {"full_plain_control.compile_s": 0.3188324979855679} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/full_plain_control/compile_s` |
| `measurement/e7a4e0fcea245b427db6` | solved_chi_squared.compile_s | {"solved_chi_squared.compile_s": 0.7672846140339971} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_chi_squared/compile_s` |
| `measurement/e8cef4613ca0880ef9bb` | floor_params_pytree_solved.compile_s | {"floor_params_pytree_solved.compile_s": 0.026712168008089066} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_solved/compile_s` |
| `measurement/eb62b59fceb5d1b2b629` | solved_precision_tensor.lower_s | {"solved_precision_tensor.lower_s": 0.37594991899095476} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/solved_precision_tensor/lower_s` |
| `measurement/f14509f446ecb1b20f1c` | floor_params_pytree_plain.first_call_s | {"floor_params_pytree_plain.first_call_s": 0.0003172560245729983} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/point_source_source/source_plane/simple/jax_cpu_na_fp64/hpc_a100/history/2026.10.25.1_2026-10-28/breakdown.json) `/jit_phases/floor_params_pytree_plain/first_call_s` |

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
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
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
      "autolens_profiling": "2f98e01a150d2f59aebb5735c4c12ae286a7b1d3",
      "autolens_profiling.library_revisions": "2f98e01a150d2f59aebb5735c4c12ae286a7b1d3",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.10.2",
      "jaxlib.dependency_versions": "0.10.2",
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
    "measured_at": "2026-10-28T21:54:57Z",
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

- [backward_pass_ab](../../../../scripts/point_source_source/source_plane/backward_pass_ab.py)
- [gradient_mode_crossover](../../../../scripts/point_source_source/source_plane/gradient_mode_crossover.py)
- [gradient_mode_library_ab](../../../../scripts/point_source_source/source_plane/gradient_mode_library_ab.py)
- [pytree_input_ab](../../../../scripts/point_source_source/source_plane/pytree_input_ab.py)
- [likelihood_breakdown](../../../../scripts/point_source_source/source_plane/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/point_source_source/source_plane/likelihood_runtime.py)
- [likelihood_runtime_solved](../../../../scripts/point_source_source/source_plane/likelihood_runtime_solved.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
