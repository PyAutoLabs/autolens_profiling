<!-- generated: build_setup_wiki.py; do not edit -->
# source_plane · simple

[Model index](index.md)

Exact setup ID: `point_source_source/source_plane/simple/9377bab772704aa02e72`.

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
| weighting | {"plain_control": "magnification", "solved": "jacobian"} | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>24 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/01322f3eed102db5ad4a` | plain_model_data.steady_per_call_s | {"plain_model_data.steady_per_call_s": 0.00017363774601835758} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_model_data/steady_per_call_s` |
| `measurement/031dfefab406261fe4db` | solved_source_plane_coordinate.steady_per_call_s | {"solved_source_plane_coordinate.steady_per_call_s": 0.00025208887801272793} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_source_plane_coordinate/steady_per_call_s` |
| `measurement/0e43ef38f738991db885` | floor_params_pytree_plain.steady_per_call_s | {"floor_params_pytree_plain.steady_per_call_s": 0.00019309866800904275} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_plain/steady_per_call_s` |
| `measurement/39385789315081aef3f1` | plain_log_likelihood.steady_per_call_s | {"plain_log_likelihood.steady_per_call_s": 0.0002864188420062419} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_log_likelihood/steady_per_call_s` |
| `measurement/4d2881ca0ab6b22e26c1` | full_plain_control.steady_per_call_s | {"full_plain_control.steady_per_call_s": 0.00028939444600837303} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_plain_control/steady_per_call_s` |
| `measurement/5eaaf4f5344cf7f4b510` | steps.Ray trace: _beta_hat (deflections at observed positions) | {"steps.Ray trace: _beta_hat (deflections at observed positions)": 0.00018711249867919832} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/steps/Ray trace: _beta_hat (deflections at observed positions)` |
| `measurement/6032a7788046b92a4ea7` | solved_marginalization_term.steady_per_call_s | {"solved_marginalization_term.steady_per_call_s": 0.00027302624401636424} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_marginalization_term/steady_per_call_s` |
| `measurement/62bd44247b54614f9c4f` | value_and_grad_solved.steady_per_call_s | {"value_and_grad_solved.steady_per_call_s": 0.0006408821840013843} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/value_and_grad_solved/steady_per_call_s` |
| `measurement/64d112ee0cd12a1561f6` | plain_chi_squared_map.steady_per_call_s | {"plain_chi_squared_map.steady_per_call_s": 0.0002549540220061317} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_chi_squared_map/steady_per_call_s` |
| `measurement/6840c8486f8ddbd8089c` | solved_precision_tensor.steady_per_call_s | {"solved_precision_tensor.steady_per_call_s": 0.00027508682000916454} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_precision_tensor/steady_per_call_s` |
| `measurement/788e6d0871349e17647c` | component_total | {"component_total": 0.00022172799799591303} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/total_step_by_step` |
| `measurement/8aa431db72e4ed18e52a` | full_solved.steady_per_call_s | {"full_solved.steady_per_call_s": 0.00025577629200415684} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_solved/steady_per_call_s` |
| `measurement/97292e552e620303029b` | solved_beta_hat.steady_per_call_s | {"solved_beta_hat.steady_per_call_s": 0.00020315614598803223} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_beta_hat/steady_per_call_s` |
| `measurement/a1d9af79212e8ad9bcf3` | solved_log_likelihood.steady_per_call_s | {"solved_log_likelihood.steady_per_call_s": 0.0002735909979965072} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_log_likelihood/steady_per_call_s` |
| `measurement/bc751938d86b79f18322` | steps.Precision tensor W_i (jacfwd Hessian -> A^-T Theta A^-1) | {"steps.Precision tensor W_i (jacfwd Hessian -> A^-T Theta A^-1)": 6.12244984949939e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/steps/Precision tensor W_i (jacfwd Hessian -> A^-T Theta A^-1)` |
| `measurement/c437a922654ca3f4db15` | cse_precision_thrice.steady_per_call_s | {"cse_precision_thrice.steady_per_call_s": 0.0002582870599871967} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_thrice/steady_per_call_s` |
| `measurement/ca6ee5d8a56535a2c29a` | floor_scalar_array.steady_per_call_s | {"floor_scalar_array.steady_per_call_s": 0.00012500681201345289} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_scalar_array/steady_per_call_s` |
| `measurement/ced8edf89f04a8a48a1d` | solved_chi_squared.steady_per_call_s | {"solved_chi_squared.steady_per_call_s": 0.0002876000499818474} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_chi_squared/steady_per_call_s` |
| `measurement/d98decfc24938a47ebb4` | floor_params_pytree_solved.steady_per_call_s | {"floor_params_pytree_solved.steady_per_call_s": 0.0001922015599848237} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_solved/steady_per_call_s` |
| `measurement/da86b8067750a2e3eb61` | cse_magnifications_once.steady_per_call_s | {"cse_magnifications_once.steady_per_call_s": 0.0002244600899866782} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_once/steady_per_call_s` |
| `measurement/dbc67468552338d3db18` | cse_precision_once.steady_per_call_s | {"cse_precision_once.steady_per_call_s": 0.00023129600999527612} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_once/steady_per_call_s` |
| `measurement/e594d046d6cb07eccfd1` | steps.Chi-squared (rebuilds W_i, beta*, _beta_hat) | {"steps.Chi-squared (rebuilds W_i, beta*, _beta_hat)": 3.6033496144227684e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/steps/Chi-squared (rebuilds W_i, beta*, _beta_hat)` |
| `measurement/ee0afa37a121a3e40575` | plain_magnifications_at_positions.steady_per_call_s | {"plain_magnifications_at_positions.steady_per_call_s": 0.0002527566679927986} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_magnifications_at_positions/steady_per_call_s` |
| `measurement/fe27a8914edf72cceba6` | cse_magnifications_twice.steady_per_call_s | {"cse_magnifications_twice.steady_per_call_s": 0.0002551126400067005} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_twice/steady_per_call_s` |

</details>

### compile

<details><summary>60 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/00d515847b9bcfe8c885` | solved_beta_hat.compile_s | {"solved_beta_hat.compile_s": 0.37345619199913926} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_beta_hat/compile_s` |
| `measurement/0a3a4e6d3bebf64008df` | plain_log_likelihood.compile_s | {"plain_log_likelihood.compile_s": 0.7365044399921317} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_log_likelihood/compile_s` |
| `measurement/1731474b61efcf26f5c1` | solved_precision_tensor.first_call_s | {"solved_precision_tensor.first_call_s": 0.0016658600070513785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_precision_tensor/first_call_s` |
| `measurement/1f40216cc2de65fe7d7a` | plain_model_data.first_call_s | {"plain_model_data.first_call_s": 0.0014566409954568371} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_model_data/first_call_s` |
| `measurement/27de1f30eeeb73b946d0` | plain_chi_squared_map.compile_s | {"plain_chi_squared_map.compile_s": 0.5548777419899125} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_chi_squared_map/compile_s` |
| `measurement/29c21cad8c483d7c10d1` | cse_magnifications_once.first_call_s | {"cse_magnifications_once.first_call_s": 0.0015487110067624599} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_once/first_call_s` |
| `measurement/34e569b6711d6a75aacc` | plain_model_data.compile_s | {"plain_model_data.compile_s": 0.374366170988651} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_model_data/compile_s` |
| `measurement/377d912250315b7a833f` | plain_log_likelihood.lower_s | {"plain_log_likelihood.lower_s": 0.9082484059908893} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_log_likelihood/lower_s` |
| `measurement/37f1163515db68dc29b5` | value_and_grad_solved.compile_s | {"value_and_grad_solved.compile_s": 4.250155496993102} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/value_and_grad_solved/compile_s` |
| `measurement/39ba8b7647a72e269e01` | solved_log_likelihood.first_call_s | {"solved_log_likelihood.first_call_s": 0.0024860789999365807} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_log_likelihood/first_call_s` |
| `measurement/3baa2fc3ed903c44205d` | solved_log_likelihood.compile_s | {"solved_log_likelihood.compile_s": 1.1695996620110236} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_log_likelihood/compile_s` |
| `measurement/3f892fa46b696180ee63` | cse_magnifications_twice.compile_s | {"cse_magnifications_twice.compile_s": 0.39587766099430155} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_twice/compile_s` |
| `measurement/5073bcb1abe1f6db6156` | plain_chi_squared_map.first_call_s | {"plain_chi_squared_map.first_call_s": 0.0019718359981197864} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_chi_squared_map/first_call_s` |
| `measurement/539f1762e5c84bb9fe4f` | solved_chi_squared.first_call_s | {"solved_chi_squared.first_call_s": 0.0019525370007613674} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_chi_squared/first_call_s` |
| `measurement/53af3e82245f87448123` | solved_marginalization_term.compile_s | {"solved_marginalization_term.compile_s": 0.8927268240076955} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_marginalization_term/compile_s` |
| `measurement/54483370cbaa2b14e7f7` | solved_marginalization_term.lower_s | {"solved_marginalization_term.lower_s": 1.1454128850018606} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_marginalization_term/lower_s` |
| `measurement/5856e08979d31d92eb97` | floor_params_pytree_plain.compile_s | {"floor_params_pytree_plain.compile_s": 0.03054743200482335} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_plain/compile_s` |
| `measurement/594d13e2f67ff1638c56` | value_and_grad_solved.first_call_s | {"value_and_grad_solved.first_call_s": 0.00679990800563246} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/value_and_grad_solved/first_call_s` |
| `measurement/5da5b6cdb9a1b6b7c567` | cse_precision_once.compile_s | {"cse_precision_once.compile_s": 0.36512199298886117} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_once/compile_s` |
| `measurement/611c141dd4561ab914f3` | floor_scalar_array.lower_s | {"floor_scalar_array.lower_s": 0.004256339001585729} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_scalar_array/lower_s` |
| `measurement/629256eb441a99143080` | cse_precision_thrice.compile_s | {"cse_precision_thrice.compile_s": 0.4507614389876835} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_thrice/compile_s` |
| `measurement/6503ed99d0e4634013e1` | cse_magnifications_twice.lower_s | {"cse_magnifications_twice.lower_s": 0.33810937899397686} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_twice/lower_s` |
| `measurement/6563ef3c65adb37a764c` | cse_precision_once.first_call_s | {"cse_precision_once.first_call_s": 0.0016157209902303293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_once/first_call_s` |
| `measurement/6f50c24ed9ea5f35dbf1` | solved_beta_hat.first_call_s | {"solved_beta_hat.first_call_s": 0.0013477040047291666} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_beta_hat/first_call_s` |
| `measurement/74264fd69aa3a37e9e9a` | plain_chi_squared_map.lower_s | {"plain_chi_squared_map.lower_s": 0.44347695699252654} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_chi_squared_map/lower_s` |
| `measurement/79caad7481f4f93417a0` | full_plain_control.first_call_s | {"full_plain_control.first_call_s": 0.0019021769985556602} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_plain_control/first_call_s` |
| `measurement/7f4f4c00e75bd4276257` | solved_beta_hat.lower_s | {"solved_beta_hat.lower_s": 0.06041200000618119} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_beta_hat/lower_s` |
| `measurement/84a40669502ce16776fa` | floor_scalar_array.first_call_s | {"floor_scalar_array.first_call_s": 0.0005586030019912869} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_scalar_array/first_call_s` |
| `measurement/8acbdd4cf639bc682568` | solved_log_likelihood.lower_s | {"solved_log_likelihood.lower_s": 1.7524035170063144} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_log_likelihood/lower_s` |
| `measurement/8b5504cf54bfb0063ca5` | floor_params_pytree_solved.lower_s | {"floor_params_pytree_solved.lower_s": 0.0065452909911982715} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_solved/lower_s` |
| `measurement/8cbeb27afcc32090353d` | floor_params_pytree_plain.lower_s | {"floor_params_pytree_plain.lower_s": 0.006251824990613386} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_plain/lower_s` |
| `measurement/8d743edc918c60e50d64` | plain_log_likelihood.first_call_s | {"plain_log_likelihood.first_call_s": 0.0024052410008152947} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_log_likelihood/first_call_s` |
| `measurement/8f55b4671830d3e47dce` | solved_source_plane_coordinate.lower_s | {"solved_source_plane_coordinate.lower_s": 0.4552688349940581} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_source_plane_coordinate/lower_s` |
| `measurement/930199ebab505d3cee52` | plain_magnifications_at_positions.compile_s | {"plain_magnifications_at_positions.compile_s": 0.5471453459904296} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_magnifications_at_positions/compile_s` |
| `measurement/94bba209f56142f6a8ba` | cse_magnifications_twice.first_call_s | {"cse_magnifications_twice.first_call_s": 0.0015254409954650328} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_twice/first_call_s` |
| `measurement/9bf6af65df1b52c16b39` | full_solved.compile_s | {"full_solved.compile_s": 0.8724405089888023} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_solved/compile_s` |
| `measurement/9d8063a6e7ea1a70c433` | plain_magnifications_at_positions.lower_s | {"plain_magnifications_at_positions.lower_s": 0.23659305500041228} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_magnifications_at_positions/lower_s` |
| `measurement/ade2fc0ef17d46b65d58` | floor_scalar_array.compile_s | {"floor_scalar_array.compile_s": 0.024713881008210592} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_scalar_array/compile_s` |
| `measurement/b01238ac3a4611890116` | solved_marginalization_term.first_call_s | {"solved_marginalization_term.first_call_s": 0.0021426239982247353} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_marginalization_term/first_call_s` |
| `measurement/b3cd6a4bc03db9d67403` | full_solved.first_call_s | {"full_solved.first_call_s": 0.005487923990585841} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_solved/first_call_s` |
| `measurement/bc5cd729014363a7e02e` | floor_params_pytree_solved.compile_s | {"floor_params_pytree_solved.compile_s": 0.028206039001815952} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_solved/compile_s` |
| `measurement/bcc166f72f182f262638` | floor_params_pytree_solved.first_call_s | {"floor_params_pytree_solved.first_call_s": 0.0006921219901414588} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_solved/first_call_s` |
| `measurement/c397d88e03da9938bcca` | cse_precision_thrice.lower_s | {"cse_precision_thrice.lower_s": 0.6101155249925796} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_thrice/lower_s` |
| `measurement/c7828905afedb6e1fdcd` | solved_precision_tensor.compile_s | {"solved_precision_tensor.compile_s": 0.58152384099958} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_precision_tensor/compile_s` |
| `measurement/c7d2cf97d1f64b74d9ed` | solved_source_plane_coordinate.first_call_s | {"solved_source_plane_coordinate.first_call_s": 0.0020553859940264374} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_source_plane_coordinate/first_call_s` |
| `measurement/c8b8dc31d60dbe3132f6` | cse_precision_once.lower_s | {"cse_precision_once.lower_s": 0.19470217000343837} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_once/lower_s` |
| `measurement/c9782588a5f95b0ed0c0` | solved_chi_squared.compile_s | {"solved_chi_squared.compile_s": 0.8065905339899473} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_chi_squared/compile_s` |
| `measurement/cce59394e7a7aa5f5ad7` | solved_precision_tensor.lower_s | {"solved_precision_tensor.lower_s": 0.3280191210069461} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_precision_tensor/lower_s` |
| `measurement/cd181936b31ff5d2584e` | floor_params_pytree_plain.first_call_s | {"floor_params_pytree_plain.first_call_s": 0.000678611992043443} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/floor_params_pytree_plain/first_call_s` |
| `measurement/cf2df6bfd8efa397ae01` | value_and_grad_solved.lower_s | {"value_and_grad_solved.lower_s": 1.3092260959965643} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/value_and_grad_solved/lower_s` |
| `measurement/d0ab905ee51413bd87de` | full_solved.lower_s | {"full_solved.lower_s": 0.8810702649934683} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_solved/lower_s` |
| `measurement/d14a5e0ef7f4c89054e4` | plain_model_data.lower_s | {"plain_model_data.lower_s": 0.06262422500003595} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_model_data/lower_s` |
| `measurement/dcccb4ce7d1ee68da3ab` | full_plain_control.lower_s | {"full_plain_control.lower_s": 0.40987121300713625} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_plain_control/lower_s` |
| `measurement/df5f863572f27f33dd03` | cse_magnifications_once.lower_s | {"cse_magnifications_once.lower_s": 0.18323579900607} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_once/lower_s` |
| `measurement/e30d9d7e7b43f053e29d` | full_plain_control.compile_s | {"full_plain_control.compile_s": 0.5282084950013086} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/full_plain_control/compile_s` |
| `measurement/ea12991c819d227dbd19` | solved_chi_squared.lower_s | {"solved_chi_squared.lower_s": 0.9725010609981837} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_chi_squared/lower_s` |
| `measurement/eb69b1996127a3046579` | plain_magnifications_at_positions.first_call_s | {"plain_magnifications_at_positions.first_call_s": 0.0017444989935029298} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/plain_magnifications_at_positions/first_call_s` |
| `measurement/ebe4ac4f09f008b18ec2` | solved_source_plane_coordinate.compile_s | {"solved_source_plane_coordinate.compile_s": 0.6180229099991266} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/solved_source_plane_coordinate/compile_s` |
| `measurement/f3f4e5623741b563c4b7` | cse_magnifications_once.compile_s | {"cse_magnifications_once.compile_s": 0.34139902899914887} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_magnifications_once/compile_s` |
| `measurement/ffa12040e686cb414391` | cse_precision_thrice.first_call_s | {"cse_precision_thrice.first_call_s": 0.0016195810021599755} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/point_source_source/source_plane_hpc_a100_fp64.json) `/jit_phases/cse_precision_thrice/first_call_s` |

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
      "cache_fresh": false,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-1",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 1163 MiB, 81920 MiB",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray": "3de624b5b91a4efa130dc3cb0a3a0f06f85025a3",
      "PyAutoFit": "dd9fbe0aab60c4a57702b1444ca031b317c91331",
      "PyAutoGalaxy": "70a61e26cd715cefe7012a38285698091246a8a4",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoNerves": "1fa613aa8d89a79613009fb66f5b4e2d21deb57a",
      "autolens_profiling": "ec29705a51ab108b14a4869056a7a1758c104c8c"
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
    "host": "euclid-ral-gpu-1",
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

- [backward_pass_ab](../../../../scripts/point_source_source/source_plane/backward_pass_ab.py)
- [gradient_mode_crossover](../../../../scripts/point_source_source/source_plane/gradient_mode_crossover.py)
- [gradient_mode_library_ab](../../../../scripts/point_source_source/source_plane/gradient_mode_library_ab.py)
- [pytree_input_ab](../../../../scripts/point_source_source/source_plane/pytree_input_ab.py)
- [likelihood_breakdown](../../../../scripts/point_source_source/source_plane/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/point_source_source/source_plane/likelihood_runtime.py)
- [likelihood_runtime_solved](../../../../scripts/point_source_source/source_plane/likelihood_runtime_solved.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
