# Fixed lens light on the numba CPU path — the three levers (2026-09-16)

autolens_profiling issue [#267](https://github.com/PyAutoLabs/autolens_profiling/issues/267),
epic `fixed-lens-light-numba-cpu` phase 3, branch `feature/fixed-light-numba-levers`.
Phase 2 ([`fixed_lens_light_numba_2026_09.md`](./fixed_lens_light_numba_2026_09.md)) closed the
solver and left a memo-invariant, non-solver residue in the production numba CPU likelihood call:
`inversion.regularization_matrix` at 111.5 ms, the two log-dets at 40.2 + 37.5 ms, and
`sparse_numba.curvature_matrix` at 87.4 ms of a 404.6 ms call. Phase 3 attacks that residue in three
levers, each paired with an A100 row: **lever 1** numba-jit the split-regularization assembly,
**lever 2** log det H from sparsity, **lever 3** one shared Cholesky of `F + λH`.
**Lever 1 is measured; levers 2 and 3 are pending** and will be appended here as their own sections.

**Lever 1's result: 1.379× on the production numba CPU route (413.301 → 299.709 ms), and identity
on the A100.** `inversion.regularization_matrix` falls 110.599 → 5.924 ms — 26.8 % of the call to
2.0 % — with the regularization matrix bit-identical, the interpolators' cached stencil tables
unmutated, and the fit's `figure_of_merit` unchanged at relative difference exactly 0.0. The A100
arm is identity to 0.036 % on the whole call because lever 1 changed only the `xp==np` branch, which
the GPU cell never reaches; its one finding is negative and structural — **the `#536` compaction
constants `SPLIT_REG_COMPACT_WIDTH` and `SPLIT_REG_WIDE_ROW_BUDGET` are inert at this
configuration**, because `InterpolatorDelaunay`'s stencil table is K = 4 wide.

## Scope — read this before quoting a number

- **Route b, memo ON, is what production pays.** Both arms are phase 2's Leg D cell and nothing
  else: `fixed_light_numba.py --mesh delaunay --dataset hst --formalism sparse_numba --routes b
  --threads 1 --n-repeats 16 --instances iid --nnls-warm-start on`. Route **b** is the source-only
  **S3** system (1500 params) the `fixed-lens-light` campaign established; the cross-evaluation NNLS
  memo is **on**, which is the library default and therefore production. `configuration.routes_selected`
  is `["b"]` and `nnls_warm_start_mode` is `"on"` in both JSONs — check them before quoting, because
  `parse_profile_cli` uses `parse_known_args` and silently ignores a mistyped flag.
- **S3 source-only only.** Route **a** (the joint S0 system the library solves today) was not
  re-measured in this phase, so **no "a → lever 1" ratio exists** and none is quoted. Route **c**
  (positivity off) **is not present in either arm** — it is a diagnostic, it silently turns edge
  zeroing off as well, and phase 2's prohibition on quoting its milliseconds carries here unchanged.
- **`log_likelihood` is never compared across legs.** `--instances iid` rotates a call index, so the
  last-timed instance depends on the route count and the leg. Both arms here printed the *same*
  `log_likelihood` **18665.014334145228** and the same 32-entry `log_likelihood_sequence`, which is
  what a deterministic iid stream from `iid_seed: 263` on the same route set must do. **Record it as
  an observation of stream determinism, never as a cross-leg comparison and never as the equivalence
  evidence** — the equivalence evidence is the witness's W1/W3 and the gates.
- **1.379× is a lower bound.** The scheduler placed both jobs on `euclid-ral-gpu-1`, so the feature
  arm's timed rows started at 1-minute loadavg **2.25** against the control arm's **1.90** (both on
  124 cores; `contention.load_average_at_start`). The asymmetry is small and it works *against* the
  feature arm. Both arms stamp `timing_status: "measured"` and `contention_warning: false`.
- **The A100 numbers are `delaunay.py` prefix differences, not standalone timings.** The cell times
  nested jitted prefixes of one inversion setup; `steps["Regularization matrix (H)"]` is the
  **difference of two jitted prefixes** — `regularization_matrix_prefix_s` minus
  `interpolator_prefix_s` (0.0073318 − 0.0071154 = 0.0002164 s in the control arm, exactly the H
  step). A 20 µs "H difference" between the two arms is 0.3 % of a 7.1 ms prefix and is prefix
  scatter, not an H change. Judge the A100 arms on the prefixes and on `total_step_by_step`.
- **The compaction constants are unobservable in this row.** `InterpolatorDelaunay` gives a stencil
  table of width **K = 4**, so `kc = min(K, SPLIT_REG_COMPACT_WIDTH=12) = 4`, the head scatter
  already covers every column and the `if K > kc` supplement is a **static Python branch XLA never
  traces**. Neither constant can be measured, tuned or validated on this cell — see
  "Lever 1 — the A100 row and the compaction finding".
- **No pin exists here.** `pinned_expected` is `null` and `pinned_drift` is empty in both arms; the
  `no_pin_note` block says the same. Every millisecond is RECORDED. The hard verdicts are the
  structural gates (`gates`), the witness's W1–W3, and the A100 cell's own eager regression assertion.
- **Single-threaded only**, as in phase 2: production runs one single-threaded numba likelihood per
  pool worker. `n_threads: 1`, `NUMBA_NUM_THREADS: "1"` and the whole BLAS family pinned to `"1"` in
  both arms; `cpu_over_wall` 1.0003 / 1.0004, so no pool went unpinned.
- **The control is a private merge-base checkout, not the shared install.** The shared
  `/mnt/ral/jnightin/PyAuto` venv also carries PyAutoArray `5e2bc0f4`, but it was **not** used as the
  control: each arm prepends its own PyAutoArray checkout to `PYTHONPATH` inside its own subshell and
  prints `autoarray.__file__`, so the A/B is symmetric in how the library is reached.
- **The process is not JAX-free**, as in phase 2: `jax_in_sys_modules` is `true` in the witness JSON.
  Recorded, not fixed (PyAutoArray bug, still unfiled).

## Provenance

Two SLURM jobs, one node, 2026-09-16.

| | CPU A/B + witness | A100 A/B |
|---|---|---|
| Job | **343345**, `COMPLETED 0:0` | **343346**, `COMPLETED 0:0` |
| Elapsed / MaxRSS | 00:02:16 / 2 079 124 K of 32 GB | 00:03:29 / 3 692 788 K of 64 GB |
| Node | **euclid-ral-gpu-1** | **euclid-ral-gpu-1** |
| Partition | `gpu` CPUs-only, `--cpus-per-task=4 --mem=32gb`, **no `--gres`** | `gpu`, `--gres=gpu:1 --cpus-per-task=4 --mem=64gb` |
| Device | AMD EPYC 7702 64-Core, `os.cpu_count()` 124, kernel 5.14.0-687.39.1.el9_8 | NVIDIA A100 80GB PCIe (`nvidia_smi`), `cuda:0` |
| Job loadavg, entry → exit | `0.00 0.00 0.00` → `2.16 0.96 0.37` | `0.00 0.00 0.00` → `1.96 1.18 0.49` |
| Cell | `scripts/imaging/likelihood_breakdown/fixed_light_numba.py` | `scripts/imaging/likelihood_breakdown/delaunay.py` |
| Python / autolens | 3.12.4 / 2026.8.17.1 | 2026.8.17.1 |
| `AP_ROOT` | `/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-levers` @ `7c7968c4`, branch `feature/fixed-light-numba-levers` | same worktree |

Per-arm checkouts and config names:

| Arm | PyAutoArray checkout | rev | config name |
|---|---|---|---|
| CPU control | `/mnt/ral/jnightin/PyAuto_wt/fixed-light-numba-levers/PyAutoArray_control` | `5e2bc0f42940adfe04ff30fa42a7ce13a86ecbe0` (merge base) | `hpc_ral_cpu_fp64_fixed_light_numba_lever1_control_b_warm_t1` |
| CPU feature | `.../PyAutoArray_feature` | `bae9296ed528da6812613edc2ebc6ea367361ebb` (lever 1) | `hpc_ral_cpu_fp64_fixed_light_numba_lever1_b_warm_t1` |
| CPU witness | `.../PyAutoArray_feature` | `bae9296e` | `hpc_ral_cpu_fp64_lever1` |
| A100 control | `.../PyAutoArray_control` | `5e2bc0f4` | `hpc_a100_fp64_lever1_control` |
| A100 feature | `.../PyAutoArray_feature` | `bae9296e` | `hpc_a100_fp64_lever1` |

The rest of the stack is the shared RAL install, untouched and identical for all four arms:

| Repo | rev |
|---|---|
| PyAutoNerves | `fac8b17b962adc272e0aaae81e951d58b13bcce3` |
| PyAutoFit | `27d41e7c8d235a8aedbc1afa88cadf0863d68fe8` |
| PyAutoArray (shared install, **not** the control) | `5e2bc0f42940adfe04ff30fa42a7ce13a86ecbe0` |
| PyAutoGalaxy | `840ffde063fae1c7bfb41986f7c62b6ca96df3b7` |
| PyAutoLens | `ccf9295f32ec3ad897d4c809c76f33987ce2b0ed` |

System under test, identical in both CPU arms: HST, `pixel_scale` 0.05″, mask 3.5″, **15 361** masked
image pixels, 62 752 over-sampled pixels, Hilbert/Delaunay mesh at **1500** source pixels,
`AdaptSplit(inner=0.1, outer=10.0, signal_scale=0.1)`, fp64, `InversionImagingSparseNumba`,
`use_jax=False`, `n_edge_zeroed` 0, `--instances iid` from `iid_seed` 263.
The A100 cell runs the same dataset and mesh on the **dense** JAX path (`inversion_path: "dense"`,
`total_params` 1560, `vmap_batch` 16, XLA flags
`--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false`).

## Gate roll-up

Cell gates, both CPU arms (`gates` in each JSON; thresholds in `gate_thresholds`):

| Gate | control | feature | value (identical in both arms) |
|---|---|---|---|
| `P1_rebaked_operator_equals_original` | **PASS** | **PASS** | 3/3 arrays identical |
| `S3_mapper_block_equals_S0` | **PASS** | **PASS** | log-dets 8326.70017133483 / 7670.876894312947, rel diff **0.0**, edge-zeroed 0 vs 0 |
| `P2_dense_equals_sparse_numba_system` | **PASS** | **PASS** | D 1.297e-15, F(mapper) 3.728e-15 (rtol 1e-09) |
| `P3_dense_equals_sparse_numba_evidence` | **PASS** | **PASS** | Δ −9.094947e-11 nats (rel 4.615e-15), **0** passive-set differences, 1485 passive both sides |
| `instrumentation_overhead_ratio` (≤ 1.03) | 1.0147 **PASS** | 1.0163 **PASS** | 8 ABBA blocks each |
| `unattributed_fraction` (≤ 0.05) | 0.00308 | 0.00329 | explicit measured remainder |
| `cached_site_call_count_violations` | `{}` | `{}` | — |
| `timing_status` / `contention_warning` | measured / false | measured / false | peak 1-min loadavg 2.13 / 2.25 on 124 cores |

Every `gates` field is **byte-identical between the two arms**, including
`P3.figure_of_merit_sparse_numba` = 19705.71758591176 and the S3 = S0 log-dets. That is the first
equivalence statement: the two library revisions produce the same numbers on the same system.

Solver state, both arms: `seed_source` `"memo"`, `warm_start_fallback` `false`, `n_passive` **1484**,
`outer_iterations` 2, `inner_iterations` 1, `memo_cleared_within_block: false` — the rows are
genuinely warm, and the solver saw the same problem in both arms.

Witness (`fixed_light_numba_levers_witness.py`, **verdict PASS**, run inside the feature arm's
subshell, wall 12.75 s, `slurm_job_id` 343345, `nnls_warm_start_env: "0"` — the witness takes no
timing, so the memo setting is irrelevant to it):

| Gate | status | evidence |
|---|---|---|
| **W1** H bit-identical | **PASS** | `np.array_equal` retained pure-Python reference chain vs the library's `inversion.regularization_matrix`: `array_equal` **true**, `max_abs_diff` **0.0**, shape (1500, 1500), `n_zero_size_stencil_rows` **0**. Same matrix: reference 106.8 ms, library **6.5 ms** |
| **W2** interpolator tables unmutated | **PASS** | `mappings_unchanged` / `sizes_unchanged` / `weights_unchanged` all **true** after the library inversion |
| **W3** evidence unchanged | **PASS** | `figure_of_merit` kernel **19705.71758591176** vs reference **19705.71758591176**, `d_log_evidence_nats` **0.0**, `relative_difference` **0.0** (rtol 1e-09); `monkeypatch_calls` `{reg_split: 1, pixel_splitted: 1}`, `monkeypatch_fired` true |

Witness system block: `InversionImagingSparseNumba`, `InterpolatorDelaunay`, `AdaptSplit`, `n_params`
1500, `n_funcs` 0, `edge_zeroed_pixels` 0, stencil tables **mappings (6000, 4) int64, sizes (6000,)
int32, weights (6000, 4) float64**, `stencil_sizes_max` **3**, H shape (1500, 1500) not sliced,
subtracted light flux 4945.524141998385.

A100 assertions, both arms (`output/output.343346.out`): "cell-driven NNLS reproduces the library
reconstruction to 1e-8" **PASSED**, "inversion-matrix log_evidence matches FitImaging.log_evidence"
**PASSED**, "Eager regression assertion PASSED: log_evidence matches 29140.295882". Every one of the
six `log_evidence_terms` is bit-identical across the two arms, and `nnls.iterations` is 22 with
`reconstruction_max_abs_diff_vs_library` **7.592702022662934e-10** in both — the same digits.

## Lever 1 — headline

### Host 1: `euclid-ral-gpu-1`, numba CPU

`use_jax: false`, `backend: "numba_cpu"`, `InversionImagingSparseNumba`, `n_threads` 1,
`NUMBA_NUM_THREADS` `"1"`, `OMP`/`OPENBLAS`/`MKL`/`VECLIB`/`NUMEXPR` all `"1"`,
`cpu_over_wall` 1.0003 (control) / 1.0004 (feature). Clean (uninstrumented) calls, n-repeats 16.

| arm | PyAutoArray | mean ms | median ms | min | max | clean spread | ABBA | `inversion.regularization_matrix` ms | H as % of call |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | `5e2bc0f4` | **413.301** | 411.135 | 398.487 | 466.934 | 16.56 % | 1.0147 | **110.599** | 26.76 % |
| feature | `bae9296e` | **299.709** | 297.262 | 285.081 | 350.800 | 21.93 % | 1.0163 | **5.924** | 1.98 % |

| | value |
|---|---:|
| speedup on means | **1.379×** |
| speedup on medians | **1.383×** |
| whole-call saving | **113.592 ms** (means), 113.873 ms (medians) |
| H saving | **104.675 ms** (110.599 → 5.924, **18.7×** on that site) |

Anchors on phase 2's Leg D (job 343311, the same cell on the same node type, pre-lever):
mean 407.864 → control 413.301 (**+1.33 %**), median 404.597 → 411.135 (**+1.62 %**),
H 111.501 → 110.599 (**−0.81 %**). The control reproduces phase 2.

Both arms ran 16 clean and 16 instrumented calls in 8 counterbalanced ABBA blocks; the decomposition
closes exactly — the 48 instrumented sites (37 of them reached) plus the explicit
`unattributed (call − sum of exclusive times)` row, 49 rows in all, sum to the clean call to the last
printed digit: **413.301** and **299.709**. `unattributed` is 1.273 ms (0.31 %) and 0.986 ms
(0.33 %) of the two calls, on a 5 % gate.

### Host 2: `euclid-ral-gpu-1`, A100 80GB PCIe, JAX/XLA

`backend: "gpu"`, `device: "cuda:0"`, fp64, `--split-setup --vmap-batch 16`, dense inversion path,
`jax_compilation_cache_dir` `/home/jnightin/.cache/pyauto_jax` with `cache_fresh: true` and
`autotune_cache_entries_at_start` 0 in both arms.

| arm | PyAutoArray | `total_step_by_step` s | `regularization_matrix_prefix_s` | `interpolator_prefix_s` | H step (prefix difference) s | H @ vmap/16 s | `regularization_matrix_jit.compile_s` |
|---|---|---:|---:|---:|---:|---:|---:|
| control | `5e2bc0f4` | **0.067071** | 0.0073318 | 0.0071154 | 0.00021639 | 5.877e-05 | 3.043 |
| feature | `bae9296e` | **0.067047** | 0.0073298 | 0.0070933 | 0.00023652 | 5.976e-05 | 3.261 |
| difference | — | **−0.036 %** | **−0.028 %** | −0.312 % | +20 µs (see Scope) | +1.7 % | +0.218 s |

**Identity, as expected.** The whole call moves 0.036 % and the H prefix 0.028 %. The H *step* moves
+20 µs only because the interpolator prefix it is subtracted from moved −0.312 %; both prefixes are
~7.1–7.3 ms, so the step is a small difference of two large jitted numbers.

## Lever 1 — decomposition

Exclusive (self) time per instrumented site, rescaled by `decomposition_rescale_factor`
(0.98604 control / 0.98476 feature) so the rows sum to the clean call. `n_sites_instrumented` 48 and
`n_sites_reached` 37 in both arms. Top rows by control time:

| site | group | control ms | feature ms | Δ ms | Δ % |
|---|---|---:|---:|---:|---:|
| `inversion.regularization_matrix` | F + λH | **110.599** | **5.924** | **−104.675** | **−94.6** |
| `sparse_numba.curvature_matrix` | D and F assembly | 89.615 | 87.153 | −2.462 | −2.7 (scatter) |
| `solver.fnnls_cholesky` | Positivity solve | 65.050 | 62.602 | −2.448 | −3.8 (scatter) |
| `inversion.log_det_curvature_reg_matrix_term` | Log dets + evidence | 40.595 | 39.983 | −0.613 | −1.5 |
| `inversion.log_det_regularization_matrix_term` | Log dets + evidence | 37.809 | 37.200 | −0.609 | −1.6 |
| `delaunay.triangulation` | Mesh | 18.816 | 18.662 | −0.155 | −0.8 |
| `sparse_numba.psf_weighted_data` | D and F assembly | 8.399 | 8.211 | −0.188 | −2.2 |
| `solver.reconstruction_positive_only_from` | Positivity solve | 7.748 | 7.471 | −0.277 | −3.6 |
| `inversion.reconstruction` | Positivity solve | 7.322 | 7.043 | −0.279 | −3.8 |
| `to_inversion.lp_linear_func_list_galaxy_dict` | Mapper | 7.265 | 6.049 | −1.216 | −16.7 |
| `inversion.curvature_reg_matrix` | F + λH | 4.261 | 4.108 | −0.152 | −3.6 |
| `unattributed (call − sum of exclusive times)` | — | 1.273 | 0.986 | −0.287 | −22.5 |

**Only one row moved.** `sparse_numba.curvature_matrix` (−2.462 ms) and `solver.fnnls_cholesky`
(−2.448 ms) are the only other sites that moved more than 2 ms, and both are **−2.7 % and −3.8 % of
their own rows against clean spreads of 16.6 % and 21.9 % — that is scatter, not an effect.** Lever 1
cannot touch either site: it changed two functions in
`autoarray/inversion/regularization/regularization_util.py` and nothing else.
`to_inversion.lp_linear_func_list_galaxy_dict` is the largest relative mover after H (−16.7 %) but
only −1.216 ms of a 7 ms row, and is likewise scatter. The whole-call saving (113.592 ms) exceeds
the H saving (104.675 ms) by **8.917 ms**, which is — by the additive construction of the
decomposition — precisely the sum of every non-H row's delta. **That 8.917 ms is scatter, not a
secondary effect**, so the honest reading of the headline is "H fell 104.675 ms and the call fell
113.592 ms on a host whose rows scatter 17–22 %", not "104.675 ms plus 8.9 ms of side benefits".

Groups, before and after (percentages of each arm's own clean call):

| group | control ms | control % | feature ms | feature % |
|---|---:|---:|---:|---:|
| **F + λH** | **114.931** | **27.81** | **10.094** | **3.37** |
| D and F assembly | 98.183 | 23.76 | 95.528 | 31.87 |
| Log determinants + evidence | 81.354 | 19.68 | 80.024 | 26.70 |
| Positivity solve | 80.126 | 19.39 | 77.122 | 25.73 |
| Mesh (ray trace, placement, triangulation) | 20.145 | 4.87 | 19.992 | 6.67 |
| Mapper + mapping matrix | 10.797 | 2.61 | 9.516 | 3.18 |
| Mapper weights (Delaunay) | 3.781 | 0.91 | 3.745 | 1.25 |
| Image + blurring | 2.712 | 0.66 | 2.702 | 0.90 |
| unattributed | 1.273 | 0.31 | 0.986 | 0.33 |

The `F + λH` group was the largest in the call and is now the second smallest of the four big ones.
**After lever 1 the call's shape is: assembly 32 %, log-dets 27 %, solve 26 %, mesh 7 %.**

## Lever 1 — the A100 row and the compaction finding

Lever 1 was expected to be invisible on the GPU and it is. The mechanism, verified against the code
rather than assumed:

1. `delaunay.py`'s `_setup_prefix_fn` (`scripts/imaging/likelihood_breakdown/delaunay.py:927`) has an
   `upto == 11` branch (line 992) which calls
   `pixelization_obj.regularization.regularization_matrix_from(linear_obj=m, xp=jnp)` **inside
   `jax.jit`** — `jit_profile(_setup_prefix_fn(11), "regularization_matrix_jit", …)` at line 1338.
2. With `xp=jnp`, `AdaptSplit` dispatches to the **JAX** branches of `reg_split_from` and
   `pixel_splitted_regularization_matrix_from` — `#536`'s compacted scatter. The numpy branch lever 1
   rewrote (`reg_split_np_from` / `pixel_splitted_regularization_matrix_np_from`) is **never reached**
   on this cell.

**The finding: the two compaction constants are inert at this configuration.**
`InterpolatorDelaunay` (`delaunay.py:747`, `_INTERPOLATOR_CLS = al.InterpolatorDelaunay`) produces a
stencil table of width **K = 4** — three barycentric columns plus one reserved pad column for the
centre-pixel insertion — which the witness JSON records as `stencil_mappings_shape` **(6000, 4)** with
`stencil_sizes_max` **3**. In `pixel_splitted_regularization_matrix_from`
(`regularization_util.py:462`):

- `kc = min(K, int(compact_width))` = `min(4, 12)` = **4** (line 568), so the compact head scatter
  `reg_mat.at[rows, cols].add(outer_scaled)` already covers **every** column of the table;
- `if K > kc:` (line 583) is a **static Python branch on a shape**, so with K = 4 it is not taken and
  XLA never traces it: **no `jax.lax.top_k`, no `(W, K, K)` wide-row supplement, no `head_block`
  mask, `overflow = None`, and no `jnp.where(overflow, jnp.nan, reg_mat)` poison guard is emitted at
  all.**

Therefore **`SPLIT_REG_COMPACT_WIDTH = 12` and `SPLIT_REG_WIDE_ROW_BUDGET = 256` do nothing at HST
Delaunay N=1500 with `InterpolatorDelaunay`.** They bite only when K > 12 — the natural-neighbour
Sibson path, where the constants' own docstring records K = 33 (`SIBSON_MAX_NEIGHBORS` 32 + 1 spare
column) and an occupied width of min 1 / median 5 / p99 9 / max 11. **Any retune of either constant
must be measured on a `DelaunayNN` cell; a measurement on this Delaunay row cannot see them.**
It follows that the A100 identity result here is *not* evidence that the compaction is correctly
tuned — it is evidence that the compaction is not exercised.

For context (not a control): the pre-existing `delaunay_hpc_a100_fp64.json` row reads
`total_step_by_step` 0.067474 s, `regularization_matrix_prefix_s` 0.0073808, H step 0.00023435 s and
H @ vmap/16 **0.00034227 s**. It ran on **`euclid-ral-gpu-2`** with a different JAX cache directory
and a different PyAutoArray revision, so its 5.8× larger vmap H row is **not attributable to lever 1**
and is not part of this A/B.

## Lever 1 — verdict

1. **Lever 1 ships and is worth 1.379× on the production numba CPU route.** 413.301 → 299.709 ms
   mean (1.383× on medians), a saving of **113.592 ms** on a 404–413 ms call, single-threaded, HST
   Delaunay N=1500, sparse numba operator, route b, memo ON. Take it as a **lower bound**: the
   feature arm was the arm with the busier host.
2. **What shipped** (PyAutoArray `bae9296e`, merge base `5e2bc0f4`, branch
   `feature/fixed-light-numba-levers`): the two pure-Python loops behind `reg_split_np_from` and
   `pixel_splitted_regularization_matrix_np_from` in
   `autoarray/inversion/regularization/regularization_util.py` are now `@numba_util.jit()` kernels
   (`_reg_split_kernel`, `_pixel_splitted_regularization_matrix_kernel`) behind thin numpy wrappers,
   reproducing the reference accumulation order bit for bit. The Python bodies are **retained** as
   `_reg_split_reference` and `_pixel_splitted_regularization_matrix_reference` and pinned against the
   kernels with `np.array_equal` in
   `test_autoarray/inversion/regularizations/test_regularization_util_numba.py` (7 test functions, 8
   collected cases with the `NARROW_SIZES`/`WIDE_SIZES` parametrize). The **JAX branch is untouched**
   — PyAutoArray#536 owns it. PyAutoArray suites: regularizations 118, interpolator 58, inversion 212
   passed.
3. **Three deliberate behaviour changes, all recorded in the commit message and all covered by a
   test:**
   - `reg_split_np_from` **copies `mappings` and `sizes` on entry** (`np.array(...)`) instead of
     mutating the interpolators' `cached_property` stencil tables in place. Witness **W2** is the
     measurement of that on the production system: all three tables byte-identical after the
     inversion. Test: `test__reg_split__does_not_mutate_its_inputs`.
   - a **`size == 0` row inserts its centre pixel at column 0**, not at a stale `j + 1` — the old
     code leaked the loop variable from the previous row. Not exercised here
     (`n_zero_size_stencil_rows` is **0** on this system, from both the witness and W1), so this is a
     latent-correctness change, not a source of the speedup. Test:
     `test__reg_split__empty_row__inserts_at_column_zero`.
   - a **full stencil with no self pixel raises `exc.InversionException`** instead of writing past the
     table width; numba does no bounds checking, so the kernel returns the first overflowing row index
     and the wrapper raises. Test: `test__reg_split__full_row_without_self_pixel__raises`.
4. **The equivalence is bit-level, not tolerance-level, and it was witnessed on the production
   system.** W1: `np.array_equal` **true**, `max_abs_diff` **0.0** on the (1500, 1500) matrix. W3:
   `figure_of_merit` **19705.71758591176** on both paths, relative difference **0.0** against a 1e-09
   gate, with the monkeypatch confirmed fired (`{reg_split: 1, pixel_splitted: 1}`). Every `gates`
   field in the two A/B arms is identical. No number changed.
5. **On the A100 it is identity, and the negative finding is the useful one.** 0.036 % on the whole
   call, 0.028 % on the H prefix, `log_evidence_terms` bit-identical. The `#536` compaction constants
   are **unmeasurable** at K = 4 (see above), so nothing about them is validated or invalidated by
   this job.
6. **Where the H site now sits.** `inversion.regularization_matrix` was the **single largest site in
   the S3 call** in phase 2 (111.5 ms, 27 %); it is now **5.924 ms, 2.0 %**, and the witness measures
   the same matrix at 6.5 ms against the reference's 106.8 ms. The local laptop sanity check before
   the cluster run read 106.3 ms → 1.23 ms at P=1500, K=4 in isolation; the in-call site is larger
   than that because it includes the `AdaptSplit` weight construction and the split-point walk, not
   only the two kernels. **The lever the phase-2 note named as the biggest untouched one is closed.**

## Next

Lever 1 removed the largest site and did not touch the others. The residue of the 299.709 ms feature
call, from its own decomposition:

| site | feature ms | % of the 299.709 ms call | lever |
|---|---:|---:|---|
| `sparse_numba.curvature_matrix` | 87.153 | 29.08 | — (phase 2 concluded; not a phase-3 lever) |
| `solver.fnnls_cholesky` | 62.602 | 20.89 | — (phase 2 closed the solver) |
| `inversion.log_det_curvature_reg_matrix_term` | 39.983 | 13.34 | **lever 3** |
| `inversion.log_det_regularization_matrix_term` | 37.200 | 12.41 | **lever 2** |
| `delaunay.triangulation` | 18.662 | 6.23 | — |
| `inversion.regularization_matrix` | 5.924 | 1.98 | lever 1, done |

1. **Lever 2 — log det H from sparsity.** `inversion.log_det_regularization_matrix_term`, 37.468 ms in
   phase 2's Leg D and 37.200 ms in the lever 1 feature arm (memo-invariant, as the two agree to
   0.7 %). H is a split-regularization matrix: sparse and, for Delaunay, structurally banded once the
   pixels are ordered. Options to measure: `scipy.sparse.linalg.splu` on the CSR form; a **banded
   Cholesky after an RCM (reverse Cuthill-McKee) permutation**, whose log det is permutation-invariant;
   or caching the factor across evaluations keyed on the mesh **and** the adapt weights, since
   `AdaptSplit`'s weights move with the model while the mesh connectivity does not — the cache key is
   the whole question, and a wrong key is a silent wrong evidence.
2. **Lever 3 — one shared Cholesky of `F + λH`.** `inversion.log_det_curvature_reg_matrix_term`,
   40.172 ms in Leg D and 39.983 ms here. The solver already factorises `F + λH`
   (`solver.fnnls_cholesky`, 62.602 ms) and the log-det term then factorises it again. Lever 3 folds in
   the `curvature_reg_matrix` rebuild draft, **on the numpy/numba path only** — the JAX side is the GPU
   session's HLO census to answer, not this lever's. Together levers 2 and 3 address **77.183 ms,
   25.8 %** of the feature call.
3. **Both levers need the same A/B rig this one used**, and it worked: two private PyAutoArray
   checkouts prepended to `PYTHONPATH` in separate subshells, both arms in one SLURM job on one node,
   a bit-identity witness inside the feature arm, and an A100 row to prove the JAX branch did not move.
   Re-use `hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_ral_hst_fp64` and
   `hpc/batch_gpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_a100_hst_fp64`.
4. **If either compaction constant is ever retuned**, the cell must be `DelaunayNN`. This Delaunay row
   cannot observe them (K = 4 ⇒ `kc` = 4 ⇒ the wide-row branch is never traced), so a Delaunay
   measurement is not evidence about them in either direction.
5. **Still unfiled, carried from phase 2**: the PyAutoArray bug that
   `abstract_ndarray.__getitem__` imports `jax.numpy`, so a numba-only process cannot stay JAX-free
   after the first `FitImaging` (`jax_in_sys_modules: true` in this phase's witness JSON too).
6. **Out of scope throughout, as in phase 2**: route a, route c, rectangular meshes, Euclid, the
   source-pixel sweep, and the sparse-operator/profile-subtracted-image bug.

## Artifacts

Result artifacts added by this commit (**9 files**):

```
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_lever1_control_b_warm_t1.{json,png}  # CPU control, 5e2bc0f4
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_lever1_b_warm_t1.{json,png}          # CPU feature, bae9296e
results/breakdown/imaging/fixed_light_numba_levers_witness_hpc_ral_cpu_fp64_lever1.json                                      # W1/W2/W3, feature arm
results/breakdown/imaging/delaunay_hpc_a100_fp64_lever1_control.{json,png}                                                   # A100 control, 5e2bc0f4
results/breakdown/imaging/delaunay_hpc_a100_fp64_lever1.{json,png}                                                           # A100 feature, bae9296e
```

Read alongside (already versioned, **not** part of this A/B):

```
results/breakdown/imaging/delaunay_hpc_a100_fp64.json                                              # prior A100 row, context only (ran on gpu-2)
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_sparse_b_warm_t1.json  # phase 2 Leg D, the anchor
```

Two laptop witness JSONs written while developing the witness script
(`fixed_light_numba_levers_witness_local_laptop_lever1{,_postlint}.json`) were **deliberately not
versioned**: they are laptop test artifacts, not cluster rows, and the laptop's clean spreads make it
useless as a timing host for this cell (phase 2, Scope).

Cells and submit scripts:

```
scripts/imaging/likelihood_breakdown/fixed_light_numba.py                 # the CPU cell (phase 2's Leg D)
scripts/imaging/likelihood_breakdown/fixed_light_numba_levers_witness.py  # the W1/W2/W3 witness
scripts/imaging/likelihood_breakdown/delaunay.py                          # the A100 cell
scripts/misc/likelihood_breakdown/call_accounting.py                      # the decomposition instrument
hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_ral_hst_fp64   # job 343345, both CPU arms + witness
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_a100_hst_fp64  # job 343346, both A100 arms
```

Library change: **PyAutoArray `bae9296ed528da6812613edc2ebc6ea367361ebb`** on
`feature/fixed-light-numba-levers`, merge base
`5e2bc0f42940adfe04ff30fa42a7ce13a86ecbe0` — `autoarray/inversion/regularization/regularization_util.py`
(+181/−17) and `test_autoarray/inversion/regularizations/test_regularization_util_numba.py` (+295).

Job logs (SLURM, **not versioned** — `output/` is gitignored): `output/output.343345.out`,
`output/output.343346.out` and `error/error.34334{5,6}.err`, on
`/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-levers` and mirrored into this worktree's
untracked `output/`.

JSON keys specific to these arms: `rows["b_sparse_numba"]` with `call_ms` / `call_ms_median` /
`call_ms_sequence` / `clean_relative_spread` / `abba` / `instrumentation_overhead_ratio`,
`steps[*]` (`excl_s` already rescaled, plus `excl_s_unscaled`, `incl_s`, `n_calls`, `group`) including
the explicit `unattributed (call − sum of exclusive times)` row, `decomposition_rescale_factor`,
`solver_stats["solver.fnnls_cholesky"]`, `gates`, `contention`; and on the A100 side `steps`,
`steps_vmap_per_call`, `setup_split`, `regularization_matrix_prefix_s`, `interpolator_prefix_s`,
`jit_phases[*].compile_s`, `log_evidence_terms`, `nnls`, `total_step_by_step`.

<!-- Levers 2 and 3 append their own "Lever N — headline / decomposition / A100 row / verdict"
     sections below this line, and extend the Provenance, Gate roll-up and Artifacts sections. -->
