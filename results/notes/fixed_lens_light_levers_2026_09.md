# Fixed lens light on the numba CPU path — the three levers (2026-09-16)

autolens_profiling issue [#267](https://github.com/PyAutoLabs/autolens_profiling/issues/267),
epic `fixed-lens-light-numba-cpu` phase 3, branches `feature/fixed-light-numba-levers`
(lever 1) and `feature/fixed-light-numba-levers-l2` (lever 2, stacked on it).
Phase 2 ([`fixed_lens_light_numba_2026_09.md`](./fixed_lens_light_numba_2026_09.md)) closed the
solver and left a memo-invariant, non-solver residue in the production numba CPU likelihood call:
`inversion.regularization_matrix` at 111.5 ms, the two log-dets at 40.2 + 37.5 ms, and
`sparse_numba.curvature_matrix` at 87.4 ms of a 404.6 ms call. Phase 3 attacks that residue in three
levers, each paired with an A100 row: **lever 1** numba-jit the split-regularization assembly,
**lever 2** log det H from sparsity, **lever 3** one shared Cholesky of `F + λH`.
**Levers 1 and 2 are measured; lever 3 is pending** and will be appended here as its own sections.

**Lever 1's result: 1.379× on the production numba CPU route (413.301 → 299.709 ms), and identity
on the A100.** `inversion.regularization_matrix` falls 110.599 → 5.924 ms — 26.8 % of the call to
2.0 % — with the regularization matrix bit-identical, the interpolators' cached stencil tables
unmutated, and the fit's `figure_of_merit` unchanged at relative difference exactly 0.0. The A100
arm is identity to 0.036 % on the whole call because lever 1 changed only the `xp==np` branch, which
the GPU cell never reaches; its one finding is negative and structural — **the `#536` compaction
constants `SPLIT_REG_COMPACT_WIDTH` and `SPLIT_REG_WIDE_ROW_BUDGET` are inert at this
configuration**, because `InterpolatorDelaunay`'s stencil table is K = 4 wide.

**Lever 2's result: a further 1.132× on top of lever 1 (302.709 → 267.448 ms), identity on the
A100's jitted rows, and a CPU-only verdict.** `inversion.log_det_regularization_matrix_term` falls
37.480 → 6.508 ms — 12.38 % of the call to 2.43 % — with the sibling `F + λH` log-det term flat at
−1.6 % and every other row inside the host's scatter. Cumulatively levers 1 and 2 take the
merge-base call **413.301 → 267.448 ms, 1.545×**, removing 145.853 ms. Unlike lever 1, lever 2 is
**not** bit-identical: SuperLU and a dense Cholesky are two different fp64 factorisations of the same
matrix, whose condition number here is 6.491e12, so the log determinant moves −9.213e-6 nats
(relative 1.201e-9) and the fit's `figure_of_merit` −4.606e-6 nats (relative 2.338e-10). The A100
arm's six `log_evidence_terms` are bit-identical, and its own **dense** Cholesky of the same H costs
**1.126 ms** against 37.480 ms on one CPU core — so the sparse route is not worth carrying to the
GPU, and lever 2 is recorded as a CPU-only lever on the strength of that measurement rather than of
an argument.

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

Six more caveats belong to **lever 2** only:

- **Lever 2's control arm is lever 1, not the merge base.** Lever 2's commit is stacked on lever 1,
  so a merge-base control would measure both levers at once and attribute the sum to lever 2. The
  A/B therefore measures the **increment**: control PyAutoArray `bae9296e` (lever 1's feature
  revision) against feature `0b17c292`. That control reproduces lever 1's feature arm to **+1.00 %**
  (302.709 against 299.709 ms; the log-det site 37.480 against 37.200 ms, +0.75 %), which is what
  makes the two jobs comparable. The cumulative figure against the merge base is quoted two ways —
  the direct ratio 413.301 → 267.448 ms = **1.545×** and the product of the two measured rows
  1.379 × 1.132 = 1.561× — and they differ by exactly that 1.0 %. **Quote 1.545×**; the product is
  the same number's upper end.
- **Lever 2 changes numbers, at the size of the matrix's own round-off — "bit-identical" is lever
  1's claim, not lever 2's.** SuperLU in symmetric mode and a dense Cholesky are two different fp64
  factorisations of one matrix, and `regularization_matrix_reduced` on this system has eigenvalues
  1.000e-08 … 6.492e+04, **condition number 6.491e12** (witness, `eigvalsh`). The log determinant
  moves **−9.213e-06 nats (relative 1.201e-09)** and the fit's `figure_of_merit`
  **−4.606e-06 nats (relative 2.338e-10)**. Always quote those with the conditioning beside them: at
  cond 6.5e12 an fp64 factorisation already carries orders of magnitude more relative error in the
  *matrix* than 1e-9, so this is the matrix's round-off and not an error introduced on top of it.
- **The two arms' `log_likelihood` differ, and it is the same round-off.** Control
  **18665.014334145228**, feature **18665.014330695830**, relative **1.848e-10**, on the same iid
  instance (`iid_seed` 263, same route set, same 32-entry sequence shape). Record it as an
  observation of the value change; it is still **never** a cross-leg comparison and still not the
  equivalence evidence — that is W2/W3 and the gates.
- **"Identity on the A100" means the jitted rows.** The call site is
  `self._log_det_symmetric_from(self.regularization_matrix_reduced, sparse=self._xp is np)`, so
  `sparse` is false whenever `xp` is `jnp`, and the helper is SciPy — neither traceable nor
  differentiable. All six `log_evidence_terms` of both systems are bit-identical across the two A100
  arms. But `fixed_light.py` also builds an **eager `xp=np`** fit, and that one *does* take the
  sparse route: `log_evidence_eager` and both systems' `log_evidence_figure_of_merit` move
  **−3.4466e-06 nats (relative 1.183e-10)**. That was predicted in the submit header and is not a
  finding.
- **Lever 2 is a CPU-only lever, and that rests on a measurement.** The A100's *dense* Cholesky of
  the same (1500, 1500) H is **1.126 ms** (`s3/rows["Log det Cholesky (H reduced)"]`, feature arm)
  against **37.480 ms** on one CPU core — **33×**. The CPU's *sparse* route, at 6.265 ms (witness
  W5), is still **5.6× slower than the GPU's dense one**. At ~1.1 ms the GPU log det is launch-bound,
  not FLOP-bound, and a sparse triangular factorisation has no dense-BLAS kernel to beat there.
- **The two regime switches are constants, not parameters this row tunes.**
  `SPARSE_LOG_DET_MIN_PIXELS = 256` and `SPARSE_LOG_DET_MAX_NNZ_PER_ROW = 32` were calibrated on
  off-cluster sweeps (recorded in "Lever 2 — verdict"). What this row measures is only that **both
  gates admit the production matrix, with margin**: W4 records pixels 1500 ≥ 256 by 1244 and
  nnz/row 8.493 ≤ 32 with a per-row maximum of 15. A retune of either constant needs its own sweep
  on its own cell, exactly as the lever 1 compaction constants do.

## Provenance

### Lever 1 — jobs 343345 / 343346

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

### Lever 2 — jobs 343353 / 343354

Two SLURM jobs, the same node, the same day. The A100 job was submitted only **after** the CPU job
had finished, so — unlike lever 1 — neither job shared the host with the other.

| | CPU A/B + witness | A100 A/B |
|---|---|---|
| Job | **343353**, `COMPLETED` | **343354**, `COMPLETED` |
| Elapsed / MaxRSS | 00:01:56 / 1 974 132 K of 32 GB | 00:01:58 / 5 431 032 K of 64 GB |
| Node | **euclid-ral-gpu-1** | **euclid-ral-gpu-1** |
| Partition | `gpu` CPUs-only, `--cpus-per-task=4 --mem=32gb`, **no `--gres`** | `gpu`, `--gres=gpu:1 --cpus-per-task=4 --mem=64gb` |
| Device | AMD EPYC 7702 64-Core, `os.cpu_count()` 124, kernel 5.14.0-687.39.1.el9_8 | NVIDIA A100 80GB PCIe (`nvidia_smi`), `cuda:0` |
| Job loadavg, entry → exit | `0.00 0.00 0.04` → `0.86 0.32 0.15` | `0.34 0.27 0.14` → `1.75 0.82 0.36` |
| Cell | `scripts/imaging/likelihood_breakdown/fixed_light_numba.py` | `scripts/imaging/likelihood_breakdown/fixed_light.py` (**not** `delaunay.py`) |
| Python / autolens | 3.12.4 / 2026.8.17.1 | 2026.8.17.1 |
| `AP_ROOT` | `/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-levers` @ `289cea39`, branch `feature/fixed-light-numba-levers-l2` | same worktree |

Per-arm checkouts and config names:

| Arm | PyAutoArray checkout | rev | config name |
|---|---|---|---|
| CPU control (= **lever 1**) | `/mnt/ral/jnightin/PyAuto_wt/fixed-light-numba-levers/PyAutoArray_feature` | `bae9296ed528da6812613edc2ebc6ea367361ebb` | `hpc_ral_cpu_fp64_fixed_light_numba_lever2_control_b_warm_t1` |
| CPU feature | `.../PyAutoArray_l2` | `0b17c292afc467dff1dbb0c171888daf8bb62731` | `hpc_ral_cpu_fp64_fixed_light_numba_lever2_b_warm_t1` |
| CPU witness | `.../PyAutoArray_l2` | `0b17c292` | `hpc_ral_cpu_fp64_lever2` |
| A100 control (= **lever 1**) | `.../PyAutoArray_feature` | `bae9296e` | `hpc_a100_fp64_fixed_light_lever2_control` |
| A100 feature | `.../PyAutoArray_l2` | `0b17c292` | `hpc_a100_fp64_fixed_light_lever2` |

The merge-base checkout `PyAutoArray_control` (`5e2bc0f4`) still exists on the cluster from lever 1
and was deliberately **not** used by either job.

**The in-job proof that the two arms differ by exactly lever 2** is a per-arm echo taken before the
cell runs: each arm prints `autoarray.__file__`, its checkout's `rev-parse HEAD`, and whether
`inversion_util.SPARSE_LOG_DET_MIN_PIXELS` and `SPARSE_LOG_DET_MAX_NNZ_PER_ROW` exist in the
*imported* module. Control arm: `bae9296e…`, `PyAutoArray_feature/autoarray/__init__.py`, both
constants **absent**. Feature arm: `0b17c292…`, `PyAutoArray_l2/autoarray/__init__.py`, both
constants **present**, `256` and `32` (`output/output.343353.out`, lines 36–40 and 142–149).

The rest of the stack is the shared RAL install, untouched and identical for all five arms — the
same five revisions as lever 1's table above (PyAutoNerves `fac8b17b`, PyAutoFit `27d41e7c`,
PyAutoArray shared-install `5e2bc0f4`, PyAutoGalaxy `840ffde0`, PyAutoLens `ccf9295f`) — and the
system under test is the same HST / Hilbert-Delaunay N=1500 / `AdaptSplit(inner=0.1, outer=10.0,
signal_scale=0.1)` / fp64 / `InversionImagingSparseNumba` configuration, with
`image_pixels_masked` **15 361**, `over_sampled_pixels` **62 752** and `n_edge_zeroed` 0 in both CPU
arms. The A100 arms run that same dataset and mesh on the dense JAX path
(`inversion_path: "dense"`, `vmap_batch` 16, the same three XLA flags as lever 1) with a **fresh
compilation and autotune cache per arm** (`autotune_cache_entries_at_start` 0, `cache_fresh: true`,
separate `output/jax_cache/fixed_light_lever2_{control,feature}_343354` directories).

## Gate roll-up

### Lever 1

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

### Lever 2

Cell gates, both CPU arms of job 343353 (same cell, same `gate_thresholds`):

| Gate | control | feature | value |
|---|---|---|---|
| `P1_rebaked_operator_equals_original` | **PASS** | **PASS** | 3/3 arrays identical, identical in both arms |
| `S3_mapper_block_equals_S0` | **PASS** | **PASS** | log-dets rel diff **0.0** both arms, edge-zeroed 0 vs 0; the `log_det_regularization` *value* differs between arms (below) |
| `P2_dense_equals_sparse_numba_system` | **PASS** | **PASS** | D 1.297e-15, F(mapper) 3.728e-15 (rtol 1e-09), identical in both arms |
| `P3_dense_equals_sparse_numba_evidence` | **PASS** | **PASS** | Δ −9.094947e-11 nats, **0** passive-set differences, 1485 passive both sides, identical in both arms |
| `instrumentation_overhead_ratio` (≤ 1.03) | 1.01671 **PASS** | 1.02206 **PASS** | 8 ABBA blocks each |
| `unattributed_fraction` (≤ 0.05) | 0.00334 | 0.00328 | explicit measured remainder |
| `cached_site_call_count_violations` | `{}` | `{}` | — |
| `timing_status` / `contention_warning` | measured / false | measured / false | peak 1-min loadavg **0.57** / **0.81** on 124 cores |

**Exactly five `gates` fields differ between the two arms, and every one of them is derived from the
log determinant of H:**

| field | control | feature | Δ |
|---|---|---|---|
| `S3_mapper_block_equals_S0.log_dets.log_det_regularization_matrix_term.{s0,s3}` | 7670.876894312947 | 7670.876885100281 | −9.213e-06 nats, rel 1.201e-09 |
| `P3_dense_equals_sparse_numba_evidence.figure_of_merit_sparse_numba` | 19705.71758591176 | 19705.717581305427 | −4.606e-06 nats, rel 2.338e-10 |
| `P3_dense_equals_sparse_numba_evidence.figure_of_merit_dense` | 19705.71758591185 | 19705.71758130552 | −4.606e-06 nats |
| `P3_dense_equals_sparse_numba_evidence.rel_diff` | 4.6153848384752585e-15 | 4.615384839554133e-15 | last two digits |

`figure_of_merit_dense` moving too is the expected reading, not an anomaly: the P3 gate's *dense*
comparison fit is also a numpy fit, so it takes the same sparse route, and the gate stays PASS
because it compares the two formalisms **within** an arm. Every other `gates` field — P1, P2, the
passive sets, the `log_det_curvature_reg_matrix_term` log-dets (8326.70017133483 in both arms, rel
diff 0.0) — is **byte-identical between the arms**. `pinned_expected` is `null` and `pinned_drift`
empty in both, as in lever 1: every millisecond is RECORDED.

Solver state, both arms, identical: `seed_source` `"memo"`, `warm_start_fallback` `false`,
`n_passive` **1484**, `outer_iterations` 2, `inner_iterations` 1, `warm_start_errors` 5,
`memo_cleared_within_block: false`. The solver saw the same problem in both arms, warm, which is
what makes the −2.041 ms on `solver.fnnls_cholesky` readable as noise rather than as an effect.

Witness (`fixed_light_numba_levers_l2_witness.py`, **verdict PASS**, feature arm's subshell,
`slurm_job_id` 343353, wall 13.63 s, scipy 1.17.1, `nnls_warm_start_env: "0"`):

| Gate | status | evidence |
|---|---|---|
| **W1** the sparse route fires | **PASS** | `log_det_sparse_spd_from` called **exactly once** during one `fit.figure_of_merit`, on a **(1500, 1500)** matrix, returning a float — **7670.876885100281** in **7.15 ms**. A count of 2 would mean the route leaked into `log_det_curvature_reg_matrix_term`; a `None` return would mean the fit silently took the dense Cholesky and every other check compared the dense route with itself |
| **W2** the log determinant | **PASS** | sparse 7670.876885100281 vs dense-forced 7670.876894312947, **Δ −9.213e-06 nats, rel 1.201e-09** against a **1e-08** gate, with `h_identical_between_fits` **true** (so this is two factorisations of one matrix). Conditioning recorded beside it: eigenvalues **1.0001215e-08 … 64922.433**, **cond 6.491e12** (`eigvalsh`, 0.19 s). The dense route is forced through the library's own fallback by monkeypatching the helper to return `None` (patch fired, 2 calls), not by a hand-rolled copy of the Cholesky |
| **W3** the likelihood | **PASS** | `figure_of_merit` sparse **19705.717581305427** vs dense **19705.71758591176**, **Δ −4.606e-06 nats, rel 2.338e-10** against a **1e-09** gate — about 4× of margin. This is the phase's tighter pin: a relative move in one of six evidence terms is diluted by the sum |
| **W4** both regimes admit | **PASS** | pixels **1500** ≥ `SPARSE_LOG_DET_MIN_PIXELS` **256** (margin 1244); nnz **12 740** = **8.4933/row** mean ≤ `SPARSE_LOG_DET_MAX_NNZ_PER_ROW` **32**, per-row max **15**, density **5.662e-03**. W1 says the route fired; W4 says why, and with how much margin |
| **W5** the site, in isolation | **PASS** | same matrix, median of 20 each at 1 thread: sparse **6.265 ms** vs dense **37.720 ms** = **6.02×**, same-matrix Δ −9.213e-06 nats (rel 1.201e-09). RECORDED, not pinned — the lever's number is the whole-call A/B, not this |

Witness system block: `InversionImagingSparseNumba`, `InterpolatorDelaunay`, `AdaptSplit`,
`n_params` **1500**, `n_funcs` **0**, `edge_zeroed_pixels` **0**, H shape (1500, 1500),
subtracted light flux 4945.524141998385, `log_det_method` `"cholesky"` (the default — the
`"slogdet"` mode is untouched by lever 2), `jax_in_sys_modules` **true** (the unfiled PyAutoArray
bug, as in lever 1).

A100 assertions, both arms (`output/output.343354.out`): **"Pin PASSED: S0 log_det_curvature_reg
matches 8360.401763"** and **"Pin PASSED: S0 log_det_regularization matches 7756.614959"** in each
arm, `mapper_block_log_det_status` **ASSERTED** at rtol 1e-06 and `pinned_drift` **empty** in both
JSONs. `nnls` is 17 iterations (S3) and 22 (S0) in both, `converged: true`, with
`full_system_pdip_vs_library_reconstruction_max_abs_diff` **3.254729818991109e-12** — the same
digits — and `peak_bytes.after_all` **1 810 646 016** in both arms.

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

## Lever 2 — headline

### Host 1: `euclid-ral-gpu-1`, numba CPU

`use_jax: false`, `backend: "numba_cpu"`, `InversionImagingSparseNumba`, `n_threads` 1,
`NUMBA_NUM_THREADS` `"1"`, `OMP`/`OPENBLAS`/`MKL`/`VECLIB`/`NUMEXPR` all `"1"`,
`cpu_over_wall` 1.0004 (control) / 1.0003 (feature). Clean (uninstrumented) calls, n-repeats 16,
warm-up run to a steady state in 6 calls in both arms.

| arm | PyAutoArray | mean ms | median ms | min | max | clean spread | ABBA | `inversion.log_det_regularization_matrix_term` ms | as % of call |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control (**lever 1**) | `bae9296e` | **302.709** | 299.781 | 288.551 | 352.546 | 21.14 % | 1.0167 | **37.480** | 12.38 % |
| feature (lever 2) | `0b17c292` | **267.448** | 264.081 | 254.461 | 317.170 | 23.45 % | 1.0221 | **6.508** | 2.43 % |

| | value |
|---|---:|
| speedup on means | **1.132×** |
| speedup on medians | **1.135×** |
| whole-call saving | **35.261 ms** (means), 35.700 ms (medians) |
| log-det-H site saving | **30.972 ms** (37.480 → 6.508, **5.76×** on that site) |
| cumulative vs the merge base, direct | 413.301 → 267.448 ms = **1.545×**, **145.853 ms** removed |
| cumulative as the product of the two measured rows | 1.379 × 1.132 = 1.561× |

**Both arms ran alone on the host.** `contention.load_average_at_start` is `0.49` (control) and
`0.78` (feature) on **124 cores**, rising to peaks of 0.57 and 0.81 — figures a single-threaded
process generates by itself as it accumulates into the exponential average, and the shell echo at
each arm's entry says the same (`0.00 0.00 0.04` before the control arm, `0.60 0.17 0.10` before the
feature arm, `0.86 0.32 0.15` at job exit). There was no second tenant: the A100 job was submitted
only after this one completed. So unlike lever 1's 1.379×, **1.132× is not a lower bound** — it is
the number, on a quiet host, with the usual 21–23 % per-call scatter around it. Both arms stamp
`timing_status: "measured"` and `contention_warning: false`.

Anchor on lever 1's feature arm (job 343345, the same cell, same node, same flags): mean
299.709 → **302.709** (**+1.00 %**), median 297.262 → 299.781 (+0.85 %), log-det-H site
37.200 → 37.480 (+0.75 %). **The control reproduces lever 1**, which is what licenses multiplying
the two rows; the 1.0 % gap is also exactly the difference between the direct cumulative ratio
(1.545×) and the product (1.561×).

Both arms ran 16 clean and 16 instrumented calls in 8 counterbalanced ABBA blocks, and the
decomposition closes exactly: 48 instrumented sites (37 reached) plus the explicit
`unattributed (call − sum of exclusive times)` row sum to the clean call to the last printed digit,
**302.709** and **267.448**. `unattributed` is 1.027 ms (0.33 %) and 0.896 ms (0.33 %) of the two
calls unrescaled, on a 5 % gate.

### Host 2: `euclid-ral-gpu-1`, A100 80GB PCIe, JAX/XLA

`backend: "gpu"`, `device: "cuda:0"`, fp64, `--mesh delaunay --regularization adapt_split
--vmap-batch 16`, dense inversion path, the same three XLA flags as lever 1, and a fresh JAX
compilation cache per arm (`cache_fresh: true`, `autotune_cache_entries_at_start` 0). The cell is
**`fixed_light.py`**, not lever 1's `delaunay.py`: lever 1 changed the *assembly* of H, which
`delaunay.py` times as a jitted-prefix difference, whereas lever 2 changes the *factorisation* of H,
and `delaunay.py` has no log-det row at all. **There is therefore no lever-1 → lever-2 continuity
row on the A100**; the anchor is the fiducial `fixed_light_delaunay_hpc_a100_fp64_fixed_light.json`,
and it ran on `euclid-ral-gpu-2`, so it is a scale check and not a control.

| `s3/rows` (ms) | control | feature | Δ | fiducial (gpu-2) |
|---|---:|---:|---:|---:|
| **`Log det Cholesky (H reduced)`** | **1.148** | **1.126** | −1.9 % | 1.204 |
| `Log det Cholesky (F+λH reduced)` | 1.140 | 1.134 | −0.5 % | 1.171 |
| `curvature_reg_matrix build (dense)` | 4.819 | 4.835 | +0.3 % | 4.918 |
| `Cholesky solve (unconstrained)` | 1.461 | 1.456 | −0.3 % | 1.485 |
| `NNLS PDIP (cell-driven, max_iter 50)` | 28.211 | 28.217 | +0.0 % | 28.347 |
| `log_evidence_terms (eager, PDIP solution)` | 60.967 | 54.102 | −11.3 % (**scatter**, see below) | 63.319 |
| `library_row.s3_ms` | 49.194 | 49.198 | +0.0 % | 49.169 |
| `library_row.s0_ms` | 65.225 | 65.025 | −0.3 % | 70.406 |

**Identity, as constructed.** The jitted rows move by tenths of a percent, and the numbers do not
move at all: see the next section.

## Lever 2 — decomposition

Exclusive (self) time per instrumented site, rescaled by `decomposition_rescale_factor`
(0.98439 control / 0.97942 feature) so the rows sum to the clean call. `n_sites_instrumented` 48 and
`n_sites_reached` 37 in both arms. Top rows by control time:

| site | group | control ms | feature ms | Δ ms | Δ % |
|---|---|---:|---:|---:|---:|
| `sparse_numba.curvature_matrix` | D and F assembly | 86.960 | 86.949 | −0.011 | −0.0 |
| `solver.fnnls_cholesky` | Positivity solve | 64.530 | 62.489 | −2.041 | −3.2 (noise) |
| `inversion.log_det_curvature_reg_matrix_term` | Log dets + evidence | 40.232 | 39.596 | −0.636 | **−1.6 (the sibling term — flat)** |
| `inversion.log_det_regularization_matrix_term` | Log dets + evidence | **37.480** | **6.508** | **−30.972** | **−82.6** |
| `delaunay.triangulation` | Mesh | 18.493 | 18.517 | +0.024 | +0.1 |
| `sparse_numba.psf_weighted_data` | D and F assembly | 8.232 | 8.173 | −0.059 | −0.7 |
| `solver.reconstruction_positive_only_from` | Positivity solve | 7.601 | 7.425 | −0.176 | −2.3 |
| `inversion.reconstruction` | Positivity solve | 7.541 | 6.740 | −0.801 | −10.6 (noise) |
| `to_inversion.lp_linear_func_list_galaxy_dict` | Mapper | 6.036 | 5.981 | −0.055 | −0.9 |
| `inversion.regularization_matrix` | F + λH | 5.941 | 5.965 | +0.023 | +0.4 (lever 1's site, stable) |
| `inversion.curvature_reg_matrix` | F + λH | 4.271 | 4.081 | −0.190 | −4.5 |
| `unattributed (call − sum of exclusive times)` | — | 1.011 | 0.878 | −0.133 | −13.2 |

**One row moved, and the row that must not move did not.**
`inversion.log_det_curvature_reg_matrix_term` — the `F + λH` log det, which calls
`_log_det_symmetric_from` **without** `sparse=True` because `F + λH` is dense — is
40.232 → 39.596 ms, **−1.6 %**, against a clean spread of 21–23 %. That is the check that the sparse
route did not leak into the wrong term, and it agrees with W1's call count of exactly one.

**Two rows moved by more than half a millisecond and both are noise.**
`solver.fnnls_cholesky` (−2.041 ms, −3.2 %) and `inversion.reconstruction` (−0.801 ms, −10.6 %) sit
in a group lever 2 cannot reach: it changed one method's factorisation of H and nothing in the
solver. `solver_stats` is identical in the two arms to the last field (`seed_source` `"memo"`,
`n_passive` 1484, 2 outer / 1 inner iteration, 5 warm-start errors), so the solver did the same work
in both — the row is the same work, timed twice on a host whose rows scatter 21–23 %.

The whole-call saving (35.261 ms) exceeds the site saving (30.972 ms) by **4.289 ms**, which is — by
the additive construction of the decomposition — precisely the sum of every non-log-det row's delta.
**That 4.289 ms is scatter, not a secondary effect.** The honest reading is "the log-det-H site fell
30.972 ms and the call fell 35.261 ms on a host whose rows scatter 21–23 %".

Groups, before and after (percentages of each arm's own clean call):

| group | control ms | control % | feature ms | feature % |
|---|---:|---:|---:|---:|
| D and F assembly | 95.357 | 31.50 | 95.282 | 35.63 |
| **Log determinants + evidence** | **80.596** | **26.62** | **48.854** | **18.27** |
| Positivity solve | 79.677 | 26.32 | 76.659 | 28.66 |
| Mesh (ray trace, placement, triangulation) | 19.813 | 6.55 | 19.840 | 7.42 |
| F + λH | 10.277 | 3.40 | 10.107 | 3.78 |
| Mapper + mapping matrix | 9.513 | 3.14 | 9.432 | 3.53 |
| Mapper weights (Delaunay) | 3.701 | 1.22 | 3.730 | 1.39 |
| Image + blurring | 2.763 | 0.91 | 2.665 | 1.00 |
| unattributed | 1.011 | 0.33 | 0.878 | 0.33 |

The log-determinant group was the second largest in the lever 1 call and is now the third, and
**39.596 of its remaining 48.854 ms is the single `F + λH` term lever 3 targets** — 81 % of the
group. **After lever 2 the call's shape is: assembly 36 %, solve 29 %, log-dets 18 %, mesh 7 %.**

## Lever 2 — the A100 row and the CPU-only verdict

The A100 job asks one question, and it is not "is lever 2 faster on the GPU" — it cannot be, it
never runs there (`sparse=self._xp is np`, and the helper is SciPy). The question is whether the
sparse route is worth **carrying** to the JAX path in a later lever. The row that answers it is the
A100's own **dense** Cholesky of the same (1500, 1500) H:

| factorisation of the same H | ms |
|---|---:|
| dense Cholesky, one CPU core (numba CPU arm's site, control) | **37.480** |
| dense Cholesky, one CPU core (witness W5, same matrix, median of 20) | 37.720 |
| **sparse SuperLU, one CPU core** (witness W5, median of 20) | **6.265** |
| **dense Cholesky, A100, jitted** (`s3/rows`, feature arm) | **1.126** |
| dense Cholesky, A100, jitted (fiducial record, gpu-2) | 1.204 |

**Verdict: a CPU-only lever.** The GPU's *dense* factorisation is **33× faster than the CPU's dense
one** and still **5.6× faster than the CPU's sparse one**. At ~1.1 ms a dense Cholesky of a 1500²
SPD matrix on an A100 is launch-bound, not FLOP-bound, and a sparse triangular factorisation — a
sequential, data-dependent elimination — has no dense-BLAS kernel to beat there. Its sibling row
`Log det Cholesky (F+λH reduced)` at 1.134 ms is the scale check: **on the GPU the two log dets cost
the same**, whereas on the CPU only one of them is now sparse. Carrying the sparse route to JAX
would buy at most 1.1 ms of a 49 ms call and would cost the differentiability of the term.

**The numbers did not move on the jitted path.** All six `log_evidence_terms` are bit-identical
across the two arms **and** against the fiducial record, in both systems. S3's six, in full:
`chi_squared` 19821.951110654016, `regularization_term` 928.9375270506275,
`log_det_curvature_reg` 8360.401762997288, `log_det_regularization` **7756.614958909804**,
`noise_normalization` −79635.26723742482, `log_evidence` **29140.295897816348**. S0's differ from
those only where the joint system differs from the source-only one (`chi_squared`
19821.951110653972, `regularization_term` 928.9375270502924, `log_evidence` 29140.295897816537, the
two log-dets identical), and are likewise the same digits in both arms.
`mapper_block_log_dets`, `pinned_expected`, `log_evidence_pdip`,
`log_evidence_cholesky`, `log_evidence_library_edge_zeroed`, the `nnls` block and `peak_bytes` are
identical too, and both arms' S0 log-det pins PASSED.

**What did move is the eager `xp=np` build, exactly as the submit header predicted.**
`fixed_light.py` builds its S0 fit eagerly with `xp=np` and derives S3 from it, so those values take
lever 2's sparse route:

| field | control | feature | Δ |
|---|---|---|---|
| `log_evidence_eager` (= `s0.log_evidence_figure_of_merit`) | 29140.29587964608 | 29140.295876199474 | **−3.4466e-06 nats, rel 1.183e-10** |
| `s3.log_evidence_figure_of_merit` | 29140.295879645884 | 29140.295876199274 | −3.4466e-06 nats |
| `s0`/`s3` `d_log_evidence_terms_minus_figure_of_merit` | 1.8170e-05 | 2.1617e-05 | the same 3.45e-06, by construction |

That move is attributable, and cleanly: the control arm's `log_evidence_eager` is **byte-identical
to the fiducial record's** (29140.29587964608) even though the fiducial ran on a different node with
a third PyAutoArray revision, so the eager value is reproducible run-to-run and the feature arm's
−3.4466e-06 nats is lever 2 and nothing else. One refinement of the header's prediction:
`log_evidence_library_edge_zeroed`, which it also listed as an eager-np value, is **bit-identical in
both arms and both systems** — on this cell it is not reached through the numpy route.

**Two rows to name as scatter rather than as findings.** `s3/rows["log_evidence_terms (eager,
PDIP)"]` 60.967 → 54.102 ms (−11.3 %) and `s0/rows["Log det Cholesky (F+λH reduced)"]`
1.283 → 1.135 ms (−11.5 %) are the two largest relative movers in the A100 pair. Both are
eager-Python-dispatch rows around values that are **bit-identical**, and the fiducial's own figures
(63.319 ms and 1.159 ms) sit *between* the two arms in both cases. A row whose value does not change
and whose fiducial lies between the arms is being timed, not changed.

**`library_row`'s log likelihoods are not bit-reproducible on this cell, at ~1e-06 nats.** Control
vs feature: S0 −2.151e-06, S3 +1.601e-06 nats (relative ~7e-11), with opposite signs. The fiducial
record differs from the *control* by +4.615e-07 (S0) and −7.355e-07 (S3) nats on the same code path,
so that row scatters at this size across jobs and **its movement is not attributable to lever 2** —
which is why the identity claim above rests on `log_evidence_terms`, the block that does reproduce
bit-for-bit including against the fiducial.

## Lever 2 — verdict

1. **Lever 2 ships and is worth 1.132× on top of lever 1.** 302.709 → 267.448 ms mean
   (1.135× on medians), a saving of **35.261 ms**, single-threaded, HST Delaunay N=1500, sparse
   numba operator, route b, memo ON, on a host with no other tenant. **Cumulatively with lever 1 the
   production numba CPU call is 413.301 → 267.448 ms, 1.545×, with 145.853 ms removed.**
2. **What shipped** (PyAutoArray `0b17c292afc467dff1dbb0c171888daf8bb62731`, stacked on lever 1's
   `bae9296e`, branch `feature/fixed-light-numba-levers-l2`):
   `AbstractInversion.log_det_regularization_matrix_term`, **on the numpy path only and only with the
   default `"cholesky"` method**, now offers `regularization_matrix_reduced` to
   `inversion_util.log_det_sparse_spd_from`. That function does **one pass over the dense matrix** to
   build the CSC triple — not `csc_matrix(dense)`, which alone costs 13 ms and would have eaten a
   third of the win — then factorises with **SuperLU in symmetric mode**
   (`permc_spec="MMD_AT_PLUS_A"`, `diag_pivot_thresh=0.0`), whose `diag(U)` are the pivots with `L`
   unit-diagonal, so `log det H = Σ log diag(U)`. A non-positive pivot or a singular factor raises
   `np.linalg.LinAlgError`, **exactly as the dense Cholesky did**, so the test-mode guard downstream
   is unchanged. `"slogdet"` mode, the JAX path and `log_det_curvature_reg_matrix_term` are
   **byte-identical to lever 1**. The stale `":903 uses scipy sparse"` docstring is replaced. Tests:
   **17 new** in `test_log_det_sparse.py`; `pytest test_autoarray/inversion` **570 passed**.
3. **Two measured regime switches, both riding the same non-zero scan** (so neither costs an extra
   pass over the matrix):
   - **`SPARSE_LOG_DET_MIN_PIXELS = 256`** — below it SuperLU's fixed ~0.14 ms setup loses to a small
     dense Cholesky. Sparse/dense speed on the split-stencil pattern: **0.07× at P=9, 0.38× at 128,
     1.77× at 256, 4.99× at 512, 10.77× at 1500**.
   - **`SPARSE_LOG_DET_MAX_NNZ_PER_ROW = 32`** — geometric-stencil H (≈5–30 nnz/row) goes sparse;
     **kernel** H (`MaternKernel` and family, `coefficient·C⁻¹`, fully dense) stays dense, because a
     real Matern H at P=1500 is **34.9 ms dense against 335 ms forced-sparse**. Random sparsity
     patterns at 11–153 nnz/row lose **2–10×** to fill-in, so **structure, not density, decides** —
     and no library scheme produces a random pattern, which is why a mean-nnz threshold is a safe
     proxy for "this is a mesh adjacency".
4. **One option measured and rejected: RCM + banded Cholesky.** A reverse Cuthill-McKee permutation
   does reduce the bandwidth to 143, and the log det is permutation-invariant, but the pattern scan
   plus the permute eat the win — **43 ms**, worse than the 37 ms dense route it was meant to
   replace. SuperLU's own fill-reducing ordering does that job for free inside the factorisation.
5. **The equivalence is tolerance-level, not bit-level, and the tolerance is quoted with the
   conditioning.** W2: Δ **−9.213e-06 nats**, relative **1.201e-09** against a 1e-08 gate, on a
   matrix of **cond 6.491e12** (eigenvalues 1.0001e-08 … 6.492e+04) — two fp64 factorisations of that
   matrix cannot agree more closely, and NumPy's own `slogdet` differs from the dense Cholesky by up
   to 2.8e-08 on matrices of this conditioning, so the sparse route sits **well inside the round-off
   the dense route already carried**. W3: `figure_of_merit` Δ **−4.606e-06 nats**, relative
   **2.338e-10** against a 1e-09 gate — about 4× of margin. In the A/B arms the same change shows up
   as `log_det_regularization` 7670.876894312947 → 7670.876885100281 and `figure_of_merit`
   19705.71758591176 → 19705.717581305427. **Never write "lever 2 is bit-identical."**
6. **A 1e-10 cross-backend pin is only met when both sides run the same algorithm.** At P=9/16 the
   numpy-SuperLU log det differed from the jax-dense one by **4.3e-10 relative** in an existing
   parity test pinned at **1e-10**. That pin was **not loosened**: `SPARSE_LOG_DET_MIN_PIXELS = 256`
   keeps those small fixtures on the dense route, so numpy and JAX are still running the same
   factorisation there and the pin still means what it meant. The size floor is therefore a
   correctness boundary as well as a speed one — worth remembering before anyone lowers it.
7. **The route was witnessed firing on the production system, not assumed.** W1: exactly **one** call
   during one `figure_of_merit`, on the (1500, 1500) matrix, returning a float in 7.15 ms — which is
   simultaneously the proof that it did **not** leak into the `F + λH` term, and it agrees with that
   term's flat −1.6 % row in the decomposition. W4 records the margin against both regime switches
   (1500 ≥ 256 by 1244; 8.493 ≤ 32, per-row max 15). W5 measures the two routes on **one** matrix:
   **6.265 vs 37.720 ms, 6.02×**, against the in-call site ratio of 5.76×. The two agree to 4 %,
   which is the closest thing this phase has to a cross-check of the site instrument.
8. **On the A100 the jitted path is identity and the lever is CPU-only.** All six
   `log_evidence_terms` bit-identical in both systems and against the fiducial; the A100's dense
   Cholesky of the same H is **1.126 ms** against 37.480 ms on one CPU core and 6.265 ms sparse, so
   there is nothing for a sparse factorisation to win on the GPU. The eager `xp=np` build does take
   the route and moves −3.4466e-06 nats (relative 1.183e-10) — predicted in the submit header, not a
   finding.
9. **Where the site now sits.** `inversion.log_det_regularization_matrix_term` was **12.4 % of the
   lever 1 call** and the phase-3 note's lever 2 target; it is now **6.508 ms, 2.43 %**. The
   log-determinant group falls 80.596 → 48.854 ms, and **39.596 ms of what is left is the single
   `F + λH` term** — which is lever 3, and is now the third-largest site in the call.

## Next

Levers 1 and 2 removed the two H sites and did not touch the others. The residue of the
267.448 ms lever 2 call, from its own decomposition:

| site | feature ms | % of the 267.448 ms call | lever |
|---|---:|---:|---|
| `sparse_numba.curvature_matrix` | 86.949 | 32.51 | — (phase 2 concluded; not a phase-3 lever) |
| `solver.fnnls_cholesky` | 62.489 | 23.36 | — (phase 2 closed the solver; its **factor** is lever 3's input) |
| `inversion.log_det_curvature_reg_matrix_term` | 39.596 | 14.80 | **lever 3** |
| `delaunay.triangulation` | 18.517 | 6.92 | — |
| `inversion.log_det_regularization_matrix_term` | 6.508 | 2.43 | lever 2, done |
| `inversion.regularization_matrix` | 5.965 | 2.23 | lever 1, done |

1. **Lever 3 — one shared Cholesky of `F + λH`.** `inversion.log_det_curvature_reg_matrix_term`,
   40.172 ms in phase 2's Leg D, 39.983 ms in the lever 1 feature arm and **39.596 ms here** — the
   three agree to 1.4 %, so the site is memo-invariant and lever-invariant, and it is now the
   **third-largest site in the call and 81 % of the log-determinant group**. The solver already
   factorises `F + λH` inside `solver.fnnls_cholesky` (62.489 ms) and the log-det term then
   factorises the same matrix again. Lever 3 reads the log determinant off the factor the solver
   already has, via a **Schur complement** for the rows the active set drops: the design measures
   **≤ 1.8e-12 nats** of disagreement and **~5–7 ms against ~40 ms**. It folds in **caching
   `curvature_reg_matrix`** in the same change (4.081 ms across 2 calls here, and the rebuild is what
   makes the second factorisation necessary at all). Expected landing: **≈ −37 ms, taking the call to
   ≈ 230 ms** and the cumulative figure against the merge base to ≈ 1.80×. **On the numpy/numba path
   only** — the JAX side is the GPU session's HLO census to answer, not this lever's.
2. **Lever 3's tolerance is tighter than lever 2's, and that is deliberate.** Lever 2 accepted a
   round-off-level value change because SuperLU and a dense Cholesky are genuinely different
   factorisations. Lever 3 re-uses **the same** factor the solver computed, so its design target is
   ≤ 1.8e-12 nats — three orders tighter than lever 2's 9.2e-06 — and its witness should pin at that
   scale rather than inherit lever 2's 1e-08/1e-09 gates. A lever-3 arm that lands at lever 2's
   tolerance is a sign the factor is being rebuilt somewhere, not shared.
3. **Re-use lever 2's A/B rig, which is now the template.** Two private PyAutoArray checkouts
   prepended to `PYTHONPATH` in separate subshells; both arms in one SLURM job on one node; the
   **control arm is the previous lever**, not the merge base, so the row measures the increment; a
   correctness witness inside the feature arm with the route's call count as its first gate; the
   per-arm "does this constant exist in the imported module" echo as the cheapest proof the arms
   differ by exactly the lever; and an A100 row on the cell that actually times the site. Copy
   `hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_l2_delaunay_ral_hst_fp64` and
   `hpc/batch_gpu/submit_breakdown_imaging_fixed_light_numba_levers_l2_delaunay_a100_hst_fp64`, and
   **check which A100 cell times the site before copying the GPU job** — lever 1 needed
   `delaunay.py` (a jitted-prefix difference over H's assembly) and lever 2 needed `fixed_light.py`
   (which has the log-det rows at all). Lever 3's site is `F + λH`, so `fixed_light.py` again.
4. **Do not carry the sparse log det to the JAX path.** That question is now answered with a
   measurement rather than an argument: the A100's dense Cholesky of the same H is **1.126 ms**, 5.6×
   faster than the CPU's sparse route, and the term is launch-bound there. It is not a lever.
5. **If either compaction constant is ever retuned**, the cell must be `DelaunayNN`. This Delaunay row
   cannot observe them (K = 4 ⇒ `kc` = 4 ⇒ the wide-row branch is never traced), so a Delaunay
   measurement is not evidence about them in either direction.
6. **Never lower `SPARSE_LOG_DET_MIN_PIXELS` without re-reading the cross-backend parity pins.** The
   floor of 256 is what keeps the small numpy and JAX fixtures on the same algorithm; at P=9/16 the
   two differ by 4.3e-10 relative against a 1e-10 pin (see "Lever 2 — verdict", point 6).
7. **Still unfiled, carried from phase 2**: the PyAutoArray bug that
   `abstract_ndarray.__getitem__` imports `jax.numpy`, so a numba-only process cannot stay JAX-free
   after the first `FitImaging` (`jax_in_sys_modules: true` in both of this phase's witness JSONs).
8. **Out of scope throughout, as in phase 2**: route a, route c, rectangular meshes, Euclid, the
   source-pixel sweep, and the sparse-operator/profile-subtracted-image bug.

## Artifacts

### Lever 2 — result artifacts added by this commit (**9 files**)

```
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_lever2_control_b_warm_t1.{json,png}  # CPU control = LEVER 1, bae9296e
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_lever2_b_warm_t1.{json,png}          # CPU feature, 0b17c292
results/breakdown/imaging/fixed_light_numba_levers_l2_witness_hpc_ral_cpu_fp64_lever2.json                                   # W1-W5, feature arm
results/breakdown/imaging/fixed_light_delaunay_hpc_a100_fp64_fixed_light_lever2_control.{json,png}                            # A100 control = lever 1, bae9296e
results/breakdown/imaging/fixed_light_delaunay_hpc_a100_fp64_fixed_light_lever2.{json,png}                                    # A100 feature, 0b17c292
```

Read alongside (already versioned, **not** part of lever 2's A/B):

```
results/breakdown/imaging/fixed_light_delaunay_hpc_a100_fp64_fixed_light.json   # the A100 fiducial, the anchor for the log-det rows (ran on gpu-2)
```

### Lever 1 — result artifacts (versioned by lever 1's commit, **9 files**)

```
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_lever1_control_b_warm_t1.{json,png}  # CPU control, 5e2bc0f4
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_lever1_b_warm_t1.{json,png}          # CPU feature, bae9296e
results/breakdown/imaging/fixed_light_numba_levers_witness_hpc_ral_cpu_fp64_lever1.json                                      # W1/W2/W3, feature arm
results/breakdown/imaging/delaunay_hpc_a100_fp64_lever1_control.{json,png}                                                   # A100 control, 5e2bc0f4
results/breakdown/imaging/delaunay_hpc_a100_fp64_lever1.{json,png}                                                           # A100 feature, bae9296e
```

Read alongside (already versioned, **not** part of that A/B):

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
scripts/imaging/likelihood_breakdown/fixed_light_numba.py                    # the CPU cell, both levers (phase 2's Leg D)
scripts/imaging/likelihood_breakdown/fixed_light_numba_levers_witness.py     # lever 1's W1/W2/W3 witness
scripts/imaging/likelihood_breakdown/fixed_light_numba_levers_l2_witness.py  # lever 2's W1-W5 witness
scripts/imaging/likelihood_breakdown/delaunay.py                             # lever 1's A100 cell
scripts/imaging/likelihood_breakdown/fixed_light.py                          # lever 2's A100 cell (the one with log-det rows)
scripts/misc/likelihood_breakdown/call_accounting.py                         # the decomposition instrument
hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_ral_hst_fp64      # job 343345, lever 1 CPU arms + witness
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_numba_levers_delaunay_a100_hst_fp64     # job 343346, lever 1 A100 arms
hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_levers_l2_delaunay_ral_hst_fp64   # job 343353, lever 2 CPU arms + witness
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_numba_levers_l2_delaunay_a100_hst_fp64  # job 343354, lever 2 A100 arms
```

Library changes, both on PyAutoArray, lever 2 stacked on lever 1, merge base
`5e2bc0f42940adfe04ff30fa42a7ce13a86ecbe0`:

| lever | commit | branch | files |
|---|---|---|---|
| 1 | `bae9296ed528da6812613edc2ebc6ea367361ebb` | `feature/fixed-light-numba-levers` | `autoarray/inversion/regularization/regularization_util.py` (+181/−17), `test_autoarray/inversion/regularizations/test_regularization_util_numba.py` (+295) |
| 2 | `0b17c292afc467dff1dbb0c171888daf8bb62731` | `feature/fixed-light-numba-levers-l2` | `AbstractInversion.log_det_regularization_matrix_term` (the numpy, `"cholesky"`-method branch) and `inversion_util` (the new `log_det_sparse_spd_from`, the two `SPARSE_LOG_DET_*` constants, and the stale `":903 uses scipy sparse"` docstring it replaces), plus `test_log_det_sparse.py` — 17 new tests, `pytest test_autoarray/inversion` **570 passed** |

Job logs (SLURM, **not versioned** — `output/` is gitignored): `output/output.34334{5,6}.out`
(lever 1) and `output/output.34335{3,4}.out` (lever 2), with `error/error.34334{5,6}.err` and
`error/error.34335{3,4}.err`, on
`/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-levers` and mirrored into this worktree's
untracked `output/`.

Lever 2's A100 JSON is a `fixed_light.py` record, so its keys differ from lever 1's `delaunay.py`
ones: read `s3/rows` and `s0/rows` (the explicit `Log det Cholesky (H reduced)` and
`(F+λH reduced)` rows), `s3/log_evidence_terms` and `s0/log_evidence_terms` (six terms each, the
identity evidence), `log_evidence_eager`, `{s0,s3}/log_evidence_figure_of_merit` (the eager `xp=np`
values that move), `library_row`, `mapper_block_log_dets`, `pinned_expected`, `pinned_drift`,
`jit_phases[*].compile_s`, `nnls`, `peak_bytes` and `vmap16`. Lever 2's witness JSON carries
`results.W1_sparse_route_fires` … `results.W5_site_timing_sparse_vs_dense`, plus a `system` block
with `h_nnz_total`, `h_nnz_per_row_{mean,max}`, `h_density` and
`h_eigenvalue_{min,max}`/`h_condition_number`.

JSON keys specific to the CPU arms of both levers: `rows["b_sparse_numba"]` with `call_ms` /
`call_ms_median` / `call_ms_sequence` / `clean_relative_spread` / `abba` /
`instrumentation_overhead_ratio`,
`steps[*]` (`excl_s` already rescaled, plus `excl_s_unscaled`, `incl_s`, `n_calls`, `group`) including
the explicit `unattributed (call − sum of exclusive times)` row, `decomposition_rescale_factor`,
`solver_stats["solver.fnnls_cholesky"]`, `gates`, `contention`; and on lever 1's `delaunay.py` A100
side `steps`,
`steps_vmap_per_call`, `setup_split`, `regularization_matrix_prefix_s`, `interpolator_prefix_s`,
`jit_phases[*].compile_s`, `log_evidence_terms`, `nnls`, `total_step_by_step`.

<!-- Lever 3 appends its own "Lever 3 — headline / decomposition / A100 row / verdict" sections
     below this line, and extends the Scope, Provenance, Gate roll-up, Next and Artifacts sections,
     exactly as lever 2 did. -->
