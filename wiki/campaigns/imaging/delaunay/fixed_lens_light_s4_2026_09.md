# Fixed lens light on the numba CPU path — lever 4a, the curvature-kernel A/B on Delaunay (2026-09-17)

> **Status correction — 2026-09-18:** Both curvature and permutation rounds concluded NO_LEVER; the CPU epic is now complete. Memo-policy research and phase 7 are shelved for current planning. Historical Next sections below do not reopen them. See the [current campaign summary](profiling_campaign_status_2026_09.md).

autolens_profiling issue [#274](https://github.com/PyAutoLabs/autolens_profiling/issues/274),
epic `fixed-lens-light-numba-cpu` **phase 4, wave A**, branch `feature/fixed-light-numba-s4`.
Phase 3 ([`fixed_lens_light_levers_2026_09.md`](./fixed_lens_light_levers_2026_09.md)) took the
production numba CPU fixed-light call from 413.301 ms to **230.031 ms** and left two dominant sites:
`sparse_numba.curvature_matrix` at 87.785 ms (38 %) and `solver.fnnls_cholesky` at 61.133 ms (27 %).
Lever 4a takes the first of those, and it is a **measurement, not a rewrite**:
`curvature_matrix_via_sparse_operator_from` already dispatches between a two-stage kernel and a
direct quadruple loop on a pixel-count cap (`CURVATURE_TWO_STAGE_MAX_PIX_PIXELS = 4096`) that was
calibrated on the HST **rectangular** fiducial and had never been measured on this Delaunay cell.

**Lever 4a's result: there is no lever. Both candidates are slower than production's two-stage
kernel, and by more than the noise.** On one node, in one process, under the cell's ABBA harness at
`--n-repeats 64`: route `b` (production two-stage) **230.149 ms**, route `b_direct` (the library's
own direct kernel) **244.924 ms — +14.775 ms, +6.42 %**, route `b_touched` (the candidate
touched-index two-stage kernel) **236.913 ms — +6.765 ms, +2.94 %**. The verdict rule written before
the run — *a kernel is a lever iff it beats `b` by ≥ 5 % on the whole call with the ABBA gate PASS* —
is not met in either direction, and neither candidate is even neutral. At the site in isolation
(witness W4, medians of 20) the same ordering holds with the same margins: two-stage **85.043 ms**,
touched **97.019 ms (1.14× slower)**, direct **102.052 ms (1.20× slower)**.

**So "two-stage is right on Delaunay too" at N = 1500, u0 = 1.604 — and the kernel's own docstring
is refuted at this cell.** The two-stage docstring predicts it "loses when the source space is large
and the mappings narrow (barycentric Delaunay, u0 ~ 1.55)"; measured here, that is exactly the cell
where it wins by 1.20× against the direct kernel it is supposed to lose to.
**All three kernels are numerically equivalent** — the two two-stage forms bit-identical, `direct` at
relative 4.100e-13 on the matrix and 7.385e-16 on the log evidence — so the whole result is a timing
result and nothing about the fit moves. The PyAutoArray work this phase promised therefore shrinks
to its docstring: **PyAutoArray main `91240e43`, no kernel change; docstring-only PR
[PyAutoArray#557](https://github.com/PyAutoLabs/PyAutoArray/pull/557) (#274 wave A)**.

## Scope — read this before quoting a number

- **One job, one node, one process, one library.** Every millisecond below comes from RAL job
  **343394** on `euclid-ral-gpu-1`: rows `b`, `b_direct`, `b_touched` of
  `scripts/imaging/likelihood_breakdown/fixed_light_numba.py --mesh delaunay --dataset hst
  --formalism sparse_numba --routes b,b_direct,b_touched --threads 1 --n-repeats 64 --instances iid
  --nnls-warm-start on`, then the s4 witness in its own process. There is **no control checkout and
  no feature checkout**: the A/B is three rows of one installed library, differing only in which
  kernel is bound to `inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from` for
  the duration of the row. `configuration.routes_selected` is `["b", "b_direct", "b_touched"]`,
  `n_repeats` is **64** and `timing_status` is `"measured"` — check them before quoting, because
  `parse_profile_cli` uses `parse_known_args` and silently ignores a mistyped flag.
- **Route `b` is what production pays.** It is the source-only **S3** system (1500 params) with the
  cross-evaluation NNLS memo **on** (`nnls_warm_start_mode: "on"`, the library default). Route `a`
  (the joint S0 system) and route `c` (positivity off) were not run in this job; no ratio to either
  exists here and none is quoted.
- **`log_likelihood` is never compared across rows.** `--instances iid` rotates a call index, so each
  row's last-timed instance is a *different* draw: 18665.014330695605 (`b`), 19651.457884689760
  (`b_direct`), 19319.244426792313 (`b_touched`). Those three numbers are three different problems,
  not a disagreement. The equivalence evidence is the witness's W2/W3 and the cell's gates, both of
  which compare the kernels on *identical* inputs. The same caveat covers `solver_stats`: `b`,
  `b_direct` and `b_touched` end on `outer_iterations` 2 / 5 / 3 and `n_passive` 1484 / 1487 / 1486
  because they are solving different draws.
- **Row `b`'s `log_likelihood` is bit-identical to lever 3's** (18665.014330695605, phase 3's job
  343356, both arms). Both are the **first-timed row** of their job from `iid_seed: 263`, so the
  stream starts at the same draw. Record it as stream determinism, never as a cross-job comparison.
- **The rows are ordered `b`, `b_direct`, `b_touched` and the first row pays a one-time cost the
  later two do not.** `to_inversion.lp_linear_func_list_galaxy_dict` is 7.020 ms in `b` against
  4.254 / 4.205 ms in the two injected rows — a −2.8 ms step in a group no kernel touches, present in
  *both* later rows at the same size. It is process warm-up landing on the first-timed row. It works
  **against** the candidates' case being understated: `b` carries ~2.8 ms that `b_direct` and
  `b_touched` do not, so the true deficits are *larger* than +14.775 and +6.765 ms, not smaller.
- **No pin exists here.** `pinned_expected` is `null`, `pinned_drift` is empty, and the cell's own
  `no_pin_note` says the same. Every millisecond is RECORDED. The hard verdicts are the structural
  gates (`gates`) and the witness's W1–W3; W4 is RECORDED, never gated.
- **Single-threaded, production pool-worker scope.** `n_threads: 1`, `NUMBA_NUM_THREADS: "1"` and the
  whole BLAS family (`OMP`, `OPENBLAS`, `MKL`, `VECLIB`, `NUMEXPR`) pinned to `"1"`; `cpu_over_wall`
  1.0005 / 1.0005 / 1.0005, so no pool went unpinned. `NUMBA_NUM_THREADS` is set cell-locally and is
  deliberately **not** in `_production_config.THREAD_ENV_VARS` — widening that tuple would change
  every existing numba cell's recorded `configuration.thread_env` block.
- **The witness ran with the memo off.** Its `nnls_warm_start_env` is `"0"` against the A/B rows'
  `"1"`. It never times a whole call, only the isolated kernel on seven fixed arrays, so the memo is
  irrelevant to W4 — but do not read W4's milliseconds as a warm-call quantity.
- **The process is not JAX-free**: `jax_in_sys_modules` is `true` in the witness JSON, as in phases 2
  and 3. Recorded, not fixed (PyAutoArray bug, still unfiled). The A/B cell deliberately does not
  call `_profile_cli.device_info_dict()`, which would import JAX and change the thread pools the
  numba rows are measured through (`device.note`).
- **`b_touched` is a cell-local kernel, `b_direct` is the library's.** `b_direct` binds
  `autoarray…inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_direct_from`;
  `b_touched` binds `likelihood_breakdown.fixed_light_numpy_kernels.
  curvature_matrix_via_sparse_operator_two_stage_touched_from`, a prototype that exists only in this
  repo. Nothing in PyAutoArray was changed to produce any row in this note.

## Provenance

### Job 343394 — the CPU A/B and the witness, one job

| | CPU A/B + witness |
|---|---|
| Job | **343394** `fl_numba_s4`, `COMPLETED 0:0` |
| Elapsed / MaxRSS | **00:02:42** / 2 805 588 K of 32 GB |
| Node | **euclid-ral-gpu-1** |
| Partition | `gpu` **CPUs-only**, `--cpus-per-task=4 --mem=32gb`, **no `--gres`** |
| Device | AMD EPYC 7702 64-Core, `os.cpu_count()` **124**, kernel 5.14.0-687.39.1.el9_8, 928 GB RAM |
| Job loadavg, entry → rows start → exit | `1.00` → `[1.49, 1.16, 0.93]` → `1.93` (JSON's own `load_average_at_end`, read at the end of the rows, is `[1.90, 1.39, 1.04]`; peak 1-min **1.90** on 124 cores) |
| Cells | `likelihood_breakdown/fixed_light_numba.py` (rows), `likelihood_breakdown/fixed_light_numba_s4_witness.py` (witness) |
| Python / autolens / autoarray / scipy | 3.12.4 / 2026.8.17.1 / 2026.8.17.1 / 1.17.1 |
| `AP_ROOT` | `/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-s4` @ **`c7b72f68`**, branch `feature/fixed-light-numba-s4` |
| Config names | `hpc_ral_cpu_fp64_fixed_light_numba_s4_b_warm_t1` (rows), `hpc_ral_cpu_fp64_s4` (witness) |
| Submit | `hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_s4_delaunay_ral_hst_fp64` |

The stack is the shared RAL install, untouched — there is no per-arm override in this phase, because
there is no per-arm library:

| Repo | rev |
|---|---|
| PyAutoNerves | `8eca4b3e8cdf72ed4c13850ed5b333ec7306524d` |
| PyAutoFit | `db0872314249fdd411a8d8cf0159ec05f4d3e7c5` |
| PyAutoArray | `91240e43abfdca08000bfca5b307d340fcf1edd3` (**= the local `main` the branch survey recorded**) |
| PyAutoGalaxy | `67f6ca68f0d6abad012e774550d05a40315f40cf` |
| PyAutoLens | `223132c0b524ee7819ff5648c77fbd825cdba10a` |

Kernel provenance, probed in-job against the *imported* module before the cell ran
(`output/output.343394.out`, lines 29–35):

| probe | value |
|---|---|
| `autoarray.__file__` | `/mnt/ral/jnightin/PyAuto/PyAutoArray/autoarray/__init__.py` |
| `has curvature_matrix_via_sparse_operator_from` | **True** |
| `has …_two_stage_from` | **True** |
| `has …_direct_from` | **True** |
| `CURVATURE_TWO_STAGE_MAX_PIX_PIXELS` | **4096** |
| dispatcher takes `two_stage_max_pix_pixels` | **True** |

System under test: HST, `pixel_scale` 0.05″, mask 3.5″, **15 361** masked image pixels, **62 752**
over-sampled pixels, Hilbert/Delaunay mesh at **1500** source pixels,
`AdaptSplit(inner=0.1, outer=10.0, signal_scale=0.1)`, fp64, `InversionImagingSparseNumba`,
`InterpolatorDelaunay`, `use_jax=False`, `n_edge_zeroed` **0**, subtracted light flux
**4945.524141998385**, `--instances iid` from `iid_seed` **263**. Geometry, from the witness:
`pix_pixels` **1500**, `data_pixels` **15 361**, **11 080 573** stored pairs, mean mappings per data
pixel **u0 = 1.604127335459931**. The production dispatch branch at this size is **`two_stage`**
(1500 ≤ cap 4096), which is what route `b` runs.

**Measured wall: 162 s.** The submit's `# WALL-BASIS:` block estimated **7200 s** from job 343356's
two-arm shape. That estimate mistook the rows' prologue share — 343356 ran the cell *twice* with a
prologue each; this job runs it once and adds two more timed rows, which is the cheap part. The
submit text is **not** edited in this PR, because the recorded run depends on it; the re-pin is in
"Next".

## Gate roll-up

Cell gates, job 343394 (one arm, so one column; `gate_thresholds` in the JSON):

| Gate | status | value |
|---|---|---|
| `P1_rebaked_operator_equals_original` | **PASS** | 3/3 arrays identical (`psf_precision_operator_sparse`, `indexes`, `lengths`) |
| `S3_mapper_block_equals_S0` | **PASS** | log-dets rel **2.185e-16** (`log_det_curvature_reg_matrix_term` 8326.70017133483 vs 8326.700171334833) and **0.0** (`log_det_regularization_matrix_term` 7670.876885100281 both sides); edge-zeroed **0 vs 0**, rtol 1e-06 |
| `P2_dense_equals_sparse_numba_system` | **PASS** | D **1.297e-15**, F(mapper) **3.728e-15**, rtol **1e-09**, `n_mapper` 1500 |
| `P3_dense_equals_sparse_numba_evidence` | **PASS** | Δ **−8.731149e-11 nats**, rel **4.431e-15**, rtol 1e-06, **0** passive-set differences (1485 passive both formalisms), `seed_source` `"dense"` both sides |
| `timing_status` / `contention_warning` | `measured` / `false` | peak 1-min loadavg **1.90** on 124 cores |
| `unattributed_fraction` (≤ 0.05) | 0.00267 / 0.00259 / 0.00294 | explicit measured remainder, all three rows |
| `cached_site_call_count_violations` | `{}` | all three rows |

S0's eager `figure_of_merit` is **19705.717581305344**; S3's is 19705.717581305515 (dense) and
**19705.717581305427** (sparse numba), which is the number the witness compares every kernel against.

**The overhead gate is a millisecond budget, and this is the first job to run under it.** Phase 3's
ratio gate (`≤ 1.03`) is a fixed 6–8 ms instrument cost read as a proportion, which tightens as the
call shortens and killed job 343355 at 224 ms. It is now
`MAX_INSTRUMENTATION_OVERHEAD_MS = 12.0` on `(ratio − 1) × mean clean call`, with the ratio still
recorded beside it and `MIN_BLOCKS_FOR_OVERHEAD_ASSERT = 3` unchanged:

| row | clean ms | instrumented ms | overhead ms | ratio | status | ABBA blocks |
|---|---:|---:|---:|---:|---|---:|
| `b` | 230.149 | 228.639 | **−1.129** | 0.9951 | **PASS** | 32 |
| `b_direct` | 244.924 | 242.964 | **−1.616** | 0.9934 | **PASS** | 32 |
| `b_touched` | 236.913 | 236.838 | **+0.017** | 1.0001 | **PASS** | 32 |

All three sit inside a ±1.7 ms band of a 12.0 ms budget, and all three are *below* 1.0 — the
estimator is centred on a host whose clean rows scatter 16–36 %, which is the same reading phase 3
recorded at 32 blocks.

Witness (`fixed_light_numba_s4_witness.py`, **verdict PASS**, `slurm_job_id` 343394, wall 19.27 s):

| Gate | status | evidence |
|---|---|---|
| **W1** each kernel fires exactly once | **PASS** | `two_stage`, `direct`, `two_stage_touched` each `n_calls` **1** per `fit.figure_of_merit`, each at `last_pix_pixels` **1500**, each recording the dotted name that actually ran; `patched_dotted_name` is `…curvature_matrix_via_sparse_operator_from`, production branch at this size **`two_stage`** (cap 4096) |
| **W2** the curvature matrix agrees | **PASS** | `direct` vs `two_stage`: max abs **2.151e-06**, max rel **4.100e-13** against a **1e-09** gate, Frobenius 7283350.905624719 vs 7283350.905626731. `two_stage_touched` vs `two_stage`: **bit-identical**, max abs **0.0**, gated at `np.array_equal` and not at a tolerance. All three (1500, 1500) and symmetric; the **un-injected dispatcher reproduces the `two_stage` row exactly**, so the control really is the production path |
| **W3** the log evidence agrees | **PASS** | library (un-injected) **19705.717581305427**; `two_stage` and `two_stage_touched` **bit-identical** to it (Δ 0.0); `direct` **19705.717581305442**, Δ **+1.455e-11 nats**, rel **7.385e-16** against a 1e-09 gate |
| **W4** the site in isolation | **RECORDED** | medians of 20 at 1 thread on identical inputs: `two_stage` **85.043 ms** (84.617–85.683), `two_stage_touched` **97.019 ms** (96.727–97.273, **0.88×**), `direct` **102.052 ms** (101.745–103.018, **0.83×**) |

W2 gates the two forms differently on purpose: `direct` sums the same products in a different order
and is gated at a tolerance with its value recorded; `two_stage_touched` removes additions of exact
`+0.0` and reorders nothing, so a merely *tolerant* pass there would refute its promotion case rather
than support it. It passed at exact equality.

## Lever 4a — headline

### Host: `euclid-ral-gpu-1`, numba CPU

`use_jax: false`, `backend: "numba_cpu"`, `InversionImagingSparseNumba`, `n_threads` 1,
`NUMBA_NUM_THREADS` `"1"`, `OMP`/`OPENBLAS`/`MKL`/`VECLIB`/`NUMEXPR` all `"1"`. Clean
(uninstrumented) calls, `--n-repeats 64`, warm-up run to a steady state in **6 calls** on every row.

| row | kernel bound | thread / backend | mean ms | median ms | min | max | clean spread | ABBA | Δ vs `b` (mean) | Δ % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `b` (production) | `…_two_stage_from` (dispatcher, uninjected) | 1 / numba_cpu | **230.149** | 226.240 | 214.725 | 297.333 | 35.9 % | 0.9951 **PASS** | — | — |
| `b_direct` | `…_direct_from` (library) | 1 / numba_cpu | **244.924** | 241.452 | 229.519 | 311.023 | 33.3 % | 0.9934 **PASS** | **+14.775** | **+6.42 %** |
| `b_touched` | `…_two_stage_touched_from` (cell-local) | 1 / numba_cpu | **236.913** | 235.787 | 227.834 | 265.225 | 15.8 % | 1.0001 **PASS** | **+6.765** | **+2.94 %** |

On medians the deficits are larger, not smaller: `b_direct` **+15.212 ms (+6.72 %)**, `b_touched`
**+9.547 ms (+4.22 %)**.

| | value |
|---|---:|
| the rule (written before the run) | a kernel is a lever iff it **beats `b` by ≥ 5 %** on the whole call with ABBA PASS |
| what ≥ 5 % would have required | ≤ **218.642 ms** (11.507 ms *faster* than `b`) |
| `b_direct` | **+14.775 ms slower** — 26.3 ms the wrong side of the bar |
| `b_touched` | **+6.765 ms slower** — 18.3 ms the wrong side of the bar |
| injected-kernel call counts | `b_direct` **134** calls, `b_touched` **134** calls, `last_pix_pixels` 1500 both (hard-asserted `> 0` per row) |
| warm-up first call (incl. numba compile) | 0.286 s (`b`), 0.522 s (`b_direct`), 0.922 s (`b_touched`) — the cell-local kernel compiles from cold |

**The control reproduces lever 3.** Route `b` here is the same route, the same flags and the same
node as phase 3's lever-3 feature arm, which recorded **230.031 ms** (job 343356, 2026-09-16), run
against PyAutoArray main `91240e43` rather than against lever 3's private checkout `b4322c3e`.
This job measures **230.149 ms — +0.118 ms, +0.05 %** on the mean, and 226.240 against 225.985 ms
— **+0.11 %** on the median. That is a reproducibility datum in its own right: the cell's whole-call
number is stable to a part in two thousand across jobs, days and 32-block ABBA runs on the same node.

## Lever 4a — decomposition

Exclusive (self) time per instrumented site, rescaled by `decomposition_rescale_factor` (1.006601 /
1.008067 / 1.000318) so the rows sum to the clean call. `n_sites_instrumented` 48, `n_sites_reached`
**37** in all three rows. The nine groups:

| group | label | `b` ms | `b_direct` ms | `b_touched` ms | Δ direct | Δ touched |
|---|---|---:|---:|---:|---:|---:|
| `DF` | **D and F assembly** | **96.325** | **114.271** | **109.047** | **+17.946** | **+12.722** |
| `solve` | Positivity solve | 71.960 | 71.243 | 69.505 | −0.717 | −2.455 |
| `mesh` | Mesh (ray trace, placement, triangulation) | 20.642 | 20.454 | 20.259 | −0.188 | −0.383 |
| `evidence` | Log determinants + evidence | 15.419 | 15.604 | 15.172 | +0.185 | −0.247 |
| `mapper` | Mapper + mapping matrix | 10.566 | 7.818 | 7.818 | −2.748 | −2.748 |
| `F_plus_H` | F + λH | 8.131 | 8.402 | 7.970 | +0.271 | −0.161 |
| `mapper_weights` | Mapper weights (Delaunay) | 3.815 | 3.830 | 3.783 | +0.015 | −0.032 |
| `image` | Image + blurring | 2.677 | 2.666 | 2.663 | −0.011 | −0.014 |
| — | `unattributed` | 0.614 | 0.635 | 0.697 | +0.021 | +0.083 |

**One group carries the whole result.** `D and F assembly` moves +17.946 and +12.722 ms, and the
groups sum to the clean call to the last printed digit (230.149 / 244.924 / 236.913). The `mapper`
group's −2.748 ms is the first-row warm-up described in Scope, identical in both injected rows.

Top sites by `b` time:

| site | group | `b` ms | `b_direct` ms | `b_touched` ms | Δ direct | Δ touched |
|---|---|---:|---:|---:|---:|---:|
| `sparse_numba.curvature_matrix` | DF | **87.740** | **105.690** | **100.547** | **+17.950 (+20.5 %)** | **+12.807 (+14.6 %)** |
| `solver.fnnls_cholesky` | solve | 61.509 | 61.883 | 60.750 | +0.374 | −0.759 |
| `delaunay.triangulation` | mesh | 19.227 | 19.220 | 19.025 | −0.007 | −0.202 |
| `sparse_numba.psf_weighted_data` | DF | 8.418 | 8.408 | 8.336 | −0.010 | −0.082 |
| `inversion.reconstruction` | solve | 8.400 | 8.237 | 8.605 | −0.163 | +0.205 |
| `to_inversion.lp_linear_func_list_galaxy_dict` | mapper | 7.020 | 4.254 | 4.205 | −2.766 | −2.815 |
| `inversion.log_det_regularization_matrix_term` | evidence | 6.543 | 6.575 | 6.493 | +0.032 | −0.050 |
| `inversion.log_det_curvature_reg_matrix_term` | evidence | 6.043 | 6.144 | 5.889 | +0.101 | −0.154 |
| `inversion.regularization_matrix` | F + λH | 5.923 | 6.125 | 5.753 | +0.202 | −0.170 |
| `fit.blurred_image` | image | 2.614 | 2.608 | 2.602 | −0.006 | −0.012 |
| `delaunay.mappings_sizes_weights` | mapper_weights | 2.571 | 2.578 | 2.551 | +0.007 | −0.020 |
| `mapper.unique_mappings` | mapper | 2.260 | 2.263 | 2.241 | +0.004 | −0.020 |
| `inversion.curvature_reg_matrix` | F + λH | 2.158 | 2.226 | 2.163 | +0.068 | +0.005 |
| `solver.reconstruction_positive_only_from` | solve | 2.046 | 1.118 | 0.144 | −0.928 | −1.902 |
| `inversion.mapped_reconstructed_operated_data` | evidence | 1.761 | 1.794 | 1.734 | +0.033 | −0.027 |
| `unattributed (call − sum of exclusive times)` | — | 0.614 | 0.635 | 0.697 | +0.021 | +0.083 |

**The site moved and nothing else did.** `sparse_numba.curvature_matrix` is the only site whose
delta exceeds 3 ms, and its two deltas (+17.950, +12.807) account for the group deltas (+17.946,
+12.722) and for most of the whole-call deltas (+14.775, +6.765) — the remainder being the −2.8 ms
first-row warm-up that flatters both candidates. As a fraction of the call the site goes
**38.1 % → 43.2 %** (direct) and **38.1 % → 42.4 %** (touched).

**The whole-call site delta reproduces the witness's isolated site delta.** W4 measures the three
kernels on seven fixed arrays, outside any fit: `direct − two_stage` = **+17.009 ms**,
`touched − two_stage` = **+11.976 ms**, against the A/B rows' **+17.950** and **+12.807 ms**. Two
independent measurements — one inside a 230 ms production call under ABBA, one in isolation at
20 repeats — agree to about 1 ms. There is no room left for the difference to be an artefact of the
injection seam.

`solver.reconstruction_positive_only_from` (2.046 → 1.118 → 0.144 ms) is the *exclusive* time of a
wrapper whose inclusive time is 63.556 ms and is dominated by `fnnls_cholesky`; its three rows are
three different iid draws with different iteration counts, and the `solve` group total moves only
−0.7 / −2.5 ms. It carries no signal about the kernel.

## Lever 4a — verdict

**No lever. Neither candidate ships, and the dispatcher's existing cap is correct at this cell.**

- **The rule.** Written into the issue before the run: a kernel is a lever iff it beats route `b` by
  **≥ 5 %** on the whole call with the ABBA gate PASS. `b_direct` is **+6.42 %** slower and
  `b_touched` **+2.94 %** slower, both with ABBA PASS. The gate did its job in the negative
  direction: these are clean, believable numbers that say no.
- **The margins are outside the noise, and the two measurements agree.** The whole-call deltas
  (+14.775, +6.765 ms) are reproduced by the isolated-site deltas (+17.009, +11.976 ms) at a
  completely different repeat count and outside the fit, and both candidates' deficits survive the
  −2.8 ms first-row warm-up that biases *in their favour*.
- **The docstring's prediction is refuted at this cell.** `curvature_matrix_via_sparse_operator_
  two_stage_from` says the two-stage form "wins when the source space is small relative to the PSF
  overlap and the mappings are wide (bilinear, u0 = 4), and loses when the source space is large and
  the mappings narrow (barycentric Delaunay, u0 ~ 1.55)". This cell **is** that case — barycentric
  Delaunay, u0 = **1.604**, 1500 source pixels, 11 080 573 stored pairs — and two-stage wins it by
  **1.20×** at the site. The prediction was never measured here; it is now, and it is wrong.
- **Why two-stage still wins at u0 ≈ 1.6.** The cost model behind the docstring counts stage 2's
  dense work — `(sum_data u0 + data_pixels) × pix_pixels` operations — as if dense work and scattered
  read-modify-writes cost the same per element. They do not. A `pix_pixels = 1500` fp64 AXPY is
  **12 KB**: it is L1-resident and vectorises, so stage 2's ~1.6 contiguous AXPYs per data pixel run
  at streaming rates, while the direct kernel's `sum_pairs u0 × u1` scattered RMWs into a 1500 × 1500
  (18 MB) matrix miss L2 on nearly every write. At this geometry the dense-but-local work is cheaper
  per element by more than the factor the element counts suggest.
- **Why `two_stage_touched` loses too — and it is the more interesting negative.** The touched
  variant's premise is that stage 2 wastes work adding exact zeros, so it should scatter and zero
  only the indices stage 1 touched. It is **bit-identical** to two-stage (W2, exact equality), so the
  premise about the arithmetic is right. It is still **+11.976 ms** slower at the site: maintaining
  the touched-index list is itself a scatter with a data-dependent length, and the gather it enables
  replaces a contiguous, vectorised AXPY with an indexed one. **The index bookkeeping costs more than
  the dense zero-and-AXPY it saves** — which is the same lesson as the previous point, measured from
  the other side.
- **Nothing about the fit moves.** All three kernels return the library's log evidence to
  **≤ 7.385e-16** relative (W3), the two two-stage forms bit-identically; the matrix agrees to
  **4.100e-13** (direct) and exactly (touched); the four cell gates PASS at 1e-15-to-1e-16 residuals.
  This is a timing result and only a timing result.
- **Reproducibility datum.** Route `b` reproduces lever 3's 230.031 ms at **230.149 ms (+0.05 %)** on
  a different job, on a different day, on the same node. That is worth as much to the campaign as
  the negative verdict: it says the cell's whole-call number is a stable instrument, and it licenses
  chaining future phases onto phase 3's chain without re-measuring the base.
- **Library state: PyAutoArray main `91240e43`, no kernel change; docstring-only PR pending
  (#274 wave A).** The conditional dispatch-rule change the issue scoped (a `u0`-based geometry rule
  replacing the pixel-count cap) is **not** made: nothing in this measurement supports moving the
  boundary, and the cap already routes this cell to the kernel that wins. What remains is the
  docstring fix at `inversion_imaging_numba_util.py:781`, which names a
  `CURVATURE_TWO_STAGE_COST_RATIO_THRESHOLD` that does not exist (the code branches on `pix_pixels`)
  — and which must now also correct the refuted Delaunay sentence above it.

## Phase 4 wave A — the whole campaign

| step | PyAutoArray | call ms | row ratio | cumulative | job |
|---|---|---:|---:|---:|---|
| merge base | `5e2bc0f4` | **413.301** | — | 1.000× | 343345 |
| lever 1 — numba split-reg assembly | `bae9296e` | **299.709** | 1.379× | 1.379× | 343345 |
| lever 2 — sparse log det of H | `0b17c292` | **267.448** | 1.132× | 1.545× | 343353 |
| lever 3 — shared Cholesky log det of `F + λH` | `b4322c3e` | **230.031** | 1.188× | **1.797×** | 343356 |
| **lever 4a control `b`** (reproduction) | `91240e43` | **230.149** | 0.999× | 1.796× | **343394** |
| lever 4a `b_direct` | `91240e43` | 244.924 | 0.940× | — | 343394 |
| lever 4a `b_touched` | `91240e43` | 236.913 | 0.971× | — | 343394 |

**Wave A adds 0 ms and removes 0 ms.** The campaign's number is still phase 3's **413.301 → 230.031
ms, 1.797×**, and lever 4a's contribution is a negative result plus a reproduction of the endpoint to
+0.05 %. The residue is unchanged and now measured twice:
`sparse_numba.curvature_matrix` **87.740 ms (38.1 %)** — and it stays on the two-stage kernel —
`solver.fnnls_cholesky` **61.509 ms (26.7 %)**, `delaunay.triangulation` **19.227 ms (8.4 %)**, the
two log dets **12.586 ms (5.5 %)** and `sparse_numba.psf_weighted_data` **8.418 ms (3.7 %)**.

## Next

- **Wave B — lever 4b, A′ (permute active last), issue #274 steps 9–11.** Step 9: the witness
  definition first, as a note section and a test, no kernel — per-draw active-set Jaccard against the
  library `fnnls_cholesky` on the seeded iid draws, Δ log evidence in nats at a 1e-9 relative gate,
  PDIP fallback count unchanged, solver iteration counts recorded; reviewed by the human before step
  10. Step 10: `fnnls_cholesky_permuted` in `fixed_light_numpy_solvers.py`, injected as route
  `d_perm` through the existing `numpy_solver_injected` seam. Step 11: its own RAL A/B (`b` vs
  `d_perm`, `--n-repeats 64`) and its own witness; promote to PyAutoArray only if it beats `b` by
  **≥ 5 %** on the whole call **and** the witness passes on every draw. Wave B starts after wave A
  merges — never stacked (the #267 lesson).
- **The PyAutoArray docstring PR (wave A, the only library change).** Fix
  `inversion_imaging_numba_util.py:781`: it names a non-existent
  `CURVATURE_TWO_STAGE_COST_RATIO_THRESHOLD` (the dispatcher branches on `pix_pixels` against
  `CURVATURE_TWO_STAGE_MAX_PIX_PIXELS = 4096`), and its "loses … barycentric Delaunay, u0 ~ 1.55"
  sentence is refuted by this job's 1.20×. Its own small PR, `/ship_library`, not stacked on this
  one.
- **Re-pin the s4 `WALL-BASIS`.** `wall: 7200` (`source: measured-wall`, `ref: RAL-job-343356`,
  `headroom: 1.5`) against a **measured 162 s**; 162 × 1.5 ≈ **250 s** is the honest replacement, and
  the `ref:` becomes `RAL-job-343394`. The submit text is deliberately **not** edited in this PR
  because the run recorded here depends on it verbatim; it is a one-line follow-up before the next
  s4-family submission.
- **Laptop vs RAL, W4.** The same witness run on the laptop (`DESKTOP-H143S82`, untracked,
  wiring-only) gives `two_stage` **101.534 ms**, `direct` **114.916 ms (0.88×)**,
  `two_stage_touched` **126.352 ms (0.80×)**. The two hosts agree on the verdict — **two-stage is
  fastest on both, and both candidates lose** — but they do **not** agree on the order of the two
  losers: RAL has `touched` (1.14×) ahead of `direct` (1.20×), the laptop has `direct` (1.13×) ahead
  of `touched` (1.24×). The agreement that matters for #274 is the first fact; the disagreement is a
  reminder that the candidate ranking below the winner is host-dependent and must never be quoted as
  a portable ordering.
- **The overhead gate's first measured row.** The ms budget passed all three rows at ±1.7 ms of
  12.0 ms with clean spreads of 16–36 %. Nothing further is needed here; the phase-3 ratio gate would
  have killed `b_touched`'s row at 1.0001 only by luck of sign, and the budget is now the contract.

## Artifacts

Result artifacts added by this commit (**3 files**, RAL job 343394):

```
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s4_b_warm_t1.json  # rows b, b_direct, b_touched, 64 repeats, 32 ABBA blocks
results/breakdown/imaging/fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s4_b_warm_t1.png   # the three-row bar chart
results/breakdown/imaging/fixed_light_numba_s4_witness_hpc_ral_cpu_fp64_s4.json                            # W1-W4, verdict PASS
```

Job logs, kept on the cluster and **not** versioned:
`hpc/batch_cpu/output/output.343394.out` (251 lines) and `hpc/batch_cpu/error/error.343394.err`
(16 lines, all benign `UserWarning`s) in
`/mnt/ral/jnightin/autolens_profiling_wt/fixed-light-numba-s4`.

The laptop's own smoke run of the same cell and witness
(`…_local_cpu_fp64_fixed_light_numba_s4_smoke.{json,png}`,
`fixed_light_numba_s4_witness_local_cpu_fp64_s4.json`) is **deliberately not versioned**: its cell
JSON stamps `timing_status: "wiring_only_not_measured"` at `--n-repeats 6` and its milliseconds are
wiring evidence, not measurements. Its witness W4 medians are quoted once, in "Next", as the
host-ordering cross-check and for nothing else.

Cells, seam and submit that produced these artifacts (added earlier on this branch):

```
scripts/imaging/likelihood_breakdown/fixed_light_numba.py                                 # routes b_direct / b_touched, the ms overhead budget, the injected_kernel JSON block
scripts/misc/likelihood_breakdown/fixed_light_numpy_kernels.py                            # curvature_kernel_injected + the three kernels, incl. the touched prototype
scripts/imaging/likelihood_breakdown/fixed_light_numba_s4_witness.py                      # W1-W4
scripts/misc/test/test_fixed_light_s4.py                                                  # kernel agreement on the tiny fixture, seam restores by identity, submit bash -n
hpc/batch_cpu/submit_breakdown_imaging_fixed_light_numba_s4_delaunay_ral_hst_fp64          # job 343394; its WALL-BASIS block is the one to re-pin
```


## Lever 4b — approved witness definition (2026-09-18)

The user approved continuation with these correctness checks before the permutation
prototype is judged. Each of eight seed-263 central-20-percent iid draws traverses
three independent production memo streams: library, identity injection, candidate.
The lens-light subtraction, sparse operator, thread count and model stream match
route b. Every lane starts empty; no candidate inherits the reference answer.

- W1: the identity injection reproduces the reconstruction and evidence exactly,
  with identical iteration counters. Both lanes must solve identical matrices/RHS.
- W2: final passive-set Jaccard **must equal 1.0 on every draw**. Report both
  solutions' KKT residuals even on failure; do not waive a disagreement after seeing it.
- W3: evidence difference is reported in nats and must be <=1e-9 relative.
- W4: seed source, memo-discard flag, kernel exception sequence and invocation
  count must agree. There is no PDIP fallback on this NumPy entrypoint; record that
  explicitly rather than treating an unobserved PDIP counter as zero.
- W5: record outer/inner iterations, passive count and warm-start errors.
- W6: both published factors must reconstruct the full log determinant to
  <=2e-12 nats against slogdet, with the expected keys and matrix shape.

The witness retains all failed rows and exits nonzero after writing its JSON.
Promotion additionally requires >=5% improvement on the whole call in the RAL
64-repeat b/d_perm comparison with both ABBA overhead gates passing. A failed
identity or candidate witness blocks promotion regardless of measured speed.

### W6 identity calibration and explicit approval

Before using the candidate verdict, the unchanged library failed the original
2e-12-nat threshold on 7/8 local N=1500 draws (maximum 1.2732925824820995e-11
nats). Identity evidence, reconstruction, passive sets and counters were exact.
The user explicitly approved on 2026-09-18: **max(2e-12 nats, 32 floating-point
spacings of abs(slogdet))**, together with a **1e-12 relative factor-reconstruction
residual**. The residual is max(abs(U.T@U - M[P,P])) / max(abs(M[P,P])).
This additional check catches determinant-preserving factor-order corruption.
Each factor record retains `original_within_atol`, the effective tolerance, and
the reconstruction residual. No active-set or evidence criterion changed.

### Timing-stream correction and execution provenance

Independent review found that variable warm-up lengths shifted the original
cell's global instance cursor between rows. The s4b comparison now clears the
memo after warm-up, primes each kernel untimed on instance 0, then resets the
cursor to instance 1. Both timed rows replay the same stream; the row records
its start, length, seed and priming. Other campaigns retain their existing
behavior. Timing status is explicitly `timing_candidate` only above 5% with
both ABBA gates PASS; a measured `NO_LEVER` is a valid experiment outcome.
Neither status grants promotion without the separate witness and human review.

The RAL run uses a private 48-file source snapshot under
`autolens_profiling_wt/fixed-light-numba-s4b-run`, with a SHA-256 manifest and
per-script hashes in its output, against the canonical shared libraries. The
full git-worktree checkout was cancelled because the shared filesystem made
copying the unrelated historical results very slow. No shared library changed.

## Lever 4b verdict — no lever (RAL job 343397, 2026-09-18)

**Do not promote the permutation prototype to PyAutoArray.** On
`euclid-ral-gpu-1`, one thread, HST Delaunay N=1500, memo ON, 64 clean calls
per row (32 ABBA blocks), it misses the pre-declared >=5% whole-call threshold.
The observed 0.80% slowdown is not a claim of a statistically resolved regression;
it is sufficient evidence that this run provides no qualifying improvement.

| row | whole call (ms) | fnnls site (ms) | curvature site (ms) | ABBA overhead |
|---|---:|---:|---:|---|
| b: library | 226.772 | 60.886 | 85.744 | +3.321 ms PASS |
| d_perm: symmetric permutation | 228.576 | 66.809 | 85.663 | +2.977 ms PASS |

The candidate costs 1.804 ms more on the whole call; the solver site itself
does not improve. It still allocates a full permuted matrix and factor buffer,
and SciPy factors the leading passive block rather than eliminating the block
copy entirely. The prior ~41 ms factorization-only estimate did not predict the
whole solver or whole likelihood. No library follow-up is justified by this run.

### Correctness and provenance

- SLURM **COMPLETED 0:0**, elapsed **601 s**, CPUs only on the gpu partition.
- Timed instance streams and memo priming metadata match exactly across rows.
- Load average 1.00 on 124 CPUs; no contention warning; both overhead gates PASS.
- P5 evidence comparison exact; injected kernel observed 135 times (dispatch,
  warm-up, priming, and 128 clean/instrumented timed calls); memo seed retained
  and all four factor keys published.
- Both identity and candidate witnesses PASS on **all eight seeded draws**.
  Evidence and reconstruction differences are exactly zero; passive-set
  Jaccard is 1.0 throughout; fallback sequences and iteration counts agree.
- Maximum factor reconstruction residual **3.5514e-16**; maximum determinant
  difference against slogdet **1.2733e-11 nats**, passing the explicitly approved
  roundoff-aware criterion. The original 2e-12 failures remain in each JSON row.
- Shared PyAutoArray commit `192d4b70215830ad3ad3c8c83550e477bd674b72`. Source
  snapshot base `92f1fadd`, actual files identified by the archived
  `fixed_light_numba_s4b_source_job343397.json` manifest in this notes directory.
  Its Python-file hashes match the tested local implementation.

| draw | identity | candidate | passive Jaccard | delta evidence (nats) |
|---|---|---|---:|---:|
| 0 | PASS | PASS | 1.0 | 0.0 |
| 1 | PASS | PASS | 1.0 | 0.0 |
| 2 | PASS | PASS | 1.0 | 0.0 |
| 3 | PASS | PASS | 1.0 | 0.0 |
| 4 | PASS | PASS | 1.0 | 0.0 |
| 5 | PASS | PASS | 1.0 | 0.0 |
| 6 | PASS | PASS | 1.0 | 0.0 |
| 7 | PASS | PASS | 1.0 | 0.0 |

Artifacts under `results/breakdown/imaging/`:

- `fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s4b_warm_t1.{json,png}`
- `fixed_light_numba_s4b_witness_hpc_ral_cpu_fp64_s4b.json`

The source-snapshot launch printed two misleading `FATAL ... 128` diagnostics
from git provenance probes inside command substitutions (the snapshot has no
`.git`). The enclosing echo commands continued and the numerical processes
completed normally, with no gate bypass. The submit now checks whether the
source is a git worktree before those probes and requires the snapshot manifest
otherwise. Logs are retained locally under `output/s4b/`.

### Phase 4 — the whole campaign

| stage | production whole call (ms) | attributable gain |
|---|---:|---|
| Phase 3 start | 413.301 | baseline |
| Phase 3, levers 1–3 | 230.031 | 1.80x cumulative |
| Phase 4a, curvature alternatives | 230.149 control | no new lever |
| Phase 4b, permutation prototype | 226.772 control / 228.576 candidate | no new lever |

The campaign's established improvement remains **1.80x**. Do not attribute
230.031 -> 226.772 to a source change: it is a fresh control, with aligned
timed streams. The next planned phase is the memo warm-start robustness study
over the graded draw set, followed by source-pixel scaling and the HST/Euclid
production verdict. The existing curvature and solver arithmetic remain intact.

Validation: **92 focused tests passed** (wave A, wave B, NumPy solvers, route
harness), both changed-cell import smokes passed, repository Ruff check/format,
README idempotence, submit wall contracts and shell syntax passed. Independent
review: **CLEAN** after the timed-stream correction.
