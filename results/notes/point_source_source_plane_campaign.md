# Point-source source-plane likelihood campaign

Date: 2026-09-26  
Issue: [autolens_profiling #315](https://github.com/PyAutoLabs/autolens_profiling/issues/315) (phase 1)  
Instrument commit: `44c43d9` (branch `feature/point-source-source-plane-breakdown`)  
Scope: single-source only (`simple` point-source dataset). Cluster cells, rows
and levers are out of scope; see "Carried to cluster epic".

## What landed

- `scripts/point_source_source/likelihood_breakdown/source_plane.py`: the shared
  CPU/GPU breakdown instrument for the source-plane point-source likelihood,
  mirroring `scripts/point_source_image/likelihood_breakdown/image_plane.py`: same
  helpers, JSON contract, `--config-name` rows, two-panel PNG and
  `AUTOLENS_PROFILING_SMOKE=1` exit.
  - Primary lane: `PointSolved` + `FitPositionsSourceSolved` (the adopted
    default; `weighting="jacobian"`, analytic marginalisation over the solved
    centre). Plain control lane: `PointFlux` + `FitPositionsSource`
    (`weighting="magnification"`). The two lanes are labelled separately.
  - `AnalysisPoint(solver=None)`: the source-plane fit types `solver` as
    `Optional` and never calls it, so no solver is built.
  - Cumulative JIT prefixes are cut at the fit object's own boundaries.
    Solved lane: `_beta_hat` → `precision_tensor_components_from` →
    `source_plane_coordinate` → `chi_squared` → `marginalization_term` →
    `log_likelihood`. Plain lane: `model_data` → `magnifications_at_positions` →
    `chi_squared_map` → `log_likelihood`.
  - Controls: fused JIT against eager at rtol 1e-4 in both lanes;
    `jit(vmap)` with batch 2 matches single JIT; regression literals;
    `jax.grad` is finite and non-zero; `value_and_grad` cost; dispatch-floor
    probes; CSE probes; host load average.
- `scripts/point_source_source/likelihood_runtime/source_plane.py`: the stale
  `TracerArrayConversionError` guard is gone. On the live stack
  (PyAutoArray `1bf641e4`, PyAutoLens `4487eb47`),
  `jax.jit(AnalysisPoint(FitPositionsSource).log_likelihood_function)` now
  traces and returns `-33788.35531560625`, which matches eager to 4e-9
  relative. The runtime cell ran with `full_pipeline_jits: true` and passed
  its pinned-value check. Its regenerated runtime JSON was not committed.
- README dashboards regenerated. The row reads `point_source_source/source_plane`.

## CPU fp64 reference

Canonical artifact:
`results/breakdown/point_source_source/source_plane_local_cpu_fp64.json` and
its PNG.

- Backend: JAX CPU, fp64, JAX/jaxlib 0.10.2. Threads were exported to 8
  (`NPROC`/`OMP`/`OPENBLAS`/`MKL`) and recorded. `XLA_FLAGS` includes
  `--xla_disable_hlo_passes=constant_folding` from the session environment;
  the CSE pass stays enabled.
- **Host contended:** the load average was about 16 on 8 cores for the whole
  run, and is recorded in `host_load_average`. Every timing is the per-call
  median over 500 calls, and the p10/p90 spread is about ±0.1 ms. Treat the
  rows below as the *magnitude* of a ~0.4 ms call, not as stage-resolved costs.
- Dataset: 4 observed positions, σ = 0.05 arcsec. Free parameters: solved 5,
  plain 8.

| Quantity | Solved (`FitPositionsSourceSolved`) | Plain control (`FitPositionsSource`) |
|---|---|---|
| Fused likelihood, median per call | **0.438 ms** (p10 0.385, p90 0.543) | **0.457 ms** (p10 0.405, p90 0.538) |
| Lower / compile | 1.67 s / 0.78 s | 0.83 s / 0.63 s |
| JIT log L (= regression literal) | `0.5986504555536349` | `-33788.35531560625` |
| Eager log L (finite-difference Hessian) | `0.598650453950647` | `-33788.3554397447` |
| `vmap(2)` | equal to single JIT | equal to single JIT |

Solved-lane cumulative prefixes, median ms per call (successive difference in
brackets):

| Prefix | Absolute | Step |
|---|---|---|
| Ray trace `_beta_hat` | 0.374 | 0.374 |
| + precision tensor W_i (jacfwd Hessian) | 0.632 | 0.258 |
| + solved centre β* | 0.551 | −0.080 |
| + chi-squared | 0.519 | −0.032 |
| + marginalisation term | 0.420 | −0.099 |
| + `fit.log_likelihood` | 0.506 | 0.086 |
| Fused `log_likelihood_function` | 0.438 | −0.068 (wrapper residual) |

Plain-lane prefixes: `model_data` 0.494, + magnifications 0.525,
+ chi-squared map 0.671, + `log_likelihood` 0.608, fused 0.457 ms.

- Gradient: `jax.grad` of the fused solved likelihood is finite, with
  L2 = 337.62 over 5 leaves. `jit(value_and_grad)` takes **1.065 ms**, which
  is **2.43×** the forward call. Lower/compile for it: 4.08 s / 3.17 s.
- Dispatch floor: `jit(sum of ModelInstance leaves)` takes 0.086 ms
  (solved pytree) and 0.069 ms (plain pytree). `jit(x + 1)` on a scalar
  takes 0.019 ms.
- CSE probes, run as the same computation once or repeated inside one JIT:
  precision tensor ×3 vs ×1 = **1.15**; magnifications ×2 vs ×1 = **0.98**.

To refresh the literals, follow the procedure in the cell docstring: set both
to `None`, run once, paste back the printed values, and state the cause in the
commit.

## Reading the rows

The `steps` table is built from successive differences of cumulative-prefix
timings. Each prefix returns the sum of every intermediate reached so far, so
earlier stages cannot be dead-code eliminated. Independent prefixes get their
own XLA fusion and scheduling, so negative rows are kept and are not read as
negative work. The absolute prefixes are in `prefix_steady_per_call_s`, and the
fused solved likelihood is the authoritative end-to-end runtime. By
construction the step sum telescopes to the fused value, so it must not be
presented as independently additive kernel timing.

The telescoping caveat matters here more than for the image-plane cell. Every
prefix, including the bare ray trace, sits inside the fused call's p10–p90
band. The step rows are therefore dominated by fixed per-call overhead and
host noise, not by stage cost.

## Evidence on the repeated evaluations (phase-2 hypothesis #1)

The fit properties are uncached. The plain lane evaluates
`magnifications_at_positions` twice per likelihood: once in the chi-squared map
and once in the noise normalisation. The solved lane rebuilds the precision
tensor three times (β*, chi-squared map, marginalisation term) and `_beta_hat`
twice.

What was observed, rather than assumed:

- CSE timing probes: ×3 precision costs 1.15× of ×1, and ×2 magnifications
  costs 0.98×. Both are within the p10–p90 noise, so no cost scales with the
  repeat count.
- Optimized-HLO fusion counts, taken from a scratch diagnostic that is not
  versioned: precision ×1 has 40 fusions and ×3 has 42; magnifications ×1 and
  ×2 both have 35; the `_beta_hat` prefix has 25 and the fused solved
  likelihood has 50. XLA merges the repeated Hessian and precision
  subgraphs.
- The fused solved call (50 fusions) and the bare ray-trace prefix (25 fusions)
  have the same median, 0.41 vs 0.40 ms in the scratch run with 2000 calls.
  Doubling the math does not move the time.

Conclusion: on CPU at 4 positions, the repeated evaluations are merged by XLA
and are not a lever. Caching the properties would tidy the Python-side trace
and compile time, but would not change steady per-call time.

## Ranked residue → phase-2 candidates

1. **Per-call fixed overhead: argument handling, dispatch and executable
   launch.** This accounts for about 100% of the measured call. The bare
   ray-trace prefix costs the fused time. The pytree floor is 0.07–0.09 ms of
   the 0.44 ms. In a scratch run, passing the 5 leaves flat instead of the
   registered `ModelInstance` cut the minimum from 0.25–0.31 ms to
   0.17–0.20 ms. That run was noisy: roughly 0.1 ms is Python-side pytree
   flattening. Candidate levers: vector-in likelihood entry points (the
   sampler path already works on vectors), batching via `vmap` so the
   overhead is amortised, and measuring on an uncontended host first.
2. **Backward pass.** `value_and_grad` is 2.43× forward, adding 0.63 ms over
   the forward call, with 4× the compile time. This matters for
   gradient-based searches. It is the only cost here that clearly scales with
   the math, because the reverse-mode pass runs through the jacfwd Hessian
   (forward-over-reverse). Candidate: an analytic Hessian for the SIE
   `Isothermal`, or a jacrev/jacfwd ordering study.
3. **Hessian / precision-tensor re-evaluation.** Measured share is about 0.
   XLA CSE merges the repeats (see above). This ranks below 1–2 and is kept
   only as a code-tidiness item that would reduce trace and compile time.
4. **Ray trace.** Its separable share cannot be resolved: it equals the whole
   call. Four positions through one SIE is negligible arithmetic, so it is not
   a lever until the overhead in item 1 is removed.
5. **A100 launch bound.** Unmeasured. At about 50 small fusions, one call on a
   GPU will be launch-latency-bound at roughly 50 × 5–10 µs, which is likely
   no faster than CPU. The useful GPU number is the `vmap` throughput, not the
   single call.

## Unmeasured / RAL handoff

The RAL rows were not run in phase 1 (they were not attempted); phase 2a
ran them — see "Phase 2a" below. Run the
committed instrument there after syncing the RAL checkout and PyAuto stack
(`HPCPullPyAuto`). Record the CPU model, pin `--nodelist`, and use a quiet
host:

```bash
python scripts/point_source_source/likelihood_breakdown/source_plane.py \
  --config-name hpc_ral_cpu_fp64
python scripts/point_source_source/likelihood_breakdown/source_plane.py \
  --config-name hpc_a100_fp64
```

The laptop row should also be re-run on an uncontended host before any phase-2
lever is judged against it.

## Carried to cluster epic

These are notes only, for epic `cluster-pointsolver-speed`; no action was
taken here.

- The closure-based `scripts/cluster/likelihood_breakdown/source_plane.py`
  closes over a fixed tracer, so its steps are exposed to constant folding.
  Its docstring still claims the plain end-to-end source-plane fit is
  JIT-blocked. That is no longer true on the live stack (see "What landed").
- The cluster-scale levers belong to that epic: the 13-profile dPIE/NFW
  deflection stack, and Hessian re-evaluation per plane. At cluster scale the
  arithmetic is large enough that the CSE and overhead conclusions above may
  not transfer.

## Phase 2a — RAL rows + pytree-input A/B (2026-09-26)

Issue: [autolens_profiling #322](https://github.com/PyAutoLabs/autolens_profiling/issues/322).
Branch `feature/point-source-source-plane-p2a`; the RAL jobs ran commit `ec29705`.
Single-source only.

### What landed

- `scripts/point_source_source/likelihood_breakdown/pytree_input_ab.py` is an
  interleaved in-process A/B of how the parameters enter the likelihood. It copies
  the protocol of `point_source_image/likelihood_breakdown/vertex_dedup_ab.py`:
  a fresh closure and a fresh `AnalysisPoint` per route after
  `jax.clear_caches()`, rotated route order, and a 16-instance fixed-seed
  stream. The three routes are:
  - `pytree`: production. The registered `ModelInstance` is the traced argument.
  - `flat_vector`: the physical vector is the argument, and
    `instance_from_vector(xp=jnp)` runs inside the trace.
  - `flat_leaves`: the leaves tuple is the argument, and `tree_unflatten` runs
    inside the trace.

  Each lane (solved, plain) gets `forward`, `value_and_grad` and `floor` rows.
  The hard asserts are: every route agrees on log L at rtol 1e-10, and every
  route's gradient is finite and non-zero. All asserts passed in every run, and
  the log-L deltas were exactly 0.
- Two submits: `hpc/batch_cpu/submit_breakdown_point_source_source_source_plane_ral_cpu_fp64`
  and `hpc/batch_gpu/submit_breakdown_point_source_source_source_plane_a100_fp64`.
  Their WALL-BASIS rows are measured on the RAL jobs. Known gaps, carried and
  not fixed here: `check_submits.py`'s `_PYTHON_CALL` misses `python3 -u`, and
  `hpc/sync pull` does not fetch `batch_cpu` logs. The CPU log was copied by
  hand to `results/notes/point_source_source_plane_2026_09_26_ral_job_356368.out`.
- New rows: `source_plane_{hpc_ral_cpu_fp64,hpc_a100_fp64}` and
  `pytree_input_ab_{hpc_ral_cpu_fp64,hpc_a100_fp64}` (JSON + PNG) under
  `results/breakdown/point_source_source/`.

### Hosts and revisions

- **RAL CPU**, job 356368, finished in 1:10. It ran on `euclid-ral-compute-10-2`
  (Intel Xeon Platinum 8490H), pinned with `--nodelist`. The node was idle, was
  responding, and hosted no other jnightin job. Settings: 8 CPUs,
  `sched_affinity` 8, BLAS threads 1, `JAX_PLATFORMS=cpu`, fp64. Load average
  was 0.08 at job start and 0.62 → 1.77 across the A/B, which is quiet. The
  process thread count after the first compile was 42.
- **A100**, job 356369, finished in 2:01. It ran on `euclid-ral-gpu-1`
  (A100 80GB PCIe, host AMD EPYC 7702) with `JAX_PLATFORMS=cuda` and
  `JAX_PLATFORM_NAME=cuda`. The backend was asserted to be `gpu`, and the JSON
  `device` field reads `cuda:0`.
- **Libraries**: the shared RAL stack, used as-is. The revisions were PyAutoFit
  `dd9fbe0a`, PyAutoArray `3de624b5`, PyAutoGalaxy `70a61e26`, PyAutoLens
  `86054bbc` and PyAutoNerves `1fa613aa`, with JAX 0.10.2. Against the local
  mains, the diff is the 2026.9.26.1 tag bumps plus interferometer/inversion
  and MultiStart changes. None of it touches `autolens/point/`, `AnalysisPoint`,
  `autofit/jax/pytrees.py` or `mapper/model.py`, so the stack is
  release-equivalent for these cells. Both JSON `device` fields match their
  labels. Every breakdown assert passed: eager vs JIT, `vmap`, the regression
  literals (A100 solved `0.5986504555536865`, 9e-14 relative from the literal)
  and the non-zero gradient (L2 337.62 on both devices).
- **Laptop**: the 1-minute load average was 5.06 at run time, above the 2.0
  bar, so the phase-1 `source_plane_local_cpu_fp64` row was **not** refreshed.
  It was measured at load ~16. The A/B ran only as a `--quick` functional
  check, and no local A/B JSON is committed. **RAL is the reference.**

### Breakdown rows (median ms per call)

| Quantity | local (phase 1, load ~16) | RAL CPU 8490H | A100 |
|---|---|---|---|
| Fused solved (`FitPositionsSourceSolved`) | 0.438 | **0.1465** (p10 0.130, p90 0.165) | **0.2217** (p10 0.202, p90 0.237) |
| Fused plain (`FitPositionsSource`) | 0.457 | **0.1450** (p10 0.126, p90 0.162) | **0.2654** (p10 0.257, p90 0.279) |
| `value_and_grad` solved | 1.065 (2.43×) | 0.331 (**2.26×**) | 0.412 (1.86×) |
| Floor: jit(sum of `ModelInstance` leaves), solved / plain | 0.086 / 0.069 | 0.032 / 0.035 | 0.201 / 0.203 |
| Floor: jit(x + 1) scalar | 0.019 | **0.0074** | **0.128** |
| CSE: precision ×3/×1; magnifications ×2/×1 | 1.15; 0.98 | 0.99; 1.00 | 1.27; 1.28 |
| Lower + compile, fused solved | 1.67 + 0.78 s | 0.70 s lower | 0.88 s lower |

Cumulative prefixes, absolute median ms:

| Solved prefix | RAL CPU | A100 |
|---|---|---|
| Ray trace `_beta_hat` | 0.147 | 0.187 |
| + precision tensor | 0.153 | 0.248 |
| + solved centre β* | 0.149 | 0.224 |
| + chi-squared | 0.148 | 0.260 |
| + marginalisation term | 0.152 | 0.238 |
| + `fit.log_likelihood` | 0.155 | 0.236 |
| Fused | 0.1465 | 0.2217 |

On the plain lane, RAL CPU is flat at 0.146 → 0.146 → 0.152 → 0.154 ms. On the
A100 it climbs from 0.160 → 0.227 → 0.222 → 0.257 ms.

What this shows:
- On a quiet host, the phase-1 picture holds and is sharper. On RAL CPU the
  bare ray-trace prefix costs the whole fused call, 0.147 vs 0.1465 ms. The
  CSE probes are 0.99 and 1.00, so XLA merges the repeated Hessian and
  precision work on CPU.
- The A100 is **slower than one RAL CPU core-group** on a single call:
  0.22–0.27 ms against 0.146 ms. Its scalar dispatch floor alone is 0.128 ms,
  and its CSE ratios of 1.27–1.28 show that the repeats cost extra launches
  there. This is the launch-bound regime that phase 1 predicted.

### Pytree-input A/B (20 rounds × 20 calls; median ms, ratio with bootstrap 90 % CI)

RAL CPU (`pytree_input_ab_hpc_ral_cpu_fp64.json`):

| Row | pytree | flat_vector | flat_leaves | pytree/flat_vector [CI] | saved ms | pytree/flat_leaves [CI] | saved ms |
|---|---|---|---|---|---|---|---|
| solved forward | 0.1451 | 0.1066 | 0.1084 | **1.361** [1.343, 1.385] | **0.0385** | 1.339 [1.321, 1.365] | 0.0367 |
| solved value_and_grad | 0.3154 | 0.2776 | 0.2924 | 1.136 [1.128, 1.146] | 0.0378 | 1.079 [1.068, 1.088] | 0.0230 |
| solved floor | 0.0337 | 0.0086 | 0.0101 | 3.937 [3.913, 3.965] | 0.0252 | 3.344 [3.323, 3.369] | 0.0237 |
| plain forward | 0.1472 | 0.1020 | 0.1072 | **1.443** [1.425, 1.462] | **0.0452** | 1.374 [1.360, 1.391] | 0.0400 |
| plain value_and_grad | 0.2669 | 0.2324 | 0.2450 | 1.148 [1.134, 1.158] | 0.0345 | 1.089 [1.080, 1.099] | 0.0219 |
| plain floor | 0.0360 | 0.0086 | 0.0117 | 4.185 [4.161, 4.218] | 0.0274 | 3.076 [3.055, 3.105] | 0.0243 |

A100 (`pytree_input_ab_hpc_a100_fp64.json`):

| Row | pytree | flat_vector | flat_leaves | pytree/flat_vector [CI] | saved ms | pytree/flat_leaves [CI] | saved ms |
|---|---|---|---|---|---|---|---|
| solved forward | 0.2933 | 0.2344 | 0.2405 | 1.251 [1.243, 1.262] | 0.0588 | 1.219 [1.212, 1.230] | 0.0527 |
| solved value_and_grad | 0.6850 | 0.5373 | 0.6371 | 1.275 [1.269, 1.280] | 0.1477 | 1.075 [1.071, 1.078] | 0.0479 |
| solved floor | 0.2182 | 0.1605 | 0.1710 | 1.360 [1.352, 1.373] | 0.0577 | 1.276 [1.268, 1.288] | 0.0472 |
| plain forward | 0.2635 | 0.2065 | 0.2149 | 1.276 [1.261, 1.291] | 0.0570 | 1.226 [1.216, 1.239] | 0.0486 |
| plain value_and_grad | 0.5588 | 0.4408 | 0.5080 | 1.268 [1.262, 1.273] | 0.1180 | 1.100 [1.096, 1.104] | 0.0508 |
| plain floor | 0.2218 | 0.1540 | 0.1735 | 1.440 [1.430, 1.450] | 0.0678 | 1.279 [1.268, 1.288] | 0.0484 |

The minimum detectable improvement on the RAL CPU pytree forward rows is
0.25–0.26. That is p10/p90 spread over the 16-instance stream, and every CI
above is tight and excludes 1. The laptop `--quick` check ran at load 5, with
5 × 5 calls. It gave solved forward 0.578 / 0.405 / 0.420 ms (1.43×, saving
0.17 ms) and plain 0.491 / 0.322 / 0.366 ms (1.53×). That is the same
direction, load-inflated about 4×.

What the A/B shows:
- The pytree cost is a **fixed per-call cost of 0.035–0.045 ms on RAL CPU**,
  and it is the same in the forward and `value_and_grad` rows. It is **Python
  flattening of the `ModelInstance`**, not the instance rebuild:
  - `flat_leaves` is within 2–5 % of `flat_vector` (`flat_leaves/flat_vector`
    is 1.017 for solved forward and 1.051 for plain).
  - So passing the same leaves without the custom-node flatten recovers almost
    all of the saving.
  - `instance_from_vector` inside the trace costs nothing at run time.
- In relative terms it is 26.5 % (solved) and 30.7 % (plain) of the fused
  forward call. It is the largest single separable piece of the forward
  call, because the compute itself sits at about 0.10 ms.

### Corrected ranked residue (RAL CPU, fused solved 0.1465 ms)

1. **Backward pass.** `value_and_grad` takes 0.331 ms, which is 2.26× forward
   and **+0.185 ms** over the forward call. This is the largest absolute
   residue, and the only one that scales with the math: forward-over-reverse
   through the jacfwd Hessian. The pytree fix would remove only 0.038 ms of it
   (1.14×).
2. **`ModelInstance` pytree flatten.** 0.0385 ms solved and 0.0452 ms plain,
   which is 27–31 % of the forward call. It is Python-side and fixed-cost.
3. **Executable launch plus the remaining compute.** About 0.10 ms on the
   flat routes, against a scalar floor of 0.0074 ms. It is not separable from
   the ray trace: the ray-trace prefix already equals the fused call.
4. **Hessian / precision-tensor repeats.** They are merged on CPU (0.99 /
   1.00), so they are not a lever there. On the A100 they cost 1.27–1.28×, but
   the A100 single call is launch-bound and slower than CPU in any case.
5. **A100 single call.** 0.22–0.27 ms, above RAL CPU. The GPU number that
   matters is `vmap` throughput, which is still unmeasured here.

### Phase-2b go/no-go

The rule from issue #322 is **go** if `pytree − flat_vector` ≥ 0.05 ms **AND**
≥ 15 % of the fused call on RAL CPU.

| Lane | saved ms | fraction of fused | ≥ 0.05 ms | ≥ 15 % | Verdict |
|---|---|---|---|---|---|
| solved | 0.0385 | 26.5 % | no | yes | no-go |
| plain | 0.0452 | 30.7 % | no | yes | no-go |

**Verdict: NO-GO for phase 2b as the PyAutoFit flatten fast path.** The
fraction criterion passes comfortably, but the absolute criterion fails. On a
quiet 8490H the whole call is only 0.146 ms, so a 27–31 % fixed cost is
0.04 ms. Per the rule, the **backward-pass lever is promoted**: an analytic SIE
Hessian, or a jacrev/jacfwd ordering study, attacking the +0.185 ms of
`value_and_grad`.

For the human reading this: the threshold was set against the load-inflated
~0.44 ms laptop call. The A100 rows meet both criteria (0.057–0.059 ms,
20–22 %), and so does the loaded laptop. The flatten fix is small and is
already located: `autofit/jax/pytrees.py` `_partition`/`flatten`, and
`ModelInstance.tree_flatten` in `mapper/model.py`. `pytree_input_ab.py` is its
ready red control. Revisiting the 0.05 ms bar is a policy call, not a
measurement gap.

## Phase 2b — backward-pass A/B (2026-09-27)

Issue: [autolens_profiling #325](https://github.com/PyAutoLabs/autolens_profiling/issues/325).
Branch `feature/point-source-source-plane-p2b`; the RAL jobs ran commit `eee7d5b`.
Single-source only. Workspace-only: every lever is prototyped inside the cell,
no library was edited.

### What landed

- `scripts/point_source_source/likelihood_breakdown/backward_pass_ab.py` is an
  interleaved in-process A/B of how the gradient of the production likelihood
  is computed. It copies the `pytree_input_ab.py` harness (flat-script
  convention: model/analysis builders, `register_model`, the 16-instance
  stream, `_stats_ms`, bootstrap ratios, fresh closure + fresh `AnalysisPoint`
  per route after `jax.clear_caches()`, rotated route order). Every route takes
  the same registered `ModelInstance` pytree argument and returns
  `(log L, gradient)`:
  - `rev`: production, `jax.value_and_grad(ll)`, Hessian by `jax.jacfwd`.
  - `fwd`: `jax.jacfwd(ll, has_aux=True)` over the pytree's 5 (solved) / 7
    (plain) scalar leaves, i.e. the flat parameter vector, value as aux.
  - `rev_jacrev`: `rev` with the Hessian built by `jax.jacrev` over
    `deflections_yx_scalar`.
  - `rev_analytic`: `rev` with the closed-form SIE Hessian.
  - `fwd_analytic`: `fwd` with the closed-form SIE Hessian.

  A secondary `forward` row times the likelihood alone under the three Hessian
  builders (`jacfwd` / `jacrev` / `analytic`), to split any saving into its
  forward and backward parts.
- Hessian swaps are a scoped monkeypatch of `LensCalc._hessian_via_jax`
  (context manager; the restore is asserted). Each route is traced, lowered and
  compiled inside its patch scope, and the timing loop calls those `Compiled`
  executables. Every patched builder counts its trace-time calls (asserted
  > 0; 3 per trace), and each row asserts the routes' StableHLO hashes are
  pairwise distinct, so no route can be served another's trace.
- Submits `hpc/batch_cpu/submit_backward_pass_ab_point_source_source_ral_cpu_fp64`
  (pinned to `euclid-ral-compute-10-2`; a command-line override runs the same
  file on the idle gpu-partition host CPUs as a quiet cross-check) and
  `hpc/batch_gpu/submit_backward_pass_ab_point_source_source_a100_fp64`.
- Rows `backward_pass_ab_{local_cpu_fp64,hpc_ral_cpu_fp64,hpc_ral_gpunode_cpu_fp64,hpc_a100_fp64}`
  (JSON + PNG) under `results/breakdown/point_source_source/`.

**Analytic Hessian: sign/order finding.** `Isothermal.shear_yx_2d_from`
returns `[:, 0] = γ₂` and `[:, 1] = γ₁`, with autogalaxy's `γ₁ = (H_xx − H_yy)/2`
and `γ₂ = H_xy`. So `H_xx = κ + γ₁`, `H_yy = κ − γ₁` and `H_xy = H_yx = γ₂`,
returned in `_hessian_via_jax`'s `(yy, xy, yx, xx)` order, with **no sign
flips**. This matches the `jacfwd` Hessian to ≤ 9.8e-16 (max component error
over the largest |component| per point).

**Library bug found (not fixed here).** The library `Isothermal.convergence_2d_from`
and `shear_yx_2d_from` **cannot be traced under `jax.jit` with a traced
`ell_comps`**. `PowerLawCore.convergence_2d_from` calls
`self.convergence_func(grid_radius=grid_eta)` without `xp`, so
`einstein_radius_rescaled` → `axis_ratio` runs `np.logical_and` on a tracer
(`TracerArrayConversionError`, `autogalaxy/convert.py:80`). `shear_yx_2d_from`
inherits it through its internal `convergence_2d_from` call. The cell therefore
rebuilds the same formulas from the profile's own geometry helpers with
`xp=jnp` threaded through (`_sie_kappa_gamma`). The gate checks that function
against the library methods evaluated eagerly with NumPy: max relative error
5.5e-16. A library-side analytic Hessian would need this `xp` fix first.

The analytic builder **refuses** (raises at trace time) unless the tracer is two
planes with exactly one mass profile, of type `Isothermal`, in the image plane.
It never approximates.

### Hosts and revisions

- **RAL CPU 8490H** — job 357380, `euclid-ral-compute-10-2`, pinned with
  `--nodelist`. **Never ran: cancelled at the human's decision (see "Phase-2c
  go/no-go").** It was submitted at 12:14 and was still `PD (Priority)` at 13:10. On
  2026-09-27 every 10-* node was `mix` and carried jnightin DR1 array tasks:
  10-2 had 18–20 of them, with 144–160 of 236 CPUs allocated, so it was not quiet
  the way it was for phase 2a. Thousands of same-priority DR1 array tasks were
  queued ahead of the job, and `scontrol top` is not permitted for users. There
  is therefore no `backward_pass_ab_hpc_ral_cpu_fp64` row.
- **RAL gpu-partition host CPUs (quiet cross-check)** — job 357381,
  `euclid-ral-gpu-2` (AMD EPYC 7702), `--partition=gpu` with no `--gres`, 8 CPUs,
  `sched_affinity` 8, BLAS threads 1. The partition was idle, and load average
  was 0.10 at start and 1.52 at end. Process wall was 358 s.
- **A100** — job 357382, `euclid-ral-gpu-1` (A100, host AMD EPYC 7702),
  `JAX_PLATFORMS=cuda`. The backend was asserted to be `gpu`, and the JSON
  device is `cuda:0`. Process wall was 601 s.
- **Laptop (lead only)** — i9-10885H, WSL2, 8 threads
  (`OMP/OPENBLAS/MKL/NPROC=8`, matching the phase-1 laptop row). Load average
  was 1.56 at start and 8.32 at end: another session started work mid-run.
  Process wall was 527 s.
- **Libraries.** Every JSON records PyAutoNerves `eb27da24`, PyAutoFit
  `5468c6ce`, PyAutoArray `14d63360`, PyAutoGalaxy `879a9308`, PyAutoLens
  `def4decf` and JAX 0.10.2. These are the library mains. Both the laptop
  canonical mains (11:55:39, mid-run) and the RAL shared stack (before the jobs
  started at 12:14) were fast-forwarded by other sessions from Fit `cf83504e` /
  Galaxy `0e4b89cf` / Lens `4487eb47` to the `2026.9.27.1` release commits. Those
  commits touch docs only (Colab URL tag bumps, `docs/`, `paper/`), so the code
  that ran is identical on every host. PyAutoArray did not move.

### Correctness gate (worst over laptop, gpu-node CPU and A100)

The gate ran before any timing, and no route failed on any host. Its three parts:

- **Hessian.** The analytic and jacrev Hessians are compared with jacfwd at
  rtol 1e-10, at three point sets: the 4 dataset positions; a 28-point
  near-critical set, made by moving each position along its ray from the lens
  centre onto the tangential critical curve (bisection on `1 − κ − |γ|`, with a
  residual ≤ 6e-16) and then ±1e-2 / 1e-4 / 1e-6 relative off it; and 17
  models (the prior medians plus `PRNGKey` 0..15 draws).
- **Route agreement.** Each route's log L is compared with `rev` at rtol 1e-10,
  and its gradient at rtol 1e-8, over the same 17 draws. The gradient must be
  finite and non-zero.
- **Eager vs JIT.** Each route runs under `jax.disable_jit()` on the prior
  medians and `PRNGKey` 0 and 1, and is compared with its JIT result.

| Lane | Route | Hessian: worst rel err vs jacfwd | log L vs `rev` | gradient vs `rev` | eager vs JIT (log L / grad) | min ‖∇‖ | Result |
|---|---|---|---|---|---|---|---|
| solved | `rev` | — (jacfwd) | 0.0e+00 | 0.0e+00 | 1.5e-12 / 3.0e-11 | 103.4 | PASS |
| solved | `fwd` | — (jacfwd) | 6.9e-13 | 5.5e-12 | 1.5e-12 / 2.3e-12 | 103.4 | PASS |
| solved | `rev_jacrev` | 8.6e-16 | 1.7e-12 | 8.7e-12 | 7.5e-13 / 2.2e-11 | 103.4 | PASS |
| solved | `rev_analytic` | 9.8e-16 | 1.5e-12 | 3.0e-11 | 6.8e-13 / 1.7e-11 | 103.4 | PASS |
| solved | `fwd_analytic` | 9.8e-16 | 1.5e-12 | 1.8e-11 | 1.1e-12 / 2.0e-11 | 103.4 | PASS |
| plain | `rev` | — (jacfwd) | 0.0e+00 | 0.0e+00 | 2.9e-14 / 1.0e-13 | 7.2e+04 | PASS |
| plain | `fwd` | — (jacfwd) | 8.1e-14 | 1.5e-13 | 5.2e-14 / 1.0e-13 | 7.2e+04 | PASS |
| plain | `rev_jacrev` | 8.6e-16 | 5.5e-13 | 9.6e-12 | 1.1e-13 / 2.0e-13 | 7.2e+04 | PASS |
| plain | `rev_analytic` | 9.8e-16 | 2.6e-13 | 5.2e-12 | 1.0e-13 / 1.6e-13 | 7.2e+04 | PASS |
| plain | `fwd_analytic` | 9.8e-16 | 2.6e-13 | 5.2e-12 | 1.4e-14 / 1.5e-13 | 7.2e+04 | PASS |

The in-cell κ/γ matches the library `convergence_2d_from` / `shear_yx_2d_from`
evaluated eagerly to ≤ 5.5e-16. The gradient L2 norms agree to 12+ digits across
routes and hosts, so the model is registered and the gradients are not zero.

### Backward-pass A/B (20 rounds × 20 calls, interleaved; median ms, ratio and saving vs control with bootstrap 90 % CI)

The `grad` rows use `rev` as the control. The `forward` rows time the likelihood
alone, with `jacfwd` as the control.

**RAL CPU 8490H (`hpc_ral_cpu_fp64`): not run (job 357380 cancelled; see "Phase-2c go/no-go").**

RAL gpu-partition host CPUs, AMD EPYC 7702, quiet (`backward_pass_ab_hpc_ral_gpunode_cpu_fp64.json`):

| Lane / row | Route | median ms (p10–p90) | ratio vs control [90 % CI] | saved ms [90 % CI] | lower + compile s |
|---|---|---|---|---|---|
| solved grad | `rev` | 0.6356 (0.537–0.742) | control | — | 1.51 + 1.33 |
| solved grad | `fwd` | 0.3411 (0.309–0.371) | 0.537 [0.526, 0.544] | +0.2945 [+0.2856, +0.3064] | 1.27 + 1.11 |
| solved grad | `rev_jacrev` | 0.6184 (0.529–0.704) | 0.973 [0.955, 0.993] | +0.0172 [+0.0051, +0.0290] | 1.67 + 1.37 |
| solved grad | `rev_analytic` | 0.4754 (0.422–0.530) | 0.748 [0.735, 0.761] | +0.1602 [+0.1506, +0.1707] | 0.52 + 1.00 |
| solved grad | `fwd_analytic` | 0.3584 (0.328–0.405) | 0.564 [0.554, 0.573] | +0.2771 [+0.2677, +0.2875] | 0.51 + 0.85 |
| solved forward | `jacfwd` | 0.2624 (0.235–0.295) | control | — | 0.75 + 0.45 |
| solved forward | `jacrev` | 0.2602 (0.233–0.294) | 0.992 [0.975, 1.006] | +0.0022 [-0.0016, +0.0067] | 0.76 + 0.48 |
| solved forward | `analytic` | 0.2591 (0.234–0.293) | 0.987 [0.974, 1.002] | +0.0033 [-0.0005, +0.0070] | 0.18 + 0.30 |
| plain grad | `rev` | 0.5066 (0.456–0.572) | control | — | 1.04 + 0.94 |
| plain grad | `fwd` | 0.3145 (0.290–0.344) | 0.621 [0.612, 0.627] | +0.1921 [+0.1885, +0.1982] | 0.94 + 0.81 |
| plain grad | `rev_jacrev` | 0.5007 (0.452–0.555) | 0.988 [0.978, 0.998] | +0.0060 [+0.0010, +0.0111] | 1.10 + 0.94 |
| plain grad | `rev_analytic` | 0.4237 (0.384–0.470) | 0.836 [0.827, 0.844] | +0.0830 [+0.0788, +0.0878] | 0.34 + 0.65 |
| plain grad | `fwd_analytic` | 0.3090 (0.286–0.341) | 0.610 [0.603, 0.618] | +0.1976 [+0.1932, +0.2027] | 0.38 + 0.65 |
| plain forward | `jacfwd` | 0.2544 (0.230–0.287) | control | — | 0.61 + 0.32 |
| plain forward | `jacrev` | 0.2566 (0.231–0.286) | 1.009 [0.995, 1.022] | -0.0022 [-0.0054, +0.0012] | 0.46 + 0.34 |
| plain forward | `analytic` | 0.2547 (0.231–0.283) | 1.001 [0.985, 1.016] | -0.0004 [-0.0041, +0.0038] | 0.12 + 0.22 |

A100 (`backward_pass_ab_hpc_a100_fp64.json`):

| Lane / row | Route | median ms (p10–p90) | ratio vs control [90 % CI] | saved ms [90 % CI] | lower + compile s |
|---|---|---|---|---|---|
| solved grad | `rev` | 0.5272 (0.510–0.562) | control | — | 1.52 + 4.13 |
| solved grad | `fwd` | 0.3571 (0.344–0.396) | 0.677 [0.675, 0.680] | +0.1702 [+0.1683, +0.1716] | 1.28 + 1.79 |
| solved grad | `rev_jacrev` | 0.5259 (0.509–0.559) | 0.997 [0.995, 1.003] | +0.0013 [-0.0015, +0.0028] | 1.68 + 4.25 |
| solved grad | `rev_analytic` | 0.4604 (0.441–0.494) | 0.873 [0.870, 0.879] | +0.0669 [+0.0638, +0.0688] | 0.52 + 2.65 |
| solved grad | `fwd_analytic` | 0.3498 (0.337–0.390) | 0.664 [0.662, 0.667] | +0.1774 [+0.1753, +0.1786] | 0.52 + 1.31 |
| solved forward | `jacfwd` | 0.2465 (0.238–0.267) | control | — | 0.79 + 0.68 |
| solved forward | `jacrev` | 0.2472 (0.240–0.264) | 1.003 [0.996, 1.009] | -0.0007 [-0.0023, +0.0008] | 0.76 + 0.72 |
| solved forward | `analytic` | 0.2384 (0.231–0.253) | 0.967 [0.960, 0.974] | +0.0082 [+0.0064, +0.0097] | 0.18 + 0.47 |
| plain grad | `rev` | 0.4661 (0.452–0.491) | control | — | 1.05 + 2.45 |
| plain grad | `fwd` | 0.3528 (0.340–0.385) | 0.757 [0.750, 0.766] | +0.1133 [+0.1090, +0.1165] | 0.95 + 1.47 |
| plain grad | `rev_jacrev` | 0.4634 (0.449–0.488) | 0.994 [0.989, 0.999] | +0.0027 [+0.0003, +0.0054] | 1.12 + 2.52 |
| plain grad | `rev_analytic` | 0.4426 (0.428–0.472) | 0.950 [0.944, 0.957] | +0.0235 [+0.0200, +0.0266] | 0.35 + 1.91 |
| plain grad | `fwd_analytic` | 0.3556 (0.341–0.377) | 0.763 [0.757, 0.768] | +0.1105 [+0.1079, +0.1136] | 0.39 + 1.01 |
| plain forward | `jacfwd` | 0.2413 (0.233–0.267) | control | — | 0.65 + 0.56 |
| plain forward | `jacrev` | 0.2437 (0.236–0.266) | 1.010 [1.003, 1.017] | -0.0024 [-0.0042, -0.0009] | 0.47 + 0.57 |
| plain forward | `analytic` | 0.2304 (0.208–0.251) | 0.955 [0.949, 0.961] | +0.0109 [+0.0096, +0.0123] | 0.12 + 0.37 |

Laptop lead, i9-10885H, 8 threads, load 1.6 → 8.3 (`backward_pass_ab_local_cpu_fp64.json`):

| Lane / row | Route | median ms (p10–p90) | ratio vs control [90 % CI] | saved ms [90 % CI] | lower + compile s |
|---|---|---|---|---|---|
| solved grad | `rev` | 0.7504 (0.652–0.950) | control | — | 1.64 + 1.38 |
| solved grad | `fwd` | 0.4737 (0.410–0.582) | 0.631 [0.618, 0.645] | +0.2767 [+0.2634, +0.2907] | 1.78 + 1.25 |
| solved grad | `rev_jacrev` | 0.7455 (0.653–0.918) | 0.994 [0.975, 1.014] | +0.0048 [-0.0101, +0.0182] | 1.98 + 1.78 |
| solved grad | `rev_analytic` | 0.6214 (0.540–0.757) | 0.828 [0.810, 0.846] | +0.1290 [+0.1138, +0.1434] | 0.80 + 1.04 |
| solved grad | `fwd_analytic` | 0.4928 (0.443–0.653) | 0.657 [0.644, 0.669] | +0.2576 [+0.2445, +0.2707] | 0.54 + 0.82 |
| solved forward | `jacfwd` | 0.3380 (0.306–0.378) | control | — | 1.26 + 0.81 |
| solved forward | `jacrev` | 0.3377 (0.304–0.385) | 0.999 [0.989, 1.012] | +0.0002 [-0.0038, +0.0037] | 1.37 + 0.60 |
| solved forward | `analytic` | 0.3296 (0.299–0.369) | 0.975 [0.968, 0.986] | +0.0083 [+0.0046, +0.0108] | 0.41 + 0.47 |
| plain grad | `rev` | 0.8020 (0.707–0.931) | control | — | 1.44 + 1.14 |
| plain grad | `fwd` | 0.5799 (0.499–0.682) | 0.723 [0.709, 0.736] | +0.2221 [+0.2107, +0.2340] | 1.21 + 1.02 |
| plain grad | `rev_jacrev` | 0.8054 (0.698–0.977) | 1.004 [0.986, 1.023] | -0.0034 [-0.0180, +0.0121] | 1.86 + 1.36 |
| plain grad | `rev_analytic` | 0.7012 (0.623–0.857) | 0.874 [0.860, 0.892] | +0.1008 [+0.0856, +0.1132] | 0.53 + 0.91 |
| plain grad | `fwd_analytic` | 0.5500 (0.487–0.672) | 0.686 [0.678, 0.696] | +0.2520 [+0.2414, +0.2607] | 0.50 + 0.86 |
| plain forward | `jacfwd` | 0.4808 (0.420–0.605) | control | — | 0.81 + 0.38 |
| plain forward | `jacrev` | 0.5043 (0.429–0.615) | 1.049 [1.030, 1.070] | -0.0235 [-0.0332, -0.0147] | 0.72 + 0.42 |
| plain forward | `analytic` | 0.4927 (0.416–0.627) | 1.025 [0.999, 1.043] | -0.0119 [-0.0202, +0.0014] | 0.27 + 0.26 |

What the A/B shows (all hosts that ran):
- **`fwd` is the large lever.** Forward-mode over the 5 or 7 scalar leaves
  saves 0.11–0.29 ms per `value_and_grad`-equivalent call: 24–46 % of `rev`,
  with CIs well clear of zero on every host. It brings the gradient call from
  about 2.4× the forward call to about 1.3× (gpu-node CPU solved:
  0.636 → 0.341 ms, against a forward of 0.262 ms).
- **The analytic Hessian helps only under `rev`.**
  - `rev_analytic` saves 0.08–0.16 ms on the CPUs, but only 0.02–0.07 ms on
    the A100.
  - `fwd_analytic` ≈ `fwd` on every host (within 0.03 ms, in both directions). Once the gradient is forward-mode, differentiating
    through the jacfwd Hessian is no longer the cost.
  - In the `forward` rows the analytic Hessian saves ≤ 0.011 ms. The Hessian
    is not a forward-pass cost; the reverse-over-forward tape is.
- **`rev_jacrev` ≈ `rev`** everywhere: ratio 0.97–1.00, saving ≤ 0.017 ms.
  Hessian ordering is not a lever.
- **XLA FLOPs barely move** (267–298 k on every `grad` route, against
  263–345 k for the forward rows). The saving is dispatch and tape structure, not arithmetic.
- **Compile.** Both analytic routes lower 2–3× faster than the jacfwd routes
  (0.5 vs 1.3–1.7 s on CPU). On the A100, `fwd`/`fwd_analytic` compile in
  1.3–1.8 s against 4.1 s for `rev`. Compile is not part of the rule.
- **The gpu-node host is not the 8490H.** Its forward call is 0.26 ms, against
  phase 2a's 0.1465 ms on 10-2, so its absolute savings are not the reference.
  They are interleaved in-process on a quiet host, though, so the ratios are
  sound.

### Phase-2c go/no-go

The rule from issue #325: a route is **GO** if, on RAL CPU (8490H), it saves
≥ 0.05 ms **AND** ≥ 15 % against `rev` on the `value_and_grad`-equivalent
call, with the 90 % CIs excluding the bar (saved-ms CI low ≥ 0.05; ratio CI
high ≤ 0.85) and a green correctness gate.

**Deciding row, re-based by the human (2026-09-27): the quiet RAL gpu-partition
host CPUs (EPYC 7702, job 357381).** The pre-registered 8490H row could not run in
the session: job 357380 sat behind thousands of DR1 array tasks, and node 10-2
carried 18–20 of them, so it would not have been quiet even if it had started.
Every host that did run agrees on the direction and clears the bar for `fwd` by a
wide margin. The human chose to decide on the EPYC row and cancel 357380. Its
absolute milliseconds are higher than the 8490H's (forward 0.26 vs 0.146 ms), so
the savings are quoted as ratios first.

The mechanical evaluation on every row that ran (the EPYC table decides):

RAL gpu-partition host CPUs (EPYC 7702, quiet):

| Lane | Route | saved ms | fraction | ≥ 0.05 ms (CI low) | ≥ 15 % (CI high ≤ 0.85) | gate | rule |
|---|---|---|---|---|---|---|---|
| solved | `fwd` | +0.2945 | 46.3% | yes (+0.2856) | yes (0.544) | green | GO |
| solved | `rev_jacrev` | +0.0172 | 2.7% | no (+0.0051) | no (0.993) | green | no-go |
| solved | `rev_analytic` | +0.1602 | 25.2% | yes (+0.1506) | yes (0.761) | green | GO |
| solved | `fwd_analytic` | +0.2771 | 43.6% | yes (+0.2677) | yes (0.573) | green | GO |
| plain | `fwd` | +0.1921 | 37.9% | yes (+0.1885) | yes (0.627) | green | GO |
| plain | `rev_jacrev` | +0.0060 | 1.2% | no (+0.0010) | no (0.998) | green | no-go |
| plain | `rev_analytic` | +0.0830 | 16.4% | yes (+0.0788) | yes (0.844) | green | GO |
| plain | `fwd_analytic` | +0.1976 | 39.0% | yes (+0.1932) | yes (0.618) | green | GO |

A100:

| Lane | Route | saved ms | fraction | ≥ 0.05 ms (CI low) | ≥ 15 % (CI high ≤ 0.85) | gate | rule |
|---|---|---|---|---|---|---|---|
| solved | `fwd` | +0.1702 | 32.3% | yes (+0.1683) | yes (0.680) | green | GO |
| solved | `rev_jacrev` | +0.0013 | 0.3% | no (-0.0015) | no (1.003) | green | no-go |
| solved | `rev_analytic` | +0.0669 | 12.7% | yes (+0.0638) | no (0.879) | green | no-go |
| solved | `fwd_analytic` | +0.1774 | 33.6% | yes (+0.1753) | yes (0.667) | green | GO |
| plain | `fwd` | +0.1133 | 24.3% | yes (+0.1090) | yes (0.766) | green | GO |
| plain | `rev_jacrev` | +0.0027 | 0.6% | no (+0.0003) | no (0.999) | green | no-go |
| plain | `rev_analytic` | +0.0235 | 5.0% | no (+0.0200) | no (0.957) | green | no-go |
| plain | `fwd_analytic` | +0.1105 | 23.7% | yes (+0.1079) | yes (0.768) | green | GO |

Laptop (lead):

| Lane | Route | saved ms | fraction | ≥ 0.05 ms (CI low) | ≥ 15 % (CI high ≤ 0.85) | gate | rule |
|---|---|---|---|---|---|---|---|
| solved | `fwd` | +0.2767 | 36.9% | yes (+0.2634) | yes (0.645) | green | GO |
| solved | `rev_jacrev` | +0.0048 | 0.6% | no (-0.0101) | no (1.014) | green | no-go |
| solved | `rev_analytic` | +0.1290 | 17.2% | yes (+0.1138) | yes (0.846) | green | GO |
| solved | `fwd_analytic` | +0.2576 | 34.3% | yes (+0.2445) | yes (0.669) | green | GO |
| plain | `fwd` | +0.2221 | 27.7% | yes (+0.2107) | yes (0.736) | green | GO |
| plain | `rev_jacrev` | -0.0034 | -0.4% | no (-0.0180) | no (1.023) | green | no-go |
| plain | `rev_analytic` | +0.1008 | 12.6% | yes (+0.0856) | no (0.892) | green | no-go |
| plain | `fwd_analytic` | +0.2520 | 31.4% | yes (+0.2414) | yes (0.696) | green | GO |

**Verdict and recommendation (main session, 2026-09-27).**

- **`fwd` is GO.** Forward-mode gradients cut the `value_and_grad`-equivalent call
  by 46 % (solved) and 38 % (plain) on the deciding EPYC row, with CIs far from
  the bar. They also cut it by 24–34 % on the A100 and 28–37 % on the laptop, and
  roughly halve the A100 compile (4.1 → 1.8 s). The XLA flop count barely moves
  (267–298 k), so the saving is reverse-mode tape and dispatch structure, not
  arithmetic. With 5 free parameters, 5 JVPs beat one reverse sweep through the
  jacfwd Hessian.
- **`rev_analytic` passes on the CPU rows but is not pursued.** It fails the 15 %
  bar on the A100, and it adds nothing on top of `fwd`: `fwd_analytic` is within
  0.03 ms of `fwd` everywhere. The Hessian costs time only through reverse mode,
  so switching the AD mode removes the reason for an analytic Hessian.
- **`rev_jacrev` is NO-GO** on every host (≤ 3 %): the inner Hessian ordering does
  not matter. This is a negative result.
- **Phase 2c = the forward-mode gradient entry point, and it is sampler-facing.**
  Forward mode costs one JVP per free parameter, whereas reverse mode costs a
  roughly constant multiple of the forward call. So `fwd` wins only below a
  crossover `n_params`, and a production switch must be conditional. Phase 2c
  should first measure the crossover on this likelihood: add external shear,
  multipoles and a second lens to take the model from 5 to about 20 free
  parameters, in the same A/B harness. Only after that should it touch the
  PyAutoFit gradient entry point (the `value_and_grad` / `grad` call the
  gradient searches use), library-first with a GPU regression check. Where that
  switch lives, and how the threshold is chosen, is a human design decision
  before phase 2c is issued.
- **Carried library bug (separate from phase 2c).** `Isothermal.convergence_2d_from`
  and `shear_yx_2d_from` cannot be traced under `jax.jit` when `ell_comps` is
  traced: `PowerLawCore.convergence_2d_from` calls `convergence_func` without
  `xp`, and `einstein_radius_rescaled` runs NumPy on a tracer
  (`autogalaxy/convert.py:80`). The cell works around it in-cell. File it through
  intake as a PyAutoGalaxy bug.


## Phase 2c — gradient-mode crossover (2026-09-27)

Issue: [autolens_profiling #329](https://github.com/PyAutoLabs/autolens_profiling/issues/329).
Branch `feature/point-source-source-plane-p2c`; the RAL jobs ran commit `94dcd1d`.
Single-source only. Workspace-only: no library was edited.

### What landed

- `scripts/point_source_source/likelihood_breakdown/gradient_mode_crossover.py`
  measures the forward/reverse gradient ratio along a model-complexity ladder
  on the phase-2b `simple` dataset. It copies the `backward_pass_ab.py` harness
  (flat-script convention: `register_model`, fixed-seed 16-vector stream,
  fresh closure + fresh `AnalysisPoint` per executable after
  `jax.clear_caches()`, rotated interleaving, bootstrap ratio CIs, StableHLO
  hash guard, lower/compile/first-call timing). Per rung × lane it compiles four
  executables and times them interleaved, 20 rounds × 20 calls:
  - `rev_single`: `jax.value_and_grad(ll)`;
  - `fwd_single`: `jax.jacfwd(ll, has_aux=True)`, value as aux, one tangent per
    free parameter;
  - `rev_batched` / `fwd_batched`: `jax.jit(jax.vmap(route))` over B = 8
    vectors, the shape the multi-start gradient search uses
    (`multi_start_gradient/search.py:1089`).
- **Argument = the flat physical vector, not the pytree.** Every route builds
  the instance inside the trace with `model.instance_from_vector(vector, xp=jnp)`,
  as PyAutoFit's JAX `Fitness.call` does. From L9 on this is not a detail: the
  linked multipole priors make the `ModelInstance` pytree carry more leaves than
  free parameters (12 leaves for 9 parameters at L9, 30 for 24 at L24), so
  `jacfwd` over the pytree (phase 2b's argument, where leaves = parameters)
  would push one tangent per leaf. The L5 ratios are also lower than phase
  2b's (EPYC solved 0.41 here vs 0.54 there): both calls are faster with the
  flat vector (`rev` 0.60 vs 0.64 ms, `fwd` 0.24 vs 0.34 ms), `fwd` much more
  so. The likely reason is that `jacfwd` over one vector argument is cheaper
  than over five scalar pytree leaves, but this cell does not isolate it.
- Submits `hpc/batch_cpu/submit_gradient_mode_crossover_point_source_source_ral_cpu_fp64`
  (the **reference CPU row**: `--partition=gpu`, no `--gres`, pinned to
  `euclid-ral-gpu-2`, label `hpc_ral_gpunode_cpu_fp64`) and
  `hpc/batch_gpu/submit_gradient_mode_crossover_point_source_source_a100_fp64`
  (pinned to `euclid-ral-gpu-1`). WALL-BASIS = the laptop's measured 1906 s.
- Rows `gradient_mode_crossover_{local_cpu_fp64,hpc_ral_gpunode_cpu_fp64,hpc_a100_fp64}`
  (JSON + PNG) under `results/breakdown/point_source_source/`; the CPU job log
  is `results/notes/point_source_source_plane_2026_09_27_ral_job_358770.out`.

### The ladder

| Rung | Components (lens galaxy unless noted) | n_params solved / plain | `ModelInstance` pytree leaves solved / plain |
|---|---|---|---|
| L5 | Isothermal | 5 / 8 | 5 / 8 |
| L7 | Isothermal + ExternalShear | 7 / 10 | 7 / 10 |
| L9 | + PowerLawMultipole m=4 (comps free; centre/theta_E linked, slope 2) | 9 / 12 | 12 / 15 |
| L11 | + PowerLawMultipole m=3 (comps free, linked) | 11 / 14 | 17 / 20 |
| L16 | + satellite Isothermal at z=0.5 (centre (-1, 1), theta_E 0.1) | 16 / 19 | 22 / 25 |
| L19 | main Isothermal -> PowerLaw (free slope, multipole slopes linked) + satellite shear | 19 / 22 | 25 / 28 |
| L24 | + second satellite Isothermal at z=0.5 (centre (1, -1), theta_E 0.1) [extension] | 24 / 27 | 30 / 33 |

Build notes (each checked against the installed stack):

- The multipoles' centre and Einstein radius are linked to the main lens with af
  prior linking; their slope is fixed at 2.0 (`PowerLawMultipole`'s default,
  the SIE value) and linked to the main `PowerLaw` slope at L19/L24.
- **Perturbation priors are centred on 1e-3, not 0.** `ExternalShear`,
  `PowerLawMultipole` comps and the satellite `Isothermal`'s `ell_comps` are
  parameterised by a magnitude `sqrt(c₀² + c₁²)` and an angle, and
  `jax.grad` of their deflections is **NaN at exactly (0, 0)** (checked
  directly; a first ladder with means of 0 had NaN gradients at the prior
  medians from L7 up). With means of 1e-3 (σ 0.01) every draw is finite.
- **`PowerLawMultipole` m=1 is singular at slope exactly 2**: its deflections
  are `-inf` / NaN at the SIE prior median, so it cannot sit on an SIE-centred
  slope prior. L19 therefore uses the plan's other option, a satellite
  `ExternalShear` (exactly degenerate with the main shear in deflection, which is
  harmless for timing and the gradient gate).
- **The plain lane is +3, not +2.** `PointFlux` adds a flux parameter that the
  positions-only likelihood never reads (its gradient entry is 0). The phase-2b
  JSONs already recorded `plain: 8`; the phase-2b note's "7 (plain)" was a slip.
- **L24 is an extension beyond the planned ladder**: a second satellite
  `Isothermal` at (1.0, −1.0). It was added after the laptop lead run showed
  `fwd` still winning at L19, so that a crossing, if any, would be bracketed
  rather than extrapolated.
- Satellite centres are 1.4″ from the lens and ≥ 1.49″ from every image, with
  θ_E 0.1″, so the likelihood stays well-defined.

### Hosts and revisions

- **RAL gpu-partition host CPUs (reference)** — job 358770, `euclid-ral-gpu-2`
  (AMD EPYC 7702), `--partition=gpu`, no `--gres`, 8 CPUs, `sched_affinity` 8,
  BLAS threads 1. The node was otherwise idle: load average 0.28 at start and
  1.26 at end. Process wall 1348 s; `COMPLETED` in 22:51.
- **A100** — job 358771, `euclid-ral-gpu-1` (host EPYC 7702),
  `JAX_PLATFORMS=cuda`, backend asserted `gpu`, JSON device `cuda:0`. Process
  wall 1546 s; `COMPLETED` in 26:07.
- **Laptop (lead)** — i9-10885H, WSL2, 8 threads (`OMP/OPENBLAS/MKL/NPROC=8`).
  Load average 8.2 at start (other sessions) and 4.2 at end; process wall 1906 s.
  The laptop's absolute times and its batched ratios are load-exposed.
- **8490H (`ral`)** — not run: `sinfo` showed all 28 `ral` nodes allocated at
  submit time (not quiet), so the optional extra row was skipped.
- **Libraries.** PyAutoNerves `eb27da24`, PyAutoFit `5468c6ce`, PyAutoArray
  `14d63360`, PyAutoGalaxy `879a9308`, PyAutoLens `def4decf`, JAX 0.10.2 on
  every host: the library mains, and the RAL shared stack matched them before
  submission. The laptop JSON's end-of-run `source_revisions` reads PyAutoArray
  `4383ea81`, because another session fast-forwarded the canonical checkout at
  17:12:46, after this process had imported the libraries (16:43). The two
  commits touch only `autoarray/structures/triangles/` (PointSolver routes; this
  cell runs `solver=None`). The JSON carries a `source_revisions_note` saying so.
- The PYTHONPATH inherited by the session pointed at a stale
  `interferometer-sparse-cache` library worktree. The laptop run used an
  explicit PYTHONPATH of the five canonical mains instead.

### Correctness gate (worst over all three hosts)

Before any timing, per rung × lane, over the prior medians plus `PRNGKey` 0..15
draws (17 vectors): `fwd_single`, `rev_batched` and `fwd_batched` must equal
`rev_single` (log L rtol 1e-10, gradient rtol 1e-8 with atol 1e-12 × max|∇|),
all finite with non-zero L2 gradient; eager (`jax.disable_jit()`) must equal JIT
for both single routes on the prior medians and `PRNGKey` 0 and 1; and the four
StableHLO hashes must be pairwise distinct. No rung failed on any host, so every
rung was timed.

| Rung | Lane | n | log L fwd/batched vs `rev` single | gradient vs `rev` single | eager vs JIT (log L / grad) | min ‖∇‖ | HLO distinct | Result (3 hosts) |
|---|---|---|---|---|---|---|---|---|
| L5 | solved | 5 | 1.1e-12 | 1.7e-11 | 1.5e-12 / 3.0e-11 | 103 | yes | PASS |
| L5 | plain | 8 | 8.1e-14 | 2.7e-13 | 5.2e-14 / 1.0e-13 | 7.2e+04 | yes | PASS |
| L7 | solved | 7 | 3.1e-13 | 2.6e-10 | 3.1e-13 / 4.5e-12 | 166 | yes | PASS |
| L7 | plain | 10 | 4.7e-14 | 1.0e-12 | 3.3e-14 / 1.8e-13 | 2.39e+05 | yes | PASS |
| L9 | solved | 9 | 1.4e-12 | 1.5e-11 | 2.5e-12 / 1.5e-11 | 423 | yes | PASS |
| L9 | plain | 12 | 1.3e-13 | 7.7e-13 | 8.9e-14 / 1.8e-13 | 1.95e+05 | yes | PASS |
| L11 | solved | 11 | 4.1e-12 | 4.5e-11 | 6.2e-12 / 2.4e-11 | 680 | yes | PASS |
| L11 | plain | 14 | 5.2e-13 | 7.9e-13 | 3.5e-13 / 8.8e-13 | 1.65e+05 | yes | PASS |
| L16 | solved | 16 | 3.5e-13 | 3.5e-09 | 1.8e-12 / 1.7e-10 | 809 | yes | PASS |
| L16 | plain | 19 | 2.5e-13 | 2.4e-12 | 3.8e-14 / 1.3e-12 | 9.03e+05 | yes | PASS |
| L19 | solved | 19 | 1.1e-12 | 2.6e-09 | 6.7e-14 / 2.6e-10 | 1.67e+03 | yes | PASS |
| L19 | plain | 22 | 1.8e-13 | 4.0e-12 | 2.1e-14 / 5.7e-13 | 1.83e+06 | yes | PASS |
| L24 | solved | 24 | 6.0e-13 | 1.7e-10 | 1.9e-14 / 1.9e-11 | 469 | yes | PASS |
| L24 | plain | 27 | 1.4e-13 | 2.7e-12 | 6.4e-15 / 5.5e-12 | 4.31e+05 | yes | PASS |

The largest gradient disagreement, 3.5e-09 (L16 solved), is below the
1e-8 gradient rtol. The gradient norms agree across hosts to the printed
digits, so every model is registered and no gradient is silently zero.

### Ratio tables (20 rounds × 20 calls, interleaved; median ms per call; ratio fwd / rev with bootstrap 90 % CI)

"B=8 ms/call" is the time of one vmapped call over 8 vectors (divide by 8 for
per-vector cost). "compile s" is lower + compile.

RAL gpu-partition host CPUs, AMD EPYC 7702, quiet (`gradient_mode_crossover_hpc_ral_gpunode_cpu_fp64.json`) — **reference**:

| Rung | Lane | n | rev single ms | fwd single ms | fwd/rev single [90% CI] | rev B=8 ms/call | fwd B=8 ms/call | fwd/rev batched [90% CI] | compile s rev / fwd (single) | compile s rev / fwd (batched) |
|---|---|---|---|---|---|---|---|---|---|---|
| L5 | solved | 5 | 0.5973 | 0.2449 | 0.410 [0.404, 0.418] | 0.6228 | 0.2790 | 0.448 [0.442, 0.453] | 3.4 / 2.4 | 3.6 / 2.7 |
| L5 | plain | 8 | 0.4775 | 0.2331 | 0.488 [0.481, 0.497] | 0.5083 | 0.2760 | 0.543 [0.534, 0.548] | 2.1 / 1.7 | 2.4 / 1.9 |
| L7 | solved | 7 | 0.7441 | 0.2816 | 0.379 [0.375, 0.384] | 0.7669 | 0.3247 | 0.423 [0.418, 0.429] | 4.8 / 3.0 | 5.5 / 3.5 |
| L7 | plain | 10 | 0.5551 | 0.2589 | 0.466 [0.461, 0.473] | 0.5963 | 0.3318 | 0.556 [0.549, 0.563] | 3.0 / 2.1 | 3.2 / 2.5 |
| L9 | solved | 9 | 0.9810 | 0.2847 | 0.290 [0.287, 0.293] | 1.0776 | 0.3741 | 0.347 [0.343, 0.353] | 8.1 / 4.0 | 9.1 / 4.7 |
| L9 | plain | 12 | 0.7227 | 0.2862 | 0.396 [0.391, 0.400] | 0.8049 | 0.3935 | 0.489 [0.484, 0.497] | 4.4 / 2.8 | 5.1 / 3.4 |
| L11 | solved | 11 | 1.1358 | 0.2987 | 0.263 [0.259, 0.266] | 1.3160 | 0.4987 | 0.379 [0.374, 0.383] | 11.2 / 5.0 | 12.8 / 5.5 |
| L11 | plain | 14 | 0.8794 | 0.3183 | 0.362 [0.358, 0.366] | 0.9896 | 0.5294 | 0.535 [0.530, 0.540] | 5.8 / 3.5 | 6.5 / 3.9 |
| L16 | solved | 16 | 1.3418 | 0.3370 | 0.251 [0.247, 0.255] | 1.5373 | 0.6077 | 0.395 [0.388, 0.399] | 21.4 / 6.1 | 23.7 / 6.9 |
| L16 | plain | 19 | 1.0032 | 0.3289 | 0.328 [0.323, 0.334] | 1.1338 | 0.6276 | 0.554 [0.547, 0.560] | 9.0 / 4.2 | 10.1 / 4.9 |
| L19 | solved | 19 | 2.0954 | 0.4979 | 0.238 [0.234, 0.240] | 2.4509 | 0.9016 | 0.368 [0.363, 0.373] | 47.2 / 7.9 | 53.9 / 9.1 |
| L19 | plain | 22 | 1.4797 | 0.4809 | 0.325 [0.322, 0.328] | 1.6950 | 0.8520 | 0.503 [0.498, 0.508] | 16.3 / 5.5 | 19.1 / 6.5 |
| L24 | solved | 24 | 2.2938 | 0.5078 | 0.221 [0.218, 0.225] | 2.5860 | 1.0510 | 0.406 [0.401, 0.412] | 79.6 / 9.1 | 91.7 / 10.5 |
| L24 | plain | 27 | 1.6339 | 0.5339 | 0.327 [0.323, 0.331] | 1.9518 | 1.0918 | 0.559 [0.553, 0.565] | 26.1 / 6.4 | 29.7 / 7.5 |

A100 (`gradient_mode_crossover_hpc_a100_fp64.json`):

| Rung | Lane | n | rev single ms | fwd single ms | fwd/rev single [90% CI] | rev B=8 ms/call | fwd B=8 ms/call | fwd/rev batched [90% CI] | compile s rev / fwd (single) | compile s rev / fwd (batched) |
|---|---|---|---|---|---|---|---|---|---|---|
| L5 | solved | 5 | 0.4469 | 0.2857 | 0.639 [0.635, 0.643] | 0.4291 | 0.2896 | 0.675 [0.669, 0.681] | 6.9 / 3.1 | 6.5 / 3.6 |
| L5 | plain | 8 | 0.3368 | 0.2431 | 0.722 [0.719, 0.725] | 0.3421 | 0.2481 | 0.725 [0.721, 0.733] | 4.9 / 2.5 | 5.1 / 2.7 |
| L7 | solved | 7 | 0.5320 | 0.2565 | 0.482 [0.477, 0.486] | 0.4510 | 0.2648 | 0.587 [0.583, 0.593] | 7.6 / 4.0 | 8.4 / 4.8 |
| L7 | plain | 10 | 0.4039 | 0.2808 | 0.695 [0.693, 0.699] | 0.4052 | 0.2799 | 0.691 [0.687, 0.693] | 5.7 / 3.3 | 5.7 / 3.5 |
| L9 | solved | 9 | 0.7699 | 0.3407 | 0.443 [0.440, 0.444] | 0.7481 | 0.3471 | 0.464 [0.460, 0.467] | 10.5 / 6.1 | 10.9 / 7.2 |
| L9 | plain | 12 | 0.5871 | 0.3653 | 0.622 [0.618, 0.625] | 0.6176 | 0.3549 | 0.575 [0.572, 0.577] | 7.7 / 4.9 | 9.3 / 5.5 |
| L11 | solved | 11 | 0.9459 | 0.3591 | 0.380 [0.377, 0.382] | 0.9687 | 0.3857 | 0.398 [0.396, 0.400] | 12.6 / 7.2 | 14.2 / 8.6 |
| L11 | plain | 14 | 0.7031 | 0.3245 | 0.462 [0.457, 0.464] | 0.7303 | 0.3486 | 0.477 [0.474, 0.480] | 9.2 / 5.8 | 10.9 / 7.0 |
| L16 | solved | 16 | 1.1792 | 0.4357 | 0.369 [0.367, 0.372] | 1.1484 | 0.4532 | 0.395 [0.390, 0.396] | 16.7 / 9.9 | 17.8 / 11.3 |
| L16 | plain | 19 | 0.8413 | 0.4443 | 0.528 [0.524, 0.532] | 0.8671 | 0.4467 | 0.515 [0.512, 0.519] | 11.8 / 8.6 | 13.4 / 9.5 |
| L19 | solved | 19 | 2.7198 | 0.9474 | 0.348 [0.347, 0.350] | 2.6776 | 1.0588 | 0.395 [0.395, 0.396] | 21.7 / 12.0 | 27.4 / 13.6 |
| L19 | plain | 22 | 2.0495 | 0.9194 | 0.449 [0.447, 0.451] | 1.9939 | 0.9929 | 0.498 [0.497, 0.499] | 15.2 / 10.3 | 16.8 / 10.7 |
| L24 | solved | 24 | 2.8550 | 1.0185 | 0.357 [0.356, 0.358] | 2.7137 | 1.0613 | 0.391 [0.389, 0.393] | 26.3 / 14.7 | 31.3 / 15.7 |
| L24 | plain | 27 | 2.1697 | 0.9936 | 0.458 [0.457, 0.458] | 2.0454 | 1.0211 | 0.499 [0.498, 0.501] | 18.4 / 11.6 | 20.4 / 13.4 |

Laptop lead, i9-10885H, 8 threads, load 8.2 → 4.2 (`gradient_mode_crossover_local_cpu_fp64.json`):

| Rung | Lane | n | rev single ms | fwd single ms | fwd/rev single [90% CI] | rev B=8 ms/call | fwd B=8 ms/call | fwd/rev batched [90% CI] | compile s rev / fwd (single) | compile s rev / fwd (batched) |
|---|---|---|---|---|---|---|---|---|---|---|
| L5 | solved | 5 | 0.6948 | 0.4086 | 0.588 [0.577, 0.597] | 0.7501 | 0.4657 | 0.621 [0.610, 0.635] | 7.5 / 6.3 | 7.6 / 7.5 |
| L5 | plain | 8 | 0.4788 | 0.2699 | 0.564 [0.543, 0.624] | 0.5046 | 0.3430 | 0.680 [0.653, 0.706] | 3.4 / 4.7 | 4.4 / 3.6 |
| L7 | solved | 7 | 1.0159 | 0.5271 | 0.519 [0.505, 0.528] | 1.0490 | 0.5543 | 0.528 [0.517, 0.539] | 12.6 / 5.7 | 7.8 / 4.9 |
| L7 | plain | 10 | 0.6184 | 0.3909 | 0.632 [0.619, 0.640] | 0.6534 | 0.4347 | 0.665 [0.651, 0.673] | 5.0 / 3.7 | 5.5 / 4.7 |
| L9 | solved | 9 | 0.8826 | 0.3714 | 0.421 [0.413, 0.427] | 0.9410 | 0.4791 | 0.509 [0.498, 0.519] | 12.0 / 5.8 | 13.7 / 8.1 |
| L9 | plain | 12 | 0.7043 | 0.3949 | 0.561 [0.551, 0.571] | 0.7817 | 0.4778 | 0.611 [0.603, 0.625] | 5.4 / 3.4 | 6.0 / 3.9 |
| L11 | solved | 11 | 1.2243 | 0.4736 | 0.387 [0.377, 0.395] | 1.4194 | 0.7015 | 0.494 [0.482, 0.500] | 13.0 / 5.7 | 14.9 / 6.5 |
| L11 | plain | 14 | 0.7915 | 0.4184 | 0.529 [0.521, 0.538] | 0.8909 | 0.5888 | 0.661 [0.652, 0.670] | 6.9 / 4.4 | 7.7 / 4.3 |
| L16 | solved | 16 | 1.3306 | 0.4621 | 0.347 [0.343, 0.352] | 1.5145 | 0.8003 | 0.528 [0.520, 0.537] | 26.3 / 6.8 | 31.1 / 8.8 |
| L16 | plain | 19 | 0.9937 | 0.4540 | 0.457 [0.443, 0.472] | 1.1380 | 0.8598 | 0.756 [0.722, 0.783] | 12.7 / 6.1 | 17.1 / 9.7 |
| L19 | solved | 19 | 2.5893 | 0.7846 | 0.303 [0.297, 0.308] | 3.0511 | 1.5530 | 0.509 [0.500, 0.520] | 64.2 / 9.5 | 69.3 / 10.4 |
| L19 | plain | 22 | 1.3168 | 0.5944 | 0.451 [0.446, 0.457] | 1.5879 | 1.2689 | 0.799 [0.790, 0.808] | 23.5 / 7.7 | 27.9 / 9.0 |
| L24 | solved | 24 | 2.2066 | 0.6817 | 0.309 [0.304, 0.314] | 2.5605 | 1.5810 | 0.617 [0.609, 0.626] | 117.2 / 11.0 | 134.1 / 17.2 |
| L24 | plain | 27 | 1.5956 | 0.6817 | 0.427 [0.417, 0.433] | 2.0760 | 1.7122 | 0.825 [0.810, 0.837] | 38.5 / 8.5 | 51.7 / 10.5 |

### Crossover n*

The ratio `fwd / rev` never reaches 1 on any host, lane or call shape. In all
2000 bootstrap curves of every host × lane × shape, not one crosses, so there is
no interpolated `n*` inside the ladder:

| Host | Lane | single call | batched (B=8) | batched: extrapolated n at ratio 1 |
|---|---|---|---|---|
| RAL gpu-node CPU (EPYC 7702) — reference | solved | fwd wins through n=24 (ratio 0.41 → 0.22; 0/2000 bootstrap curves cross) | fwd wins through n=24 (ratio 0.45 → 0.41; 0/2000 bootstrap curves cross) | ~319 |
| RAL gpu-node CPU (EPYC 7702) — reference | plain | fwd wins through n=27 (ratio 0.49 → 0.33; 0/2000 bootstrap curves cross) | fwd wins through n=27 (ratio 0.54 → 0.56; 0/2000 bootstrap curves cross) | ~278 |
| A100 | solved | fwd wins through n=24 (ratio 0.64 → 0.36; 0/2000 bootstrap curves cross) | fwd wins through n=24 (ratio 0.67 → 0.39; 0/2000 bootstrap curves cross) | none (ratio falling) |
| A100 | plain | fwd wins through n=27 (ratio 0.72 → 0.46; 0/2000 bootstrap curves cross) | fwd wins through n=27 (ratio 0.73 → 0.50; 0/2000 bootstrap curves cross) | none (ratio falling) |
| Laptop (lead) | solved | fwd wins through n=24 (ratio 0.59 → 0.31; 0/2000 bootstrap curves cross) | fwd wins through n=24 (ratio 0.62 → 0.62; 0/2000 bootstrap curves cross) | ~56 |
| Laptop (lead) | plain | fwd wins through n=27 (ratio 0.56 → 0.43; 0/2000 bootstrap curves cross) | fwd wins through n=27 (ratio 0.68 → 0.82; 0/2000 bootstrap curves cross) | ~48 |

The extrapolation column is a least-squares line through the last three rungs'
point ratios. It is reported only for completeness and is not a crossover
estimate: where the ratio is still falling, there is no crossing to extrapolate
to. The only rising trends are in the batched shape on CPU, and they project
past n ≈ 280 on the reference host (≈ 50 on the loaded laptop).

### Compile times (lower + compile, single call)

| Host | Rung (n solved) | `rev` s | `fwd` s |
|---|---|---|---|
| RAL gpu-node CPU (EPYC 7702) | L5 (5) | 3.4 | 2.4 |
| RAL gpu-node CPU (EPYC 7702) | L7 (7) | 4.8 | 3.0 |
| RAL gpu-node CPU (EPYC 7702) | L9 (9) | 8.1 | 4.0 |
| RAL gpu-node CPU (EPYC 7702) | L11 (11) | 11.2 | 5.0 |
| RAL gpu-node CPU (EPYC 7702) | L16 (16) | 21.4 | 6.1 |
| RAL gpu-node CPU (EPYC 7702) | L19 (19) | 47.2 | 7.9 |
| RAL gpu-node CPU (EPYC 7702) | L24 (24) | 79.6 | 9.1 |
| A100 | L5 (5) | 6.9 | 3.1 |
| A100 | L7 (7) | 7.6 | 4.0 |
| A100 | L9 (9) | 10.5 | 6.1 |
| A100 | L11 (11) | 12.6 | 7.2 |
| A100 | L16 (16) | 16.7 | 9.9 |
| A100 | L19 (19) | 21.7 | 12.0 |
| A100 | L24 (24) | 26.3 | 14.7 |
| Laptop (lead) | L5 (5) | 7.5 | 6.3 |
| Laptop (lead) | L7 (7) | 12.6 | 5.7 |
| Laptop (lead) | L9 (9) | 12.0 | 5.8 |
| Laptop (lead) | L11 (11) | 13.0 | 5.7 |
| Laptop (lead) | L16 (16) | 26.3 | 6.8 |
| Laptop (lead) | L19 (19) | 64.2 | 9.5 |
| Laptop (lead) | L24 (24) | 117.2 | 11.0 |

### What the curve shows

- **`fwd` wins at every rung, on every host, in both lanes and both call shapes.**
  On the reference EPYC host, the single-call ratio *falls* from 0.41 (L5) to
  0.22 (L24) in the solved lane and from 0.49 to 0.33 in the plain lane. On the
  A100 it falls from 0.64 to 0.36 (solved) and from 0.72 to 0.46 (plain). The
  reverse-mode call grows about 3.8× from L5 to L24 on the EPYC
  (0.60 → 2.29 ms), while the forward-mode call grows about 2.1×
  (0.24 → 0.51 ms).
- **Why the naive crossover does not appear:** reverse mode here is reverse over
  the forward-mode lensing Hessian (`LensCalc._hessian_via_jax` is a per-position
  `jax.jacfwd`), and its cost grows with the number of mass-profile terms, not
  only with `n_params`. XLA flops tell the same story from the other side:
  `fwd` flops overtake `rev` from L9 on (EPYC solved L24: 528 k vs 340 k), yet
  `fwd` stays 4.5× faster. The cost is tape and dispatch structure, not
  arithmetic, as phase 2b found at L5.
- **The batched shape narrows the gap on CPU.** Under `vmap` over 8 vectors,
  `fwd` pushes 8 × n tangents. On the EPYC the solved-lane batched ratio is flat
  at 0.35–0.45 across the ladder, and the plain lane stays at 0.49–0.56. On the
  loaded laptop the plain batched ratio rises to 0.83 at n = 27. The A100
  batched ratios track its single-call ones (0.39–0.73).
- **Compile time is the other half of the result.** The `rev` compile grows
  steeply with the model: on the EPYC it goes from 3.4 s (L5) to 47 s (L19) and
  80 s (L24), and on the laptop to 117 s. `fwd` stays at 2.4–9 s. On the A100
  it is 6.9 → 26 s for `rev` and 3.1 → 15 s for `fwd`. For a real model
  (L16–L24 is a realistic galaxy-scale point-source model), the `rev` compile
  alone is tens of seconds to minutes on CPU.

### Design memo

Written by the main session, 2026-09-27.

**What the curve says.** There is no crossover to design a threshold around. Forward mode wins
through n = 24 (solved) / 27 (plain) on every host and call shape, and on the single call its lead
*grows* with model size: 0.41 → 0.22 on the reference EPYC, 0.64 → 0.36 on the A100. This is the
opposite of the textbook "forward mode loses as n grows". The reason is structural: this likelihood
contains an inner forward-mode derivative (the lensing Hessian, via `jax.jacfwd` in
`LensCalc._hessian_via_jax`). Reverse mode therefore runs reverse-over-forward through every mass
profile, and its cost grows with the number of profiles (rev ×3.8 across the ladder vs fwd ×2.1).
Forward-over-forward stays cheap, even with more flops than rev from L9 on. The only projected
crossing is for vmapped batches on CPU, at n ≈ 280–320, far beyond any realistic
single-source point-source model. The compile-time result is just as large: rev lower + compile
reaches 80 s at L24 on the EPYC (117 s on the laptop), while fwd stays at 2.4–11 s.

**Consequence for the design.** An automatic switch on `n_params < threshold` is the wrong shape.
The win comes from the *likelihood's structure* (an inner jacfwd), not from the parameter count.
The same switch would lose on a pixelized-source imaging likelihood with many parameters and no
inner Hessian. The choice belongs to the analysis, which knows its structure, and the user should
be able to override it.

**Where a switch would live** (PyAutoFit gradient call sites):
- `Fitness.grad`: `autofit/non_linear/fitness.py:933` (`jax.grad(self.call)`).
- multi-start gradient: `autofit/non_linear/search/mle/multi_start_gradient/search.py:974` and
  `:1072` (`jax.value_and_grad`), vmapped at `:1089`. This is the batched shape measured here.
- blackjax NUTS / SMC: `autofit/non_linear/search/mcmc/blackjax/{nuts,smc}/search.py`. blackjax
  differentiates the log-density itself, so forward mode there means supplying a custom
  `value_and_grad`. That is a later step.

**Options, for the human to choose:**
1. **Analysis-declared default plus a search override (recommended).** Add a
   `gradient_mode: "reverse" | "forward"` attribute. `af.Analysis` defaults to `"reverse"`, so there is
   no behaviour change anywhere else. `AnalysisPoint` declares `"forward"`. A search keyword
   overrides it. PyAutoFit builds the gradient through one helper that returns
   `value_and_grad` for reverse, or `jacfwd(has_aux=True)` over the flat vector for forward, and the
   three call sites above use that helper.
2. **Opt-in only.** The same helper and keyword, default `"reverse"` everywhere, with point-source
   users opting in. This is safest, but the 2–4.5× speed-up and the 8× compile saving stay hidden
   unless people read the docs.
3. **Self-calibrating `"auto"`.** Compile both modes once at search start and keep the faster one.
   This is robust across likelihoods, but it pays the reverse compile (up to 80 s here) that forward
   mode exists to avoid. It is only worth it as an explicit `"auto"` value on top of option 1 or 2.

Either library phase should take the flat parameter vector, not the `ModelInstance` pytree. Linked
priors give the pytree more leaves than free parameters (12 for 9 at L9), which would make jacfwd
push one tangent per leaf. `Fitness.call` already works on the vector. The library phase also needs
a GPU regression check and gradient parity tests (fwd ≡ rev) on a non-point-source analysis.

**Carried findings (separate from the switch, for intake):**
- `jax.grad` is NaN at exactly zero for `ExternalShear`, multipole `multipole_comps` and
  `ell_comps` (the magnitude/angle parameterisation). A gradient search that starts at prior
  medians of 0 would see NaN gradients. This is a PyAutoGalaxy robustness bug, independent of AD
  mode.
- `PowerLawMultipole` m=1 is singular at slope 2 (`-inf`/NaN deflections).
- `Isothermal.convergence_2d_from` / `shear_yx_2d_from` cannot be jit-traced with traced
  `ell_comps` (carried from phase 2b).

## Phase 2e — gradient_mode through the library (2026-09-27)

Issue: [autolens_profiling #334](https://github.com/PyAutoLabs/autolens_profiling/issues/334).
Branch `feature/point-source-source-plane-p2e`; the RAL jobs ran commit `2bf293a`.
Single-source only. Workspace-only: no library was edited. The library side is phase 2d
(PyAutoFit#1649, merge `867af1c`; PyAutoLens#752, merge `b3c9b68`): an analysis-declared
`gradient_mode`, `al.AnalysisPoint.gradient_mode = "forward"`, and a
`MultiStartGradient(gradient_mode=...)` override.

### What landed

- `scripts/point_source_source/likelihood_breakdown/gradient_mode_library_ab.py` asks whether the
  phase-2b/2c forward-mode speed-up survives the real `af.MultiStartAdam` path. It uses phase 2c's
  `L5` and `L24` rungs, both lanes, the same `simple` dataset, the same ladder builders and the same
  per-rung seeds, so the timed start batches are the ones phase 2c timed.
- **The real path, not a re-implementation.** `MultiStartGradient._fit` builds its objective
  inline. The chain is `Fitness(fom_is_log_likelihood=False, convert_to_chi_squared=True)`, then
  `value_and_grad_from(fitness.call, gradient_mode)`, then the local `_value_and_grad_finite`
  wrapper (which adds `all(isfinite(grad))` and the constraint violation as outputs), then
  `_vmapped = jax.jit(jax.vmap(_value_and_grad_finite))`. None of this is exposed, so the cell runs
  `search.fit(model, analysis)` with `jax.jit` temporarily wrapped by a recorder. When the search
  jits a function named `_value_and_grad_finite`, the recorder keeps the jitted object and aborts
  the fit, before any compile or likelihood evaluation. The kept object *is* the search's
  `_vmapped`, closing over the search's own `Fitness`. The timed objective is therefore
  `-2 log posterior` (`Fitness.call`, with its value guards), not phase 2c's bare `log L`.
- **Modes.** `forward` = `af.MultiStartAdam()` with no override, so the mode comes from
  `AnalysisPoint`'s declaration. `reverse` = `af.MultiStartAdam(gradient_mode="reverse")`. Both use
  B = `n_starts` = 8 with `batch_size=None`, one vmap over all starts, as in phase 2c's batched row.
  The defaults are `ScalerNone`, `BijectorNone` and `ClipperNone`, so the stepped vector is the
  physical one.
- **Routes timed.** `<mode>_exe` is `_vmapped.lower(example).compile()`, the AOT executable,
  comparable to phase 2c's `*_batched`. `<mode>_jit` is the captured jitted callable itself, called
  the way the search's step loop calls it. Both modes are interleaved, 20 rounds × 20 calls with
  the order rotated each round, and every call is `block_until_ready`'d. The ratio forward/reverse
  comes with a bootstrap 90 % CI.
- **End-to-end fits.** At `L5`, per lane, one complete `search.fit` per mode (`n_starts=8`,
  `n_steps=20`, `seed=334`, no abort). A throwaway warm-up fit runs first. Two walls are recorded:
  the whole `fit()`, and the `_fit` call alone (compiles + steps, no output I/O).
- Submits `hpc/batch_cpu/submit_gradient_mode_library_ab_point_source_source_ral_cpu_fp64`
  (reference CPU: `--partition=gpu`, no `--gres`, pinned to `euclid-ral-gpu-2`) and
  `hpc/batch_gpu/submit_gradient_mode_library_ab_point_source_source_a100_fp64` (pinned to
  `euclid-ral-gpu-1`). Both prepend the scratch library clones and refuse to run unless `autofit`
  and `autolens` import from them at the merge commits. WALL-BASIS = the laptop's measured 315 s.
- Rows `gradient_mode_library_ab_{local_cpu_fp64,hpc_ral_gpunode_cpu_fp64,hpc_a100_fp64}` (JSON +
  PNG) under `results/breakdown/point_source_source/`. The CPU job log is
  `results/notes/point_source_source_plane_2026_09_27_ral_job_359192.out`.

### Hosts and revisions

- **RAL gpu-partition host CPUs (reference)**: job 359192 on `euclid-ral-gpu-2` (AMD EPYC 7702),
  `--partition=gpu`, no `--gres`, 8 CPUs, `sched_affinity` 8, BLAS threads 1. The node was quiet:
  load average 0.22 at start and 1.37 at end. Process wall 266 s; `COMPLETED` in 4:45.
- **A100**: job 359193 on `euclid-ral-gpu-1` (host EPYC 7702), `JAX_PLATFORMS=cuda`, backend
  asserted `gpu`, JSON device `cuda:0`. Process wall 307 s; `COMPLETED` in 5:18. The host was
  *not* quiet: another 8-CPU job (`vispix_cores_diag`) was running on the node, with load average
  7.1 at start and 3.4 at end. It used no GPU, but host-side dispatch was shared (see the A100
  L24 note below).
- **Laptop (lead)**: i9-10885H, WSL2, 8 threads (`OMP/OPENBLAS/MKL/NPROC=8`). Load average 1.8 at
  start (other sessions) and 2.4 at end; process wall 315 s.
- **Libraries, laptop**: the canonical mains, imported through the task worktree's symlinks.
  PyAutoNerves `bf10410`, PyAutoFit `867af1c`, PyAutoArray `e281abf3`, PyAutoGalaxy `152695e0`,
  PyAutoLens `b3c9b68`; JAX 0.10.2. The worktree `activate.sh` put the task symlinks first on
  PYTHONPATH, and they resolve to the canonical mains, so no stale library worktree was involved.
- **Libraries, RAL (both jobs)**: the shared `/mnt/ral/jnightin/PyAuto` stack did not carry the
  merged `gradient_mode` commits (PyAutoFit `c156a9d8`, PyAutoLens `dcbd4b71`). It was **not**
  touched, because live DR1 array jobs use it. Instead the scratch clones
  `/mnt/ral/jnightin/p2d_check/PyAutoFit` and `/mnt/ral/jnightin/p2d_check/PyAutoLens` were
  fast-forwarded by git bundle from the phase-2d feature commits to the merge commits `867af1c6`
  and `b3c9b68e` (the merge trees are identical to the feature trees), then prepended to
  PYTHONPATH. PyAutoNerves `bf10410` and PyAutoArray `e281abf3` came from the shared stack, both
  equal to the local mains. PyAutoGalaxy `ba8a08fa` also came from the shared stack. It is one PR
  behind main `152695e0`, and that PR touches only `PowerLawCore`, which these models do not use.
  JAX 0.10.2. The JSON `source_revisions` and `library_files` record exactly this.

### Correctness gate (all three hosts)

- **Declared default.** On every host, `AnalysisPoint.gradient_mode`,
  `resolve_gradient_mode(AnalysisPoint)` and `af.MultiStartAdam()._resolved_gradient_mode(...)` all
  resolve to `forward`, and the `gradient_mode="reverse"` override resolves to `reverse`. Every
  captured and end-to-end search logged `MultiStartGradient gradient mode: forward (declared by
  AnalysisPoint).` or `... reverse (search override).` as appropriate. PASS.
- **Batched step, forward vs reverse.** One B = 8 start batch per `PRNGKey` 0..15
  (`U(0.25, 0.75)` unit cube, 128 starts per rung × lane). Checks: objective rtol 1e-10,
  gradient rtol 1e-8 (atol 1e-12 × max|∇|), all finite, the search's own `grad_finite` flag True,
  no zero gradient row, and distinct StableHLO hashes:

| Rung | Lane | n | objective fwd vs rev (worst host) | gradient fwd vs rev (worst host) | min ‖∇‖ | Result (3 hosts) |
|---|---|---|---|---|---|---|
| L5 | solved | 5 | 6.2e-12 | 1.0e-10 | 219 | PASS |
| L5 | plain | 8 | 1.6e-13 | 2.0e-12 | 7.49e+04 | PASS |
| L24 | solved | 24 | 4.8e-13 | 2.1e-09 (A100) | 786 | PASS |
| L24 | plain | 27 | 1.1e-12 | 7.4e-12 | 6.89e+05 | PASS |

  On the A100 the forward and reverse objectives were bit-identical (rel err 0).
- **End-to-end `MultiStartAdam` fits (L5).** The forward and reverse fits reach the same best vector
  (worst rel err 1.4e-12, tolerance 1e-6) and the same max log likelihood (worst 2.7e-14,
  tolerance 1e-8) in both lanes. The max log likelihood is identical across all three hosts
  (solved 2.107660, plain −485.923685). PASS.

### Timing (20 rounds × 20 calls, interleaved; median ms per batched step, B = 8; ratio forward / reverse with bootstrap 90 % CI)

"2c batched" is phase 2c's `fwd_batched / rev_batched` for the same host and rung × lane
(`gradient_mode_crossover_<host>.json`), timed on bare `log L` with no library wrapper. The `jit`
ratio (the search's own dispatch) matches the `exe` ratio everywhere within ~0.02 except A100
L24 solved, so the table shows `exe`; both are in the JSONs.

| Host | Rung | Lane | n | fwd ms (2c) | rev ms (2c) | **fwd/rev library** [90 % CI] | 2c batched [90 % CI] | jit fwd/rev | lower+compile s fwd / rev (2c batched) |
|---|---|---|---|---|---|---|---|---|---|
| RAL EPYC (reference) | L5 | solved | 5 | 0.312 (0.279) | 0.686 (0.623) | **0.455** [0.450, 0.465] | 0.448 [0.442, 0.453] | 0.469 | 2.7 / 3.8 (2.7 / 3.6) |
| RAL EPYC (reference) | L5 | plain | 8 | 0.309 (0.276) | 0.532 (0.508) | **0.580** [0.575, 0.588] | 0.543 [0.534, 0.548] | 0.573 | 2.0 / 2.5 (1.9 / 2.4) |
| RAL EPYC (reference) | L24 | solved | 24 | 1.083 (1.051) | 2.976 (2.586) | **0.364** [0.361, 0.368] | 0.406 [0.401, 0.412] | 0.359 | 11.2 / 95.9 (10.5 / 91.7) |
| RAL EPYC (reference) | L24 | plain | 27 | 1.110 (1.092) | 1.975 (1.952) | **0.562** [0.552, 0.569] | 0.559 [0.553, 0.565] | 0.550 | 7.7 / 34.8 (7.5 / 29.7) |
| A100 | L5 | solved | 5 | 0.287 (0.290) | 0.446 (0.429) | **0.643** [0.638, 0.647] | 0.675 [0.669, 0.681] | 0.642 | 3.8 / 6.5 (3.6 / 6.5) |
| A100 | L5 | plain | 8 | 0.267 (0.248) | 0.379 (0.342) | **0.704** [0.699, 0.708] | 0.725 [0.721, 0.733] | 0.695 | 3.2 / 5.4 (2.7 / 5.1) |
| A100 | L24 | solved | 24 | 1.264 (1.061) | 4.837 (2.714) | **0.261** [0.254, 0.267] | 0.391 [0.389, 0.393] | 0.286 | 15.3 / 33.5 (15.7 / 31.3) |
| A100 | L24 | plain | 27 | 1.116 (1.021) | 2.264 (2.045) | **0.493** [0.490, 0.496] | 0.499 [0.498, 0.501] | 0.495 | 15.8 / 20.8 (13.4 / 20.4) |
| Laptop (lead, loaded) | L5 | solved | 5 | 0.471 (0.466) | 0.726 (0.750) | **0.649** [0.640, 0.661] | 0.621 [0.610, 0.635] | 0.648 | 4.1 / 4.6 (7.5 / 7.6) |
| Laptop (lead, loaded) | L5 | plain | 8 | 0.415 (0.343) | 0.526 (0.505) | **0.789** [0.780, 0.798] | 0.680 [0.653, 0.706] | 0.801 | 2.4 / 3.0 (3.6 / 4.4) |
| Laptop (lead, loaded) | L24 | solved | 24 | 1.556 (1.581) | 3.063 (2.561) | **0.508** [0.502, 0.517] | 0.617 [0.609, 0.626] | 0.512 | 14.2 / 121.7 (17.2 / 134.1) |
| Laptop (lead, loaded) | L24 | plain | 27 | 1.608 (1.712) | 2.058 (2.076) | **0.781** [0.776, 0.789] | 0.825 [0.810, 0.837] | 0.770 | 8.8 / 40.5 (10.5 / 51.7) |

Reading it (reference host first):

- **Forward mode wins through the library at every host, rung and lane.** The step is 1.7–2.7×
  faster on the reference EPYC and 1.4–3.8× faster on the A100. No ratio's CI comes near 1.
- **Reference EPYC vs phase 2c.** L5 solved (0.455 vs 0.448) and L24 plain (0.562 vs 0.559) match
  within noise. L5 plain is slightly less favourable (0.580 vs 0.543), and L24 solved is *more*
  favourable (0.364 vs 0.406). The library wrapper costs ~+0.02–0.06 ms per batched step at L5 in
  both modes (fwd 0.312 vs 0.279 ms; rev 0.686 vs 0.623 ms). At L24 the forward step is within 3 %
  of phase 2c, while the reverse step is 15 % slower in the solved lane (2.976 vs 2.586 ms). That
  extra reverse cost, not a slower forward step, is why the library ratio beats phase 2c's there.
  In short, the library path adds no overhead to forward mode beyond a fixed ~0.03 ms at L5, and
  the phase-2c ratios carry over.
- **A100 L24 solved is the outlier**: reverse is 4.84 ms through the library vs 2.71 ms in phase
  2c (+78 %), and forward is 1.26 vs 1.06 ms (+19 %), giving a ratio of 0.26 vs 0.39. The other
  three A100 cells match phase 2c to within 0.03. The A100 host was shared with an 8-CPU job
  (load 7.1), and this is also the only cell where `jit` and `exe` differ (0.286 vs 0.261). So
  part of this may be host-side contention rather than the wrapper, and a re-run on a quiet
  node would separate the two. Either way, it moves the ratio in forward mode's favour.
- **The laptop is load-exposed.** Its L5 plain (0.79 vs 0.68) and L24 solved (0.51 vs 0.62) differ
  from phase 2c in opposite directions. The reference row decides.
- **Compile.** Through the library, the reverse lower + compile at L24 solved is 95.9 s on the
  reference EPYC (121.7 s on the laptop) against 11.2 s forward, 8.6×. On the A100 it is 33.5 vs
  15.3 s. This matches phase 2c's batched compiles (91.7 / 10.5 s EPYC). Forward mode's compile
  saving arrives through the library intact.

### End-to-end `MultiStartAdam` fits at L5 (n_starts 8, n_steps 20, seed 334)

`_fit` = the search's `_fit` call alone: the single-point compile for the broad starts, the batched
compile and 20 steps, with no output I/O. `fit()` = the whole `search.fit`, including pre-fit
output, samples, latent draw and zip. On the A100, `_fit` also includes the search's GPU
batched-memory probe (two throwaway compiles), in both modes. Warm-up fit (discarded): 19.9 s
EPYC, 41.9 s A100, 19.5 s laptop.

| Host | Lane | `_fit` fwd s | `_fit` rev s | fwd/rev | `fit()` fwd s | `fit()` rev s | same best vector / max log L |
|---|---|---|---|---|---|---|---|
| RAL EPYC (reference) | solved | 4.98 | 6.54 | 0.76 | 10.54 | 12.04 | yes (6.7e-13 / 1.6e-14) |
| RAL EPYC (reference) | plain | 4.32 | 4.55 | 0.95 | 11.04 | 10.86 | yes (4.9e-14 / 8.2e-16) |
| A100 | solved | 19.43 | 25.05 | 0.78 | 25.57 | 31.19 | yes (1.4e-12 / 2.7e-14) |
| A100 | plain | 15.53 | 18.93 | 0.82 | 22.69 | 25.81 | yes (3.3e-14 / 1.8e-15) |
| Laptop (lead) | solved | 5.60 | 8.09 | 0.69 | 10.62 | 13.26 | yes (6.7e-13 / 1.6e-14) |
| Laptop (lead) | plain | 5.07 | 5.02 | 1.01 | 12.00 | 10.95 | yes (4.9e-14 / 8.2e-16) |

A 20-step L5 fit is compile-dominated: 20 × ~0.3–0.7 ms of stepping is ≤ 15 ms against several
seconds of compile. These walls therefore mostly show the compile saving (solved lane −24 % on the
EPYC). The plain lane's reverse compile is already small at L5, so there the saving is within
run-to-run noise. The per-step saving is in the timing table. For a production-length fit
(hundreds of steps at L16–L24), both savings add up: roughly 85 s less compile at L24 solved on
the EPYC, plus ~1.9 ms per step.

### Verdict

<!-- PLACEHOLDER: written by the main session. -->
