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
