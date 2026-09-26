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

The RAL rows were not run in this session (they were not attempted). Run the
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
