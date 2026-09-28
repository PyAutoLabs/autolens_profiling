# Point-source A100 breakdown — phase 0+1 combined (lean), 2026-09-28

Campaign page: [wiki/campaigns/point_source_gpu_breakdown.md](../../wiki/campaigns/point_source_gpu_breakdown.md).
Issue: autolens_profiling#350. Cell:
`scripts/point_source_image/likelihood_breakdown/gpu_bottleneck_map.py` with the stage map
`_point_solver_stage_map.py` (unit tests `scripts/misc/test/test_point_solver_stage_map.py`).

**Quoted run: RAL job 366916**, `euclid-ral-gpu-1`, A100 80 GB PCIe, submitted with
`--nodelist=euclid-ral-gpu-1 --exclusive` (the node had no other job 21:12:38–21:19:50),
job wall 7:12 (fp64 process 344 s, fp32 process 78 s). Results:
`results/breakdown/point_source_image/gpu_bottleneck_map_hpc_ral_a100_fp64.{json,png}`,
`..._hpc_ral_a100_fp32_whatif.{json,png}`, log
`results/logs/point_source_image/point_source_gpu_2026_09_28_ral_job_366916_gpu_bottleneck_map.out`.

**`all_gates_pass: false`** on the fp64 JSON — two real findings, kept failing (see
"Gates"): forward-mode gradients are NaN, and two smooth points fail the strict FD rule. Every
instrument gate (fiducial, command-buffer bit identity, vmap, trace join, reverse finiteness)
passes. The fp32 what-if JSON passes its reported gates.

## Setup and provenance

- Library: the RAL mirror at current `main` — PyAutoArray 9428eca2 (includes #580, #584),
  PyAutoLens 21b520be (includes #753), PyAutoGalaxy c9609825, PyAutoFit 404b3e5f, PyAutoNerves
  bf10410; the JSON `source_revisions` equal the mirror HEADs (asserted by the submit). No
  `HPCPullPyAuto` was needed: the mirror already equalled `origin/main` on all five repos.
- Likelihood: `FitPositionsImagePairAllSolved` through `PointSolver.for_grid` (100 x 100 x
  0.2", precision 1e-3, magnification 0.1, `neighbor_degree` 1), MCS 20, `structured` step-0
  route, 8 refinement steps, unpatched. Builders copied from `solver_config_sweep.py`'s control
  (that script runs its sweep at import, so it is not importable; it was not refactored).
- Env: `JAX_PLATFORMS=cuda`, `XLA_FLAGS` byte-identical to jobs 357322 / 359102
  (`constant_folding` disabled, autotune 0, triton gemm off), no compile cache, fp64 then a
  second process with `JAX_ENABLE_X64=0`.
- Timing: interleaved 20 rounds x 20 calls over a 16-vector fixed-seed stream (seed 314),
  `block_until_ready`, per-call medians, bootstrap 90 % CI. **MDI 5.45 %** (split-half
  bootstrap of the scalar command-buffers-ON samples).

### Runs and co-tenancy (why 366916 is the quoted run)

| Job | Node | Co-tenants overlapping in time | Scalar ON | Status |
|---|---|---|---|---|
| 366913 | gpu-2, GPU 0 | 366911 (CPU-only, 20:46:05–:21), 366912 (1 A100, 20:46:22–:46, overlapped the baseline leg), 366914 (1 A100, 20:49:33–:46, the grad leg) | 0.904 ms | caveat only; strict FD gate failed |
| 366915 | gpu-1, `--exclusive` | none | 0.927 ms | ran with a widened FD gate (8c12e57), superseded |
| **366916** | gpu-1, `--exclusive` | none | **0.910 ms** | quoted; final code 75833a9 |

**Scalar vs job 359102 (control 0.827 ms / its MCS-20 row 0.838 ms, vmap-16 0.071 ms/L, same
node class and XLA flags):** the quiet runs reproduce 0.91–0.93 ms, +8.6 % over 359102's MCS-20
row, and vmap-16 0.0766 vs 0.0707 ms/L (+8 %). The contended run was not slower than the quiet
ones, so co-tenancy does not explain it. The code differs: since 359102 the mirror gained
PyAutoGalaxy#634 ("finite, correct jax.grad at zero shear / multipole / ell_comps"), #633,
PyAutoLens#752/#753/#754 and PyAutoFit#1649. The fiducial log L is still bit-exact
(7.743201200876806), so the forward value is unchanged; an extra-op guard in the Isothermal
path (#634 / #754) is the likely candidate. **Not bisected** — a follow-up, not a result.

## Leg 1 — baseline (command buffers ON vs OFF)

| Program | median ms | p10 | p90 | compile s |
|---|---|---|---|---|
| command buffers ON (production) | **0.910** | 0.881 | 0.971 | 6.42 (lower 3.31, first call 0.018) |
| command buffers OFF | 1.097 | 1.061 | 1.156 | 6.67 |

OFF / ON = 1.206 [1.197, 1.211]: CUDA graphs already remove 0.187 ms of launch overhead per
call. FLOPs 3.45e6 per call, XLA temp 389 kB. Gates: fiducial bit-exact; OFF bit-identical to
ON on 64 stream points; all 64 finite (every point 4 images).

## Leg 2 — vmap scaling

| batch | ms / likelihood | likelihoods / s | per-L speed-up vs 1 | batch ms | compile s | XLA temp | device peak |
|---|---|---|---|---|---|---|---|
| 1 | 0.970 | 1 031 | 1.0x | 0.970 | 7.6 | 0.39 MB | 0.42 MB |
| 4 | 0.274 | 3 644 | 3.5x | 1.098 | 7.8 | 1.5 MB | 2.1 MB |
| 16 | 0.0766 | 13 051 | 12.7x | 1.226 | 8.1 | 6.1 MB | 8.4 MB |
| 64 | 0.0253 | 39 476 | 38.3x | 1.621 | 7.9 | 24 MB | 34 MB |
| 256 | **0.0120** | **83 445** | **80.9x** | 3.068 | 7.8 | 97 MB | 134 MB |

No OOM (largest batch tried, 256, fits in 134 MB of 80 GB). Every lane equals the scalar
program exactly (max |Δ| = 0). A 256-wide batch costs 3.2x one call: the call is overhead,
not arithmetic.

## Leg 3 — device trace (command buffers OFF, 10 traced calls)

| Program | kernels / call | device busy | untraced wall OFF | wall ON | busy / wall OFF | ON program busy / wall ON |
|---|---|---|---|---|---|---|
| scalar | **173** | 0.616 ms | 1.153 ms | 0.937 ms | 53 % | **57 %** (0.537 ms busy) |
| vmap-16 | 179 | 0.862 ms | 1.363 ms | 1.515 ms | 63 % | 52 % (0.789 ms busy) |

(Walls here are 50-call steady means; the leg-1/leg-2 medians are the timing numbers.) The ON
program's busy time is measured from the same CUPTI kernels inside the graphs (named by graph
node, so not attributable to stages). Inter-kernel gaps (traced, scalar): 172 per call,
median 5.1 µs, p10 1.2, p90 8.0, max 72 µs; the trace join is complete (unjoined 0 ms).

Stage table (scalar, kernel ms per call; `_active` = MCS-padded active set of every step,
`_lattice` = the 23 283-triangle step-0 lattice):

| stage | ms | kernels | share of kernel ms |
|---|---|---|---|
| mixed_fusion | 0.412 | 110 | 66.9 % |
| neighbourhood_active | 0.115 | 28 | 18.7 % |
| containment_active | 0.061 | 25 | 9.9 % |
| magnification_filter | 0.017 | 6 | 2.8 % |
| step0_containment | 0.004 | 1 | 0.7 % |
| beta_star | 0.004 | 2 | 0.7 % |
| chi_squared | 0.002 | 1 | 0.3 % |

Two-thirds of the device time sits in fusions spanning several stages (largest sets:
containment + neighbourhood 0.077 ms; containment + deflections + magnification + model mapping
+ geometry 0.056 ms). Prorated **estimate** of the mixed fusions (equal split, never summed
into the table): containment 0.091, triangle geometry 0.063, neighbourhood 0.062, magnification
0.036, up-sample 0.028, kept gather 0.028, **deflections 0.026**, model mapping 0.023 ms. The
step-0 lattice is not where the time is: its containment is one 4 µs kernel. The Python step
loop is unrolled and shares source lines, so no per-step split exists in HLO metadata (hence
the size split).

## Leg 4 — fp32 what-if (whole-program `JAX_ENABLE_X64=0`, NOT a supported mode)

| | fp64 | fp32 | fp64 / fp32 |
|---|---|---|---|
| scalar ON | 0.910 ms | 0.719 ms | 1.266 [1.252, 1.276] |
| vmap-16 per L | 0.0766 | 0.0723 | 1.06 |
| vmap-256 per L | 0.0120 | 0.0139 | 0.86 (fp32 slower) |

Over the same 64 stream points: |Δ log L| max 1.6e-5, median 3.3e-6; finite image count agrees
on 64/64; max position Δ 2.2e-7" (median 1.6e-7"). Precision is not the constraint; speed
headroom is ~21 % scalar and none once batched.

## Leg 5 — gradients

| | scalar ms | / primal [90 % CI] | compile s | vmap-16 ms / L | / primal | compile s |
|---|---|---|---|---|---|---|
| primal | 0.937 | 1 | 6.4 | 0.0803 | 1 | 8.0 |
| reverse `value_and_grad` | 1.801 | 1.92 [1.90, 1.94] | 17.2 | 0.141 | 1.76 | 16.7 |
| forward `jacfwd` (**NaN**) | 1.210 | 1.29 [1.28, 1.30] | 16.6 | 0.144 | 1.80 | 20.7 |

**Forward mode is NaN.** `jax.jacfwd` of this likelihood returns NaN in all five components at
all 16 stream points on the A100 (and on the laptop CPU, production and 1e-5 fine solvers
alike); reverse mode is finite everywhere. `AnalysisPoint.gradient_mode = "forward"`
(PyAutoLens#752, unreleased) makes forward mode the default for gradient searches on every
point-source analysis, on the strength of source-plane measurements (#327/#331). The forward
timing row is a cost measurement of a program that returns NaN, not a usable gradient.

FD certification (reverse mode, workspace_test `jax_grad/gradient.py` method: fine 1e-5
solver, relative steps 1e-4 .. 5e-3 x max(|x|, 0.1) in physical parameter space, FD closest to
AD, strict per-component |ad − fd| <= 1e-4 + 2 % max(|ad|, |fd|)): **4 of 6 gated smooth points
pass**; points 3 and 4 fail; the symmetric prior-median point 0 (lens centre (0,0),
ell_comps_1 = 0) is a reported diagnostic and fails too. No point has a topology transition:
the finite image count is 4 at x−h and x+h at every step for every component.

Evidence at the failing components (A100 job 366916; the laptop CPU re-derivation reproduces
the A100 AD to 1e-11 and adds h at relative 1e-6 / 1e-5):

| point / param | AD (fine) | h = 1e-7 | 1e-6 | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | 5e-4 | ‖∇‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 / centre_1 | −0.244 | 0 | 74.9 | −0.78 (0.69) | 2.03 (1.12) | −0.45 (0.46) | 0.96 (1.25) | −0.74 (0.67) | −0.09 (0.62) | 406 |
| 4 / centre_0 | 7.161 | 0 | 63.5 | 4.43 (0.38) | 15.1 (0.53) | 6.68 (0.068) | 7.49 (0.044) | 7.49 (0.044) | 7.00 (0.023) | 1482 |
| 0 / ell_comps_0 (diag.) | 2.876 | 0 | 41.1 | 3.11 (0.075) | 6.94 (0.59) | −1.99 (1.69) | 2.59 (0.10) | 3.11 (0.076) | 2.59 (0.10) | 1728 |

FD value (relative error). The solved-image set moves smoothly with h at every step (max
image shift 1.4e-5" at h = 1e-5 to 9.4e-4" at h = 5e-4 for point 3; 3.7e-5" to 2.0e-3" for
point 4; counts 4/4/4 throughout). Below the stair width FD reads exactly 0 (h = 1e-7) or one
stair jump (h = 1e-6). Forward vs reverse cannot be compared there: forward is NaN.

**These are smooth points that disagree at every step size, so they are recorded as a
finding and kept failing.** What the data says about them: (a) they are the smallest
components of their gradients (6e-4 and 5e-3 of ‖∇‖); (b) the FD values at point 3 change
sign from step to step (−0.78 .. 2.03) and never converge, while point 4 approaches AD from
38 % to 2.3 % as h grows; (c) the whole-vector disagreement is 5.9e-4 (point 3) and 3.2e-4
(point 4) of ‖fd‖, versus 1.4e-4 .. 1.0e-3 at the passing points. The residual stair noise of
the 1e-5 fine solver is larger than these components can resolve; whether reverse mode is
also slightly off there is not decidable with this FD instrument. A finer reference solver
(1e-6) or a Richardson sweep would decide it.

Gate history, recorded so it cannot be misread: job 366913 ran the strict gate and failed
points 3/4; commit 8c12e57 widened the gate to "strict or within the FD sweep's spread, plus a
2 % vector criterion" and job 366915 passed under it; that widening was reverted in 75833a9 (a
gate changed after a failure has to be justified from the data, and a smooth point failing at
every step is a finding). The widened classification is kept per point in the JSON as
`diagnostic_noise_floor_classification`, never as the gate.

## Phase-2 upper bounds (per contract lever)

Every ceiling is "the whole measured cost the lever could remove", against MDI 5.45 %. Wall =
production ON median (0.910 ms scalar, 1.226 ms per vmap-16 batch); busy = ON-program device
busy (0.537 ms scalar, 0.789 ms vmap-16).

| Lever | What bounds it | Arithmetic | Ceiling | vs MDI |
|---|---|---|---|---|
| Fewer launches / kernel fusion (launch-bound) | non-busy fraction of the ON wall | scalar 0.910 − 0.537 = 0.373 ms (41 %); vmap-16 1.226 − 0.789 = 0.437 ms (36 %) | ≤ 1.69x scalar, ≤ 1.55x vmap-16 (only if every launch and host gap vanished; CUDA graphs already took 0.187 ms) | clears |
| Redundant sort / `unique` (neighbourhood) | neighbourhood measured + prorated | 0.115 + 0.062 = 0.177 ms kernel of 0.910 wall | ≤ 19 % → ≤ 1.24x | clears (kernel time only; its kernels' launches are in the row above) |
| Static lattice / step-0 gather | step-0 containment + lattice rows | 0.004 ms + prorated 0.015 ms | ≤ 2 % | **below MDI** |
| Deflections | prorated estimate (no clean row: deflections fuse) | 0.026 / 0.910 scalar; 0.054 / 1.226 vmap-16 | ≤ 2.9 % / ≤ 4.4 % | **below MDI** |
| `vmap` break-even | leg 2 | per-L 0.970 → 0.0120 ms at 256 | already 3.5x at B = 4, 81x at 256 | not a library lever: it pays only if the sampler batches |
| Loop form for compile | cold compile | 6.4 s primal, 7.6–8.1 s vmap, 16.6–20.7 s gradients | share of a fit = compile / (compile + N_eval x t): e.g. 1e5 serial evals x 0.91 ms = 91 s → 6.6 %; 1e5 at vmap-256 = 1.2 s → 84 % | depends on N_eval (unmeasured) |
| Implicit-gradient Jacobian | gradient cost above primal | reverse 1.801 − 0.937 = 0.864 ms (48 % of the gradient call); vmap-16 0.061 ms/L (43 %) | ≤ 1.92x per reverse gradient | clears; forward is NaN, so its lever is moot until fixed |
| Mixed precision (what-if only) | leg 4 | 0.910 / 0.719 | ≤ 21 % scalar, 6 % vmap-16, negative at 256 | clears scalar only; no switch exists |

## Go / no-go memo (decision is the human's)

- The GPU call is launch-bound as the CPU campaign's A100 rows said: 173 kernels per call,
  device busy 57 % of the production wall, 81x per-likelihood gain from `vmap`-256 at 3.2x the
  single-call cost. The FLOP levers the contract lists last (deflections, step-0 lattice) are
  **below MDI** even at a 100 % saving. The levers with room are launch count / fusion
  (≤ 1.69x), the neighbourhood sort (≤ 1.24x) and the reverse gradient's implicit Jacobian
  (≤ 1.92x per gradient).
- **No image-plane fit has been timed.** The 09-27 review's admission bar is the likelihood's
  share of a fit. Every ceiling above is per call; whether any of them moves a fit depends on
  the eval count, on batched vs serial evaluation (a batched sampler already gets 81x), and on
  compile vs run time. The prerequisite, if any lever is to be pursued, is one
  `autolens_inference` measurement: likelihood share of an image-plane point-source fit and its
  eval count, batched vs serial.
- **Blocking finding for any gradient work:** forward-mode `jacfwd` of the image-plane solved
  likelihood is NaN everywhere, while `AnalysisPoint` now defaults gradient searches to forward
  mode (PyAutoLens#752, unreleased). That is a library defect to route through intake before
  any gradient lever, and before #752 ships in a release.
- Open follow-ups: the +8.6 % scalar vs job 359102 (candidate PyAutoGalaxy#634 / PyAutoLens#754,
  not bisected); points 3/4 need a finer FD reference to decide.

## Leftovers

- RAL scratch worktree `/mnt/ral/jnightin/autolens_profiling_wt/point-source-gpu-p01`
  (branch `feature/point-source-gpu-p01`), with superseded run outputs in
  `/mnt/ral/jnightin/autolens_profiling_wt/_p01_run1_366913/` and `_p01_run2_366915/`, traces
  and optimized HLO under its `output/gpu_bottleneck_map_trace_<job>/`.
- Laptop witnesses (`--quick`, not quotable): `gpu_bottleneck_map_laptop_cpu_fp64.json`
  (`all_gates_pass: false` — forward NaN reproduces on CPU) and
  `gpu_bottleneck_map_laptop_cpu_fp32_whatif.json`.
- Shared-module defects noticed, not edited (other tasks work in that file):
  `xla_attribution.idle_gaps` compares event dicts on an exact (start, end) tie (crashes on the
  multi-threaded CPU timeline), and `Instruction.dims` does not parse `pred[...]` or tuple
  shapes. The stage map carries local fixes (`inter_kernel_gaps_ms`, `shape_dims`).
- `solver_config_sweep.py` keeps its own copies of the builders (not refactored).
