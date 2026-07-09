# NNLS positive-only solver: the optimization ledger (closed 2026-07-09)

Findings of [PyAutoArray#369](https://github.com/PyAutoLabs/PyAutoArray/issues/369)
(solver knobs) and [PyAutoArray#370](https://github.com/PyAutoLabs/PyAutoArray/issues/370)
(BPP/ADMM experiment). All measurements on **real extracted production
systems** — the inversion's `curvature_reg_matrix` (Q) and `data_vector` (q)
from HST pixelization + MGE-60 fits (rect 39×39 `RectangularAdaptImage`
n=1581; Delaunay 1500-vertex Hilbert n=1560), post PyAutoArray#368, fp64,
Jacobi-preconditioned as production runs it (cond(Q) ≈ 4e10 raw / 2e11
preconditioned). Probe scripts and extracted systems: `scratch/nnls_speedup/`
(untracked; `extract_system.py` regenerates the `.npz` systems).

## Where the solve sits in the likelihood budget

| step (rect pixelization+MGE HST) | laptop CPU fp64 | RTX 2060 fp64 | RTX 2060 mp |
|---|---|---|---|
| Curvature matrix (F) | 3.01 s (51%) | 0.397 s (45%) | 0.395 s (48%) |
| Inversion setup | 1.58 s (27%) | 0.125 s (14%) | 0.077 s (9%) |
| NNLS reconstruction | 1.21 s (20%) | 0.298 s (34%) | 0.294 s (36%) |
| whole eval | 5.96 s | 0.881 s | 0.820 s |

- The NNLS **share grows on accelerators** (F is one big GEMM and melts on
  GPU; the PDIP solve is a sequential Cholesky chain and does not).
- **Consumer laptop GPUs are viable**: 6.8× the same laptop's CPU, sub-second
  per eval, evidence parity passing. Mixed precision is *not* the consumer
  rescue — F and the NNLS solve stay fp64 by design (cond ~1e10) and are 84%
  of the GPU eval.
- The solve is **latency-bound, not flops-bound** at n≈1600: an RTX 2060
  does a PDIP iteration in ~14 ms despite ~1/100th the A100's fp64 flops.

## The solver itself: what was tried

PDIP (jaxnnls predictor-corrector interior point) needs 19–21 iterations on
these systems; each iteration is one fresh dense Cholesky of the (n, n) KKT
system (~14 ms on RTX 2060) plus two cheap triangular-solve passes (~2 ms).

| lever | outcome |
|---|---|
| **solver tolerance** (upstream `n·eps·5e3` ≈ 1.7e-9 → 1e-6 / 1e-4) | −3 to −6 iterations (~15–30% of solve); rel Δobjective ≤ 4e-13 / 2e-10 ⇒ Δlog_evidence ~1e-8 / 1e-6. **The one real win — shipped as `Settings(nnls_solver_tol=…, nnls_max_iter=…)`, default off.** |
| warm start from the unconstrained solve (PDIP init) | **Hurts** (17→38 iterations): un-centers the interior point; jaxnnls's (Q+I) init is already optimal. |
| fp32 / iterative refinement | Diverges: cond × eps₃₂ ≈ 600 ≫ 1, and the KKT matrix worsens near convergence. |
| Gondzio multiple centrality corrections | Iterations −15% (real, transfers) but wall-time wash — extra solves + line searches eat it; *worse* under `vmap` (all lanes pay every corrector while acceptance diverges). |
| block principal pivoting (BPP) | Sign-pattern init is 95% correct and masked-Cholesky reduced solves are exact (rel Δobj ~1e-15) — but the degenerate rect outskirts + collinear MGE block force pivot thrash: best damped variant needs 36 (Delaunay) – 53 (rect) factorizations vs PDIP's 19–21. Strict Kim–Park: 151. **Slower.** |
| ADMM (factor `(Q+ρI)` once, ~2 ms iterations) | Reaches rel Δobj ~1.5e-5 in 25 iterations then plateaus (3.3e-6 at 3000; ρ ∈ 0.1–100, over-relax 1.7). Evidence needs ≤1e-10. Sublinear tail at this conditioning; real preconditioning is blocked by the non-negativity geometry. **Fails.** |

## Conclusions

1. **PDIP at ~20 iterations × fresh Cholesky is near-optimal for this problem
   class** (cond ~1e10, n ~1600, degenerate active sets) among GPU-viable
   algorithms. Do not re-litigate the table above without new structure.
2. The only shipped lever is per-fit and default-off:
   `aa.Settings(nnls_solver_tol=1e-6)` (~15–20% of solve, Δlog_ev ~1e-8);
   `nnls_max_iter` additionally caps the vmap worst-case lane.
3. Further speedups must change the **problem**, not the solver: fewer source
   pixels (n³ scaling), regularization design that improves conditioning, or
   relaxing the positivity model.
4. Under `vmap`, `lax.while_loop` runs every batch to its slowest lane
   (measured spread 18–22 iterations) — solver-iteration savings survive
   batching; per-lane early exit does not exist.

## Pending

A100 rows (job 330046, `hpc/batch_gpu/submit_probe_nnls_a100`) — queued while
`euclid-ral-gpu-[1-2]` are down; delivers the A100 per-step shares, the
post-#368 Delaunay re-profile, and on-A100 PDIP probe numbers into
`scratch/nnls_speedup/`.
