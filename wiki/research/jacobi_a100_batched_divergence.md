# Jacobi PDIP: batched vs unbatched on the A100

Research note for linear-solver phase 4a (autolens_profiling#397), 2026-10-08. Campaign:
[Linear-solver accuracy](../campaigns/linear_solver_accuracy.md). Full provenance is in the
[ledger, phase 4a](../../results/notes/linear_solver_accuracy_2026_09.md#phase-4a-2026-10-08--jacobi-batched-vs-unbatched-on-the-a100).
This page prepares a decision. It does not take one, and nothing in the library changed.

## Question

Phase 3b found that on the A100 the Jacobi-preconditioned PDIP (`pdip_jacobi`), solved inside
`jax.jit(jax.vmap(...))` over 50 distinct SLaM systems, gave different results from the same
systems solved one at a time. Iterations differed on 19/50 lanes and the convergence flag on
13/50. On CPU, batched and unbatched were bit-identical. Why does this happen, and what does it
mean for the library?

## Why it matters

Jacobi is the preconditioning that every pixelized (Mapper) inversion uses. MGE systems moved to
the raw solver in PyAutoArray#571/#595. If a batched GPU solve and a single solve of the same
system follow different paths, then a Nautilus `vmap` batch on the A100 and a single-evaluation
re-fit at the same parameters are not the same computation. Reproducibility, the meaning of the
convergence flag, and any CPU-vs-GPU or batched-vs-single parity test for pixelized fits all
inherit that.

## Method

[`scripts/lens/solver/batched_divergence.py`](../../scripts/lens/solver/batched_divergence.py)
is a dataset-free probe that uses the solver corpus. It calls only the library's own solver
entry points, composed the way `_solvers` composes them. It tests `pdip_jacobi`, with the
released `pdip_raw` and the `certified` active-set solve as controls, on the phase-3b batch: the
first 50 systems of `slam_fixture_571` + `slam48_hst`, all n = 60, fp64. The probes are:

1. **Determinism.** The batched solve is run twice, each time through a fresh `jit(vmap)` (new
   trace and compile), and the first executable is called again. Each lane is also solved
   unbatched twice through a fresh `jit`. The probe compares the results bit for bit.
2. **Lowering.** Each lane goes through `jit(vmap)` at B = 1 and through plain `jit`.
3. **Tiling.** One system (`euclid_vis_lp_k0` or `slam_fixture_571/k0`) is copied into B = 2, 8
   and 50 lanes. The probe checks whether the lanes match each other and whether they match the
   unbatched solve.
4. **Trajectory.** The library's `solve_nnls` runs on the lane's Jacobi system with
   `max_iter = k` for k = 0…50, batched and unbatched. This is the `while_loop` that the
   `pdip_jacobi` candidate's `solve_nnls_primal_with_status` runs. k = 0 is jaxnnls's
   `initialize` alone. A consistency check confirms that k = 50 reproduces the candidate's own
   `x` bit for bit on all 50 lanes, batched and unbatched. Because the loop is deterministic
   (probe 1), k indexes the actual PDIP path.
5. **Primitives** (context). The probe runs the linear-algebra calls jaxnnls's PDIP is built
   from (`jax.scipy.linalg.cho_factor`, `cho_solve` and a matrix-vector product) on each lane's
   Jacobi system, `jit` vs `jit(vmap)`.
6. **Join.** Each lane is joined to cond(Q) from the corpus manifest, cond(Q_pc) of the
   Jacobi-scaled system, and the phase-3a and phase-3b convergence flags on both devices.

There was one A100 job (RAL 399050, `euclid-ral-gpu-2`, 2:30). It ran the probe twice: once
with the stack's default XLA flags and once with `--xla_gpu_deterministic_ops=true`. The CPU
control ran on the laptop. Both used the five libraries at tag 2026.10.7.1.

## Evidence

**Per-probe results** (number of lanes, out of 50, where the two sides differ):

| Probe | CPU | A100 | A100, deterministic ops |
|---|---|---|---|
| batched run 1 vs run 2 (fresh jit), all three candidates | 0 | 0 | 0 |
| unbatched run 1 vs run 2 | 0 | 0 | 0 |
| `jit(vmap)` at B = 1 vs `jit`, all three candidates | 0 | 0 | 0 |
| tiled B = 2 / 8 / 50: lanes vs each other | identical | identical | identical |
| tiled B = 2 / 8 / 50: lane vs unbatched | identical | **differs at every B ≥ 2** | same as A100 |
| batched vs unbatched, `x` bits — `pdip_jacobi` / `pdip_raw` / `certified` | 0 / 0 / 0 | **50 / 50 / 50** | identical to A100 |
| batched vs unbatched, iterations — `pdip_jacobi` / `pdip_raw` / `certified` | 0 / 0 / 0 | **19** / 0 / 0 | identical |
| batched vs unbatched, flag — `pdip_jacobi` / `pdip_raw` / `certified` | 0 / 0 / 0 | **13** / 0 / 0 | identical |
| primitives `jit` vs `jit(vmap)`: `cho_factor` / `cho_solve` / matvec / `Q_pc` | 0 / 0 / 0 / 0 | **50 / 50** / 0 / 0 | identical |

The deterministic-ops run reproduced the default run exactly: every per-lane comparison and
every trajectory difference is the same.

**Size of the batched-vs-unbatched difference (A100):**

| Candidate | max ‖Δx‖/‖x‖ over the 50 lanes | iterations / flag change |
|---|---|---|
| `pdip_raw` (released, MGE default) | 1.3e-13 | none |
| `certified` | 3.4e-12 | none |
| `pdip_jacobi` (Mapper default) | 61 | 19 / 13 lanes |

**Trajectory of `pdip_jacobi`** (A100). On all 50 lanes the first difference is at k = 0, in
`initialize`, before any PDIP iteration. All three of x, s and z differ there, on 55–60 of the
60 components, by ‖Δy‖/‖y‖ = 1.1e-15 to 2.1e-15 (fp64 rounding level). After that the lanes
fall into two groups that do not overlap:

| Lane group | Lanes | ‖Δx‖/‖x‖ at k = 1 | Largest ‖Δx‖/‖x‖ at any k | Iterations differ | Unconverged: CPU 3a / A100 unbatched / A100 batched |
|---|---|---|---|---|---|
| quiet | 26 | 1.2e-15 – 1.0e-14 | 8.2e-13 | 0 | 0 / 0 / 0 |
| sensitive | 24 | 2.9e-15 – 0.99 | 2.8e3 | 19 | 19 / 13 / 4 |

In the 19 lanes whose iteration count changes, ‖Δx‖/‖x‖ passes 1e-3 by k = 1 (8 lanes) or
k = 2 (11 lanes). So a rounding-level difference becomes an order-one difference in one or two
PDIP iterations. In the 26 quiet lanes the difference stays at or below 8.2e-13 for all 50
iterations.

**What the sensitive group is.** The 24 sensitive lanes include every lane on which Jacobi fails
to converge on any device or in any mode: all 7 CPU-only, all 12 both-device and the 1 A100-only
member of the phase-3a divergence set, plus 4 lanes that converge everywhere. cond(Q) does not
separate the groups. Across the 50 lanes cond(Q) spans only 9.64e10–9.75e10 and cond(Q_pc) spans
1.44e11–1.55e11 (same model, nearby parameters). The AUC of cond(Q) for predicting "iterations
differ" is 0.61, and 0.56 for cond(Q_pc). For predicting "sensitive" it is 0.62 and 0.54. Both
are close to chance (0.5).

## The explanation the evidence supports

1. **It is not run-to-run nondeterminism.** Batched and unbatched results are each bit-stable
   across fresh compiles and repeat calls. XLA's deterministic-ops flag changes nothing. A
   determinism flag therefore cannot fix it.
2. **It is a difference in arithmetic that depends on the batch shape, and it starts in the
   Cholesky factorisation.** On the A100, the batched (B ≥ 2) lowering of `cho_factor`, and of
   `cho_solve` on top of it, gives different bits from the unbatched lowering on every lane. The
   matrix-vector product and the Jacobi scaling are bit-identical. B = 1 through `vmap` matches
   plain `jit` exactly. The lanes of a tiled batch match each other exactly but not the unbatched
   solve. jaxnnls's `initialize` starts with exactly this factorisation, which is why the PDIP
   state differs at k = 0, at the rounding level. On CPU the batched and unbatched Cholesky are
   bit-identical, so nothing propagates.
3. **Jacobi's ill-behaved systems amplify that difference into a different outcome.** On the
   24 sensitive systems, a ~1e-15 perturbation grows to order one within two iterations. This is
   the same group where Jacobi fails to converge on CPU or A100. Whether a given run converges
   there behaves like a function of the rounding: 19 unconverged on CPU, 13 on the A100
   unbatched, 4 on the A100 batched. On the 26 well-behaved systems, the same perturbation stays
   below 1e-12. The released `pdip_raw` and `certified` see the same Cholesky difference and stay
   at 1e-13 and 3e-12. So the batch-dependence of `pdip_jacobi` is a symptom of the instability
   already known from PyAutoArray#571. It is not a separate defect of `vmap`.

**What the evidence does not show yet:**

- Which GPU kernel the batched Cholesky lowers to, and why it rounds differently. For example,
  it could be a batched cuSOLVER routine against the single-matrix one. That is a 4b
  kernel-selection probe. The probe ran with the stack's own flags (`--xla_gpu_autotune_level=0`,
  Triton GEMM off, set by PyAutoNerves). They were not varied.
- Whether the CPU-vs-A100 difference in the phase-3a divergence set (29 vs 19 systems) comes from
  the same mechanism (LAPACK vs cuSOLVER Cholesky rounding). It is consistent with this result
  but was not tested.
- Whether a full Mapper likelihood (a pixelized inversion with regularisation) produces systems
  in the sensitive group. The corpus systems here are SLaM MGE systems (linear light profiles),
  not Mapper systems. Phase 4a used them because they are the batch where phase 3b saw the
  effect.
- Why the 4 sensitive systems that converge everywhere are sensitive. cond(Q) does not explain
  it, and no other per-system quantity was tested.
- Per-element ulp counts in the committed A100 JSONs are rounded to multiples of ~512–1024 ulp.
  A probe bug, fixed after the run, caused this. The relative norms quoted on this page are
  exact.

## Decision table (for the human; nothing here is decided)

| Option | What it does | Cost | Consequence | Evidence that would settle it |
|---|---|---|---|---|
| A. Keep Jacobi as the Mapper GPU default and document it | Nothing changes. Document that batched and single GPU solves agree only to the rounding level, and fully only on systems where Jacobi converges. | Docs only | Systems that already fail to converge stay batch-dependent on GPU: a `vmap` batch and a re-fit can disagree on the flag and on x. On convergent systems the two agree to ~1e-12. | A Mapper corpus (real pixelized systems) showing how often Mapper systems fall in the sensitive group. If almost never, A is enough. |
| B. XLA deterministic flag | Set `--xla_gpu_deterministic_ops=true` | Some GPU throughput, not measured here | **Ruled out by this evidence**: the flag changed no bit. The effect is not nondeterminism. | Already settled (job 399050). |
| C. Tolerance or iteration-cap change for Jacobi | Tighten or loosen the stop, or the 50 cap | One `accuracy.py` + `timing.py` rerun | Does not remove the rounding-level difference. It only moves which lanes look different, because the sensitive lanes go order-one within 2 iterations, long before any stopping test. Very unlikely to help. | An early-stopping sweep for `pdip_jacobi` on the sensitive lanes. Not expected to change the picture. |
| D. Move the Mapper default off Jacobi on GPU | Route Mapper inversions to `pdip_raw` + polish (#595) or `certified` | A library change and a Mapper-corpus accuracy study (`certified` already serves Mapper-only inversions) | Removes the amplification: `pdip_raw` and `certified` see the same Cholesky difference and stay at ≤ 3e-12 with identical flags. This also removes the known Jacobi divergence. | A Mapper corpus showing that raw or certified meet the accuracy rule on Mapper systems, and a timing row. This is the step to take if A's evidence shows that Mapper systems do become sensitive. |
| E. Make batched and single solves share one lowering | For example, always run the solve through the batched kernel | Engineering, and possibly speed | Gives bit-parity between a batch and a re-fit, but not stability. The sensitive lanes would still depend on rounding, which differs between devices and library versions. | A 4b kernel-selection probe. Only worth it if bit-parity itself becomes a requirement. |

**This research favours** gathering the evidence for A or D next: capture a Mapper corpus and
measure how often its systems are sensitive. B is excluded, and C is not expected to help.

## Mapper corpus (phase 5)

Phase 5 (autolens_profiling#399, 2026-10-08) gathered the evidence the decision table above asked
for: a Mapper corpus run through the same three cells on the laptop CPU and one A100 job (RAL
job 399225, 6:20). Full provenance and tables:
[ledger, phase 5](../../results/notes/linear_solver_accuracy_2026_09.md#phase-5-2026-10-08--mapper-corpus).

**Corpus.** 24 systems captured from the JAX likelihood on `dataset/imaging/hst` at tag
2026.10.7.1, each asserted to be the positive-only Mapper solve in `"jacobi"` mode:
`delaunay_hst` (Delaunay, Hilbert 1500, `AdaptSplit`; n = 1500), `rectangular_hst`
(`RectangularBilinearAdaptImage` 39×39; n = 1369 after edge zeroing) — both Mapper only, lens
light fixed — and `slam_mixed_hst` (the SLaM `source_pix` case: lens 2×20 MGE + Delaunay;
n = 1540, cond(Q) up to 5.6e11). 4 near-truth vectors and 2 at each of noise ×0.3 and ×3 per
group. The group files are stored outside git (sha256 in the corpus manifest; copies on RAL
`/mnt/ral/jnightin/autolens_profiling_corpus/` and the laptop canonical checkout).

**What it shows** (accuracy, stability and cost are separate statements):

| Group | `pdip_raw` admissible (phase-1 rule, *extended*) | `pdip_jacobi` unconverged (CPU / A100) | A100 batched-vs-unbatched sensitive lanes | A100 per-eval ms at B = 8, raw / jacobi | CPU per-eval ms at B = 8, raw / jacobi |
|---|---|---|---|---|---|
| `delaunay_hst` | **yes** | 0/8 / 0/8 | 0/8 (max ‖Δx‖/‖x‖ 1.5e-12) | 30.9 / 29.4 | 659 / 653 |
| `rectangular_hst` | **no** — criterion 2: significant-column error 0.956 (source flux 1.2e-7 passes); `pdip_jacobi` fails it too (1.05) | 0/8 / 0/8 | 0/8 (1.9e-14) | 37.8 / 23.4 | 1634 / 922 |
| `slam_mixed_hst` | **yes** | 0/8 / 0/8 | 0/8 (2.7e-13) | 37.1 / 32.4 | 1331 / 1123 |

Compile is separate: 0.4–1.1 s per batch shape on the A100, 2–43 s on the laptop. On the A100
the batched Cholesky still rounds differently from the single solve at k = 0 on every lane (the
phase-4a mechanism), but no Mapper or mixed lane amplifies it; iterations and flags are
identical batched and unbatched. `certified` is not a candidate for the mixed case (1/8
uncertified, source flux 7.8e-3 off), consistent with the library routing it to Mapper-only
inversions.

One new disagreement: on the laptop CPU at n ~ 1500 the batched PDIP `x` differs from the
unbatched one at ≤ 4.2e-13 from k = 1–3 (bit-identical at n = 60 in phase 4a); the Cholesky
primitives at `initialize` are identical there. Iterations and flags do not change. Not
localised.

### Recommendation table, with the Mapper evidence (for the human; nothing here is decided)

| Option | Mapper evidence now | Cost now measured | Verdict this evidence supports |
|---|---|---|---|
| A. Keep Jacobi as the Mapper GPU default and document | 24/24 Mapper and mixed systems converge on both devices; 0/24 A100-sensitive lanes; accuracy equal to raw except where both fail the same significance floor | none (no change) | **Supported for the systems measured.** Document that batched and single GPU solves agree to ~1e-12 on these systems and that the batch-dependence is confined to systems where Jacobi is unstable — none of which appeared here. |
| B. XLA deterministic flag | — | — | Ruled out (phase 4a). |
| C. Tolerance or cap change for Jacobi | Jacobi converges in 14–20 iterations here; nothing to fix | — | Not indicated. |
| D. Move the Mapper default to `pdip_raw` + polish (or `certified`) | Raw is admissible on Delaunay and mixed, fails criterion 2 only where Jacobi also fails it, and is equally stable batched; `certified` fails on mixed | Raw costs 1.05x / 1.62x / 1.15x Jacobi per evaluation on the A100 at B = 8 (Delaunay / rectangular / mixed); 1.01x / 1.77x / 1.19x on CPU | **Not supported on this evidence**: it buys no stability these systems need and costs 5–62 % more per Mapper evaluation. It is the option that gives one solver for all likelihoods, at that measured price. |
| E. One lowering for batched and single solves | Batched vs unbatched already agree to ≤ 1.5e-12 here | Engineering | Only if bit-parity becomes a requirement (and the CPU n ~ 1500 difference says it would need doing on CPU too). |

**Still not established.** Wide-prior systems: every vector here is near-truth, while the early
phase of a non-linear search samples far from it; that is where a Mapper or mixed system could
still turn Jacobi-unstable, and it is the cheapest next capture (same runners, wider vectors).
Other instruments and configurations (Euclid-like, JWST, interferometer — skipped, no
importable capture harness), other mesh sizes and regularizations, and the presets' MGE-60 lens
light. Why the mixed systems, which carry the n = 60 trigger (signal-free-floor MGE columns),
stay Jacobi-stable at n = 1540. So the honest answer to "fast and stable for all likelihood
functions" today is: raw+polish for MGE (shipped) and Jacobi for Mapper are each stable and the
faster (or, on Delaunay, equal) choice on every system measured; the stability of Jacobi on Mapper systems far from the
truth is the one open gap.
