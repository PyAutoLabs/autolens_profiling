# Linear-solver accuracy

**Status:** open (phase 5: Mapper corpus measured — Jacobi stable and no slower than raw on 24/24 near-truth pixelized/mixed systems; decision pending)
**Question:** On the positive-only systems real models produce, does each solver return the fnnls reference amplitudes, does its convergence flag tell the truth, and what does each digit of accuracy cost in iterations?
**Pre-registered rule:** a candidate is admissible only if it reports converged on 100 % of the 81 systems, has significant-column error ≤ 1e-3 and source-flux error ≤ 1e-4 everywhere (≤ 1e-4 on the euclid system), and has a KKT residual within 10x `pdip_jacobi`'s; lowest median iterations wins ([ledger, pre-registered rule](../../results/notes/linear_solver_accuracy_2026_09.md#pre-registered-decision-rule); committed at `327f571` before the deciding run).
**Verdict:** phase 1: no drop-in candidate; phase 2 needs a solution-based stop. No candidate is admissible. The released raw stop is blind to the flux on reference-inactive columns. Phase 2: PyAutoArray#595 shipped the forward polish (the phase-1 `pdip_raw_polish`, bit-for-bit on 81/81): `flux_inactive_rel` 0.115 -> 3.31e-4, euclid latent +5.76e-2 -> +7.47e-5 (test green); still not admissible under the rule as written (criteria 2–4, for the recorded rule weaknesses). Phase 3a: on the A100 the released `pdip_raw` (tag 2026.10.7.1) reproduces its CPU row on 81/81 systems (identical iterations, per-system abs(Δ `flux_inactive_rel`) ≤ 5.4e-14); same rule outcome on both devices. Phase 3b (timing, not admissibility): batched `jit(vmap)` on the A100 the released `pdip_raw` costs 0.190 ms per evaluation at B = 50 on distinct SLaM systems (laptop CPU 0.682 ms; A100 B = 1 3.95 ms), its lanes stop within one iteration of each other and the batched path reproduces the unbatched solve on every lane; `pdip_jacobi` costs 1.9x because one diverging lane pins the batch at the 50 cap. No pin moves. Phase 4a (research): the A100 batched-vs-unbatched Jacobi difference is deterministic, batch-shape-dependent Cholesky rounding (B ≥ 2 vs single, ~1e-15 at PDIP iteration 0), amplified to order one within 2 iterations on exactly the systems where Jacobi diverges somewhere; the XLA deterministic-ops flag changes nothing; no default changed ([research note](../research/jacobi_a100_batched_divergence.md)).
**Headline:** the released raw PDIP solve reports converged on 81/81 systems, yet on the euclid system it leaves 11.5 % of the reference's total amplitude on columns the reference holds at zero (`total_source_flux` +5.76 %); laptop WSL CPU, no RAL job.
**Library PRs:** PyAutoArray#595 (merged 2026-09-30, merge `7a89e19a0`; released in 2026.10.2.1).
**Profiling PRs:** #354 (phase 1); phase 2 record; phase 3a #394; phase 3b #396; phase 4a #397 (PR pending). Phase 5 #399 (PR pending).
**Ledger:** [linear_solver_accuracy_2026_09.md](../../results/notes/linear_solver_accuracy_2026_09.md)
**Mind contract:** epic `linear-solver-programme`; `active/raw_forward_pdip_nnls_early_stopping.md`
**Next:** human decision on the Mapper default from the [phase-5 recommendation table](../research/jacobi_a100_batched_divergence.md#recommendation-table-with-the-mapper-evidence-for-the-human-nothing-here-is-decided) — the evidence supports A (keep Jacobi, document) on near-truth systems; the open gap is wide-prior Mapper/mixed systems (a wider-vector capture with the phase-5 runners)

## Why this campaign

The euclid pipeline's latent jit test
(`tests/test_compute_latent_variable.py::test_latent_euclid_variables_traces_under_jax_jit`)
compares the jitted `total_source_flux` with the eager NumPy value at rel 1e-3. On the released
library the jitted value is +5.76 % off (3.5110934 against 3.3198795). The two paths differ only in
the positive-only solver: NumPy runs fnnls, JAX runs a primal-dual interior-point (PDIP) method.

The solver has come up again and again. PyAutoArray#571 found the Jacobi-preconditioned PDIP
diverging (to ~1e68) on SLaM MGE systems and made the raw (un-preconditioned) forward mode the
default for linear-object-only inversions, with a data-scaled tolerance (#572). #573 then added a
backward-pass polish so gradients did not inherit the raw mode's loose stop. Each fix was
measured on the system in front of it, and none of the measurements accumulated. The user asked
for one home:

> "We have had the linear solver issue / tolerance / accuracy crop up as an issue frequently, so
> I think we need to make it a dedicated package in autolens_profiling which includes a wiki,
> stats and whatnot which continually adds runs and data and info too so we can learn and recheck
> what tolerances to use."

That home is [`scripts/lens/solver/`](../../scripts/lens/solver/README.md): a frozen, growing
corpus of captured `(Q, q)` systems with stored fnnls references, a candidate registry built only
from library primitives, and cells that re-run on every release. This page is its campaign record.

Prior art: the [NNLS solver ledger](../../results/notes/nnls_solver_ledger.md), the
[certified positive solver](certified_positive_solver.md) campaign, the
[NNLS warm-start memo](../../results/nnls_warm_start/nnls_warm_start_memo.md) and the
[hazards README](../../scripts/imaging/mge/hazards.md) (where the #571 capture lives).

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 accuracy study | 2026-09-30 | Is any existing solver or constant change a drop-in fix, on 81 captured systems? | admissibility on flag, significant-column and source-flux accuracy, euclid proxy, KKT; `327f571` | no candidate admissible; the released stop is blind to flux on reference-inactive columns | none (laptop) | #354 |
| 2 PyAutoArray fix | 2026-09-30 | Fix the raw forward PDIP's amplitude bias | phase-1 rule re-applied, not re-based | forward polish shipped: `flux_inactive_rel` ≤ 3.31e-4, euclid latent +7.47e-5, 0/81 unconverged; rule still FAILs 2–4 ([ledger](../../results/notes/linear_solver_accuracy_2026_09.md#phase-2-2026-09-30--library-fix-shipped-pyautoarray595)) | none (laptop) | PyAutoArray#595 |
| 3 GPU / vmap / A100 | 3a 2026-10-07; 3b 2026-10-08 | 3a: does the released solver on the A100 reproduce the stored fnnls references (parity)? 3b: the same cells under `jit(vmap)` (timing) | 3a: the phase-1 rule and the standing no-drift rule, applied per device | 3a: parity holds for `pdip_raw` at 2026.10.7.1 (0/81 unconverged, worst `flux_inactive_rel` 3.31e-4 on both devices, iterations identical 81/81); `pdip_jacobi` diverges on a different system set (29/81 CPU, 19/81 A100) and `pdip_raw_tol_jaxnnls` flips its flag on euclid ([ledger](../../results/notes/linear_solver_accuracy_2026_09.md#phase-3a-2026-10-07--a100-parity-of-the-corpus-on-the-released-20261071)). 3b (timing, not admissibility): `pdip_raw` per-eval 0.190 ms A100 / 0.682 ms CPU at B = 50 SLaM, batch max 19 vs median 17.5 iterations, batched = unbatched on every lane; `pdip_jacobi` 0.369 ms A100, batch pinned at the 50 cap, batched A100 trajectories differ from unbatched on 19/50 lanes ([ledger](../../results/notes/linear_solver_accuracy_2026_09.md#phase-3b-2026-10-08--gpuvmap-timing-of-the-solver-corpus)) | 3a: RAL 397475 (euclid-ral-gpu-1, 1:04) + laptop CPU; 3b: RAL 398249 (euclid-ral-gpu-2, 0:38) + laptop CPU | #393/#394; #395 |
| 4a Jacobi batched divergence (research) | 2026-10-08 | Why does `pdip_jacobi` under `jit(vmap)` differ from the unbatched solve on the A100 only? | none (research; no rule) | deterministic (fresh-jit reruns and `--xla_gpu_deterministic_ops` bit-identical); B = 1 vmap = jit; tiled lanes identical to each other but not to unbatched; the batched `cho_factor` / `cho_solve` differ from unbatched on 50/50 lanes (matvec identical), so the PDIP state differs at k = 0 by ~1e-15; 24/50 lanes (every lane Jacobi diverges on, on any device, + 4) amplify that to order one within 2 iterations, 26 stay ≤ 8.2e-13; cond(Q) does not separate them; `pdip_raw` / `certified` stay ≤ 1.3e-13 / 3.4e-12 ([research note](../research/jacobi_a100_batched_divergence.md), [ledger](../../results/notes/linear_solver_accuracy_2026_09.md#phase-4a-2026-10-08--jacobi-batched-vs-unbatched-on-the-a100)) | RAL 399050 (euclid-ral-gpu-2, 2:30) + laptop CPU | #397 |
| 5 Mapper corpus (research) | 2026-10-08 | On pixelized (Delaunay, rectangular) and mixed (2×20 MGE + Delaunay) systems, is `pdip_raw` admissible, is `pdip_jacobi` stable on the A100 batched and unbatched, and what does each cost? | the phase-1 rule, explicitly *extended* to this corpus | `pdip_raw` admissible on Delaunay and mixed, fails criterion 2 (sig 0.956) on rectangular where Jacobi also fails it (1.05); Jacobi 0/24 unconverged and 0/24 A100-sensitive lanes; raw costs 1.05–1.62x Jacobi per A100 evaluation at B = 8 ([ledger](../../results/notes/linear_solver_accuracy_2026_09.md#phase-5-2026-10-08--mapper-corpus)) | RAL 399225 | #399 |
| standing | every release | Re-run `accuracy.py` (and `early_stopping.py`) over the whole corpus | no drift in the tables without a solver change | — | — | — |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#595 | forward value of the raw PDIP mode = the #573 polished iterate | 2026-09-30 (`7a89e19a0`) | 2026.10.2.1 |

The package itself (`scripts/lens/solver/`) is profiling-repo code in #354. PyAutoArray#595 is the
campaign's first library change.

## Open / parked / drafts

- Option 1 of the ledger's "What phase 2 should implement" — a solution-based stop (active-set
  certificate) — is still unbuilt; #595 shipped option 2 (the forward polish). Needed only if
  better than `flux_inactive_rel` 3.3e-4 is required on euclid-like systems.
- `pdip_jacobi` under `vmap` on the A100 converges on a batch-dependent set of SLaM systems (phase 3b guard). Localised in phase 4a ([research note](../research/jacobi_a100_batched_divergence.md)): batch-shape-dependent Cholesky rounding amplified by Jacobi's unstable systems; the Mapper-default decision is open, pending a Mapper corpus.
- A phase-2 rule should give the KKT criterion an absolute floor (as written, the reference fnnls
  fails it on 9/52 systems at residuals ≤ 4.3e-16).

## Caveats

- **Laptop timings.** Walls are WSL2 medians of five warm calls at load ~2 and move 20–50 %
  between runs; the iteration counts are exact and are what the rule ranks by.
- **Post-hoc numbers are not a verdict.** Seven candidates, two metrics (`flux_inactive_rel`,
  `active_set_mismatch`) and the euclid-latent-by-candidate table were added after the deciding
  run; the ledger and the package README keep them in separate, labelled sections.
- **One euclid system.** The euclid magnitude rests on one captured system; the SLaM groups show
  the same mechanism at 1e-3 – 1e-2.
- **Noise variants are not rescalings** (the 1e-3 diagonal add is unscaled), and `amp_rel_max` is
  dominated by 1e-9 reference columns, so it is recorded but not gated.
- **The intensity-sum flux proxy is stricter than the latent** (the forward polish: proxy 4.96e-2,
  latent 7.5e-5), because spurious amplitude lands on compact Gaussians.

## Journal

### 2026-09-30 — phase 1: package, corpus, pre-registered rule, deciding run

Built `scripts/lens/solver/` (corpus, candidate registry, metrics, `accuracy.py`,
`early_stopping.py`, `capture.py`) and captured 81 systems in four groups: the 8-system #571
fixture, all 48 #571 SLaM vectors, 24 sigma_min / noise variants and the euclid latent test's
system with its eager (3.3198795) and jitted (3.5110934) latents. After an 8-system smoke run the
decision rule was committed (`327f571`); the deciding run over all 81 systems followed on
`8adb629` and reproduced the pre-rule 11:27 run bit-for-bit on every row except wall time.

No candidate is admissible. The released raw PDIP solve is converged on 81/81 at objective gap ≤
2.6e-12, but carries 11.5 % of the reference amplitude on inactive columns on the euclid system.
Jacobi PDIP diverges on 29/81 (#571); jaxnnls's absolute tolerance is accurate but runs to the cap
on 75/81; the forward polish is 0/81 unconverged but leaves 4.96e-2 on the euclid source-flux
proxy; certified is exact on euclid in 8 passes but uncertified on 48/81.

Post-hoc (inputs to phase 2, not a verdict): a larger polish cap changes nothing; jaxnnls's
tolerance with cap 100 / 200 still runs to the cap on 74/81; the data-scaled tolerance keeps the flag
truthful down to 1e-5 (euclid latent +5.1e-4) and loses it at 1e-6 (78/81 at the cap, exact). The
euclid latent computed through the pipeline's own code from each candidate's reconstruction
(validated to 3e-8 against both stored values) turns green for the forward polish, the 1e-5
tolerance and jaxnnls's tolerance above cap 50 (51 iterations). Next: phase 2 in PyAutoArray.

Also in #354: `xla_attribution.py`'s PyAutoArray line anchors were re-pinned to `d4298445`
after PyAutoArray#591 moved `inversion/inversion/abstract.py` by +1 / +9.

### 2026-09-30 — phase 2: PyAutoArray#595 shipped the forward polish

PyAutoArray#595 (issue #594, merge `7a89e19a0`) makes the raw mode return the #573 polished
iterate as its forward value. All cells were re-run against the merged library (provenance
PyAutoArray `7a89e19a0` in every JSON): `pdip_raw` is now identical to phase 1's `pdip_raw_polish`
on 81/81 systems — 0/81 unconverged, forward iterations median 18 / max 24 (the polish's 1–7 are
not counted), worst `flux_inactive_rel` 3.31e-4 (was 0.115), worst source-flux 4.96e-2 (was 16.4,
euclid only), KKT ≤ 3.2e-16. Euclid `total_source_flux` +7.47e-5 (was +5.76e-2), so the pipeline
test is green (19/19). The rule as written still rejects it (criterion 2 on euclid and the
`noise_x3_v32` flat direction, criterion 3's proxy at 4.96e-2 against a latent of 7.5e-5,
criterion 4 at the 3e-16 floor on 5/52). `euclid_latent.py`'s `pdip_raw -> jit` validation leg
was re-based to the library-main jit value 3.320127604; `pdip_raw_tol_1e-2` (no polish) now
reproduces the *pre-fix* `pdip_raw`, and the phase-1 sentences saying otherwise are annotated.
The `_v2026.8.17.1` artefacts were overwritten in place (source-checkout version stamp); the
pre-fix ones are at profiling `3ad68af`. [Ledger, phase 2](../../results/notes/linear_solver_accuracy_2026_09.md#phase-2-2026-09-30--library-fix-shipped-pyautoarray595).

### 2026-10-07 — phase 3a: A100 parity from a private 2026.10.7.1 checkout

The shared RAL mirror is not synced (euclid_dr1 depends on it), so the five libraries were cloned
at tag 2026.10.7.1 into `/mnt/ral/jnightin/PyAuto_wt/linear-solver-p3/` and a new submit,
`hpc/batch_gpu/submit_lens_solver_accuracy_a100_fp64`, ran `accuracy.py --device gpu` from a
feature-branch worktree beside it: RAL job 397475, `euclid-ral-gpu-1`, COMPLETED in 1:04, import
guard green, 0 tracebacks. A same-tag CPU row was run on the laptop; it is bit-for-bit identical
to the phase-2 artefact except wall time. On the A100 the released `pdip_raw` reproduces its CPU
row on every system: 0/81 unconverged, iterations identical on 81/81, worst `flux_inactive_rel`
3.31e-4 (euclid) and median 1.69e-7 on both, per-system differences ≤ 5.4e-14. Two disagreements
are reported in the ledger: `pdip_jacobi` diverges on 19/81 on the A100 against 29/81 on CPU
(14 systems change flag), and `pdip_raw_tol_jaxnnls` reports converged on euclid at the 50 cap on
the A100 only. The rule as written gives the same "not admissible" for `pdip_raw` on both devices.
Walls (0.88 ms CPU, 3.96 ms A100 per unbatched n = 60 solve, compile excluded) are context only;
batched timing is phase 3b. Artefacts are the `_v2026.10.7.1` pairs (renamed from the source
checkouts' 2026.8.17.1 stamp). [Ledger, phase 3a](../../results/notes/linear_solver_accuracy_2026_09.md#phase-3a-2026-10-07--a100-parity-of-the-corpus-on-the-released-20261071).

### 2026-10-08 — phase 3b: batched `jit(vmap)` timing on CPU and the A100

New cell `scripts/lens/solver/timing.py` (with `_solvers.batched_kernel`, the existing jitted
bodies under `vmap`) timed `pdip_raw` against `pdip_jacobi` at B = 1 / 16 / 50, fp64, with compile
recorded apart from 7 interleaved steady rounds, on 56 distinct SLaM systems and the tiled euclid
system. A100: RAL job 398249 (`euclid-ral-gpu-2`, 0:38) from the phase-3a private 2026.10.7.1
clone, run from a sibling RAL worktree because the 3a one held untracked 3a artefacts; CPU: the
laptop on local tag worktrees. Per evaluation at B = 50 on SLaM the released raw solve costs
0.190 ms on the A100 and 0.682 ms on the laptop (A100 B = 1: 3.95 ms, matching phase 3a's 3.96 ms
unbatched); its batch maximum is 19 iterations against a median of 17.5, and every batched lane
reproduces its unbatched solve (A100 |Δ `flux_inactive_rel`| ≤ 1.4e-14). Jacobi costs 0.369 ms
because a diverging lane holds the batch at the 50 cap, and on the A100 its batched trajectories
differ from the unbatched ones on 19/50 lanes. Compile is 0.3–1.2 s per batch shape. A timing is
not admissibility; no pin moves. [Ledger, phase 3b](../../results/notes/linear_solver_accuracy_2026_09.md#phase-3b-2026-10-08--gpuvmap-timing-of-the-solver-corpus).

### 2026-10-08 — phase 4a: Jacobi batched vs unbatched on the A100, localised

New dataset-free probe `scripts/lens/solver/batched_divergence.py` (library entry points only)
on the phase-3b 50-lane SLaM batch, with `pdip_raw` and `certified` as controls. A100: RAL job
399050 (`euclid-ral-gpu-2`, 2:30), run once with the stack's XLA flags and once with
`--xla_gpu_deterministic_ops=true`, from the phase-3a private 2026.10.7.1 clone via a new
sibling RAL worktree `linear-solver-p4a`; CPU: the laptop on tag worktrees (every probe
identical). On the A100 the effect is deterministic (fresh-jit reruns bit-identical, the
deterministic-ops run identical to the default), absent at B = 1 and present at every B ≥ 2 (tiled
lanes identical to each other, not to the unbatched solve), and it starts in the batched
`cho_factor` / `cho_solve` (50/50 lanes differ; matvec identical), so every lane's PDIP state
differs at k = 0 by ~1e-15. 24 lanes, containing every lane Jacobi diverges on anywhere, amplify
that to order one within 2 iterations; the other 26 stay ≤ 8.2e-13 for 50 iterations. cond(Q)
(9.64e10–9.75e10 here) does not separate them. Research only; no default, pin or tolerance moved.
[Research note and decision table](../research/jacobi_a100_batched_divergence.md);
[ledger, phase 4a](../../results/notes/linear_solver_accuracy_2026_09.md#phase-4a-2026-10-08--jacobi-batched-vs-unbatched-on-the-a100).

### 2026-10-08 — phase 5: the Mapper corpus

Three new `capture.py` runners added 24 systems captured at tag 2026.10.7.1 and asserted to be
the positive-only Mapper solve in `"jacobi"` mode: `delaunay_hst` (n = 1500), `rectangular_hst`
(n = 1369) and `slam_mixed_hst` (lens 2×20 MGE + Delaunay, n = 1540, cond(Q) up to 5.6e11),
stored with a new lossless symmetric `Q` encoding (dense groups would exceed GitHub's 100 MB
limit). `accuracy.py`, `timing.py` (B = 1 / 8 / 16) and `batched_divergence.py` ran per group on
the laptop and in one A100 job (RAL 399225, `euclid-ral-gpu-2`, 6:20, private 2026.10.7.1 clone,
new sibling RAL worktree `linear-solver-p5`). Under the phase-1 rule extended to this corpus,
`pdip_raw` is admissible on Delaunay and mixed and fails criterion 2 on rectangular (significant
columns 0.956, source flux 1.2e-7), where `pdip_jacobi` fails it too (1.05). Jacobi converged on
24/24 on both devices; the A100 batched Cholesky still rounds differently at k = 0, but no lane
amplifies it (≤ 1.5e-12; iterations and flags identical). Raw costs 1.05x / 1.62x / 1.15x Jacobi
per A100 evaluation at B = 8 (18–38 ms per evaluation at these sizes; compile 0.4–1.1 s). New
disagreement: CPU batched vs unbatched now differs at ≤ 4.2e-13 at n ~ 1500 (bit-identical at
n = 60). Research only; nothing moved.
[Research note, Mapper section](../research/jacobi_a100_batched_divergence.md#mapper-corpus-phase-5);
[ledger, phase 5](../../results/notes/linear_solver_accuracy_2026_09.md#phase-5-2026-10-08--mapper-corpus).
