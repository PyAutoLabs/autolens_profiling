# Linear-solver accuracy

**Status:** open
**Question:** On the positive-only systems real models produce, does each solver return the fnnls reference amplitudes, does its convergence flag tell the truth, and what does each digit of accuracy cost in iterations?
**Pre-registered rule:** a candidate is admissible only if it reports converged on 100 % of the 81 systems, has significant-column error ≤ 1e-3 and source-flux error ≤ 1e-4 everywhere (≤ 1e-4 on the euclid system), and has a KKT residual within 10x `pdip_jacobi`'s; lowest median iterations wins ([ledger, pre-registered rule](../../results/notes/linear_solver_accuracy_2026_09.md#pre-registered-decision-rule); committed at `327f571` before the deciding run).
**Verdict:** phase 1: no drop-in candidate; phase 2 needs a solution-based stop. No candidate is admissible. The released raw stop is blind to the flux on reference-inactive columns.
**Headline:** the released raw PDIP solve reports converged on 81/81 systems, yet on the euclid system it leaves 11.5 % of the reference's total amplitude on columns the reference holds at zero (`total_source_flux` +5.76 %); laptop WSL CPU, no RAL job.
**Library PRs:** none yet (phase 2: `draft/bug/autoarray/raw_pdip_forward_amplitude_bias_fix.md`).
**Profiling PRs:** #354 (this study; PR number pending).
**Ledger:** [linear_solver_accuracy_2026_09.md](../../results/notes/linear_solver_accuracy_2026_09.md)
**Mind contract:** epic `linear-solver-programme`; `active/raw_forward_pdip_nnls_early_stopping.md`
**Next:** phase 2: a solution-based stop (an active-set certificate on the PDIP iterate) in PyAutoArray, gated on `flux_inactive_rel` and the euclid latent directly.

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
[hazards README](../../scripts/imaging/hazards/README.md) (where the #571 capture lives).

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 accuracy study | 2026-09-30 | Is any existing solver or constant change a drop-in fix, on 81 captured systems? | admissibility on flag, significant-column and source-flux accuracy, euclid proxy, KKT; `327f571` | no candidate admissible; the released stop is blind to flux on reference-inactive columns | none (laptop) | #354 |
| 2 PyAutoArray fix | not started | A solution-based stop for the raw forward PDIP | to be written before its deciding run | — | — | `draft/bug/autoarray/raw_pdip_forward_amplitude_bias_fix.md` |
| 3 GPU / vmap / A100 | not started | The same cells under `jit(vmap)` on the A100 | — | — | — | `draft/research/autoarray/mge_nnls_fix_pyautoarray_571_slam_60.md` |
| standing | every release | Re-run `accuracy.py` (and `early_stopping.py`) over the whole corpus | no drift in the tables without a solver change | — | — | — |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| none yet | — | — | — |

The package itself (`scripts/lens/solver/`) is profiling-repo code in #354; no library change has
come out of the campaign yet.

## Open / parked / drafts

- `draft/bug/autoarray/raw_pdip_forward_amplitude_bias_fix.md` — phase 2. The ledger's
  "What phase 2 should implement" ranks the options with measured iteration costs: a
  solution-based stop (the only one supported on every system; its cost is not yet measured),
  the forward polish as a stop-gap (euclid latent +7.5e-5, median 23 iterations), a tighter
  tolerance or larger cap (does not fix the flag).
- `draft/research/autoarray/mge_nnls_fix_pyautoarray_571_slam_60.md` — GPU / vmap / A100 rows.
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
