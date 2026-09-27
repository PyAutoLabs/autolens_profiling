# Post-certified-solver likelihood breakdown

**Status:** draft
**Question:** Once the certified solver is integrated in the installed stack, where does the likelihood spend its time on numba CPU, JAX A100 and JAX RTX, for rectangular, Delaunay and DelaunayNN, and which remaining bottlenecks can be optimised?
**Pre-registered rule:** the draft's witness: all nine backend/mesh combinations have measured whole-call and exclusive breakdown records that reconcile within 5 % (or quantify instrumentation uncertainty), and a ranked assessment states measured contribution, correctness constraints and whole-call speedup ceiling per candidate. No per-lever go/no-go threshold is written yet.
**Verdict:** not started
**Headline:** none yet
**Library PRs:** none
**Profiling PRs:** none
**Ledger:** none yet; context in [certified_solver_phase_c1_lane_rate_2026_09.md](../../results/notes/certified_solver_phase_c1_lane_rate_2026_09.md)
**Mind contract:** `draft/research/autolens_profiling/post_certified_solver_likelihood_breakdown.md` (formalised 2026-09-15; dependency `complete/2026/09/certified-positive-solver.md`)
**Next:** issue when scheduled; first reconcile the draft's stale `Blocked-by:` line (see Caveats)

## Why this campaign

The human asked, once the certified solver was implemented, for a look at the likelihood breakdown
to see whether the parts that had become the bottlenecks could be optimised, on numba CPU and JAX
GPU (A100 and RTX), for rectangular, Delaunay and DelaunayNN. The intake agent formalised it on
2026-09-15 from the non-solver seed of `autolens_profiling#259` (`ideas.md`): after the
[fixed lens light](fixed_lens_light.md) epic, ~21 ms of the 25 ms certified A100 call was not the
solve, and the dense `F + λH` build overtook the solve above N ≈ 2500.

It is one bounded assessment task: measure, rank, and hand any library change to a separately
scoped task. It must run on the production certified solver and dispatch as installed, not the
earlier harness monkeypatch, and record the merged library revision and the solver each
measurement actually selected. No speedup is promised.

## Phases

The draft defines no phases; the rows below are its required measurements, none started.

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| Numba CPU × 3 meshes | not started | whole-call + same-process exclusive breakdown, rectangular / Delaunay / DelaunayNN | reconcile within 5 % or quantify unexplained time; record threads, BLAS, preprocessing provenance | not started | none | none |
| JAX A100 × 3 meshes | not started | same, fp64, ~1500 source pixels | as above; check instrumentation has not changed fusion | not started | none | none |
| JAX RTX × 3 meshes | not started | same | as above; a missing path is a stated gap, never extrapolated | not started | none | none |
| Ranked assessment | not started | which remaining terms are worth optimising? | per candidate: measured contribution, whole-call ceiling, cost, correctness tradeoff; demonstrated vs hypothesis kept apart | not started | none | none |

## What shipped and where it is

Nothing shipped; the campaign has not started.

## Open / parked / drafts

- `draft/research/autolens_profiling/post_certified_solver_likelihood_breakdown.md` — the prompt itself (unissued).
- Overlap to reconcile at planning, per the draft: [HST GPU residue](hst_gpu_residue.md) (`complete/2026/09/hst-gpu-non-solver-residue.md`), [fixed-light numba CPU](fixed_light_numba_cpu.md) (`complete/2026/09/fixed-lens-light-numba-cpu.md`, `fixed-light-numba-phase1.md`, `fixed-light-numba-solver.md`).

## Caveats

- **Superseded in part before it started** (the draft's own `Superseded-in-part:` line): the A100 /
  RTX columns were measured by `autolens_profiling#268` (HST GPU residue phase 1, 2026-09-16), and
  the numba CPU columns belong to the fixed-lens-light numba CPU campaign (#263 / #265 / #267). The
  HST Delaunay CPU evidence is reusable but is not nine-cell coverage.
- **The `Blocked-by:` line is stale.** It says PyAutoArray#567 is "merged 2026-09-23, unreleased";
  the release sheet has it released 2026.9.26.1. The production default flip it names
  (`draft/feature/autoarray/certified_solver_scalar_default_flip.md`) was retired unbuilt on
  2026-09-24 (`complete/2026/09/certified-solver-scalar-default-flip.md`), and the library default
  `positive_only_solver` is still `pdip` (`complete/2026/09/certified-positive-solver.md`). The draft
  already warns not to assume certified solving is selected or fastest, so "post-certified" needs
  restating at planning.
- **Historical notes are context, not baselines**: `fixed_lens_light_{library_path,hardware,source_pixel_scaling,verdict}_2026_09.md` must not be quoted as fresh production numbers.
- **Kernel times subtracted across different runs are not a breakdown**; the draft asks for a
  same-process decomposition, separate compile from warm steady state, and fixed-light preparation
  cost stated separately.
- **Batching is out of scope** without an evidence-driven follow-up, as are wider source-size,
  dataset and precision sweeps. The context ledger (certified-solver phase C1) found `jit(vmap)` at
  B=50 returning wrong values on nearly every lane and B=100 out of memory on the A100; any batched
  row here would inherit that.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the draft's question, witness contract, required nine-cell measurements and its
own warnings. Resolved: the draft's `Blocked-by:` is stale (PyAutoArray#567 released 2026.9.26.1;
the default flip was retired). Open: no phases, thresholds or jobs exist until the prompt is issued.
