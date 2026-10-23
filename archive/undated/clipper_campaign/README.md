# Clipper validation campaign — first results (imaging/mge hst)

Phase 2 of the prior-support work. Phase 1 (`ClipperPriorBox`) shipped in
**PyAutoFit#1477** (`1f4b66a`); this measures whether it is worth flipping the
default in phase 3.

**Status: FIRST ARM PAIR RUN, on CPU, at a budget where neither arm converges.
Two of the four pre-registered falsification conditions fired. This is a starting
point for the campaign, NOT a verdict on the fix, and phase 3 must not be written
on it.**

The campaign is being re-run from scratch on a laptop/GPU. Treat everything here
as prior context to check against, not as results to build on: the machinery
(`clipper_campaign.py`, the `SEARCHES_CLIPPER` knob) is reusable, the numbers are
a CPU/float32 single-seed data point, and the "What changed" section below lists
what has already gone stale.

## The result

`multi_start_adam` / `imaging` / `mge` / `hst`, 16 starts x 150 steps, single
seed, **cloud CPU, float32**. Raw rows in
`multi_start_adam_imaging_mge_hst.json`.

| arm | steps | max_log_L | gap to truth | value-NaN | grad-NaN | clips | lanes pinned |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 150 | **-15529.587987** | 47316.370 | 2268 | 0 | 0 | — |
| `prior_box` | 150 | **-15529.587987** | 47316.370 | **47** | 9 | 1920 | **11 / 16** |

Nautilus reference (structurally immune, unit-cube): `max_log_L = 31786.782462`
(the searches framework's `nautilus/imaging/mge/hst/hpc_a100_fp64.json` row, A100 fp64,
archived in PyAutoGut ref `refs/heads/archive/condemned/autolens-profiling/inference-programme` @ `c8b605801068ec3de04314b47da8f7272a038ba1`).

### What fired

1. **"Lane deaths fall but best-fit logL does not move toward the reference."**
   Deaths fell **2268 -> 47**, a 98% reduction — and `max_log_likelihood` and
   `best_fom` are **identical to every printed digit** between the arms. On this
   cell and budget, clipping keeps lanes alive without making them useful.

2. **"Most surviving lanes end pinned to a bound."** 11 of 16 lanes end on a
   bound (31 pinned coordinates). The prompt's reading applies: the wall is
   absorbing the population, and the momentum-reset arm is indicated rather than
   optional.

### What this is NOT

**It is not the caching artefact.** That was the predicted way to get a fake
"identical" result, so it was checked first: the two runs differ in value-NaN
(2268 vs 47), gradient-NaN (0 vs 9) and wall time (286.5s vs 249.6s). Arm 2
really ran; it simply converged to the same best point. Each arm also had a
unique search `name` and a cleared output directory.

The most likely mechanism is the plain one: **the winning lane never left the
prior box**, so clipping never touched it, and every lane clipping rescued
remained worse than the incumbent best.

### The caveat that limits all of it

**Both arms are ~47,316 nats from the Nautilus bar.** Neither is remotely
converged, so the comparison is being made in a regime where the search has not
found the basin at all. This budget (16x150, CPU, float32) is the one #128
characterised the *failure* on — but the truth-recovery claim it is being graded
against came from GPU runs at larger budgets with Prodigy. A fix cannot be shown
to improve an answer that neither arm is close to finding.

So the honest statement is: **at this budget, on this cell, clipping does not
change the answer.** Whether it helps where the search actually converges is
untested, and is the next thing to run.

### The gradient-NaN move is expected, not a regression

`n_grad_nan_lane_steps` went 0 -> 9. #128 predicted exactly this: the NaN-gradient
population is a separate mechanism, and it only becomes visible once lanes survive
long enough to reach it. Same shape as the counter-finding in #128, where removing
prior deaths exposed `ell_comps` trapping that had been masked.

## What is needed before phase 3

1. **A converged budget.** Repeat where `multi_start_*` actually approaches
   31786.8 — GPU, more steps, likely Prodigy. Until then the load-bearing
   question is unanswered, not answered negatively.
2. **Multiple seeds — currently impossible.** `_broad_starts` seeds with a
   hardcoded `np.random.default_rng(0)`, so every run draws **identical** starts;
   seeding `random`/`numpy` perturbs only the initializer. Honouring the
   campaign's "at least two seeds per arm" needs a `seed` argument on
   `AbstractMultiStartGradient`. Single-seed numbers are exactly what #128 had to
   go back and re-derive.
3. **The momentum-reset arm.** 11/16 pinned makes this the decisive arm, and it
   does not exist yet — phase 1 ships the clipped mask it would need, but no
   reset.
4. **The other cells.** Pixelized meshes (GPU), `point_source`, and the unbounded
   negative control.

## What changed after these numbers were taken

These results predate three PyAutoFit merges from the same day. **Read this before
reusing the machinery**, because part of the driver's justification is now void:

| PR | merge | effect here |
|---|---|---|
| #1478 | `bbceff6` | `search.summary` now reports `Clipper`, `Clipped Lane-Steps`, `Clipped Lane-Step Rate` and `Constrained Lane-Steps`. **Prefer reading the clip count from there** rather than from this driver's capture. |
| #1479 | `b6e89cd` | the `float32` `save_json` crash is **fixed**, so counters no longer have to be captured out-of-band to survive the end of a successful run. |
| #1480 | `5c9244b` | crashed-run-poisons-resume is **fixed**. Its real symptom was a hard `JSONDecodeError` on every rerun of the same name — not the "zero-step no-op that reads as a clean result" this note's mitigation was written against. |

What still holds, and is why `clipper_campaign.py` is not simply deletable:

- the **raw** `search_internal` dict is still deleted on successful completion, so
  anything not surfaced through `samples_info` must be captured as written;
- `save_search_internal` must be patched at **class** level — `fit()` rebuilds
  `search.paths`, so an instance-level hook is silently discarded;
- the arm-collision hazard is unchanged (see the bottom of this note).

A useful consequence of #1478: **a `ClipperPriorBox` arm reporting zero clips has
not exercised the clipper.** That is a broken arm, not a null result, and it is now
visible directly in `search.summary`.

## Reproducing

Environment is the fiddly part:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install autolens jaxnnls          # pulls the released stack
pip install -e <PyAutoFit checkout> --no-deps   # MUST come after
python -c "import autofit; print(autofit.__file__)"   # MUST be the checkout
```

The clipper is unreleased, so a PyPI `autofit` silently has no `clipper` argument.
Verify the path before trusting a number.

```bash
python clipper_campaign.py  # removed with the inference programme; recover from the archive ref \
    --arms none,prior_box --seeds 0 --n-starts 16 --n-steps 150
```

`clipper_campaign.py` exists rather than using the ordinary `searches/` runner
because that runner records none of the lane counters, and `search_internal` —
where they live — is **deleted on successful completion**. The driver captures it
as it is written, patching `save_search_internal` at **class** level (`fit()`
rebuilds `search.paths`, so an instance-level hook is discarded).

`SEARCHES_CLIPPER=none|prior_box` was also added to `_samplers.py` for the
ordinary runner; it records the arm as a string in the sampler config, so no
result file can be ambiguous about which arm produced it.

## Trap worth keeping

The `clipper` does **not** enter the search identifier, so two arms differing only
in clipper resolve to the **same output directory**. Stacked with the `.completed`
short-circuit (`fit()` returns a cached result without entering `_fit`), arm 2 can
silently return arm 1's numbers. Every arm here gets a unique `name` and a cleared
directory, and the driver asserts `total_steps == n_steps`. Whether the clipper
*should* enter the identifier is an open question for phase 3's re-baseline.
