# Cluster critical-curve dispatch — phase 3a

Research for autolens_workspace_test#337. This report records the
CPU fp64 evidence and a bounded production contract. No library code or defaults
are changed by this audit.

## Scope

Two synthetic fixtures: an analytic spherical cored isothermal control and a
smooth cluster-scale cored host plus one offset member at lens redshift 0.5,
observed at source redshifts 1 and 2. This checks per-plane distance scaling,
not the general multiple-deflector-plane or realistic substructure problem.
Both critical-curve kinds and both caustic kinds are compared at two marching
resolutions, then against automatic and explicit zero-contour seeds. Explicit
seeds are taken from the numerical reference and therefore test continuation,
not independent discovery or completeness.

The approved work is evidence and a contract. The unissued concurrent
`cluster_curves_engine_dispatch.md` proposal must be reconciled against this
report before it is issued. Phase 3 is not completed by this audit.

## Current-source map

- `autogalaxy/plot/plot_utils.py` has already been removed; the canonical helper
  is `autogalaxy/util/plot_utils.py`. No duplicate-module refactor remains.
- `_critical_curves_method` reads static configuration, defaults/falls back to
  marching squares, and warns on an invalid value or missing zero-contour
  dependency. The helper docstrings still incorrectly call zero-contour default.
- `autolens/cluster/plot/cluster_plots.py` calls marching-squares methods directly
  in both plotting helpers, preserving `use_multi_plane=True, plane_j=j` but
  ignoring configured engine selection.
- `LensCalc._critical_curve_list_via_zero_contour` compiles an inner solver and
  caches it by callable identity on the LensCalc instance. It then reduces paths
  and converts them to NumPy-backed irregular-grid lists. This is not the same
  contract as `einstein_radius_jit_from` and is not an outer-JIT plotting API.
- Automatic seeds come from a 25×25 grid of half-width 3 arcsec; no cluster
  extent is forwarded from the plot grid. Explicitly seeded results do not
  establish automatic discovery.

## Reproduction

From the profiling repo, with the PyAuto environment active:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/lens/critical_curves/dispatch.py --run
python scripts/lens/critical_curves/dispatch.py --summarize
```

The audit is opt-in and outside PR smoke. No arguments print help. Each engine
worker has a 120-second process-group timeout and writes partial completed
quantities. Timeout is not a success. `--summarize` checks raw geometry and
provenance without regenerating evidence. Source hashes and Git revisions in
JSON identify the numerical inputs; all measured calls are synchronized through
host-array conversion. First-call and two warm timings are per quantity; caustic
methods can reuse the earlier critical-curve compilation within the same worker.

## Numerical admission rule

Before running, the distance bound was set to two fine-reference pixels
(0.1 arcsec control, 0.25 arcsec cluster). Component counts must agree, with
one-to-one matching by symmetric point-to-segment distance. Areas, polygon
centroids and longest segments are retained separately. The analytic control
checks both critical radii and caustic radii. Coarse/fine agreement is a local
convergence check, not proof of cluster-wide completeness. A plotting contract
must not silently claim that empty output means no physical critical curve.

## Decision: no-go for a selector-only cluster patch

The current marching-squares path remains the default. A production follow-up
must first preserve the requested field under the evaluation-grid cap; then
address seed coverage and path completion before making explicit zero-contour
cluster selection available. No production default, solver or capacity changes
are included in phase 3a.

The pinned matrix found:

- Both cluster planes lose their tangential curve/caustic with automatic seeds.
  The near plane also loses one of two radial components. Returning an empty list
  is not evidence that the physical curve is absent.
- Reference-derived explicit seeds recover the near-source tangential geometry
  before timeout (radial results unmeasured in this run), but the
  far-source tangential curve has 1002 points, a 20.65 arcsec closing segment,
  area shortfall 90.52 arcsec² and maximum reference distance 6.16 arcsec.
  This is consistent with exhausting the default 500 steps in each direction;
  a larger-step-budget A/B was not run, so sole-cause attribution remains open.
- The corresponding far-source tangential caustic meets the loose absolute
  distance bound even though its parent curve fails. **A caustic must inherit
  the admission failure of its parent critical curve.** The JSON retains the raw geometric comparison, while the summary figure
  also applies the parent-curve gate before colouring a caustic green.
- Outer `jax.jit` rejects all four explicit-seed list-returning methods with
  `NonConcreteBooleanIndexError`. Internal solver JIT remains useful; these
  wrappers must not be advertised as outer-JIT safe.
- The disabled-JIT control times out at 120 seconds. It is a bounded observation
  on this host, not proof that all disabled-JIT calls take the historical >10 min.

## Grid-cap witness

The audit invokes the real `evaluation_grid` decorator with a recorder instead
of evaluating a million-point Hessian. Input: 120×120 at 0.5 arcsec/pixel, requested
resolution 0.05, configured cap 1000. Expected preservation is the same 60 arcsec
field sampled at approximately 0.06 arcsec/pixel. Actual output is 1000×1000 at
8.333333 arcsec/pixel, coordinate centres spanning −4162.5 to +4162.5 arcsec.
The current capped formula mixes a dimensionless ratio with a pixel scale.
This demonstrates field distortion independently of either contour engine.
The normal two-resolution comparisons deliberately stay below that cap.

## Proposed production contract

| Context | Contract |
|---|---|
| One-shot plotting, no engine override | Marching squares; preserve field/centre under any grid cap and report effective resolution. |
| Explicit zero-contour, JIT enabled | Require per-source-plane seeds covering the requested field and an explicit closed-path/completion check; reject incomplete results rather than auto-closing and accepting them. |
| Missing zero-contour dependency | Preserve the existing documented warning plus marching-squares fallback, once that path preserves the field. |
| JIT globally disabled | Do not silently enter the expensive zero-contour path; recommend a warning plus marching-squares fallback for plotting. |
| Outer-jitted caller | Use a separately specified fixed-shape numerical API; do not route through Python plotting/list wrappers or choose solely from `xp`. |
| Multi-plane cluster | Retain `LensCalc.from_tracer(use_multi_plane=True, plane_j=j)` independently for every source plane; do not rebuild a generic final-plane calculator. |
| Caustics | Validate the originating critical curve first; finite/near-reference caustic points alone do not establish a valid boundary. |

These are recommendations, not implemented semantics. The method-selector spy
records static config/default/missing-dependency behavior separately from the
real numerical rows. It uses fake calculators only to observe which method is
called and cannot establish geometry correctness.

## Next bounded implementation

1. Fix `PyAutoGalaxy/autogalaxy/operate/lens_calc.py:evaluation_grid` to preserve
   field, origin and aspect ratio under the size cap. Add a NumPy regression
   for the exact witness, including non-square grids; add the stable invariant
   to a required CI lane. No engine or JAX default changes in this first patch.
2. After that ships, use this campaign to resolve grid-aware seed coverage and
   path-completion signaling. Keep the unissued cluster-dispatch prompt as a
   candidate; a selector-only patch would expose known-invalid results.
3. Then integrate cluster selection, with per-plane routing and geometry tests,
   missing-dependency/disabled-JIT policy and corrected helper docstrings.

Only the current audit issue is open. These are an ordered handoff, not an issue
queue or a declaration that the whole phase 3 has shipped.

## Research versus CI ownership

The experiment driver, full curves, timings, JSON/PNG and cumulative wiki live in
`autolens_profiling/scripts/lens/critical_curves/`, `results/` and
`wiki/campaigns/critical_curves.md`. Subsequent runs append a dated ledger entry
and update the wiki/index; preserve previous artifacts and their verdicts.

`autolens_workspace_test/scripts/cluster/critical_curves.py` is the small
independent regression: a cored spherical cluster at two source redshifts,
analytic tangential/radial radii, analytic caustic radii, actual plane-specific
ray mapping, and distinct per-plane geometry. It is wired into `smoke_tests.txt`.
It does not depend on this profiling checkout. Known failures remain research
witnesses until their production fix can promote the desired invariant into CI.

## Timing interpretation

Warm measurements reuse the same LensCalc instance. Both the generic plot
helpers and cluster per-plane helper currently create new calculators; they do
not automatically receive that warm-cache benefit on the next plotting call.
The experiment's first-call samples are not a benchmark of a whole figure, and
caustic first calls can already reuse the corresponding curve compilation.
Workers disable the persistent compilation cache; host load and thread/cache
environment accompany the final rows. This is a local diagnostic run while other
validation work exists on the host, not an isolated speed-up claim or a GPU result.


## 2026-10-02 — Pinned CPU evidence

Artifacts: [raw JSON](../lens/critical_curves/dispatch_summary_cpu_fp64_v2026.8.17.1_2026-10-02.json),
[admission figure](../lens/critical_curves/dispatch_summary_cpu_fp64_v2026.8.17.1_2026-10-02.png),
[exact measured driver](../lens/critical_curves/dispatch_measured_source.txt).

The 14-worker run completed all six marching-squares rows, both ordinary
zero-contour control rows, the outer-JIT error witness, and the near-source
automatic-seed row. Four workers reached the 120-second cap: disabled-JIT,
near-source explicit, far-source automatic, and far-source explicit. The latter
three saved completed quantities before termination; these are partial evidence,
not passing workers. Both explicit cluster rows saved only the tangential pair;
the far-source automatic row saved all four quantities but did not finish its
final provenance step. Do not infer the unmeasured explicit radial results.

Collection repaired a post-processing CPU-name check (`TFRT_CPU_0` is CPU).
The exact measured driver is SHA256
`47c730b478dd1de91873ab3a7992c9a48d6d13993f088c17c272ee7361312608`.
The collector checks that numerical function ASTs match that frozen driver and
that library/config/runtime provenance matches; collection has its own script
hash. No numerical timing/curve was rerun or replaced. A machine-local cache path
is omitted from device metadata; worker compilation caches were disabled.

Validation: profiling suite 1002 passed / 5 skipped; ruff and generated-document
checks passed. The new bounded CI example passed on both source JAX 0.10.2 and
the isolated smoke environment's JAX 0.11.2; the existing zero-contour example
also passed. The full workspace smoke did not complete; see the shipping checkpoint below.


### Observed call costs (seconds)

These are sums over saved successful quantities, not worker wall time or complete
figure costs. Partial rows cannot be compared as full workloads. First calls
share earlier curve compilation with caustics; warm cost is the sum of the mean
of two subsequent calls per saved quantity. Host load makes these diagnostic.

| Fixture | Engine / mode | Worker | Saved quantities | First-call sum | Warm-call mean sum |
|---|---|---|---:|---:|---:|
| control | marching_squares / coarse | ok | 4/4 | 0.804 | 0.722 |
| control | marching_squares / fine | ok | 4/4 | 2.212 | 2.087 |
| control | zero_contour / auto | ok | 4/4 | 44.916 | 0.322 |
| control | zero_contour / explicit | ok | 4/4 | 28.661 | 0.055 |
| control | zero_contour / disabled_jit | timeout | 0/4 | — | — |
| control | zero_contour / outer_jit | ok | 0/4 | — | — |
| cluster_z1 | marching_squares / coarse | ok | 4/4 | 4.305 | 3.464 |
| cluster_z1 | marching_squares / fine | ok | 4/4 | 14.557 | 10.731 |
| cluster_z1 | zero_contour / auto | ok | 4/4 | 60.519 | 1.401 |
| cluster_z1 | zero_contour / explicit | timeout | 2/4 | 49.842 | 2.537 |
| cluster_z2 | marching_squares / coarse | ok | 4/4 | 2.741 | 2.282 |
| cluster_z2 | marching_squares / fine | ok | 4/4 | 11.359 | 8.128 |
| cluster_z2 | zero_contour / auto | timeout | 4/4 | 91.986 | 3.197 |
| cluster_z2 | zero_contour / explicit | timeout | 2/4 | 57.467 | 2.397 |


## Shipping checkpoint — 2026-10-02

No PR opened; both implementation worktrees remain uncommitted for review.
Initial full smoke timed out unchanged Delaunay examples. The one retry passed
five scripts (Delaunay, Delaunay MGE, rectangular, MGE, LP), then was stopped
when the refreshed Heart ship gate returned RED. Remaining scripts are unrun;
standalone success of the new example does not substitute for the full gate.

Heart snapshot 2026-10-02T09:49:31.721134+00:00, score 45:

- `PyAutoFit: 1 commit(s) behind origin`
- `PyAutoGalaxy: 1 commit(s) behind origin`
- `PyAutoLens: 1 commit(s) behind origin`
- YELLOW: `manifest drift: workspace checkouts (manifest ↔ disk) — 1 mismatch(es) vs PyAutoMind/repos.yaml`
- STALE: `release validation incomplete: no rehearsal for current source`

The three behind commits are release Colab-link updates. Shared canonical
checkouts were not changed during other sessions' work. Resume by resolving the
Heart gate through the normal workflow, then rerun the full workspace smoke and
ship the two companion PRs. Do not issue the grid-cap successor before this
phase is reviewed and near shipping. Scratch logs, raw workers and PR drafts
are retained in the task bundle's `scratch/`; no process was left running.


## Resumed shipping validation — 2026-10-02

The full Heart-owned workspace smoke now passes **33/33**, including the new
cluster regression in 2.9 seconds (768.87 seconds total). This supersedes the
incomplete smoke checkpoint above. Profiling incorporated merged PR #361 without
overlap; its changed upstream test passes 46 cases. Prior 1002-test validation,
artifact checks and numerical evidence remain recorded unchanged.

The human requested `/prm and continue` and explicitly acknowledged the remaining
Heart YELLOW manifest warning (the separate PyAutoPulse registration task).
No release is authorized. The canonical PyAutoLens checkout was fast-forwarded
to merged point-solver PR #764 after smoke; the audit's historical source hashes
remain authoritative. PR CI validates against its installed dependency stack.


## Phase 3b — Preserve the capped evaluation field (2026-10-02)

Phase 3a merged in [profiling #364](https://github.com/PyAutoLabs/autolens_profiling/pull/364)
and [workspace #341](https://github.com/PyAutoLabs/autolens_workspace_test/pull/341).
The next single task is [PyAutoGalaxy #645](https://github.com/PyAutoLabs/PyAutoGalaxy/issues/645).
Its library fix is [PR #646](https://github.com/PyAutoLabs/PyAutoGalaxy/pull/646)
(commit `f19a3377`), awaiting merge/release; the broader engine-selection
NO-GO remains in force because seed coverage and path completion are unresolved.

The existing `dispatch.py` real-decorator `cap_probe()` was run unchanged against
the phase-3b library worktree, with CPU/source imports. This is a geometric probe,
not a new timing matrix; no million-point Hessian was evaluated.

| Same input: 120×120, 0.5 arcsec/pixel; request 0.05; cap 1000 | Phase 3a | Fixed branch |
|---|---:|---:|
| Output shape | 1000×1000 | 1000×1000 |
| Output spacing (arcsec/pixel) | 8.333333333 | 0.06 |
| Physical field width (arcsec) | 8333.333333 | 60.0 |
| Pixel-centre limits (arcsec, both axes) | ±4162.5 | ±29.97 |

Measured `autogalaxy/operate/lens_calc.py` SHA256:
`581ce30864e11e4b2ab22f01774c87a56863d460aa7807ceda1a4f5f47394607`.
The original phase-3a JSON/PNG and measured-source archive remain unchanged.

The capped scale now comes from physical extent divided by the axis limit;
both dimensions are bounded. Integer ceiling on the shorter axis preserves
coverage with less than one pixel total padding. The existing effective Zoom2D
centre and square-padding mask policy remain unchanged, as do below-cap
integer rounding and the already-evaluation-grid path. This is not a fix for
the independent masked-caustic ellipticity/origin investigation.

Validation: six new geometry cases failed before the patch, while two
compatibility controls passed. Afterward the focused suite passed 49 tests and
the full Galaxy suite passed 1315 tests (169.29s). The exact witness is also
added to the small required workspace example alongside its analytic per-plane
curves/caustics. Library-first shipping and merge/release gates still apply.

Full companion Heart-owned smoke passed 33/33 (501.91s); the changed
cluster example passed in 2.2s against the patched library. Galaxy PR #646 CI
is green on Python 3.12/3.13, no-JAX and docs. The linked workspace regression
remains dependent on a library version containing this fix; no release occurred.
