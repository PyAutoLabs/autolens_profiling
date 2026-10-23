# Gaussian deflections precompute

**Status:** complete
**Question:** With every Gaussian's geometry fixed and only the mass-to-light ratio free, can deflections be precomputed and rescaled?
**Pre-registered rule:** witness: later evaluations call no Faddeeva/`wofz` kernel and match the per-evaluation path to rtol 1e-12 (written into the epic prompt 2026-09-02, before any run); gates: deflection pins rtol 1e-6 unchanged, numba likelihood pins unchanged, numpy and JAX measured before/after. No speed threshold was set.
**Verdict:** witness met — `_wofz` call counts [60, 0, 0] against [60, 60, 60] controls, L2 rescale 2.4e-13 max relative, likelihood bit-identical on numpy; on JAX the field folds out of the trace (0 jnp `wofz` calls at compile). All 3 phases shipped 2026-09-03/04; no pin edited in any profiling or workspace suite. Honest negative: JAX steady-state `vmap` unchanged (inversion-dominated).
**Headline:** SLaM-shaped numba CPU likelihood (30 fixed Gaussians, one free ratio) 0.583 → 0.195 s per call (3.00×, bit-identical), laptop WSL2 quiet re-run; no RAL job.
**Library PRs:** PyAutoGalaxy#602, PyAutoArray#520, PyAutoGalaxy#605 (all released 2026.9.4.1).
**Profiling PRs:** #214, #216.
**Ledger:** sections "Fixed-geometry deflection memo — phase 1" and "— phase 2" of the predecessor ledger [numpy_deflections_cpu.md](../../results/notes/numpy_deflections_cpu.md); phase 3 has no profiling note.
**Mind contract:** epic `gaussian-deflections-precompute`; `complete/archive/epics/precompute_fixed_geometry_gaussian_deflections.md`; records `complete/2026/09/gaussian-precompute-p{1,2,3}.md`
**Next:** none

## Why this campaign

On 2026-09-02, during [Numpy deflections CPU](numpy_deflections_cpu.md), the human proposed a
follow-up: when every Gaussian light profile's values are fixed and only the mass-to-light ratio is
free, precompute the deflections and scale them by a constant, adding "likely that JAX doesn't use
this so would help there" (verbatim in the epic ledger). In the SLaM `mass_light_dark` stage every
Gaussian of an MGE lens light is fixed from the light stage and shares one free
`mass_to_light_ratio`, yet each likelihood call re-evaluated each Gaussian's Faddeeva field, which is
exactly linear in that ratio. The intake filed it as a successor, to start only after the
numpy-deflections epic completed, with the call-count witness above.

The epic was born 2026-09-03 with its design settled by a code survey: one private memo keyed on
values, never on model metadata. **L1** caches the whole field of a fully fixed mass profile;
**L2** caches the unit-ratio field of `Gaussian` / `lmp` / `lmp_linear` and returns `ratio × field`.
Three phases, issued one at a time: numpy memo (PyAutoGalaxy#601), JAX trace-time constant
(PyAutoGalaxy#604), downstream sweep (autolens_workspace#528).

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 numpy memo | 2026-09-03 | Memoise the fixed field across numpy likelihood evaluations and rescale by the free ratio | `_wofz` 0 on evaluations 2+, non-zero on both controls; L2 rescale rtol 1e-12; deflection pins unchanged rtol 1e-6; memo under its byte cap | Basis-30 hst 135.6 → 6.3 ms (21.5×), euclid 33.6 → 2.5 ms (13.5×); SLaM-shaped likelihood 0.583 → 0.195 s (3.00×), bit-identical; witness [60, 0, 0]; L2 2.4e-13; largest store 19.19 MB of 256 MB | laptop | PyAutoGalaxy#602, #214 |
| 2 JAX trace-time constant | 2026-09-03 | Fold the fixed field out of the jaxpr as a trace-time numpy/scipy constant | validate by `jax.vmap` over the free ratio, never jit-on-concrete; no `wofz` ops left in the trace; workspace_test JAX pins hold (report, do not edit); deflection pins rtol 1e-6 | jnp `wofz` compile calls 240 → 0 (scipy 180); jaxpr 53,369 → 13,289 equations (−75 %); `vmap_first_call` 10.84 → 5.36 s (2.02×; 1.64× on a second run); `vmap_steady_x10` unchanged (0.96×); likelihood 8.6e-15 relative; 15/15 workspace_test scripts pass, vmap values bit-identical | laptop | PyAutoArray#520, PyAutoGalaxy#605, #216 |
| 3 downstream sweep | 2026-09-04 | Does anything downstream move, and does the user learn the kill switch exists? | `test_autolens` green; SLaM smoke with memo on and `AUTOGALAXY_DEFLECTIONS_MEMO=0`; workspace_test pins unchanged; numba likelihood pins rtol 1e-6; doc line | `test_autolens` 610 passed; every SLaM stage bit-identical memo on vs off; `mass_light_dark[1]` 0.370 vs 0.515 s (0.72×, test mode); `pixelization_numba_mge_mass` 2.61× per call at hst resolution; no pin edited; docstring paragraph names the kill switch | local CLI (no RAL) | autolens_workspace#530 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoGalaxy#602 | private `deflections_memo.py` (L1/L2, content-keyed grid fingerprint + weakref cache, 256 MB byte cap, kill switch, `memo_disabled()`), hooked at the `Galaxy` / `Basis` summation sites | `a647aa32` | 2026.9.4.1 |
| PyAutoArray#520 | `Grid2D.subtracted_and_rotated_from` under `jax.ensure_compile_time_eval()` when offset and angle are concrete | `e36a5af4` | 2026.9.4.1 |
| PyAutoGalaxy#605 | memo JAX branch: numpy twin grid, scipy evaluation at trace time, `ratio * xp.asarray(field)`, `jax_folds` counter | `65af1122` | 2026.9.4.1 |

Phase 3 shipped no library code: autolens_workspace#530 (merge `7bbffe80`) adds one paragraph to the
`__MASS LIGHT DARK PIPELINE__` docstring of `mass_stellar_dark/slam.py` and its notebook.

## Open / parked / drafts

- `draft/bug/autofit/dataset_model_free_grid_offset_pytree_roundtrip.md` — a `DatasetModel` with a free `grid_offset` cannot round-trip `autofit.jax.register_model` (phase 2 finding).
- Not filed: `no_run.yaml` lists `mass_stellar_dark/slam` as needing JAX/CSE though it runs clean on the numpy test-mode path (phase 3 trap).
- Not filed: `Grid2DIrregular.subtracted_and_rotated_from` was deliberately left without the phase-2 treatment.
- Not filed: the `multi_galaxy` / `group` SLaM variants lack the pipeline docstring block, so the memo paragraph was not mirrored there.

## Caveats

- **All rows are laptop / local, none on RAL.** Phase 1 shared the box with the parallel `jax-faddeeva-clamp-audit` task (load ~8), inflating absolute ms ~1.7×; the headline figures are the quiet re-runs (load ~2). Memo-off / memo-on ratios held at 21× (hst) and 13–16× (euclid) contended or not.
- **The SLaM-shaped cell is a timing fiducial, not a fit** (parameters at prior medians, log likelihood −56107.56). Phase 1 measured it at 3.00× quiet (2.50–2.72× contended); phase 3 reports 2.61× on the same cell in a different session, with no profiling note behind it.
- **Phase 2's compile win is 1.6–2.0×**, not one number (contended developer host); steady state does not move.
- **Phase 3's 0.72× is test mode** (`PYAUTO_TEST_MODE=2`, one likelihood call per stage); the MGE-mass cell is the production-scale number. The interferometer `modeling_visualization_jit.py` Part 2 was OOM-killed at 10.7 GB RSS: its pre-existing `no_run.yaml` SLOW/OOM entry, not the memo.
- **Numerical gates differ in scale.** The rtol 1e-12 witness is a correctness gate on the L2 rescale (an ulp-level reordering); the deflection pins stay at rtol 1e-6. Phase 3's `pixelization_numba` hst read 27661.910206968903 against pin 27661.910133665442 (2.65e-9 relative), inside 1e-6 and unexplained in the record.
- **The memo changes what a harness times.** Before `_driver.measure_profile` held `memo_disabled()`, the driver's `tracer_s` silently became a hit-path timing (gNFW/hst 1.1 ms against a 282 ms record); the existing cells are memo-off by construction.
- The stub said "none in `results/notes/`"; phases 1–2 are in fact recorded in the predecessor ledger.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the three phases, the witness as the pre-registered rule and its outcome, and a
laptop headline (3.00× SLaM-shaped likelihood, phase 1 quiet re-run); all three library PRs released
2026.9.4.1. Corrected: phases 1–2 do have a ledger (sections of `numpy_deflections_cpu.md`), and #216 is
the phase-2 profiling PR. Phase 3 (autolens_workspace#528 → #530, merged 2026-09-04) has no profiling
note: it was a docs-only workspace sweep that verified the memo bit-identical on every SLaM stage and named
the `AUTOGALAXY_DEFLECTIONS_MEMO=0` kill switch. Open: no RAL measurement exists for any phase.
