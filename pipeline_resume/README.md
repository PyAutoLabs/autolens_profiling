# pipeline_resume

Profiles the **overhead of resuming a SLaM pipeline** — the wall time a user
pays when re-running a pipeline whose stages are already (partly) complete,
which is pure overhead between them and their science.

A completed stage skips its non-linear search, but a resume still pays for
library imports, dataset setup, per-stage completed-fit loading (samples
summary, full samples CSV, result rebuild), the inter-stage science re-run from
each loaded result (adapt images, position likelihoods, model composition —
each able to trigger a fresh JAX JIT compile), and any re-visualization.

## Scripts

| Script | What it measures |
|--------|------------------|
| `slam_resume.py` | Full 5-stage SLaM chain (`source_lp[1] → source_pix[1] → source_pix[2] → light[1] → mass_total[1]`, mirrored from `autolens_workspace/scripts/guides/modeling/slam_start_here.py`) run cold, then re-run to profile resume overhead per stage and per component. |

## Running

From the repo root:

```bash
python3 pipeline_resume/slam_resume.py            # 1st run: cold (searches sample)
python3 pipeline_resume/slam_resume.py            # 2nd run: resume (overhead only)
python3 pipeline_resume/slam_resume.py --fast     # scaled-down n_live harness run
python3 pipeline_resume/slam_resume.py --reset    # wipe pipeline output, force cold
```

Each invocation appends one run record (`mode: cold | resume`) to the versioned
summary JSON; once a cold and a resume run both exist the comparison PNG is
rendered:

```
results/pipeline_resume/slam_resume_summary_<instrument>_v<version>[_fast].{json,png}
```

## Instant runs via test mode (the recommended cold path)

`PYAUTO_TEST_MODE=2` bypasses every sampler (one likelihood call per stage) and
`PYAUTO_TEST_MODE_SAMPLES=<N>` sizes the synthetic `samples.csv` so completed
outputs are production-representative (PyAutoFit#1378/#1381). The full 5-stage
cold chain then completes in ~3 minutes instead of hours:

```bash
PYAUTO_TEST_MODE=2 PYAUTO_TEST_MODE_SAMPLES=10000 python3 pipeline_resume/slam_resume.py --reset   # instant cold
PYAUTO_TEST_MODE=2 PYAUTO_TEST_MODE_SAMPLES=10000 python3 pipeline_resume/slam_resume.py           # resume record
```

Rules and known deltas:

- **Keep the same env vars on the resume invocation** — test-mode output is
  namespaced under `output/test_mode/`, so unsetting them points the rerun at
  a different (cold) tree.
- Test-mode runs write to a separate `_testmode`-suffixed artifact (excluded
  from the auto-tables below, which are reserved for real-sampling records)
  and record `test_mode` per run.
- Size parity (2026-07-17, N=10000): `source_lp[1]` samples.csv = 10,000 rows
  × 21 cols, 8.81 MB vs the production target measured on PyAutoFit#1378
  (10,187 rows × 21 cols, 9.07 MB). Later stages scale with their model's
  parameter count (3.4–5.0 MB).
- Deltas vs a production resume: latent computation is auto-skipped; there is
  no search-internal checkpoint; adapt-image FITS values come from the
  prior-median model (right size, wrong values) — timing-honest, not
  science-honest.

### Post fast-path reference (2026-07-17 evening, after PyAutoGalaxy#504 + PyAutoLens#619 + PyAutoFit#1388/#1390)

Cold 159s → **any resume ~11–13s**: the cold run now writes the per-stage caches itself
(adapt images + solved positions memoized into each result's `files/` and preserved in
the search zips), so every later resume pays only imports (~3s), the stage-1
`check_likelihood_function` consistency recompute (~6s, config-gated) and ~0.1s/stage
of loading. The pre-fix decomposition below is retained as the historical baseline.

### Pre fast-path baseline (2026-07-17 morning, CPU, v2026.7.9.1, N=10000; moderate background load)

Cold chain 205s → resume 148s of pure overhead, decomposed:

| Component | Time | Notes |
|-----------|------|-------|
| `positions_likelihood_from` | 70.0s | point solver re-run from the upstream result (source_pix_1 40.6s + mass_total 29.4s) |
| adapt-image reconstruction | 54.7s | max-LH fit rebuild incl. JIT compile + inversion (source_pix_1 25.7s + source_pix_2 29.0s; light/mass legs free only via in-process `cached_property` reuse) |
| completed-fit path (all 5 stages) | 12.2s | 9.3s of it is stage 1's `check_likelihood_function` consistency recompute + first JAX compile; others ~0.7s each incl. unzip |
| imports + model compose | ~10.5s | once per process |
| full samples load | 0.5s | 8.8 MB / 10k rows — **not** a bottleneck |

## Latest results

<!-- BEGIN auto-table:pipeline-resume -->
_No data yet — run `pipeline_resume/slam_resume.py` twice (cold, then resume) to populate. See section README._
<!-- END auto-table:pipeline-resume -->

## Decomposition

- Script-level spans wrap each stage's `search.fit` and each inter-stage block
  (`adapt_images`, `positions`, `model_compose`).
- Runtime timing wrappers (no library edits) decompose the resume path inside
  `search.fit`: `completed_fit_total` (`NonLinearSearch.result_via_completed_fit`)
  and the `DirectoryPaths` `load_samples_summary` / `load_samples` loads.
  `search_fit − completed_fit_total` is pre/post-fit output handling;
  `completed_fit_total − loads` is result rebuild + optional re-visualization.
- `import_s` and `dataset_setup_s` are recorded per process — they are part of
  the real resume experience but paid once per run, not per stage.
