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
