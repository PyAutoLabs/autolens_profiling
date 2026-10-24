# results

Profiling artifacts written by the packages above. Layout mirrors the source
packages; the dashboard tables in every README are rendered from this tree by
`scripts/misc/tooling/build_readme.py`.

## Layout

Since the day-zero move (autolens_profiling#442, `catalogue/migration_day_zero.json`, applied
with `git mv` only -- no file changed bytes) the tree mirrors `scripts/`:

```
results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>[/<model_variant>]/<measurement>.<ext>
    .../history/<release_pin>_<YYYY-MM-DD>/<measurement>.<ext>   # older records of the same key
    .../scans/<name>/<measurement>.<ext>                         # size scans; never current
    .../experiments/<name>/<measurement>.<ext>                   # arms, experiments, job sidecars
results/<dataset_type>/<variant>/<instrument>/comparison.<ext>   # cross-setup aggregate
results/lens/<component>/...                                     # unchanged
results/misc/<group>/...                                         # no derivable cell
```

`setup_id` is `<backend>_<device>_<operator>_<precision>[_<transformer>]` (e.g.
`jax_gpu_sparse_fp64_wtilde`); `host_class` is `hpc_a100`, `hpc_ral_cpu`, `local_laptop` or
`local_gpu`; `model_variant` appears only for non-default variants (`light_fixed`, `rtu`,
`mge_mass`). Ledgers moved to [`wiki/campaigns/`](../wiki/campaigns/) and dead experiments to
[`archive/`](../archive/README.md). Each map entry records the old path, the new path and the
file's sha256. The 23 rows whose setup axis was never recorded stay at their old paths under
`runtime/` and `breakdown/` until a human rules on them.

Writers still compose the legacy name; `_profile_cli.resolve_output_paths` places a run whose
default folder is `runtime/` or `breakdown/` into its mirrored cell with the map's own
classification. The catalogue builder (`build_catalogue.py`, which emits the
`profiling-summary@3` lifecycle catalogue) and the tests read the mirrored tree natively.
`build_dashboard.py` and `build_readme.py` read it natively too, keyed on the v3 identity;
`catalogue/series_id_map.json` maps each pre-move dashboard series id to its v3 id so the
published trend history stays continuous (#442 PR 3b).

## Parity witness

The day-zero rule (lifecycle report §3.6) wants a parity witness beside every day-zero record:
did this cell's log-likelihood agree with a reference log-likelihood for the same dataset and
model? The run that measured it writes `parity.json` into the cell, next to `runtime.json`
(older witnesses go under the cell's `history/<release_pin>_<YYYY-MM-DD>/`):

```json
{
  "schema": "parity-witness",
  "version": 1,
  "status": "match",
  "log_likelihood": -12345.678,
  "reference_log_likelihood": -12345.671,
  "tolerance": 0.01,
  "reference": {"source": "jax_cpu_dense_fp64 on hpc_ral_cpu, same dataset and model", "date": "2026-10-20"},
  "measured_at": "2026-10-21T09:00:00Z",
  "autolens_version": "2026.10.17.1",
  "source_pixels": 1250
}
```

(The numbers above only show the shape; no witness has been recorded yet.)

| Field | Meaning |
|---|---|
| `status` | `match` (\|log L - reference\| <= `tolerance`), `diverged` (above it) or `not_assessed` (no comparison made; `reason` required, numbers may be `null`) |
| `log_likelihood`, `reference_log_likelihood` | the cell's and the reference's log-likelihood; finite numbers for `match`/`diverged` |
| `tolerance` | absolute tolerance on \|delta log L\|, non-negative |
| `reference.source`, `reference.date` | where the reference value came from (setup, host, run or paper) and the ISO date it was produced |
| `measured_at`, `autolens_version` | when and on which release the witness ran (its `release_pin`) |
| `source_pixels` | optional; sets the record's regime under the standard-size rule |

`python scripts/misc/tooling/parity_witness.py --check` validates every `results/**/parity.json`,
including that `status` agrees with the numbers. `build_catalogue.py` turns each witness into a
`profiling-summary@3` record with `measurement: parity`, `metric: abs_delta_log_likelihood` and
`values` `{abs_delta_log_likelihood, tolerance}` (v3 values must be non-negative, so the signed
log-likelihoods and the status sit in the record's `parity` block). A `not_assessed` witness has
no value, so it yields no record; the catalogue's `parity_witnesses` list names every witness.
`coverage_matrix.py` reports per planned setup cell whether a witness is `match`, `diverged`,
`not_assessed` or missing ([`dashboard/coverage.md`](../dashboard/coverage.md)).

## Day-zero move (redirects)

| Old prefix | New location | Files |
|---|---|---|
| `runtime/<dataset_type>/<variant>/...`, `breakdown/<dataset_type>/...` | `<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>/` (current, `history/`, `scans/`, `experiments/`) | 834 |
| `runtime/**/*.log`, `*.out` | `misc/logs/<campaign>/` | 20 |
| `breakdown/**` with no derivable cell | `misc/likelihood_breakdown/` | 91 |
| `runtime/**` with no derivable cell | `misc/likelihood_runtime/` | 2 |
| `hazards/` | `misc/hazards/` | 26 |
| `logs/` | `misc/logs/` | 25 |
| `quick_update/` | `misc/quick_update/` | 5 |
| `delaunay_nn/` | `misc/delaunay_nn/` | 3 |
| `nnls_warm_start/` | `misc/nnls_warm_start/` | 2 |
| `streaming_scaling/` | `misc/streaming_scaling/` | 20 |
| `sweep/` | `misc/likelihood_breakdown/` | 4 |
| `notes/*.md` (ledgers) | `../wiki/campaigns/<dataset_type>/[<variant>/]` | 55 |
| `../wiki/campaigns/*.md` (campaign pages) | `../wiki/campaigns/<dataset_type>/[<variant>/]` | 25 |
| dated snapshots, `_lever`, `_probe`, pre-2026-09 series, `baselines/PreOptimizationTimes/`, frozen packs (`notes/clipper_campaign/`, `notes/point_source_cpu_2026_09_17_reported/`), `simulators/`, `parallel_scaling/`, `pipeline_resume/` | `../archive/YYYY/MM/<slug>/` | 272 |
| `lens/`, `README.md`, `provenance_grandfathered.txt` | unchanged | 100 |

Look up any single file in `catalogue/migration_day_zero.json` (`old` -> `new`).

## Campaign findings

Read the [campaign index](../wiki/index.md) before reusing a headline or following a
historical note's Next section. It links every campaign's ledger in `notes/` and records
each result's library PRs and release. The fixed-light record stays in the
[fixed-light CPU/GPU summary](../wiki/campaigns/imaging/profiling_campaign_status_2026_09.md) (CPU closure,
corrected GPU budgets/attribution, and the unversioned bridge-control limitation); its
PyAutoArray #553–#555 were released in 2026.9.19.1. The run-time-over-time view of this tree is
[`../dashboard/`](../dashboard/index.html) (published at
<https://pyautolabs.github.io/autolens_profiling/>), rendered by
`scripts/misc/tooling/build_dashboard.py`: one point per release per cell × config, qualified on
the provenance block against `hpc/release_sweep.conf`. For interferometer fits, the
[decision matrix](../wiki/campaigns/interferometer/interferometer_likelihood_decision_matrix_2026_09.md) says which
likelihood path and device to use by N_vis × mask × source (MGE-20, Delaunay-1500, rectangular),
with per-call ms, setup cost, host memory and an indicative time per fit.

## Sections

The folders as they were before the day-zero move; see the redirect table above for where
each now lives.

| Folder | Written by | Contents |
|--------|-----------|----------|
| `runtime/` | [`likelihood_runtime/`](../scripts/misc/likelihood_runtime/README.md) sweeps | Per-config sweep outputs + `comparison.{json,png}` per cell; A100 logs/probes |
| `breakdown/` | [`likelihood_breakdown/`](../scripts/misc/likelihood_breakdown/README.md) | Versioned per-step decompositions |
| `simulators/` | [`simulators/`](../scripts/misc/simulators/README.md) | Versioned simulator run-time summaries |
| `pipeline_resume/` | [`pipeline_resume/`](../scripts/misc/pipeline_resume/README.md) | Versioned SLaM resume-overhead summaries (cold + resume run records) |
| `quick_update/` | [`quick_update/`](../scripts/misc/quick_update/README.md) | Unversioned fast re-profiling snapshots (scratch tier) |
| `delaunay_nn/` | [`delaunay_nn/`](../scripts/misc/delaunay_nn/README.md) | Versioned full-mapper cap and runtime benchmarks |
| `nnls_warm_start/` | [`nnls_warm_start/`](../scripts/misc/nnls_warm_start/README.md) | The NNLS cross-evaluation warm-start memo A/B experiment — per-model JSON/PNG pairs plus its two notes ([`nnls_warm_start_memo.md`](misc/nnls_warm_start/nnls_warm_start_memo.md), [`nnls_warm_start_memo_matrix.md`](misc/nnls_warm_start/nnls_warm_start_memo_matrix.md)). A diagnostic, **not** a production baseline. |
| `streaming_scaling/` | [`interferometer/delaunay/`](../scripts/interferometer/delaunay/streaming.md) | Versioned interferometer streamed (`from_stream`) vs in-memory (`apply_sparse_operator`) memory/time scaling rows and the parity check ([campaign](../wiki/campaigns/interferometer/delaunay/interferometer_streaming.md)) |
| `hazards/` | [`hazards/`](../scripts/misc/hazards/README.md) | Semantic finding records, reproducer plots, generated seed summary, and consumer index |
| `lens/` | [`scripts/lens/`](../scripts/lens/README.md) | Versioned **library-component** summaries (dataset-free axis) — today `lens/deflections/`, per-mass-profile deflection cost with pinned deflection values |
| `lens/solver/` | [`scripts/lens/solver/`](../scripts/lens/solver/README.md) | The linear-solver programme: the frozen positive-only system corpus (`corpus/manifest.json` + one `.npz` per group, with stored fnnls references) and versioned `accuracy` / `early_stopping` summaries per corpus |
| `notes/` | humans + agents | Narrative findings and design notes — **markdown only** (e.g. [`design_lock_in.md`](../wiki/campaigns/misc/design_lock_in.md), [`production_representative_cells.md`](../wiki/campaigns/imaging/production_representative_cells.md), [`nnls_solver_ledger.md`](../wiki/campaigns/lens/solver/nnls_solver_ledger.md), [`numpy_deflections_cpu.md`](../wiki/campaigns/lens/deflections/numpy_deflections_cpu_ledger.md)); see the artefact policy below |
| `logs/` | RAL submits | Committed SLURM job logs, one folder per campaign ([`logs/README.md`](misc/logs/README.md)); provenance a ledger cites, never a result |
| `baselines/` | campaign snapshots | Named, frozen baselines (e.g. `PreOptimizationTimes/`) — see below |

## Artefact policy

Decided 2026-09-27 (autolens_profiling#341), after `results/notes/` had grown to 74 files of
which 16 job logs and 10 JSON sidecars were 38 % of the tree, and measurement PRs had reached
16k–26k lines of per-repeat JSON. `scripts/misc/tooling/check_results_layout.py --check`
enforces the first and last rules in `lint.yml`.

1. **`notes/` holds ledgers only.** Its top level is `*.md`. The two folders
   `clipper_campaign/` and `point_source_cpu_2026_09_17_reported/` are frozen evidence packs
   with their own READMEs, allowlisted by name in the check; nothing new goes beside them.
2. **Job logs go to `logs/<campaign>/`** (`<campaign>_<YYYY_MM_DD>_ral_job_<jobid>[_<what>].out`),
   or stay on RAL and are cited by job id. Commit a log only when it carries something the
   result JSON does not (the node, the load line, a failed arm's stderr).
3. **JSON sidecars go beside the result JSONs they describe** — a job's source-hash or
   status sidecar for a `breakdown/imaging/` measurement lives in `breakdown/imaging/`,
   named `<campaign>_<phase>_job<jobid>.json`. A ledger links it relatively.
4. **Committed result JSON is summarised, not dumped.** A new measurement commits per-row
   medians, the bootstrap CI, the pins and the provenance block; per-repeat samples stay in
   `output/` (gitignored) or on RAL. The existing per-repeat files are history and stay.
5. **Every result JSON carries a provenance block** at `device.provenance`, written by
   `_profile_cli.device_info_dict()` (so any cell that records its JAX device gets it with no
   per-script code): `provenance_schema`, `captured_at` (UTC), `host`, `slurm`
   (`job_id`, `array_job_id`, `array_task_id`), `loadavg_at_import` (the run's start),
   `loadavg_at_write` (its end, when the JSON is assembled), `profiling_revision`,
   `library_revisions` (git HEAD of this checkout and every imported PyAuto* library, via
   `scripts/misc/likelihood_breakdown/provenance.py`), `library_versions`, and
   `dependency_versions` (`jax`, `jaxlib`, `numpy`, `scipy`, `numba`, `nufftax`). These are
   the fields the [wiki index](../wiki/index.md) and the run-time dashboard read. A
   device-recording JSON without the block fails the check unless it is listed in
   [`provenance_grandfathered.txt`](./provenance_grandfathered.txt) — the result JSONs
   written before the block existed. That manifest is a ratchet: lines are removed as files
   gain the block or are deleted, never added by hand.

## Performance artifact shapes

**Versioned summaries** — written by per-cell scripts run standalone; history
is retained side-by-side so cross-release trends stay inspectable:

```
<cell>_<purpose>_<instrument>_v<YYYY>.<M>.<D>.<PATCH>[_sparse].json   # purpose = summary | breakdown
<cell>_<purpose>_<instrument>_v<YYYY>.<M>.<D>.<PATCH>[_sparse].png
```

The version string is the PyAutoLens release that produced the numbers
(e.g. `v2026.5.29.4`).

**Per-config sweeps** — written by `scripts/misc/likelihood_runtime/sweep.py` and
aggregated by `aggregate.py`; each cell dir holds the *latest* sweep:

```
runtime/<class>/<model>[/<instrument>]/<config_name>[_sparse].{json,png,log}
runtime/<class>/<model>[/<instrument>]/comparison.{json,png}
```

Config names: `local_cpu_fp64 | local_cpu_mp | local_gpu_fp64 | local_gpu_mp |
hpc_a100_fp64 | hpc_a100_mp`, with `_sparse` as a filename suffix.

**Regularization provenance (Delaunay family).** Since 2026-09-08 the Delaunay
cells (`likelihood_breakdown/{delaunay,delaunay_nn}.py`,
`likelihood_runtime/{delaunay,delaunay_nn,delaunay_numba}.py`) default to
`AdaptSplit(inner=0.1, outer=10.0, signal_scale=0.1)` — what production pairs
Delaunay with — and record the resolved scheme in a top-level `regularization`
key. **A row with no `regularization` key was measured with
`ConstantSplit(1.0)`**, which `--regularization constant_split` still selects;
each cell pins one log-evidence per scheme. Same-node ConstantSplit control
rows are written with a `_constant_split` config name.

Since the same date the canonical A100 rows `breakdown/imaging/delaunay{,_nn}_hpc_a100_fp64`
and `runtime/imaging/delaunay{,_nn}/delaunay{,_nn}_hpc_a100_fp64` are **AdaptSplit** rows. The
ConstantSplit rows they replaced were not overwritten — they survive as
`..._hpc_a100_fp64_constant_split_2026_09_05.{json,png}` — and the 2026-09-08 same-node
ConstantSplit controls are `..._hpc_a100_fp64_constant_split.{json,png}`. See
`wiki/campaigns/imaging/delaunay_adapt_split_regularization.md`.

## Semantic hazard findings

Numerical-hazard records are keyed by stable semantic finding ID rather than by
release-version filename. Their reproducers decide whether the same finding
persists after source code moves:

```
hazards/<subject>/<name>/<hazard_class>.{json,png}
hazards/hazards_index.json
hazards/ell_comps_clamp.md
```

The per-check JSON carries typed measurements, backend reachability, scientific
impact, and source anchors. `hazards_index.json` is the consumer-facing view;
the markdown file is a generated worked example. See
[`scripts/misc/hazards/`](../scripts/misc/hazards/README.md).

## Named baselines

A **baseline** is a frozen snapshot of a full campaign under
`baselines/<BaselineName>/`, mirroring the `runtime/` layout plus a rendered
`<BaselineName>.md` with every headline number on one page. Baselines are
append-only — never edited after the campaign closes. The first baseline is
**`PreOptimizationTimes`** (the pre-optimization reference). Full convention:
[`notes/design_lock_in.md`](../wiki/campaigns/misc/design_lock_in.md).
