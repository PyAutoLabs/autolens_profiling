# results

Profiling artifacts written by the packages above. Layout mirrors the source
packages; the dashboard tables in every README are rendered from this tree by
`scripts/misc/tooling/build_readme.py`.

## Campaign findings

Read the [campaign index](../wiki/index.md) before reusing a headline or following a
historical note's Next section. It links every campaign's ledger in `notes/` and records
each result's library PRs and release. The fixed-light record stays in the
[fixed-light CPU/GPU summary](notes/profiling_campaign_status_2026_09.md) (CPU closure,
corrected GPU budgets/attribution, and the unversioned bridge-control limitation); its
PyAutoArray #553–#555 were released in 2026.9.19.1. The run-time-over-time view of this tree is
[`../dashboard/`](../dashboard/index.html) (published at
<https://pyautolabs.github.io/autolens_profiling/>), rendered by
`scripts/misc/tooling/build_dashboard.py`: one point per release per cell × config, qualified on
the provenance block against `hpc/release_sweep.conf`. For interferometer fits, the
[decision matrix](notes/interferometer_likelihood_decision_matrix_2026_09.md) says which
likelihood path and device to use by N_vis × mask × source (MGE-20, Delaunay-1500, rectangular),
with per-call ms, setup cost, host memory and an indicative time per fit.

## Sections

| Folder | Written by | Contents |
|--------|-----------|----------|
| `runtime/` | [`likelihood_runtime/`](../scripts/misc/likelihood_runtime/README.md) sweeps | Per-config sweep outputs + `comparison.{json,png}` per cell; A100 logs/probes |
| `breakdown/` | [`likelihood_breakdown/`](../scripts/misc/likelihood_breakdown/README.md) | Versioned per-step decompositions |
| `simulators/` | [`simulators/`](../scripts/misc/simulators/README.md) | Versioned simulator run-time summaries |
| `pipeline_resume/` | [`pipeline_resume/`](../scripts/misc/pipeline_resume/README.md) | Versioned SLaM resume-overhead summaries (cold + resume run records) |
| `quick_update/` | [`quick_update/`](../scripts/misc/quick_update/README.md) | Unversioned fast re-profiling snapshots (scratch tier) |
| `delaunay_nn/` | [`delaunay_nn/`](../scripts/misc/delaunay_nn/README.md) | Versioned full-mapper cap and runtime benchmarks |
| `nnls_warm_start/` | [`nnls_warm_start/`](../scripts/misc/nnls_warm_start/README.md) | The NNLS cross-evaluation warm-start memo A/B experiment — per-model JSON/PNG pairs plus its two notes ([`nnls_warm_start_memo.md`](./nnls_warm_start/nnls_warm_start_memo.md), [`nnls_warm_start_memo_matrix.md`](./nnls_warm_start/nnls_warm_start_memo_matrix.md)). A diagnostic, **not** a production baseline. |
| `streaming_scaling/` | [`interferometer/streaming_scaling/`](../scripts/interferometer/streaming_scaling/README.md) | Versioned interferometer streamed (`from_stream`) vs in-memory (`apply_sparse_operator`) memory/time scaling rows and the parity check ([campaign](../wiki/campaigns/interferometer_streaming.md)) |
| `hazards/` | [`hazards/`](../scripts/misc/hazards/README.md) | Semantic finding records, reproducer plots, generated seed summary, and consumer index |
| `lens/` | [`scripts/lens/`](../scripts/lens/README.md) | Versioned **library-component** summaries (dataset-free axis) — today `lens/deflections/`, per-mass-profile deflection cost with pinned deflection values |
| `lens/solver/` | [`scripts/lens/solver/`](../scripts/lens/solver/README.md) | The linear-solver programme: the frozen positive-only system corpus (`corpus/manifest.json` + one `.npz` per group, with stored fnnls references) and versioned `accuracy` / `early_stopping` summaries per corpus |
| `notes/` | humans + agents | Narrative findings and design notes — **markdown only** (e.g. [`design_lock_in.md`](./notes/design_lock_in.md), [`production_representative_cells.md`](./notes/production_representative_cells.md), [`nnls_solver_ledger.md`](./notes/nnls_solver_ledger.md), [`numpy_deflections_cpu.md`](./notes/numpy_deflections_cpu.md)); see the artefact policy below |
| `logs/` | RAL submits | Committed SLURM job logs, one folder per campaign ([`logs/README.md`](./logs/README.md)); provenance a ledger cites, never a result |
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
`results/notes/delaunay_adapt_split_regularization.md`.

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
[`notes/design_lock_in.md`](./notes/design_lock_in.md).
