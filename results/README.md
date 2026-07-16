# results

Profiling artifacts written by the packages above. Layout mirrors the source
packages; the dashboard tables in every README are rendered from this tree by
`scripts/build_readme.py`.

## Sections

| Folder | Written by | Contents |
|--------|-----------|----------|
| `runtime/` | [`likelihood_runtime/`](../likelihood_runtime/README.md) sweeps | Per-config sweep outputs + `comparison.{json,png}` per cell; A100 logs/probes |
| `breakdown/` | [`likelihood_breakdown/`](../likelihood_breakdown/README.md) | Versioned per-step decompositions |
| `simulators/` | [`simulators/`](../simulators/README.md) | Versioned simulator run-time summaries |
| `searches/` | [`searches/`](../searches/README.md) | Versioned sampler profiling summaries |
| `pipeline_resume/` | [`pipeline_resume/`](../pipeline_resume/README.md) | Versioned SLaM resume-overhead summaries (cold + resume run records) |
| `quick_update/` | [`quick_update/`](../quick_update/README.md) | Unversioned fast re-profiling snapshots (scratch tier) |
| `notes/` | humans + agents | Narrative findings and design notes (e.g. [`design_lock_in.md`](./notes/design_lock_in.md)) |
| `baselines/` | campaign snapshots | Named, frozen baselines (e.g. `PreOptimizationTimes/`) — see below |

## The two artifact shapes

**Versioned summaries** — written by per-cell scripts run standalone; history
is retained side-by-side so cross-release trends stay inspectable:

```
<cell>_<purpose>_<instrument>_v<YYYY>.<M>.<D>.<PATCH>[_sparse].json   # purpose = summary | breakdown
<cell>_<purpose>_<instrument>_v<YYYY>.<M>.<D>.<PATCH>[_sparse].png
```

The version string is the PyAutoLens release that produced the numbers
(e.g. `v2026.5.29.4`).

**Per-config sweeps** — written by `likelihood_runtime/sweep.py` and
aggregated by `aggregate.py`; each cell dir holds the *latest* sweep:

```
runtime/<class>/<model>[/<instrument>]/<config_name>[_sparse].{json,png,log}
runtime/<class>/<model>[/<instrument>]/comparison.{json,png}
```

Config names: `local_cpu_fp64 | local_cpu_mp | local_gpu_fp64 | local_gpu_mp |
hpc_a100_fp64 | hpc_a100_mp`, with `_sparse` as a filename suffix.

## Named baselines

A **baseline** is a frozen snapshot of a full campaign under
`baselines/<BaselineName>/`, mirroring the `runtime/` layout plus a rendered
`<BaselineName>.md` with every headline number on one page. Baselines are
append-only — never edited after the campaign closes. The first baseline is
**`PreOptimizationTimes`** (the pre-optimization reference). Full convention:
[`notes/design_lock_in.md`](./notes/design_lock_in.md).
