# results

Versioned profiling artifacts written by scripts under [`likelihood/`](../likelihood/README.md), [`simulators/`](../simulators/README.md), and [`searches/`](../searches/README.md). Layout mirrors the source folders.

## Filename convention

```
<profile_name>_summary_v<YYYY>.<M>.<D>.<PATCH>.json
<profile_name>_summary_v<YYYY>.<M>.<D>.<PATCH>.png
```

The version string is the PyAutoLens release that produced the numbers (e.g. `v2026.5.1.4`). Older versions are kept alongside newer ones so cross-release trends stay inspectable.

Example:

```
results/likelihood/imaging/imaging_summary_v2026.5.1.4.json
results/likelihood/imaging/imaging_summary_v2026.5.1.4.png
results/likelihood/imaging/mge/delaunay_sparse_cpu_likelihood_summary_hst_v2026.5.1.4.json
```

Populated as Phases 1–3 land. See the top-level [README](../README.md) for the full phase plan.
