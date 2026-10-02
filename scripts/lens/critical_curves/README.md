# Critical curves: geometry, dispatch and cost

This is the standing dataset-free research home for critical curves and
caustics. It studies correctness and robust settings alongside runtime.
The [campaign wiki](../../../wiki/campaigns/critical_curves.md) records decisions
and new questions; the [full ledger](../../../results/notes/critical_curves_dispatch.md)
records each run and its limitations. Add future experiments here instead of
starting a second campaign in the integration-test workspace.

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/lens/critical_curves/dispatch.py --run
python scripts/lens/critical_curves/dispatch.py --summarize --output results/lens/critical_curves/dispatch_summary_cpu_fp64_v2026.8.17.1_2026-10-02.json
AUTOLENS_PROFILING_SMOKE=1 python scripts/lens/critical_curves/dispatch.py
```

The source-checkout audit requires the PyAuto libraries from Git checkouts.
The filename retains the runtime `autolens.__version__`; the actual Git revisions,
source/config hashes and dependency versions in the JSON are authoritative.
It is not a measurement of the historical release named by stale version metadata.

`--run` saves a version/date-stamped JSON and PNG in `results/lens/critical_curves/`.
Use a new `--output` filename for another experiment; existing evidence is never
overwritten. Workers are CPU fp64, have a 120-second process-group timeout,
and disable the persistent JAX compilation cache. Three calls per quantity
separate first-call and warm costs; caustic calls reuse curve compilation.
Automatic seeds and reference-derived explicit seeds are separate experiments.
No arguments print help; import smoke creates no evidence; summarize is read-only.

## CI boundary

The small numerical regression lives in
[autolens_workspace_test/scripts/cluster/critical_curves.py](https://github.com/PyAutoLabs/autolens_workspace_test/issues/337)
and its required PR smoke list. It checks both source planes against analytic
critical/caustic radii. The existing workspace zero-contour regression remains
another validation leg. Slow timing sweeps and failure investigations stay here.
Production follow-ups must promote fixed grid-cap, seed-coverage and path-completion
witnesses into the appropriate CI lane; a research result alone is not a guard.


## Recovering completed workers

`--collect <worker-directory> --run-log <original-run-log>` assembles saved worker
files after a post-processing repair. It checks their original script digest
against `results/lens/critical_curves/dispatch_measured_source.txt`, verifies
numerical function AST identity, and requires unchanged library/config/runtime
provenance. The output records collection separately. This is recovery for the
pinned birth experiment, not permission to mix workers from different runs.
Use a fresh `--output` path; existing results are never overwritten. The original
120-second timeouts remain timeouts even when they contain partial geometry.
