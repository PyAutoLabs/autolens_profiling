# quick_update

Fast incremental re-profiling helpers: one thin script per dataset class that
re-runs the headline likelihood measurement in seconds, for tight
edit-profile-edit loops while iterating on the source libraries.

These are the **scratch tier** of the repo:

- Outputs land in `results/quick_update/` as **unversioned** JSONs
  (`<cell>_quick_update_<instrument>.json`) that are overwritten on every run —
  no history, no dashboard row.
- Numbers here are for steering a source-code change, never for citing: when a
  result matters, re-run the real cell under
  [`likelihood_runtime/`](../likelihood_runtime/README.md) or
  [`likelihood_breakdown/`](../likelihood_breakdown/README.md).

## Running

```bash
python quick_update/imaging.py
python quick_update/interferometer_delaunay.py
```
