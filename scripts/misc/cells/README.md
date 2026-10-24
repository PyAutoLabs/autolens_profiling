# cells

Tests for profiling cells that live outside `scripts/misc/` and have no helper
folder here: the cluster point-solver scripts in `scripts/cluster/image_plane/`
and the solver scripts in `scripts/lens/solver/`.

There is no code in this folder, only `test/`. It writes no results. Run the
tests from the repo root:

```bash
pytest scripts/misc/cells/test -q
```

Tests for a cell that uses a helper folder here sit with that folder instead,
for example the fixed-lens-light cells' tests in
[`likelihood_breakdown/test/`](../likelihood_breakdown/README.md).
