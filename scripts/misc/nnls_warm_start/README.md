# NNLS cross-evaluation warm-start memo

This package holds the **memo experiment**, not a profiling baseline. It A/Bs
`aa.Settings(nnls_warm_start_memo=...)` ([PyAutoArray#498](https://github.com/PyAutoLabs/PyAutoArray/issues/498)),
which seeds the numba-CPU fnnls active-set loop from the *previous evaluation's* final
passive set instead of from the sign of the unconstrained dense solve.

- `delaunay_numba_nnls_iterations.py` — the A/B cell. Runs two seeded instance sequences
  (`rw` = random walk, `iid` = independent draws from the central 20 % of every prior)
  with the memo OFF then ON, recording active-set iterations, solve time, evaluation time
  and memo-vs-no-memo parity. `--model <name>` selects one of `MODEL_VARIANTS`, each
  changing exactly one thing about the fiducial.
- `nnls_iterations_matrix.py` — the aggregator. Reads every JSON the cell wrote and
  renders `results/nnls_warm_start/nnls_warm_start_memo_matrix.md`.

Findings: [`nnls_warm_start_memo.md`](../../../results/nnls_warm_start/nnls_warm_start_memo.md)
(the single fiducial) and
[`nnls_warm_start_memo_matrix.md`](../../../results/nnls_warm_start/nnls_warm_start_memo_matrix.md)
(11 model variants × 2 instruments).

## Why this is not the production baseline

The memo's headline gains are measured on a **random walk** — consecutive evaluations
that differ by ~1 % of each prior's width, so the previous passive set is a near-perfect
seed. A Nautilus pool does not deliver that stream to any one worker: the `iid` column of
the same matrix is where the memo's advantage largely disappears (euclid `rw` 9.9× vs
`iid` 0.91×). The default profiling cells therefore run the memo **off** and record the
flag, so no headline row is a memo-self-seeded number; production may re-enable it if the
sampler ever hands a worker a correlated stream.

The library default (`nnls_warm_start_memo: true` in `autoarray/config/general.yaml`) is
unchanged by the move — this package is the evidence that set it, kept runnable.

## Re-running

From the repository root:

```bash
python scripts/misc/nnls_warm_start/delaunay_numba_nnls_iterations.py --instrument euclid
python scripts/misc/nnls_warm_start/delaunay_numba_nnls_iterations.py --instrument euclid --model powerlaw
python scripts/misc/nnls_warm_start/nnls_iterations_matrix.py
```

The cell writes `results/nnls_warm_start/delaunay_numba_nnls_iterations_<instrument>[_<model>]_v<version>.{json,png}`;
the aggregator rewrites the matrix note in the same directory. Neither enters the
auto-generated README tables (`scripts/misc/tooling/build_readme.py` matches
`_summary_` / `_breakdown_` basenames only), which is deliberate.
