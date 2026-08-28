# NNLS cross-evaluation warm-start memo: should it default on? (2026-08-28)

Measurement leg of [PyAutoArray#498](https://github.com/PyAutoLabs/PyAutoArray/issues/498)
phase 3a — the default for `aa.Settings(nnls_warm_start_memo=...)`, which seeds the numba-CPU
fnnls active-set loop from the **previous evaluation's** final passive set instead of the sign
of the unconstrained dense solve. Script:
`scripts/imaging/likelihood_breakdown/delaunay_numba_nnls_iterations.py`; results:
`results/breakdown/imaging/delaunay_numba_nnls_iterations_{euclid,hst}_v2026.8.17.1.{json,png}`.
Laptop CPU fp64, `OMP_NUM_THREADS=1`, `AUTOARRAY_NUMBA_OPERATED_MEMO=0`, v2026.8.17.1, Delaunay
Hilbert-1250 + MGE-60 linear lens light, sparse-operator numba path, n = 1310.

## What was measured

The memo can only be judged across *different* parameter points — refitting one fixed instance
hands it a 100%-correct seed. Two 30-instance sequences, each run twice (memo OFF then ON, memo
cleared between): **random walk** (seeded, each step N(0, 1% of each parameter's prior width) —
the sampler-like regime) and **iid** (independent draws, unit values uniform in the central 20%
of every prior — the pessimistic, uncorrelated case). `fnnls_cholesky` is wrapped at module
level so every solve's `stats` is captured from inside a real `FitImaging`. Medians over 30.

## Results

| instrument | sequence | total iters OFF→ON | ws errors OFF→ON | solve s OFF→ON | eval s OFF→ON | iter ratio |
|---|---|---|---|---|---|---|
| euclid | random walk | 69.5 → **7.0** | 73.5 → 9.5 | 0.544 → **0.051** | 1.510 → **0.564** | **9.9×** |
| euclid | iid | 86.0 → 94.0 | 94.5 → 98.0 | 0.347 → 0.309 | 0.906 → 0.838 | 0.91× |
| hst | random walk | 32.0 → **8.0** | 31.0 → 9.5 | 0.245 → **0.067** | 1.733 → **1.594** | **4.0×** |
| hst | iid | 42.0 → 69.5 | 48.0 → 68.0 | 0.268 → 0.260 | 1.687 → 1.867 | 0.60× |

**Parity.** Max relative Δlog-likelihood 2.98e-14 (gate rtol 1e-6); max |Δreconstruction| 1.6e-8.
**Retries.** 0 memo-seed failures in 240 evaluations — the fallback in
`reconstruction_positive_only_from` never fired.

**Step-0 baseline** (`delaunay_numba.py`, same env, post-#453/#497): euclid reconstruction solve
0.470 s of 1.189 s step-total (40%); hst 0.542 s of 3.223 s (17%) — no longer the ~70% it was
when #498 was written (previously recorded euclid: 3.615 s of 4.839 s). Pins PASSED both.

**Model robustness.** This measured ONE fiducial. The same A/B across 11 model variants and
both instruments is in [`nnls_warm_start_memo_matrix.md`](./nnls_warm_start_memo_matrix.md).

## Verdict

**Gate met: `nnls_warm_start_memo` defaults to `true`.** The random-walk median iteration
reduction is 9.9× (euclid) and 4.0× (hst), both ≥ 2×, with zero pin drift — the reconstruction is
identical to round-off and the pinned evidence values are unchanged.

The **iid column** costs *iterations*, not time. Uncorrelated jumps make the seed a worse guess at
the passive set (0.91× euclid, 0.60× hst), but solve seconds are ≤ memo-off in all four
sequence×instrument cells: the memo also skips the dense unconstrained `np.linalg.solve` that
produces the sign start, and that saving covers the extra active-set moves. The one adverse
eval-level number, hst iid +0.18 s, cannot come out of a 0.26 s solve that got no slower — it is
noise in the other steps. So the default is `true` with no time penalty in the measured worst
case; iid stays the regime to revisit if a search class proves uncorrelated throughout.

**Shipped default:** `nnls_warm_start_memo: true` in `autoarray/config/general.yaml`, with the
`KeyError` fallback in `Settings.nnls_warm_start_memo` returning `True` for workspaces whose own
`general.yaml` shadows autoarray's without the key.

## Fallback guard

`nnls_warm_start_error_tolerance` (config `inversion.nnls_warm_start_error_tolerance: 1.5`;
`inf`, `0` or a negative value disables it) makes the memo self-calibrating: each entry remembers
the error fraction of the most recent **dense-sign-started** solve for its key, and a seeded solve
whose own fraction exceeds `1.5 ×` that reference has its entry dropped — the next solve for that
key restarts dense and refreshes the reference. The rule is *relative* because the absolute
fraction does not separate helpful from harmful seeds (0.048–0.138 overlap) while the seed/dense
ratio does (helpful ≤ 0.89, worst 1.42), and 1.5 sits above that worst measured ratio, so the
guard is protective against unmeasured regimes rather than flapping inside the matrix.

Re-run with the guard on (`--n-instances 20`, same env, so the fiducial JSONs/PNGs on disk are now
these 20-instance cells, not the 30-instance ones tabulated above): euclid fiducial 10.00×/0.91×,
hst fiducial 4.27×/**0.62×**, hst source_complex 3.39×/0.73× (rw/iid). It fired **3 times in 240
evaluations**, all in `hst/fiducial/iid` — the one cell whose seed was worse than the dense-sign
start. No cell's ratio, solve seconds or parity moved beyond sampling noise (parity ≤ 9.5e-15).

## Phase 3b (batched active-set moves)

The residual iteration counts on the warm path are already 7–8 total (5–6 outer), so batching
outer active-set moves has almost nothing left to remove there. It retains value only for the
**cold / uncorrelated** path — 30–95 outer iterations on the first evaluation and the iid
sequence — which is also exactly the regime the memo does not help. That, not the warm path, is
3b's case.

## Relation to the JAX solver ledger

[`nnls_solver_ledger.md`](./nnls_solver_ledger.md) closed the *JAX PDIP* solver, where warm
starting from the unconstrained solve **hurt** (17 → 38 iterations) by un-centering the interior
point. That does not transfer: this is the numpy/numba Bro & De Jong active-set solver, where a
warm start is a discrete guess at the passive set and a wrong guess costs iterations rather than
degrading the iterate. The two solvers stay separately tuned.
