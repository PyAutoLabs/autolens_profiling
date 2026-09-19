# HST GPU residue phase 2 — current `jit(vmap)` measurement

## Verdict

**Inconclusive at the required numerical gate; do not change the production batching policy from
this grid.** The exact current PyAutoFit composition, `jax.jit(jax.vmap(fn))`, was measured on the
A100 against B scalar `jax.jit(fn)` evaluations of the same distinct parameter vectors. With the
production fallback enabled it was slower per completed likelihood at B=4, 8 and 16. The only
faster row disabled the fallback and is a diagnostic, not a production recommendation.

Three distinct-lane cells nevertheless failed the pre-declared three-way `1e-9` relative pin. All
production `jit(vmap)` values agree with the independently compiled vmapped library-PDIP reference;
the worst relative difference is `5.288e-10`. The scalar arm is the value outside the PDIP tolerance
on the rejected lanes, and `jit(vmap)` versus scalar consequently also fails. Those cells remain
useful timing evidence, but they are not accepted policy rows. The threshold is not relaxed after
seeing the data.

This fixed-N grid cannot establish an N-dependent crossover. It does not justify the phase-2b
batch-aware callback implementation, a PyAutoFit batching change, or a scalar-only production
policy. A follow-up would first need a separately approved numerical-reproducibility experiment
that localises the composition-dependent fp64 reduction difference without changing library
numerics.

## Experiment

- Job: RAL SLURM array `344635`, tasks 0–4, run 2026-09-19.
- Profiling source: `e2a51878588bb16d4a8619b8d920291bf5685ee0`.
- Device: NVIDIA A100 80GB PCIe; fp64; HST imaging; Delaunay; actual N=1500.
- Likelihood: fixed lens light, dense inversion, production border relocation, certified harness
  injection at budget 7. The certified solver is not shipped production code.
- Production arm: one warmed and synchronized `jax.jit(jax.vmap(fn))` call.
- Control arm: B warmed and synchronized scalar `jax.jit(fn)` calls over the same lane trees.
- Draws: seeded distinct mass-parameter vectors (`seed=0`); the JSON for each B records its draw
  hash. B16 identical is a control only.
- Timings below exclude compilation, lane preparation, pin evaluation, diagnostic callbacks and
  trace collection. Those costs remain separately recorded in each JSON.
- XLA compilation caches were fresh and had zero autotune entries at start. Direct post-run census
  found all five task cache directories empty.

Current library revisions were PyAutoNerves `1fa613aa`, PyAutoFit `a7368401`, PyAutoArray
`22e6d608`, PyAutoGalaxy `70a61e26`, and PyAutoLens `2aaa1c1a`. Imports resolved from the shared RAL
source checkouts; the full hashes and artifact checksums are in
`hst_gpu_residue_phase2_job344635.json`.

## Matched clean timing

| B | lanes | fallback | `jit(vmap)` ms/lane | scalar-jit ms/lane | scalar / vmap | three-way pin |
|---:|---|---|---:|---:|---:|---|
| 4 | distinct | on | 94.841 | 31.178 | 0.329x | PASS 4/4 |
| 8 | distinct | on | 60.932 | 31.204 | 0.512x | **FAIL 7/8** |
| 16 | distinct | on | 44.607 | 31.100 | 0.697x | **FAIL 13/16** |
| 16 | distinct | off | 24.946 | 31.029 | 1.244x | **FAIL 14/16** |
| 16 | identical control | on | 42.140 | 31.314 | 0.743x | PASS 16/16 |

`scalar / vmap` greater than one favours batching. The fallback-on B16 result is 43% slower per lane
than its matched scalar control. Disabling fallback makes B16 24% faster per lane, but that program
does not carry production fallback semantics. Every lane certified within budget 7, yet under vmap
the batched `lax.cond` becomes a select and both branches are evaluated; that structural cost is why
the fallback-off diagnostic must not be promoted as the production result.

The scalar controls, 31.03–31.31 ms, agree with the historical budget-7 single-call reference of
31.64 ms. The historical 25.39 ms value was budget 2 and is not the current control.

## Numerical gate

| cell | max vmap/scalar rel | max vmap/library-PDIP rel | max scalar/library-PDIP rel | certified |
|---|---:|---:|---:|---:|
| B4 distinct, fallback on | 6.477e-10 | 1.231e-10 | 6.397e-10 | 4/4 |
| B8 distinct, fallback on | **2.819e-9** | 2.107e-10 | **2.608e-9** | 8/8 |
| B16 distinct, fallback on | **2.764e-9** | 5.288e-10 | **2.235e-9** | 16/16 |
| B16 distinct, fallback off | **2.259e-9** | 2.648e-10 | **2.041e-9** | 16/16 |
| B16 identical, fallback on | 5.677e-10 | 1.656e-10 | 5.783e-10 | 16/16 |

The rejected seeded lanes are 7 for B8; 4, 7 and 10 for B16 fallback-on; and 7 and 10 for B16
fallback-off. Lane 4's direct vmap/scalar pin passes, but its scalar/library pin is `1.029e-9`.
Across the failures, the production batched value continues to agree with its library-PDIP reference.
Per-lane PDIP iteration counts remain unavailable without replacing library numerics; they are not
inferred here.

## Trace diagnostics

| B / mode | fallback | callbacks/lane | qhull ms/lane | tables ms/lane | vmap device-idle ms/lane |
|---|---|---:|---:|---:|---:|
| 4 distinct | on | 1 | 4.558 | 0.131 | 11.296 |
| 8 distinct | on | 1 | 4.500 | 0.125 | 7.859 |
| 16 distinct | on | 1 | 4.515 | 0.119 | 6.601 |
| 16 distinct | off | 1 | 4.486 | 0.120 | 5.427 |
| 16 identical | on | 1 | 4.551 | 0.120 | 6.483 |

The Delaunay callback remains sequential: every batched likelihood pays one callback per lane. Qhull,
not adjacency-table construction, dominates its host body. All traces have zero unjoined time. Wall
reconciliation passes on both arms of every task, with absolute error at most 0.160%; profiler
overhead versus the untraced wall stays below 5%. The stale source-line HLO census is excluded.

## Provenance and historical separation

The five validated JSON/PNG pairs use the `jitvmap` filename prefix and are committed beside this
note. Array `343376` retains its original `vmap(jit)` filenames and provenance as historical evidence
only; none of its rows were relabelled. Array `344635` tasks 0–2 exited non-zero after writing their
artifacts because their numerical gate failed. The sourced `activate.sh` ERR trap then pre-empted the
submit footer, so their logs lack `AUTOTUNE_ENTRIES` and `CELL_EXIT` lines. The JSON start census and
direct post-run directory census both show zero cache entries. The submit now temporarily removes and
restores that inherited trap around the expected cell exit so future failed gates retain the footer.

No repeat array was submitted: unchanged composition and tolerance would only repeat a structural
gate failure. CPU Numba sparse results remain a distinct execution path and do not alter any timing or
interpretation above.
