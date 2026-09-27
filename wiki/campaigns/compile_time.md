# JAX compile time

**Status:** complete
**Question:** Is JAX compile time a problem for the endorsed multi-start gradient searches (`af.MultiStartProdigy`) across MGE and the pixelized meshes, and if so can every cell be made to compile fast?
**Pre-registered rule:** Phase B (the PyAutoFit Python-loop batching fix) runs only if Phase A's census confirms the `lax.map` compile lever, and must keep `batch_size` semantics and results identical (issue #93 plan); for the multi-band fix, keep the old scan behind an opt-in only if GPU throughput regressed (PyAutoFit#1430).
**Verdict:** single-band compile is a non-problem on every endorsed model type, so Phase B was cancelled on the evidence; the multi-band `lax.map` blow-up is real, CPU-backend-specific, and was fixed by the Python-loop batching now shipped in `MultiStartGradient` (PyAutoFit#1431); the compile axis then moved from speed-up to regression surveillance (warm pins, `--axis compile`).
**Headline:** worst single-band cold compile ~77 s (MGE `laxmap_vag`, 14 s trace + 63 s compile; the ledger rounds to ~75 s), warm 1–2 s outside the Delaunay family, RAL 32-core CPU job 331379; laptop 1-core worst-case tier ~3.5 min.
**Library PRs:** PyAutoFit#1431 (issue #1430; released 2026.8.4.1).
**Profiling PRs:** #94 (census, issue #93), #95 (multi-band productization), #104 (warm-compile pins + dashboard, issue #103).
**Ledger:** [multistart_prodigy_compile_census.md](../../results/notes/multistart_prodigy_compile_census.md), [multiband_pyloop_productized.md](../../results/notes/multiband_pyloop_productized.md); instrument tables in `scripts/misc/jax_compile/README.md`.
**Mind contract:** no epic; prompts in the `## Original prompt` of `complete/2026/07/multistart-prodigy-compile.md` and `complete/2026/07/multiband-pyloop-batching.md`; surveillance arc `complete/2026/08/profiling-agent-compile-axis-arc.md` (phases `compile-axis-campaign-coverage.md`, `compile-warm-baseline-dashboard.md`, `compile-axis-triage-drift.md`); absorbed prompt `complete/2026/08/jax-compile-time-profiling-absorbed.md`.
**Next:** none for this campaign; the one defect it found is `draft/research/autoarray/delaunay_callback_persistent_cache_miss.md`.

## Why this campaign

The earlier compile arc (issues #71 → #74 → #77) had closed the single-start question with
"settings suffice": persistent compilation cache (CPU MGE `vag` 117.0 s → 2.3 s) plus
`--xla_gpu_autotune_level=0` (A100 FD probe 498 s → 29 s). It never probed the multi-start
production transform or the pixelized meshes. On 2026-07-28 the human asked for exactly that,
because `MultiStartProdigy` had become the endorsed search for MGE and mesh sources and a
multi-band experiment had found its in-XLA `jax.lax.map(value_and_grad, batch_size=)` scan
compile-intractable (OOM-killed, >55 min) on a 1-core host. Issue #93 split the work into a
measure phase and a PyAutoFit fix phase gated on it; the Feature Agent's 4-phase split was
overridden to that 2-phase shape.

The census found the blow-up needs a multi-band `FactorGraphModel` body, so the fix moved to its own
task (PyAutoFit#1430, the COSMOS-Web Ring multi-band case), which shipped Python-loop batching and a
jitted broad-start filter. On 2026-08-10 the remaining compile-time prompt was re-scoped from
speed-up to regression surveillance, because both shipped wins are settings that a config drift or
a `jax` bump can undo silently.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| A: single-band census | 2026-07-28/29 | Cold + warm compile of {mge, pixelization, knn, delaunay_matern} × {jit, vag, vmap_vag 16, pyloop_vag 4, laxmap_vag 4} at production knobs | measure first; settings-first discipline before touching source | benign everywhere: laptop worst ~3.5 min cold (MGE `vmap_vag`), meshes 35–75 s; RAL ≤ ~75 s cold; Delaunay never hits the persistent cache; rect `batch_size=4` load-bearing for memory (~9.2 GB per start) | laptop, 331379 | #94 |
| A: A100 tier | 2026-07-28/29 | Confirm on A100 | confirmatory only | not obtained: A100s saturated, JAX fell back to CPU; rows discarded | 331380 (discarded) | none |
| B: PyAutoFit pyloop (single-band) | 2026-07-29 | Hoist the batching out of XLA | only if Phase A confirms the lever | cancelled on the evidence; no PyAutoFit branch created | none | none |
| Multi-band productization | 2026-07-30 | Ship pyloop batching for the multi-band `FactorGraphModel` fit | results identical; keep scan opt-in only if GPU throughput regressed | shipped: 4-band cold fit intractable → 395.5 s CPU / 392.3 s GPU, warm 136.2 s / 198.5 s; `best_fom` bit-identical; scan explosion CPU-backend-specific, so no fallback kept | laptop (CPU + RTX 2060) | PyAutoFit#1431, #95 |
| Compile axis 1: coverage | 2026-08-10 | Can the Profiling Agent see the compile corpus? | not recorded | `campaign --axis compile`: 2 of 21 grid cells covered (`hst` only) | none | PyAutoBrain#219 |
| Compile axis 2: warm pins | 2026-08-10 | Make warm rows identifiable, pinned, dashboarded | comparability key `(hardware, hostname, jax_version, mixed_precision, cache_state)` | 25 sticky warm pins committed; `cache_state` derived from cache entries; `ingest --axis compile` | none | #104, PyAutoBrain#220 |
| Compile axis 3: triage | 2026-08-10 | Separate cache regression from host load and bookkeeping | not recorded | `triage --axis compile`, seven classes (three actionable, `host-load` added); arc closed | none | PyAutoBrain#222 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoFit#1431 (issue #1430) | `MultiStartGradient` batching as a Python sweep over `jit(vmap(value_and_grad))` chunks; jitted broad-start filter | `f5bdc7c6` | 2026.8.4.1 |

The persistent cache (PyAutoNerves#128) and autotune-off predate this page's window and
belong to the #71 → #77 arc; they are not listed here.

## Open / parked / drafts

- `draft/research/autoarray/delaunay_callback_persistent_cache_miss.md` — the Delaunay qhull `pure_callback` cache miss (~40–65 s trace + compile per process), not started.
- The multi-band confirmatory legs (A100 / multi-core / hetero GPU rows, filed as `multiband_compile_census_completion.md`) were moved to the Cortex and dropped there on 2026-09-04 (PyAutoCortex ruling R-20260904-05): the compile-axis verdict stands and the jax_compile README is the record.
- The single-band A100 tier was never obtained; it is confirmatory only and has no live draft.

## Caveats

- **The headline host matters.** Compile runs on host CPU cores, so the 1-core laptop tier is a deliberate worst case; 32 RAL cores buy 2–6×. Historical loaded-machine numbers were wrong by up to 7×; compare rows only within the comparability key.
- **"≤ 2 s warm on RAL" excludes Delaunay.** The census ledger's TL;DR says every RAL cell warms to ≤ 2 s, but its own finding 2 and the README table show `delaunay_matern` warm = cold (16.3 s `vag`, 21.2 s `laxmap_vag`) on job 331379 because of the cache miss.
- **Job 331380 is not an A100 result.** It ran on 8 CPU cores after `cuInit` failed; its rows were discarded, not committed.
- **Multi-band productization rows are laptop-only** (1-core WSL, RTX 2060 Max-Q), with no RAL job; warm time there is almost entirely re-tracing on one core.
- **The 2026-07-15 "autotuning ruled out" A/B was wrong** (it never flipped the flag); see the absorbed prompt record. Do not cite it.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the single-band census (#93 / #94), the multi-band pyloop fix and the 2026-08-10
compile-axis surveillance arc. Resolved: the library change is PR PyAutoFit#1431 (merge `f5bdc7c6`,
released 2026.8.4.1); #1430 is its issue. Profiling PRs #94 / #95 / #104 and the Mind contract
filled from the records. The "A100 tier outstanding" next step is closed: confirmatory only, and the
multi-band legs were dropped 2026-09-04. Open: the Delaunay cache-miss draft.
