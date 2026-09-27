# Interferometer likelihood campaign

**Status:** open
**Question:** Where do the interferometer likelihoods — MGE, and the Delaunay-1500 / rectangular meshes on the sparse (W~) path — spend their time on the A100 and on CPU, and which levers pay?
**Pre-registered rule:** not recorded as a single campaign rule; each breakdown ranks levers from committed JSONs, library changes must be bit-identical (or pinned to the Fit) with no CPU regression, and the mesh CPU phase evaluates the `interferometer_numba_nnz_per_source_max` gate against the measured numba/FFT crossover.
**Verdict:** MGE: the W~ route (lever 1) and the real-valued scatter (lever 3) shipped; mixed precision gives no gain. Mesh A100: certified solver −33 % at alma is opt-in only. Mesh CPU: `cached_property` F/D shipped; crossover nnz/col ≈ 66 Delaunay / ≈ 72 rectangular; the packaged gate stays 60 (retune to ~70 drafted, not urgent).
**Headline:** A100 alma MGE 939.6 ms (chunked; dense path OOMs asking 61.4 GiB) → 2.69 ms on the W~ route; jvla 24.2 s → 16.7 ms.
**Library PRs:** PyAutoArray#576 (+ PyAutoGalaxy#629, PyAutoLens#750), PyAutoArray#578 (PyAutoArray PRs released 2026.9.27.1; #629/#750 release not verified); PyAutoArray#582 (merged, UNRELEASED).
**Profiling PRs:** #312, #313, #319, #324, #328, #333.
**Ledger:** [interferometer_mge_breakdown_2026_09.md](../../results/notes/interferometer_mge_breakdown_2026_09.md), [interferometer_mesh_a100_breakdown_2026_09.md](../../results/notes/interferometer_mesh_a100_breakdown_2026_09.md), [interferometer_mesh_cpu_breakdown_2026_09.md](../../results/notes/interferometer_mesh_cpu_breakdown_2026_09.md); older context [numba_interferometer_verdict.md](../../results/notes/numba_interferometer_verdict.md).
**Mind contract:** epic `interferometer-likelihood-campaign` (named in the records; not in `epics.md`); phase map `draft/research/autolens_profiling/interferometer_mesh_breakdown_numba_cpu_decision_matrix.md`.
**Next:** mesh CPU phase 3 (A100 radius sweep) and phase 4 (decision matrix).

## Why this campaign

Filed as a three-task plan: (1) an MGE breakdown on JAX CPU and A100 with an optimisation task
list, (2) the Delaunay-1500 and rectangular mesh breakdown on the A100 through the sparse operator,
(3) the mesh numba-CPU decision matrix. The MGE record's "why" for lever 1: an MGE-only
interferometer fit always took the O(N_vis · n) dense NUFFT path even after
`apply_sparse_operator()`, because the inversion factory disabled the sparse operator when every
linear object is a func list — although the func-list W~ blocks already existed. Lever 3's why: a
complex128 slim→native scatter of the mapping matrix cost ~0.85 s per call on the A100 regardless
of N_vis, and was "the whole reason the A100 is ~10x slower than the CPU at sma". Task 3 exists
because the numba breakdown cells timed the prototype pack rather than the library dispatch that
has routed numba vs FFT by nnz/col since PyAutoArray #544/#545.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1/3 MGE breakdown | 2026-09-25/26 | Where does the MGE interferometer likelihood spend time on JAX CPU and A100? | not recorded (ranked levers) | levers ranked: W~ route, chunked transform, real scatter; mixed precision no gain | r3 351078–351083 (first round 351055–351057 invalid) | #312 |
| Lever 1: W~ route | 2026-09-26 | Route MGE-only fits through the W~ blocks | not recorded; witness + Fit agreement | alma 939.6 ms → 2.69 ms, jvla 24.2 s → 16.7 ms (A100) | not recorded here (see ledger) | PyAutoArray#576, PyAutoGalaxy#629, PyAutoLens#750, #313 |
| Lever 3: real scatter | 2026-09-26 | Scatter real then cast, instead of a complex128 scatter | bit-identical output (exact-equality test); no CPU regression (ABBA) | sma dense full pipeline 868.5 → 3.39 ms (A100) | 356364 | PyAutoArray#578, #319 |
| 2/3 mesh A100 | 2026-09-26/27 | Delaunay-1500 + rectangular on the sparse path on the A100 | not recorded (ranked levers; mp bar 0.5 nats) | certified solver −33 % at alma Delaunay (opt-in); F is FFT-bound; fixed-mapper curvature preload | 18 A100 jobs 356370–356387 | #324 |
| 3/3 p1 mesh CPU | 2026-09-27 | numba `direct_conv` vs NumPy FFT through the library dispatch | not recorded | RAL CPU r3.5 numba/FFT ratios 0.25 (sma) … 2.05 (alma_high rect) | not recorded here (see ledger) | #328 |
| 3/3 p2 crossover | 2026-09-27 | In-situ crossover on the cached library; gate verdict | gate evaluated against the interpolated crossover | crossover nnz/col ≈ 66 Delaunay / ≈ 72 rect; gate stays 60, retune to ~70 drafted | 358985, 358986, 359000 | #333 (library: PyAutoArray#582) |
| 3/3 p3 radius sweep | — | A100 radius sweep | — | not started | — | — |
| 3/3 p4 decision matrix | — | numba vs FFT decision matrix | — | not started | — | — |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#576 | W~ route for MGE-only interferometer fits (closes issue PyAutoArray#575) | `1bf641e4` | 2026.9.27.1 |
| PyAutoGalaxy#629, PyAutoLens#750 | W~ route plumbing | not recorded | not verified |
| PyAutoArray#578 | real scatter in `transform_mapping_matrix` (closes issue PyAutoArray#577) | `14d63360` | 2026.9.27.1 |
| PyAutoArray#582 | `cached_property` F / D on the interferometer sparse inversions | `e281abf3` | UNRELEASED |

`PyAutoArray#582` cut every NumPy full call to 0.51–0.88x of its phase-1 value: phase-1 rows
computed F twice and D four times per call (ledger); the library's evaluation-count test went red
on unfixed source (F=2, D=2) and green after the fix.

## Open / parked / drafts

- Mesh CPU phase 3 (A100 radius sweep) and phase 4 (decision matrix): `draft/research/autolens_profiling/interferometer_mesh_breakdown_numba_cpu_decision_matrix.md`.
- `draft/research/autolens_profiling/interferometer_nnls_memo_scattered_stream_guard.md` — fnnls warm-start memo guard.
- `draft/research/autolens_profiling/interferometer_w_tilde_fft_size_levers.md` — F extent / FFT size.
- `draft/research/autolens_profiling/interferometer_fixed_mapper_curvature_preload.md` — fixed-mapper curvature preload.
- Certified-solver default on the interferometer sparse path went into the C2 amendment `draft/feature/autofit/certified_solver_batched_guard_c2.md`.

## Caveats

- **A round was thrown away**: RAL jobs 351055–351057 ran on nufftax 0.4.0 (below the 0.6.1 floor), which routed x64 GPU NUFFTs to fp32 Pallas (0.25 nats off at alma_high). Those JSONs are invalid and not committed; every A100 number is from the 0.6.1 re-run.
- **Above sma the A100 MGE steps are not the library path**: a single `jax.jit(FitInterferometer)` OOMs at alma and above, so steps there run on the measurement-only chunked-transform arm (agrees with the laptop fit to 1.5e-8 nats at alma).
- **Mesh A100 CPU rows** (laptop, one core, loaded) are reference only.
- **Gate move 60 → 70 rests on interpolation** inside measured brackets, not on a misrouted measured cell; machine-dependent.
- Older numba interferometer revisit found machine drift up to 2.5x between runs ([verdict](../../results/notes/numba_interferometer_verdict.md)).

## Journal

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.
