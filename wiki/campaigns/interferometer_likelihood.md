# Interferometer likelihood campaign

**Status:** open
**Question:** Where do the interferometer likelihoods — MGE, and the Delaunay-1500 / rectangular meshes on the sparse (W~) path — spend their time on the A100 and on CPU, and which levers pay?
**Pre-registered rule:** not recorded as a single campaign rule; each breakdown ranks levers from committed JSONs, library changes must be bit-identical (or pinned to the Fit) with no CPU regression, and the mesh CPU phase evaluates the `interferometer_numba_nnz_per_source_max` gate against the measured numba/FFT crossover.
**Verdict:** MGE: the W~ route (lever 1) and the real-valued scatter (lever 3) shipped; mixed precision gives no gain. Mesh A100: certified solver −33 % at alma is opt-in only. Mesh CPU: `cached_property` F/D shipped; crossover nnz/col ≈ 66 Delaunay / ≈ 72 rectangular; the packaged gate stays 60 (retune to ~70 drafted, not urgent).
**Headline:** A100 alma MGE 939.6 ms (chunked; dense path OOMs asking 61.4 GiB) → 2.69 ms on the W~ route; jvla 24.2 s → 16.7 ms.
**Library PRs:** PyAutoArray#576 (+ PyAutoGalaxy#629, PyAutoLens#750), PyAutoArray#578 (released 2026.9.27.1); PyAutoArray#540, PyAutoArray#541, PyAutoArray#544, PyAutoArray#545 (released 2026.9.11.1; library speed-ups with no profiling note, see the 2026-09-27 journal entry); PyAutoArray#582 (merged, UNRELEASED).
**Profiling PRs:** #312, #313, #319, #324, #328, #333; phase 3 under issue #348 (PR not yet opened).
**Ledger:** [interferometer_likelihood_decision_matrix_2026_09.md](../../results/notes/interferometer_likelihood_decision_matrix_2026_09.md) (phase 4, the user-facing matrix), [interferometer_mge_breakdown_2026_09.md](../../results/notes/interferometer_mge_breakdown_2026_09.md), [interferometer_mesh_a100_breakdown_2026_09.md](../../results/notes/interferometer_mesh_a100_breakdown_2026_09.md), [interferometer_mesh_cpu_breakdown_2026_09.md](../../results/notes/interferometer_mesh_cpu_breakdown_2026_09.md); older context [numba_interferometer_verdict.md](../../results/notes/numba_interferometer_verdict.md).
**Mind contract:** epic `interferometer-likelihood-campaign` (named in the records; not in `epics.md`); phase map `draft/research/autolens_profiling/interferometer_mesh_breakdown_numba_cpu_decision_matrix.md`.
**Next:** phase 4 (decision matrix, issue #356) is written; ship it, then close the campaign. Any cells still marked pending in the matrix note are listed there with their RAL job ids.

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
| 3/3 p3 radius sweep | 2026-09-28 | A100 fp64 rows at mask r2.0 / r5.0 beside the r3.5 baseline | not recorded (witness: sparse class, step sum within 10 %) | F follows the extent, ~0.8–1.2 µs per extent pixel; alma_high r5.0 190.87 / 198.55 ms, no sparse OOM | 366895, 366896 (tasks 2–5), 366907, 366908 | issue #348 (PR pending) |
| 3/3 p4 decision matrix | 2026-09-30 | Which interferometer likelihood path and device, by N_vis × mask × source? | witness: numba arm on `InversionInterferometerSparseNumba`, numba vs FFT ≤ 0.5 nats, step sum within 10 %, CPU vs A100 ≤ 1e-3 nats | matrix + 5-rule draft; per-call cost follows masked pixels, not N_vis; A100 W~ wins 8–100× on meshes; MGE always W~ (see note for cells still pending) | 375977–375984 | issue #356 (PR pending) |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#576 | W~ route for MGE-only interferometer fits (closes issue PyAutoArray#575) | `1bf641e4` | 2026.9.27.1 |
| PyAutoGalaxy#629 | W~ route plumbing (PyAutoGalaxy side) | `da84468a` | 2026.9.27.1 |
| PyAutoLens#750 | W~ route plumbing (PyAutoLens side) | `e58715e9` | 2026.9.27.1 |
| PyAutoArray#578 | real scatter in `transform_mapping_matrix` (closes issue PyAutoArray#577) | `14d63360` | 2026.9.27.1 |
| PyAutoArray#582 | `cached_property` F / D on the interferometer sparse inversions (issue PyAutoArray#581) | `e281abf3` | UNRELEASED |
| PyAutoArray#540 | `apply_operator` via exact `rfft2`/`irfft2` (issue PyAutoArray#538) | `7a4cb700` | 2026.9.11.1 |
| PyAutoArray#541 | preload built as a type-1 NUFFT (issue PyAutoArray#539) | `9bd76799` | 2026.9.11.1 |
| PyAutoArray#544 | NumPy/scipy sparse-operator path, no JAX on `xp=np` (issue PyAutoArray#542) | `39d3024c` | 2026.9.11.1 |
| PyAutoArray#545 | numba `direct_conv` curvature path + nnz/col gate 60 (issue PyAutoArray#543) | `35aa681f` | 2026.9.11.1 |

`PyAutoArray#582` cut every NumPy full call to 0.51–0.88x of its phase-1 value: phase-1 rows
computed F twice and D four times per `figure_of_merit` (ledger, production solver settings); the
library's evaluation-count tests went red at F=2, D=2 (positive-negative solver of the test
config) and green at 1, 1. Both counts are right; see Caveats.

## Open / parked / drafts

- Mesh phase 4 (decision matrix, issue #356): in flight on `feature/interferometer-decision-matrix`; prompt `draft/research/autolens_profiling/interferometer_mesh_breakdown_numba_cpu_decision_matrix.md`.
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
- **F/D evaluation counts before PyAutoArray#582 depend on the solver branch, not the source.** F = 2 everywhere (the cached `curvature_reg_matrix` reads F once, `fast_chi_squared` once). D = 4 per full `figure_of_merit` on the profiling rows, which run the packaged config (`use_positive_only_solver: true`, `use_edge_zeroed_pixels: true`): the pre-fix edge-zeroed fnnls branch of `reconstruction` reads `data_vector` three times (warm-start fingerprint, `[ids_to_keep]`, `.shape`) plus once in `fast_chi_squared`. D = 2 is the unit-test scenario: the #582 tests build the inversion under `test_autoarray/config/general.yaml` (`use_positive_only_solver: false`), where the positive-negative branch reads D once plus `fast_chi_squared` once. Source: PyAutoArray `879b2be1` (parent of `e281abf3`), `inversion/abstract.py` and `interferometer/abstract.py`, and the two config files. The ledger's 2 / 4 is the number that sets the measured speed-up; the record's 2 / 2 is the test's red state.
- **The 2026-09-08 library speed-ups have no RAL measurement of their own.** Their gains come from laptop or unstated-host probes (journal, 2026-09-27); the #540 `rfft2` path was never confirmed on an A100 (waived at merge).

## Journal

### 2026-09-27 — page created from the ledger

Page created from the ledger; see the ledger for the full record.

### 2026-09-27 — library speed-ups with no profiling note

Interferometer library changes that shipped with a Mind record but no results/notes ledger of their own (the first four fell out of the [numba revisit](numba_interferometer_revisit.md) verdict).

- **PyAutoArray#540** (issue #538): `apply_operator` uses exact `rfft2`/`irfft2` instead of complex `fft2` on real input; pinned to the old route at 2.8e-14. Gain 1.27–1.61x less FFT work, laptop i9-10885H (numba revisit bake-off); A100 not measured. Merge `7a4cb700`, released 2026.9.11.1. `complete/2026/09/interferometer-apply-operator-rfft2.md`.
- **PyAutoArray#541** (issue #539): `nufft_precision_operator_from` builds the preload as a type-1 NUFFT, `eps=1e-12` default, brute-force builders kept as references. Gain alma 2101 s → 7.3 s (289x wall, 111x CPU-s), host not stated in the record (measured in autolens_profiling#229 / PR #234). Merge `9bd76799`, released 2026.9.11.1. `complete/2026/09/interferometer-preload-nufft-type1.md`.
- **Profiling-side origin of #541**: autolens_profiling#229 / PR #234 (phase 3 of the numba revisit) is a profiling task, not a library PR; it measured the builders and filed the #541 prompt. Its write-up is section 7 of [numba_interferometer_verdict.md](../../results/notes/numba_interferometer_verdict.md). `complete/2026/09/interferometer-preload-cpu.md`.
- **PyAutoArray#544** (issue #542): `InterferometerSparseOperator` takes `xp`; a NumPy fit uses `scipy.fft` / `scipy.sparse` and never imports JAX. Gain 3.8x over the JAX-CPU route on a 40x40 / S=400 probe, host not recorded. Merge `39d3024c`, released 2026.9.11.1. `complete/2026/09/interferometer-sparse-operator-numpy-cpu-path.md`.
- **PyAutoArray#545** (issue #543): new `interferometer_numba/` package, `InversionInterferometerSparseNumba` on the `direct_conv` kernel, routed when mean nnz/col ≤ `interferometer_numba_nnz_per_source_max` (60); changed the default `xp=np` route. No measurement recorded for the library PR (it cites the prototype's 2–7x); the in-situ check is this page's mesh CPU phases 1–2. Merge `35aa681f`, released 2026.9.11.1. `complete/2026/09/interferometer-numba-cpu-direct-conv.md`.
- **PyAutoArray#582** (issue #581): `data_vector`, `curvature_matrix`, `curvature_matrix_diag` `@cached_property` on the sparse, numba and mapping interferometer inversions. Gain laptop sma Delaunay numba 492 → 343 ms, NumPy FFT 1783 → 923 ms (host not recorded); the RAL after-measurement (0.51–0.88x) is in the mesh CPU ledger. Merge `e281abf3`, UNRELEASED. `complete/2026/09/interferometer-sparse-cache.md`.

### 2026-09-27 — PyAutoArray#582 evaluation counts settled

The ledger's "F twice, D four times" and the Mind record's "red at F=2, D=2" describe different runs: the full `figure_of_merit` under the packaged positive-only + edge-zeroed solver (profiling rows) versus the #582 unit tests under the test config's positive-negative solver. Read from the pre-fix source at PyAutoArray `879b2be1` and the test diff of `e281abf3`; both counts reconcile, and both are 1 / 1 after the fix. Also filled: PyAutoGalaxy#629 and PyAutoLens#750 released 2026.9.27.1 (merges `da84468a`, `e58715e9`).

### 2026-09-28 — phase 3: A100 mask-radius sweep

Issue #348. There are 12 new A100 fp64 rows (sma / alma / alma_high × Delaunay-1500 / rect 39² × mask r2.0 / r5.0), on RAL jobs 366895 and 366896 (tasks 2–5) plus the sma resubmits 366907 and 366908. The r3.5 column is the committed #324 baseline on the older mirror revisions; the ledger states the mix. The full record is in [the ledger section](../../results/notes/interferometer_mesh_a100_breakdown_2026_09.md#mask-radius-sweep--a100-fp64-phase-3-348).

- The radius moves the cost through F alone. From r2.0 to r5.0 (6.25× the extent area), F grows 4.9–7.8× while the solve stays at 17.6–26.6 ms. The full JIT goes alma Delaunay 39.05 → 70.24 ms and alma_high Delaunay 54.10 → 190.87 ms (rect 48.87 → 198.55 ms).
- F costs ~0.8–1.2 µs per extent pixel at every instrument. alma r5.0 (200², 1M visibilities, F 37.68 ms) costs more than alma_high r2.0 (160², 5M visibilities, F 20.74 ms).
- No sparse leg OOMs, including alma_high r5.0 at vmap b16. alma dense OOMs at every radius, because T is set by N_vis × S. Step sum / full JIT is 1.002–1.051.
- For phase 4: only alma has RAL CPU radius rows (r2.0 / 4.25 / 5.0 / 6.0). sma and alma_high CPU rows at r2.0 / r5.0 must be added before the CPU-vs-A100 witness.

### 2026-09-30 — phase 4: the decision matrix

Issue #356. The [decision-matrix note](../../results/notes/interferometer_likelihood_decision_matrix_2026_09.md) puts every committed interferometer row on one grid: N_vis (sma 190 / sdp81 1.08e5 / alma 1e6 / alma_high 5e6 / jvla 2.5e7) × mask r2.0 / 3.5 / 5.0 × source (MGE-20, Delaunay-1500, rect 39²) × path (CPU numba, CPU NumPy FFT, CPU JAX, A100 W~ sparse, A100 dense). Every cell is filled or listed as blocked / not measured with a citation.

- New: RAL CPU radius-gap cells at sma and alma_high × r2.0 / r5.0 (arrays 375977 Delaunay, 375978 rect) and an **sdp81** preset — the real SDP.81 uv coverage (108,384 visibilities) on the alma grid with the simulated lens — with CPU (375979 / 375980) and A100 (375981–375984: Delaunay, rect, MGE dense, MGE W~) rows.
- The new cells ran on the library mains from a private RAL clone (`PYAUTO_LIB_BASE`), because the shared mirror was behind main and in use by another campaign.
- Headline: per-call cost follows the masked-pixel count, not N_vis (alma r5.0 at 1M visibilities costs more than alma_high r2.0 at 5M, on CPU and A100); the packaged numba gate of 60 routes the measured CPU cells to the faster arm; the A100 W~ path is 8–100× faster than the best single-thread CPU arm on meshes.
- One pre-existing cell misses the new CPU-vs-A100 1e-3-nat bar: alma_high r3.5 rect, 1.5e-3 nats (PDIP vs fnnls on the edge-zeroed subset; 330× inside the 0.5-nat evidence bar).

